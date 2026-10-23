import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.attention import Attention
from nanovllm.layers.linear_attention import LinearAttention
from nanovllm.layers.layernorm import ZeroCenteredRMSNorm, GatedRMSNorm
from nanovllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.qwen3 import Qwen3MLP


class Qwen3_5Attention(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        assert dist.get_world_size() == 1, "tensor parallelism is not supported for Qwen3.5 yet"
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # q_proj also emits an output gate: every head is laid out as [q | gate]
        self.qkv_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [2 * self.q_size, self.kv_size, self.kv_size],
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            config.hidden_size,
            bias=False,
        )
        rope_parameters = config.rope_parameters
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=int(self.head_dim * rope_parameters["partial_rotary_factor"]),
            max_position=config.max_position_embeddings,
            base=rope_parameters["rope_theta"],
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim ** -0.5,
            self.num_kv_heads,
        )
        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split([2 * self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, gate = q_gate.view(-1, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
        q = self.q_norm(q.contiguous())    # same strides as k, so q_norm/k_norm share one compiled graph
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        o = o.flatten(1, -1) * torch.sigmoid(gate.reshape(-1, self.q_size))
        return self.o_proj(o)


class Qwen3_5GatedDeltaNet(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        assert dist.get_world_size() == 1, "tensor parallelism is not supported for Qwen3.5 yet"
        self.num_v_heads = config.linear_num_value_heads
        self.head_v_dim = config.linear_value_head_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * config.linear_num_key_heads * config.linear_key_head_dim + self.value_dim
        # the checkpoint stores in_proj_qkv / in_proj_z / in_proj_b / in_proj_a; two merged matmuls are enough
        self.in_proj_qkvz = MergedColumnParallelLinear(
            config.hidden_size,
            [self.conv_dim, self.value_dim],
            bias=False,
        )
        self.in_proj_ba = MergedColumnParallelLinear(
            config.hidden_size,
            [self.num_v_heads, self.num_v_heads],
            bias=False,
        )
        # only holds the depthwise conv weight; the convolution itself runs inside LinearAttention
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            config.linear_conv_kernel_dim,
            groups=self.conv_dim,
            bias=False,
        )
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.attn = LinearAttention(
            config.linear_num_key_heads,
            self.num_v_heads,
            config.linear_key_head_dim,
            self.head_v_dim,
            config.linear_conv_kernel_dim,
        )
        self.norm = GatedRMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = RowParallelLinear(
            self.value_dim,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        mixed_qkv, z = self.in_proj_qkvz(hidden_states).split([self.conv_dim, self.value_dim], dim=-1)
        b, a = self.in_proj_ba(hidden_states).chunk(2, dim=-1)
        beta = b.float().sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        o = self.attn(mixed_qkv, self.conv1d.weight, g, beta)
        o = self.norm(o, z.reshape(-1, self.num_v_heads, self.head_v_dim))
        return self.out_proj(o.flatten(1, -1))


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_type: str,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
        else:
            self.self_attn = Qwen3_5Attention(config)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5Model(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, layer_type) for layer_type in config.layer_types])
        self.norm = ZeroCenteredRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "in_proj_qkv": ("in_proj_qkvz", 0),
        "in_proj_z": ("in_proj_qkvz", 1),
        "in_proj_b": ("in_proj_ba", 0),
        "in_proj_a": ("in_proj_ba", 1),
    }
    weight_prefix_mapping = {"model.language_model.": "model."}
    skip_weight_prefixes = ("model.visual.", "mtp.")    # text only; the MTP head is unused for now

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
