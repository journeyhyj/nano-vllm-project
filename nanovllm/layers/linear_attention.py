import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context


@triton.jit
def causal_conv1d_kernel(
    x_ptr,
    x_stride,
    weight_ptr,
    out_ptr,
    conv_state_ptr,
    cu_seqlens_ptr,
    state_slots_ptr,
    has_initial_state_ptr,
    C,
    W: tl.constexpr,
    DECODE_LEN: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    i_c, i_t, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if DECODE_LEN > 0:    # every sequence has DECODE_LEN tokens; its current state is in the last slot of its group
        bos = i_n * DECODE_LEN
        eos = bos + DECODE_LEN
        slot = tl.load(state_slots_ptr + eos - 1).to(tl.int64)
        has_initial_state = True
    else:
        bos = tl.load(cu_seqlens_ptr + i_n)
        eos = tl.load(cu_seqlens_ptr + i_n + 1)
        slot = tl.load(state_slots_ptr + i_n).to(tl.int64)
        has_initial_state = tl.load(has_initial_state_ptr + i_n) != 0
    t0 = bos + i_t * BLOCK_T
    if t0 >= eos:
        return
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_c = i_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_t = offs_t < eos
    mask_c = offs_c < C
    state_ptr = conv_state_ptr + slot * (W - 1) * C
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for j in tl.static_range(W):
        src = offs_t - (W - 1) + j    # input position multiplied by tap j
        in_seq = src >= bos
        x = tl.load(x_ptr + src[:, None] * x_stride + offs_c[None, :],
                    mask=(in_seq & mask_t)[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        if has_initial_state:    # positions before bos come from the conv state (oldest first)
            idx = src - bos + (W - 1)
            x += tl.load(state_ptr + idx[:, None] * C + offs_c[None, :],
                         mask=((src < bos) & mask_t)[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs_c * W + j, mask=mask_c, other=0.0).to(tl.float32)
        acc += x * w[None, :]
    acc = acc * tl.sigmoid(acc)
    tl.store(out_ptr + offs_t[:, None] * C + offs_c[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=mask_t[:, None] & mask_c[None, :])


@triton.jit
def _conv_input(x_ptr, x_stride, state_ptr, offs_c, mask_c, pos, bos, has_initial_state, C):
    if pos >= bos:
        v = tl.load(x_ptr + pos * x_stride + offs_c, mask=mask_c, other=0.0)
    else:
        v = tl.load(state_ptr + (pos - bos + 3) * C + offs_c, mask=mask_c & has_initial_state, other=0.0)
    return v


@triton.jit
def conv_state_update_kernel(
    x_ptr,
    x_stride,
    conv_state_ptr,
    cu_seqlens_ptr,
    state_slots_ptr,
    has_initial_state_ptr,
    C,
    BLOCK_C: tl.constexpr,
):
    # keep the last 3 inputs of every sequence (kernel size 4) for its next chunk / decode step
    i_c, i_n = tl.program_id(0), tl.program_id(1)
    bos = tl.load(cu_seqlens_ptr + i_n)
    eos = tl.load(cu_seqlens_ptr + i_n + 1)
    has_initial_state = tl.load(has_initial_state_ptr + i_n) != 0
    offs_c = i_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    slot = tl.load(state_slots_ptr + i_n).to(tl.int64)
    state_ptr = conv_state_ptr + slot * 3 * C
    # read everything before writing: a short sequence keeps part of the old state
    s0 = _conv_input(x_ptr, x_stride, state_ptr, offs_c, mask_c, eos - 3, bos, has_initial_state, C)
    s1 = _conv_input(x_ptr, x_stride, state_ptr, offs_c, mask_c, eos - 2, bos, has_initial_state, C)
    s2 = _conv_input(x_ptr, x_stride, state_ptr, offs_c, mask_c, eos - 1, bos, has_initial_state, C)
    tl.store(state_ptr + offs_c, s0, mask=mask_c)
    tl.store(state_ptr + C + offs_c, s1, mask=mask_c)
    tl.store(state_ptr + 2 * C + offs_c, s2, mask=mask_c)


@triton.jit
def _window_input(x_ptr, x_stride, offs_c, mask_c, bos, p: tl.constexpr, old1, old2):
    if p >= 0:
        v = tl.load(x_ptr + (bos + p) * x_stride + offs_c, mask=mask_c, other=0.0)
    elif p == -1:
        v = old2
    else:
        v = old1
    return v


@triton.jit
def conv_state_update_decode_kernel(
    x_ptr,
    x_stride,
    conv_state_ptr,
    state_slots_ptr,
    C,
    DECODE_LEN: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # after decoded token j, write the conv state (last 3 inputs) into slot j of the sequence's group
    i_c, i_n = tl.program_id(0), tl.program_id(1)
    bos = i_n * DECODE_LEN
    offs_c = i_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    cur = tl.load(state_slots_ptr + bos + DECODE_LEN - 1).to(tl.int64)
    old1 = tl.load(conv_state_ptr + cur * 3 * C + C + offs_c, mask=mask_c, other=0.0)
    old2 = tl.load(conv_state_ptr + cur * 3 * C + 2 * C + offs_c, mask=mask_c, other=0.0)
    for j in tl.static_range(DECODE_LEN):
        state_ptr = conv_state_ptr + tl.load(state_slots_ptr + bos + j).to(tl.int64) * 3 * C
        tl.store(state_ptr + offs_c, _window_input(x_ptr, x_stride, offs_c, mask_c, bos, j - 2, old1, old2), mask=mask_c)
        tl.store(state_ptr + C + offs_c, _window_input(x_ptr, x_stride, offs_c, mask_c, bos, j - 1, old1, old2), mask=mask_c)
        tl.store(state_ptr + 2 * C + offs_c, _window_input(x_ptr, x_stride, offs_c, mask_c, bos, j, old1, old2), mask=mask_c)


@triton.jit
def gated_delta_rule_kernel(
    x_ptr,
    x_stride,
    g_ptr,
    beta_ptr,
    o_ptr,
    ssm_state_ptr,
    cu_seqlens_ptr,
    state_slots_ptr,
    has_initial_state_ptr,
    scale,
    Hk: tl.constexpr,
    Hv: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    DECODE_LEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // Hv, i_nh % Hv
    i_hk = i_h // (Hv // Hk)    # grouped value heads share one q/k head
    if DECODE_LEN > 0:
        bos = i_n * DECODE_LEN
        eos = bos + DECODE_LEN
        slot = tl.load(state_slots_ptr + eos - 1).to(tl.int64)
        has_initial_state = True
    else:
        bos = tl.load(cu_seqlens_ptr + i_n)
        eos = tl.load(cu_seqlens_ptr + i_n + 1)
        slot = tl.load(state_slots_ptr + i_n).to(tl.int64)
        has_initial_state = tl.load(has_initial_state_ptr + i_n) != 0
    offs_k = tl.arange(0, K)
    offs_v = i_v * BV + tl.arange(0, BV)
    p_state = ssm_state_ptr + (slot * Hv + i_h) * K * V + offs_k[:, None] * V + offs_v[None, :]
    S = tl.zeros((K, BV), dtype=tl.float32)
    if has_initial_state:
        S = tl.load(p_state)
    for t in range(bos, eos):
        # x row layout: [q (Hk*K) | k (Hk*K) | v (Hv*V)]
        q = tl.load(x_ptr + t * x_stride + i_hk * K + offs_k).to(tl.float32)
        k = tl.load(x_ptr + t * x_stride + (Hk + i_hk) * K + offs_k).to(tl.float32)
        v = tl.load(x_ptr + t * x_stride + 2 * Hk * K + i_h * V + offs_v).to(tl.float32)
        g = tl.load(g_ptr + t * Hv + i_h)
        beta = tl.load(beta_ptr + t * Hv + i_h)
        q = q * tl.rsqrt(tl.sum(q * q) + 1e-6) * scale
        k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
        S = S * tl.exp(g)
        v = (v - tl.sum(S * k[:, None], axis=0)) * beta    # delta rule: store only what S fails to predict
        S = S + k[:, None] * v[None, :]
        o = tl.sum(S * q[:, None], axis=0)
        tl.store(o_ptr + (t * Hv + i_h) * V + offs_v, o.to(o_ptr.dtype.element_ty))
        if DECODE_LEN > 0:    # keep the state after every token so rejected speculative tokens can be rolled back
            slot_t = tl.load(state_slots_ptr + t).to(tl.int64)
            tl.store(ssm_state_ptr + (slot_t * Hv + i_h) * K * V + offs_k[:, None] * V + offs_v[None, :], S)
    if DECODE_LEN == 0:
        tl.store(p_state, S)


class LinearAttention(nn.Module):
    """Causal conv1d + gated delta rule of Gated DeltaNet. Each sequence keeps its
    recurrent state in conv_state / ssm_state at index context.state_slots[i]."""

    def __init__(
        self,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
    ):
        super().__init__()
        assert conv_kernel_size == 4
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        self.scale = head_k_dim ** -0.5
        self.conv_state = self.ssm_state = torch.tensor([])

    def forward(self, mixed_qkv: torch.Tensor, conv_weight: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
        context = get_context()
        conv_state, ssm_state = self.conv_state, self.ssm_state
        if not ssm_state.numel():    # warmup runs before the state pool exists
            n = context.state_slots.numel()
            conv_state = mixed_qkv.new_zeros(n, self.conv_kernel_size - 1, self.conv_dim)
            ssm_state = torch.zeros(n, self.num_v_heads, self.head_k_dim, self.head_v_dim,
                                    dtype=torch.float32, device=mixed_qkv.device)
        # decode: every sequence has query_len tokens and a group of query_len state slots
        decode_len = 0 if context.is_prefill else context.query_len
        num_seqs = context.state_slots.numel() // max(decode_len, 1)
        T, C = mixed_qkv.shape
        x = torch.empty(T, C, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        BLOCK_C = 256
        BLOCK_T = triton.next_power_of_2(decode_len) if decode_len else 16
        max_seqlen = decode_len or context.max_seqlen_q
        conv_args = (context.cu_seqlens_q, context.state_slots, context.has_initial_state, C)
        causal_conv1d_kernel[(triton.cdiv(C, BLOCK_C), triton.cdiv(max_seqlen, BLOCK_T), num_seqs)](
            mixed_qkv, mixed_qkv.stride(0), conv_weight, x, conv_state, *conv_args,
            W=self.conv_kernel_size, DECODE_LEN=decode_len, BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C)
        if decode_len:
            conv_state_update_decode_kernel[(triton.cdiv(C, BLOCK_C), num_seqs)](
                mixed_qkv, mixed_qkv.stride(0), conv_state, context.state_slots, C, DECODE_LEN=decode_len, BLOCK_C=BLOCK_C)
        else:
            conv_state_update_kernel[(triton.cdiv(C, BLOCK_C), num_seqs)](
                mixed_qkv, mixed_qkv.stride(0), conv_state, *conv_args, BLOCK_C=BLOCK_C)
        o = torch.empty(T, self.num_v_heads, self.head_v_dim, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        BV = 32
        gated_delta_rule_kernel[(self.head_v_dim // BV, num_seqs * self.num_v_heads)](
            x, x.stride(0), g, beta, o, ssm_state,
            context.cu_seqlens_q, context.state_slots, context.has_initial_state, self.scale,
            Hk=self.num_k_heads, Hv=self.num_v_heads, K=self.head_k_dim, V=self.head_v_dim, BV=BV, DECODE_LEN=decode_len)
        return o
