import json
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.layers.linear_attention import LinearAttention
from nanovllm.layers.sampler import Sampler, compute_probs, sample_from_probs, rejection_sample
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.num_speculative_tokens = config.num_speculative_tokens
        self.query_len = 1 + self.num_speculative_tokens    # tokens per sequence in a decode step
        self.draft_probs = None
        self.mtp_graphs = {}

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        model_cls = {"Qwen3_5ForConditionalGeneration": Qwen3_5ForCausalLM}.get(hf_config.architectures[0], Qwen3ForCausalLM)
        self.model = model_cls(hf_config, use_mtp=True) if self.num_speculative_tokens else model_cls(hf_config)
        load_model(self.model, config.model)
        self.linear_attn_layers = [m for m in self.model.modules() if isinstance(m, LinearAttention)]
        if self.num_speculative_tokens:
            assert self.linear_attn_layers and self.world_size == 1, "MTP speculative decoding needs Qwen3.5 on one GPU"
        # drafting from a frequent subset of the vocabulary: the draft distribution is zero elsewhere,
        # which rejection sampling handles like any other draft distribution
        self.draft_ids = None
        self.draft_vocab_size = hf_config.vocab_size
        if config.draft_vocab:
            with open(config.draft_vocab) as f:
                self.draft_ids = torch.tensor(json.load(f), dtype=torch.int64)
            self.draft_vocab_size = self.draft_ids.numel()
            assert self.draft_ids.unique().numel() == self.draft_vocab_size and self.draft_ids.max() < hf_config.vocab_size
            # release cached blocks first (e.g. the lm_head weight dropped by tie_word_embeddings): carved out of one,
            # the copy would keep the rest of that block reserved, and out of the KV / state budget
            torch.cuda.empty_cache()
            self.model.set_draft_vocab(self.draft_ids)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.mtp_graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for i, seq in enumerate(seqs):
            seq.num_scheduled_tokens = seq_len
            seq.state_slots = [i]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        available = int(total * config.gpu_memory_utilization - used - peak + current)
        if self.linear_attn_layers:
            available -= self.allocate_state_cache(available)
        attn_layers = [m for m in self.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * len(attn_layers) * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = available // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, len(attn_layers), config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        for layer_id, module in enumerate(attn_layers):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]

    def allocate_state_cache(self, available: int) -> int:
        # every running sequence owns a group of query_len recurrent-state slots in each linear-attention layer
        # (one per token verified in a decode step) plus, with MTP, the probabilities of its pending draft.
        # States use at most half of the budget; the extra group absorbs CUDA-graph padding.
        config = self.config
        slot_bytes = sum(
            (m.conv_kernel_size - 1) * m.conv_dim * config.hf_config.dtype.itemsize
            + m.num_v_heads * m.head_k_dim * m.head_v_dim * 4
            for m in self.linear_attn_layers
        )
        group_bytes = self.query_len * slot_bytes + self.num_speculative_tokens * self.draft_vocab_size * 4
        num_groups = min(config.max_num_seqs, available // 2 // group_bytes - 1)
        assert num_groups > 0
        config.max_num_seqs = config.num_state_groups = num_groups
        num_slots = (num_groups + 1) * self.query_len
        for m in self.linear_attn_layers:
            m.conv_state = torch.zeros(num_slots, m.conv_kernel_size - 1, m.conv_dim)
            m.ssm_state = torch.zeros(num_slots, m.num_v_heads, m.head_k_dim, m.head_v_dim, dtype=torch.float32)
        if self.num_speculative_tokens:
            self.draft_probs = torch.zeros(num_groups + 1, self.draft_vocab_size, dtype=torch.float32)
        return (num_groups + 1) * group_bytes

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        state_slots = has_initial_state = None
        if self.linear_attn_layers:
            state_slots = self.prepare_state_slots(seqs, is_prefill=True)
            # with prefix caching off, cached tokens only exist after an earlier chunk of this sequence
            has_initial_state = [seq.num_cached_tokens > 0 for seq in seqs]
            has_initial_state = torch.tensor(has_initial_state, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, state_slots, has_initial_state)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # the last token, followed by the pending draft token when speculating
            tokens = [seq.last_token] + [seq.draft_token] * self.num_speculative_tokens
            start = len(seq) - 1
            for j, token in enumerate(tokens):
                input_ids.append(token)
                positions.append(start + j)
                block_id = seq.block_table[(start + j) // self.block_size]
                slot_mapping.append(block_id * self.block_size + (start + j) % self.block_size)
            context_lens.append(start + len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        state_slots = self.prepare_state_slots(seqs, is_prefill=False) if self.linear_attn_layers else None
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
                    state_slots=state_slots, query_len=self.query_len)
        return input_ids, positions

    def prepare_state_slots(self, seqs: list[Sequence], is_prefill: bool):
        # prefill only touches the current state (last slot); decode passes the whole group, one slot per token
        if is_prefill:
            state_slots = [seq.state_slots[-1] for seq in seqs]
        else:
            state_slots = [slot for seq in seqs for slot in seq.state_slots]
        return torch.tensor(state_slots, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def use_cudagraph(self, is_prefill: bool, num_tokens: int):
        return not (is_prefill or self.enforce_eager or num_tokens > 512 * self.query_len)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if not self.use_cudagraph(is_prefill, input_ids.size(0)):
            return self.model(input_ids, positions)
        else:
            n = input_ids.size(0)
            bs = n // self.query_len
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:n] = input_ids
            graph_vars["positions"][:n] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:n] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if context.state_slots is not None:
                graph_vars["state_slots"].fill_(self.config.num_state_groups * self.query_len)
                graph_vars["state_slots"][:n] = context.state_slots
            graph.replay()
            return graph_vars["outputs"][:n]

    @torch.inference_mode()
    def run_mtp(self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor):
        # decode only; reuses the context (and graph buffers) of the verify step that just ran
        n = input_ids.size(0)
        if not self.use_cudagraph(False, n):
            return self.model.mtp_forward(input_ids, positions, hidden_states)
        graph_vars = self.graph_vars
        graph_vars["mtp_input_ids"][:n] = input_ids
        self.mtp_graphs[next(x for x in self.graph_bs if x >= n // self.query_len)].replay()
        return graph_vars["mtp_outputs"][:n]

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool):
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        hidden_states = self.run_model(input_ids, positions, is_prefill)
        logits = self.model.compute_logits(hidden_states)
        if self.num_speculative_tokens:
            output = self.run_speculative(seqs, is_prefill, positions, hidden_states, logits, temperatures)
        else:
            output = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return output

    def run_speculative(self, seqs: list[Sequence], is_prefill: bool, positions: torch.Tensor,
                        hidden_states: torch.Tensor, logits: torch.Tensor, temperatures: torch.Tensor):
        # returns (new tokens per sequence, next draft token per sequence)
        bs = len(seqs)
        groups = [seq.state_slots[0] // self.query_len for seq in seqs]
        groups = torch.tensor(groups, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if is_prefill:
            tokens = self.sampler(logits, temperatures).tolist()
            # MTP at position p reads the target hidden state at p and the token at p + 1: shift the chunk left
            # by one; a sequence that finishes its prompt in this step ends with its sampled token
            mtp_input_ids = []
            for seq, token in zip(seqs, tokens):
                start, end = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
                mtp_input_ids.extend(seq[start + 1:end])
                mtp_input_ids.append(seq[end] if end < len(seq) else token)
            mtp_input_ids = torch.tensor(mtp_input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            draft_hidden = self.model.mtp_forward(mtp_input_ids, positions, hidden_states)
            token_ids = [[token] for token in tokens]
        else:
            target_probs = compute_probs(logits.view(bs, self.query_len, -1), temperatures.unsqueeze(1))
            draft_tokens = [seq.draft_token for seq in seqs]
            draft_tokens = torch.tensor(draft_tokens, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            draft_probs = self.draft_probs[groups]
            if self.draft_ids is not None:    # spread over the full vocabulary, zero outside the draft vocabulary
                draft_probs = draft_probs.new_zeros(bs, target_probs.size(-1)).index_copy_(1, self.draft_ids, draft_probs)
            tokens, num_accepted = rejection_sample(target_probs, draft_tokens, draft_probs)
            # run MTP on every verified position (fixed shape) and keep the row of the last accepted token;
            # rows past it write MTP KV that the next step overwrites
            draft_hidden = self.run_mtp(tokens.flatten(), positions, hidden_states)
            draft_hidden = draft_hidden.view(bs, self.query_len, -1)[torch.arange(bs, device=groups.device), num_accepted]
            token_ids = [t[:n + 1] for t, n in zip(tokens.tolist(), num_accepted.tolist())]
        draft_probs = compute_probs(self.model.compute_draft_logits(draft_hidden), temperatures)
        if self.draft_probs is not None:    # not allocated yet during warmup
            self.draft_probs[groups] = draft_probs
        draft_tokens = sample_from_probs(draft_probs)
        if self.draft_ids is not None:    # index in the draft vocabulary -> token id
            draft_tokens = self.draft_ids[draft_tokens]
        return token_ids, draft_tokens.tolist()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_tokens = max_bs * self.query_len
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_tokens, dtype=torch.int64)
        positions = torch.zeros(max_tokens, dtype=torch.int64)
        slot_mapping = torch.zeros(max_tokens, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        # padded batch entries point at the spare state group so they never touch a live sequence
        spare_slot = config.num_state_groups * self.query_len
        state_slots = torch.full((max_tokens,), spare_slot, dtype=torch.int32) if self.linear_attn_layers else None
        outputs = torch.zeros(max_tokens, hf_config.hidden_size)
        self.graph_bs = [bs for bs in [1, 2, 4, 8] if bs < max_bs] + list(range(16, max_bs, 16)) + [max_bs]
        self.graphs = {}
        self.graph_pool = None

        def set_graph_context(bs):
            n = bs * self.query_len
            set_context(False, slot_mapping=slot_mapping[:n], context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                        state_slots=state_slots[:n] if state_slots is not None else None, query_len=self.query_len)

        for bs in reversed(self.graph_bs):
            n = bs * self.query_len
            graph = torch.cuda.CUDAGraph()
            set_graph_context(bs)
            outputs[:n] = self.model(input_ids[:n], positions[:n])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:n] = self.model(input_ids[:n], positions[:n])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            state_slots=state_slots,
            outputs=outputs,
        )
        if not self.num_speculative_tokens:
            return
        # the MTP graph reads the verify graph's outputs as its hidden states and shares all context buffers
        mtp_input_ids = torch.zeros(max_tokens, dtype=torch.int64)
        mtp_outputs = torch.zeros(max_tokens, hf_config.hidden_size)
        for bs in reversed(self.graph_bs):
            n = bs * self.query_len
            graph = torch.cuda.CUDAGraph()
            set_graph_context(bs)
            mtp_outputs[:n] = self.model.mtp_forward(mtp_input_ids[:n], positions[:n], outputs[:n])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                mtp_outputs[:n] = self.model.mtp_forward(mtp_input_ids[:n], positions[:n], outputs[:n])    # capture
            self.mtp_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
        self.graph_vars.update(mtp_input_ids=mtp_input_ids, mtp_outputs=mtp_outputs)
