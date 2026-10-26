import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    num_state_groups: int = 0
    num_speculative_tokens: int = 0
    draft_vocab: str | None = None    # JSON list of token ids the MTP head drafts from (default: the full vocabulary)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.num_speculative_tokens in (0, 1), "only one MTP draft token per step is supported"
        assert self.draft_vocab is None or self.num_speculative_tokens > 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if hasattr(self.hf_config, "text_config"):    # vision-language checkpoints (e.g. Qwen3.5) nest the LLM config
            text_config = self.hf_config.text_config
            text_config.architectures = self.hf_config.architectures
            text_config.tie_word_embeddings = self.hf_config.tie_word_embeddings
            self.hf_config = text_config
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
