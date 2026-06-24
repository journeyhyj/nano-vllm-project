from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if self.enable_prefix_caching and block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence, num_tokens_to_schedule: int = -1) -> int:
        """Check if we can allocate blocks for a sequence.

        With on-demand allocation (prefix_caching disabled), only requires blocks
        for the tokens actually being scheduled this step, not the entire sequence.
        This allows more sequences to start prefilling in parallel.

        Args:
            seq: The sequence to allocate blocks for.
            num_tokens_to_schedule: Number of tokens to schedule this step.
                -1 means allocate blocks for all remaining tokens (backward compatible).

        Returns:
            -1 if not enough free blocks, otherwise the number of cached blocks
            (0 when prefix_caching is disabled).
        """
        if not self.enable_prefix_caching:
            # On-demand allocation: only require blocks for tokens being scheduled
            if num_tokens_to_schedule == -1:
                num_tokens_to_schedule = seq.num_tokens - seq.num_cached_tokens
            total_tokens_after = seq.num_cached_tokens + num_tokens_to_schedule
            total_blocks_needed = (total_tokens_after + self.block_size - 1) // self.block_size
            num_new_blocks = total_blocks_needed - len(seq.block_table)
            if num_new_blocks > 0 and len(self.free_block_ids) < num_new_blocks:
                return -1
            return 0

        # Prefix caching path: allocate all blocks upfront
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int, num_tokens_to_schedule: int = -1):
        """Allocate blocks for a sequence.

        With on-demand allocation (prefix_caching disabled), only allocates blocks
        needed for the tokens being scheduled this step. Supports incremental
        allocation across multiple prefill steps (chunked prefill).

        Args:
            seq: The sequence to allocate blocks for.
            num_cached_blocks: Number of prefix-cached blocks (0 when disabled).
            num_tokens_to_schedule: Number of tokens to schedule this step.
                -1 means allocate blocks for all remaining tokens.
        """
        if not self.enable_prefix_caching:
            # On-demand allocation: only allocate blocks for this step's tokens
            if num_tokens_to_schedule == -1:
                num_tokens_to_schedule = seq.num_tokens - seq.num_cached_tokens
            total_tokens_after = seq.num_cached_tokens + num_tokens_to_schedule
            total_blocks_needed = (total_tokens_after + self.block_size - 1) // self.block_size
            while len(seq.block_table) < total_blocks_needed:
                seq.block_table.append(self._allocate_block())
            return

        # Prefix caching path: must have empty block_table
        if seq.block_table:
            # Sequence was partially prefilled with on-demand allocation before
            # prefix caching was enabled, or chunked prefill left blocks.
            # Continue allocating incrementally.
            if num_tokens_to_schedule == -1:
                num_tokens_to_schedule = seq.num_tokens - seq.num_cached_tokens
            total_tokens_after = seq.num_cached_tokens + num_tokens_to_schedule
            total_blocks_needed = (total_tokens_after + self.block_size - 1) // self.block_size
            while len(seq.block_table) < total_blocks_needed:
                seq.block_table.append(self._allocate_block())
            return
        if num_cached_blocks > 0:
            h = -1
            for i in range(num_cached_blocks):
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h)
                block_id = self.hash_to_block_id[h]
                block = self.blocks[block_id]
                if block_id in self.used_block_ids:
                    block.ref_count += 1
                else:
                    block.ref_count = 1
                    self.free_block_ids.remove(block_id)
                    self.used_block_ids.add(block_id)
                seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        if not self.enable_prefix_caching:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
