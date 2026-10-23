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

    def __init__(self, num_blocks: int, block_size: int, num_state_groups: int = 0, slots_per_group: int = 1):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        # Hybrid models also keep a group of recurrent-state slots per sequence (one slot per token
        # verified in a step). Cached KV blocks cannot rebuild that state, so prefix caching is off for them.
        self.free_state_groups: deque[int] = deque(range(num_state_groups))
        self.slots_per_group = slots_per_group
        self.use_state_slots = num_state_groups > 0
        self.enable_prefix_caching = not self.use_state_slots

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
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        if self.use_state_slots and not self.free_state_groups:
            return -1
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        num_blocks_to_match = seq.num_blocks - 1 if self.enable_prefix_caching else 0
        for i in range(num_blocks_to_match):
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

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
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
        if self.use_state_slots:
            group = self.free_state_groups.popleft()
            seq.state_slots = [group * self.slots_per_group + i for i in range(self.slots_per_group)]

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        if seq.state_slots:
            self.free_state_groups.append(seq.state_slots[0] // self.slots_per_group)
            seq.state_slots = []
        seq.draft_token = -1

    def num_new_blocks(self, seq: Sequence, num_tokens: int) -> int:
        # blocks still needed to write num_tokens tokens starting at the last token of seq
        return max(0, (len(seq) + num_tokens - 2) // self.block_size + 1 - len(seq.block_table))

    def can_append(self, seq: Sequence, num_tokens: int = 1) -> bool:
        return len(self.free_block_ids) >= self.num_new_blocks(seq, num_tokens)

    def may_append(self, seq: Sequence, num_tokens: int = 1):
        for _ in range(self.num_new_blocks(seq, num_tokens)):
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
