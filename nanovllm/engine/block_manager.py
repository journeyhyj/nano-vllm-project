from collections import OrderedDict, deque
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


class SnapshotPool:
    """Bookkeeping for a fixed pool of recurrent-state snapshots (the tensors live in ModelRunner). A snapshot holds the
    state of every linear-attention layer right after a block boundary and is keyed by the chain hash of the block that
    ends there, so the prefix walk over KV blocks finds it too.

    An id goes through reserve() (the scheduler picks it for a boundary of a scheduled sequence; the forward writes it),
    then register() (visible to lookups) or release() (the boundary was not reached). Registered ids are evicted in LRU
    order unless a pending restore pins them; eviction happens in reserve(), which unregisters the victim at once, so an
    id is never written while a lookup can still return it."""

    def __init__(self, num_snapshots: int):
        self.free: deque[int] = deque(range(num_snapshots))
        self.index: dict[int, int] = {}                    # hash -> id
        self.hash_of = [-1] * num_snapshots                # id -> hash while registered
        self.ref = [0] * num_snapshots                     # pending restores
        self.dead: set[int] = set()                        # unregistered while pinned: freed by the last unpin
        self.lru: OrderedDict[int, None] = OrderedDict()   # registered ids, least recently used first
        self.stats = dict(saved=0, restored=0, evicted=0, dropped_with_block=0, duplicate=0, unused=0,
                          skipped_full=0, peak_in_use=0)

    @property
    def capacity(self) -> int:
        return len(self.hash_of)

    @property
    def num_in_use(self) -> int:    # registered or reserved
        return self.capacity - len(self.free)

    def lookup(self, h: int) -> int:
        return self.index.get(h, -1)

    def pin(self, sid: int):
        self.ref[sid] += 1
        self.lru.move_to_end(sid)
        self.stats["restored"] += 1

    def unpin(self, sid: int):
        assert self.ref[sid] > 0
        self.ref[sid] -= 1
        if self.ref[sid] == 0 and sid in self.dead:
            self.dead.remove(sid)
            self.free.append(sid)

    def reserve(self) -> int:
        if self.free:
            sid = self.free.popleft()
        else:
            sid = next((s for s in self.lru if self.ref[s] == 0), -1)
            if sid == -1:
                self.stats["skipped_full"] += 1
                return -1
            self._unregister(sid)
            self.stats["evicted"] += 1
        self.stats["peak_in_use"] = max(self.stats["peak_in_use"], self.num_in_use)
        return sid

    def release(self, sid: int):
        assert self.hash_of[sid] == -1 and self.ref[sid] == 0
        self.free.append(sid)
        self.stats["unused"] += 1

    def register(self, sid: int, h: int) -> bool:
        assert h != -1 and self.hash_of[sid] == -1
        if h in self.index:    # e.g. two identical prompts in one step
            self.free.append(sid)
            self.stats["duplicate"] += 1
            return False
        self.index[h] = sid
        self.hash_of[sid] = h
        self.lru[sid] = None
        self.stats["saved"] += 1
        return True

    def demote(self, sid: int, h: int):
        # the sequence reached a newer boundary: keep this snapshot but evict it first (no-op if the id moved on)
        if self.index.get(h) == sid:
            self.lru.move_to_end(sid, last=False)

    def drop(self, h: int):
        # the KV block this snapshot ends was reallocated, so no prefix walk can reach the snapshot any more
        sid = self.index.get(h, -1)
        if sid == -1:
            return
        self._unregister(sid)
        self.stats["dropped_with_block"] += 1
        if self.ref[sid]:
            self.dead.add(sid)
        else:
            self.free.append(sid)

    def _unregister(self, sid: int):
        del self.index[self.hash_of[sid]]
        del self.lru[sid]
        self.hash_of[sid] = -1


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, num_state_groups: int = 0, slots_per_group: int = 1,
                 policy: str = "branch", num_snapshots: int = 0):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        # Hybrid models also keep a group of recurrent-state slots per sequence (one slot per token verified in a step).
        # A cached KV prefix is only usable together with the recurrent state at its end, so hybrid models reuse a
        # prefix up to the deepest matched block boundary that has a state snapshot.
        self.free_state_groups: deque[int] = deque(range(num_state_groups))
        self.slots_per_group = slots_per_group
        self.use_state_slots = num_state_groups > 0
        self.policy = policy
        self.enable_prefix_caching = policy != "off"
        use_snapshots = self.use_state_slots and policy in ("last", "branch", "all") and num_snapshots > 0
        self.snapshots = SnapshotPool(num_snapshots) if use_snapshots else None
        self.counters = dict(admissions=0, prompt_tokens=0, matched_tokens=0, hit_tokens=0)

    @property
    def stats(self) -> dict:
        stats = dict(self.counters, policy=self.policy)
        if self.snapshots is not None:
            stats.update(self.snapshots.stats, capacity=self.snapshots.capacity, in_use=self.snapshots.num_in_use)
        return stats

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def prefix_hashes(self, seq: Sequence, num_blocks: int) -> list[int]:
        hashes, h = [], -1
        for i in range(num_blocks):
            h = self.compute_hash(seq.block(i), h)
            hashes.append(h)
        return hashes

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
            if self.snapshots is not None:
                self.snapshots.drop(block.hash)
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def match(self, seq: Sequence) -> list[int]:
        # chain hashes of the leading blocks whose KV is cached; the last block is always computed (its logits are needed)
        hashes = []
        if not self.enable_prefix_caching:
            return hashes
        h = -1
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            hashes.append(h)
        return hashes

    def usable_prefix(self, hashes: list[int]) -> int:
        # blocks whose cached KV can be used without recomputing them
        if not self.use_state_slots:
            return len(hashes)
        if self.snapshots is None:    # kv_only (or off)
            return 0
        for k in range(len(hashes), 0, -1):
            if self.snapshots.lookup(hashes[k - 1]) != -1:
                return k
        return 0

    def can_allocate(self, seq: Sequence) -> tuple[int, int] | None:
        """Returns (cached blocks to reuse, blocks whose KV is cached), or None if seq does not fit now."""
        if self.use_state_slots and not self.free_state_groups:
            return None
        hashes = self.match(seq)
        num_cached_blocks = self.usable_prefix(hashes)
        num_shared = len(hashes) if self.policy == "kv_only" else num_cached_blocks
        num_new_blocks = seq.num_blocks - sum(self.hash_to_block_id[h] in self.used_block_ids for h in hashes[:num_shared])
        if len(self.free_block_ids) < num_new_blocks:
            return None
        return num_cached_blocks, len(hashes)

    def allocate(self, seq: Sequence, num_cached_blocks: int, num_matched_blocks: int | None = None):
        assert not seq.block_table
        if num_matched_blocks is None:
            num_matched_blocks = num_cached_blocks
        # kv_only shares every matched block but recomputes all tokens without writing into the shared blocks
        num_shared = num_matched_blocks if self.policy == "kv_only" else num_cached_blocks
        h = -1
        for i in range(num_shared):
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
        for i in range(num_shared, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        seq.num_matched_tokens = num_matched_blocks * self.block_size
        seq.num_shared_kv_tokens = num_shared * self.block_size if self.policy == "kv_only" else 0
        if self.use_state_slots:
            group = self.free_state_groups.popleft()
            seq.state_slots = [group * self.slots_per_group + i for i in range(self.slots_per_group)]
            if num_cached_blocks:    # the GDN state at the hit boundary is restored before the first prefill chunk
                sid = self.snapshots.lookup(h)
                self.snapshots.pin(sid)
                seq.snapshot_restore = sid
        self.counters["admissions"] += 1
        self.counters["prompt_tokens"] += seq.num_tokens
        self.counters["matched_tokens"] += seq.num_matched_tokens
        self.counters["hit_tokens"] += seq.num_cached_tokens

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
        if self.snapshots is not None:
            for _, sid, _ in seq.snapshot_saves:
                self.snapshots.release(sid)
            if seq.snapshot_restore != -1:
                self.snapshots.unpin(seq.snapshot_restore)
        seq.snapshot_saves = []
        seq.snapshot_restore = -1
        seq.tail_snapshot = None
        seq.num_matched_tokens = seq.num_shared_kv_tokens = 0

    def num_new_blocks(self, seq: Sequence, num_tokens: int) -> int:
        # blocks still needed to write num_tokens tokens starting at the last token of seq
        return max(0, (len(seq) + num_tokens - 2) // self.block_size + 1 - len(seq.block_table))

    def can_append(self, seq: Sequence, num_tokens: int = 1) -> bool:
        return len(self.free_block_ids) >= self.num_new_blocks(seq, num_tokens)

    def may_append(self, seq: Sequence, num_tokens: int = 1):
        for _ in range(self.num_new_blocks(seq, num_tokens)):
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence, num_old_cached_tokens: int):
        # register the blocks that this step completed: [num_old_cached_tokens, seq.num_cached_tokens) now has valid KV.
        # Call it after appending the step's tokens, so that an accepted draft token is in seq.token_ids.
        if not self.enable_prefix_caching:
            return
        start = num_old_cached_tokens // self.block_size
        end = seq.num_cached_tokens // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        assert start == 0 or h != -1, "a hash chain may only start at the first block"
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            assert len(token_ids) == self.block_size
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    def plan_prefill_snapshots(self, seq: Sequence, step_hashes: set[int]):
        """Reserve snapshots for the block boundaries in (start, end] of the prefill chunk just scheduled for seq that
        the policy keeps. step_hashes: boundaries already reserved in this step (identical prompts save once)."""
        if self.snapshots is None:
            return
        bs = self.block_size
        start, end = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
        if self.policy == "all":
            boundaries = list(range((start // bs + 1) * bs, end + 1, bs))
        elif self.policy == "last":    # the deepest boundary a longer request can hit: the last block is never matched
            boundaries = [(seq.num_tokens - 1) // bs * bs]
        else:    # branch: the cached KV runs past the last snapshot, so another sequence went through this prefix
            boundaries = [seq.num_matched_tokens]
        boundaries = [b for b in boundaries if start < b <= end]
        if not boundaries:
            return
        hashes = self.prefix_hashes(seq, boundaries[-1] // bs)
        for b in boundaries:
            h = hashes[b // bs - 1]
            if h in step_hashes or self.snapshots.lookup(h) != -1:
                continue
            sid = self.snapshots.reserve()
            if sid == -1:
                break
            step_hashes.add(h)
            seq.snapshot_saves.append((b, sid, self.policy == "last"))

    def plan_decode_snapshots(self, seq: Sequence, num_tokens: int):
        """A decode step that crosses a block boundary saves the state there: the latest boundary of a sequence is where
        the next turn of a conversation resumes (under "all" every boundary is kept)."""
        if self.snapshots is None:
            return
        bs = self.block_size
        b = (seq.num_cached_tokens // bs + 1) * bs
        if b <= seq.num_cached_tokens + num_tokens:
            sid = self.snapshots.reserve()
            if sid != -1:
                seq.snapshot_saves.append((b, sid, self.policy != "all"))

    def finish_step(self, seq: Sequence):
        """After a step (and hash_blocks): publish the snapshots whose boundary is now computed, release the others, and
        unpin the snapshot restored for this step."""
        if self.snapshots is None:
            return
        if seq.snapshot_restore != -1:
            self.snapshots.unpin(seq.snapshot_restore)
            seq.snapshot_restore = -1
        for b, sid, is_tail in seq.snapshot_saves:
            if b > seq.num_cached_tokens:    # e.g. a rejected draft token sat right before the boundary
                self.snapshots.release(sid)
                continue
            h = self.blocks[seq.block_table[b // self.block_size - 1]].hash
            if self.snapshots.register(sid, h) and is_tail:
                if seq.tail_snapshot is not None:
                    self.snapshots.demote(*seq.tail_snapshot)
                seq.tail_snapshot = (sid, h)
        seq.snapshot_saves = []
