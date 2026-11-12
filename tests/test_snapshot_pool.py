# Prefix-cache bookkeeping for hybrid models, on the CPU: hit lengths per policy, snapshot ids through reserve / register
# / release / LRU eviction / pins, snapshots dropped with their KV block, hashing after MTP steps, and a random stress run
# of the real Scheduler against a fake model runner that records which token prefix every snapshot id was written for:
# a restore must always hand back the state of exactly the tokens the sequence skips.
# Usage: python tests/test_snapshot_pool.py   (runs on the CPU; importing nanovllm still needs a visible GPU for triton)
import random
from collections import deque

from nanovllm.engine.block_manager import BlockManager, SnapshotPool
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BS = 256


def make_scheduler(policy, num_blocks=64, num_snapshots=8, num_groups=4, k=0, max_batched=16384):
    s = Scheduler.__new__(Scheduler)
    s.max_num_seqs, s.max_num_batched_tokens, s.eos, s.block_size, s.num_speculative_tokens = num_groups, max_batched, -1, BS, k
    s.block_manager = BlockManager(num_blocks, BS, num_groups, 1 + k, policy, num_snapshots)
    s.waiting, s.running = deque(), deque()
    return s


def next_token(token_ids):
    return hash(tuple(token_ids[-8:]) + (len(token_ids),)) % 50000 + 1000    # same context -> same token, like greedy


class FakeRunner:
    """Stands in for ModelRunner: records what each snapshot id holds and checks every restore against it."""

    def __init__(self, k=0, accept=0.7, seed=0):
        self.k, self.accept, self.rng = k, accept, random.Random(seed)
        self.content = {}    # snapshot id -> token prefix whose state it holds
        self.restores = 0

    def run(self, seqs, is_prefill):
        for seq in seqs:
            if seq.snapshot_restore != -1:
                assert self.content[seq.snapshot_restore] == tuple(seq.token_ids[:seq.num_cached_tokens]), "restored a wrong state"
                self.restores += 1
            for b, sid, _ in seq.snapshot_saves:
                tokens = seq.token_ids if is_prefill or self.k == 0 else seq.token_ids + [seq.draft_token]
                self.content[sid] = tuple(tokens[:b])
        if self.k == 0:
            return [next_token(seq.token_ids) for seq in seqs], None
        token_ids, drafts = [], []
        for seq in seqs:
            if is_prefill or self.rng.random() > self.accept:
                new = [next_token(seq.token_ids)]
            else:
                new = [seq.draft_token, next_token(seq.token_ids + [seq.draft_token])]
            token_ids.append(new)
            drafts.append(next_token(seq.token_ids + new) if self.rng.random() < 0.8 else 7)
        return token_ids, drafts


def step(sched, runner):
    seqs, is_prefill = sched.schedule()
    token_ids, drafts = runner.run(seqs, is_prefill)
    sched.postprocess(seqs, token_ids, is_prefill, drafts)
    check_invariants(sched.block_manager)
    return seqs, is_prefill


def run_all(sched, runner, seqs):
    for seq in seqs:
        sched.add(seq)
    while not sched.is_finished():
        step(sched, runner)


def check_invariants(bm: BlockManager):
    pool = bm.snapshots
    if pool is not None:    # between steps no id is reserved or pinned
        assert len(pool.free) + len(pool.index) + len(pool.dead) == pool.capacity
        assert not pool.dead and all(r == 0 for r in pool.ref)
        assert set(pool.index.values()) == set(pool.lru)
        assert all(pool.hash_of[sid] == h for h, sid in pool.index.items())
        assert all(h in bm.hash_to_block_id for h in pool.index), "a snapshot outlived its KV block"
    assert len(bm.free_block_ids) + len(bm.used_block_ids) == len(bm.blocks)


def admitted(sched, seq, runner):
    # schedule until seq has been admitted; returns (cached tokens, shared KV tokens) right after admission
    sched.add(seq)
    while not seq.block_table:
        seqs, is_prefill = sched.schedule()
        got = (seq.num_cached_tokens, seq.num_shared_kv_tokens)
        token_ids, drafts = runner.run(seqs, is_prefill)
        sched.postprocess(seqs, token_ids, is_prefill, drafts)
        check_invariants(sched.block_manager)
    return got


def test_hit_lengths_per_policy():
    rng = random.Random(0)
    prefix = [rng.randint(1000, 50000) for _ in range(800)]
    prompts = [prefix + [rng.randint(1000, 50000) for _ in range(100)] for _ in range(3)]    # 900 tokens, 3 full blocks
    sp = SamplingParams(max_tokens=8)
    expect = {    # (cached, shared KV) tokens of the 2nd and 3rd request after the 1st finished
        "off": ((0, 0), (0, 0)), "kv_only": ((0, 768), (0, 768)),    # kv_only shares the KV blocks, recomputes all
        "all": ((768, 0), (768, 0)), "last": ((768, 0), (768, 0)), "branch": ((0, 0), (768, 0)),
    }
    for policy, want in expect.items():
        sched, runner = make_scheduler(policy), FakeRunner()
        run_all(sched, runner, [Sequence(prompts[0], sp)])
        got = []
        for prompt in prompts[1:]:
            got.append(admitted(sched, Sequence(prompt, sp), runner))
            run_all(sched, runner, [])
        assert tuple(got) == want, f"{policy}: {got}"
        stats = sched.block_manager.stats
        print(f"{policy:8s} hits {got}  {({k: v for k, v in stats.items() if k in ('saved', 'restored', 'matched_tokens')})}")


def test_snapshot_pool_ids():
    pool = SnapshotPool(2)
    a, b = pool.reserve(), pool.reserve()
    assert pool.register(a, 11) and pool.register(b, 22)
    pool.pin(a)
    c = pool.reserve()    # a is pinned: evicts b
    assert c == b and pool.lookup(22) == -1 and pool.lookup(11) == a
    assert pool.reserve() == -1 and pool.stats["skipped_full"] == 1    # b is reserved, a pinned
    pool.demote(b, 22)    # stale handle: b now belongs to another boundary, must be a no-op
    assert pool.register(c, 33) and pool.lookup(33) == c
    pool.drop(11)         # a's KV block went away while a restore is pending: freed only after the unpin
    assert pool.lookup(11) == -1 and a not in pool.free
    pool.unpin(a)
    assert a in pool.free
    d = pool.reserve()
    assert not pool.register(d, 33) and d in pool.free    # duplicate boundary: the second id goes back


def test_snapshot_dropped_with_block():
    bm = BlockManager(6, BS, 2, 1, "all", 4)
    seq = Sequence(list(range(1000, 1000 + 3 * BS + 10)))
    bm.allocate(seq, 0, 0)
    seq.num_scheduled_tokens = seq.num_tokens
    bm.plan_prefill_snapshots(seq, set())
    assert [b for b, _, _ in seq.snapshot_saves] == [256, 512, 768]
    seq.num_cached_tokens, seq.num_scheduled_tokens = seq.num_tokens, 0
    bm.hash_blocks(seq, 0)
    bm.finish_step(seq)
    assert len(bm.snapshots.index) == 3
    bm.deallocate(seq)    # blocks become free but stay cached; the free list hands out block 3 and 2 first
    other = Sequence(list(range(5000, 5000 + 4 * BS)))
    bm.allocate(other, 0, 0)    # takes the 2 never-used blocks and blocks 3, 2: the KV of [512, 768) is gone
    assert len(bm.snapshots.index) == 2 and bm.snapshots.stats["dropped_with_block"] == 1
    check_invariants(bm)


def test_mtp_hashing_after_accepted_draft():
    # num_cached 254: the step verifies positions 254 and 255 (the draft); once accepted, block 0 must be hashed with
    # all 256 tokens (the old code hashed it before appending the draft, with 255 tokens, and never again)
    for accept in (1.0, 0.0):
        sched, runner = make_scheduler("last", k=1), FakeRunner(k=1)
        seq = Sequence([3000 + i for i in range(254)], SamplingParams(max_tokens=20))
        sched.add(seq)
        runner.accept = accept
        step(sched, runner)                       # prefill: 254 cached, 1 sampled token, 1 draft
        runner.rng = random.Random(1)
        while seq.num_cached_tokens < 256:
            step(sched, runner)
        bm = sched.block_manager
        block = bm.blocks[seq.block_table[0]]
        assert block.hash == bm.compute_hash(seq.token_ids[:256]) and len(block.token_ids) == 256
        assert bm.match(Sequence(seq.token_ids[:256] + [1, 2])) == [block.hash]
        # accepted: saved from the draft's slot in that step; rejected: released, then saved from slot 0 a step later
        assert bm.snapshots.lookup(block.hash) != -1
        run_all(sched, runner, [])


def test_random_stress():
    rng = random.Random(1)
    total_restores = 0
    for trial in range(40):
        policy = ["all", "last", "branch", "kv_only"][trial % 4]
        k = trial // 4 % 2
        sched = make_scheduler(policy, num_blocks=rng.choice([24, 40, 80]), num_snapshots=rng.choice([1, 3, 6]),
                               num_groups=rng.choice([2, 4]), k=k, max_batched=rng.choice([300, 700, 1024, 16384]))
        runner = FakeRunner(k=k, seed=trial)
        prefixes = [[rng.randint(1000, 50000) for _ in range(rng.randint(200, 1300))] for _ in range(3)]
        seqs = []
        for _ in range(12):
            prompt = rng.choice(prefixes)[:rng.randint(100, 1300)] + [rng.randint(1000, 50000) for _ in range(rng.randint(1, 300))]
            seqs.append(Sequence(prompt, SamplingParams(max_tokens=rng.randint(1, 400))))
        pending = list(seqs)
        while pending or not sched.is_finished():
            if pending and rng.random() < 0.3:
                sched.add(pending.pop(0))
            if sched.is_finished():
                continue
            step(sched, runner)
        assert all(seq.status == SequenceStatus.FINISHED for seq in seqs)
        bm = sched.block_manager
        assert len(bm.free_state_groups) == sched.max_num_seqs and not bm.used_block_ids
        total_restores += runner.restores
    print(f"stress: {total_restores} restores checked")
    assert total_restores > 50


if __name__ == "__main__":
    test_hit_lengths_per_policy()
    test_snapshot_pool_ids()
    test_snapshot_dropped_with_block()
    test_mtp_hashing_after_accepted_draft()
    test_random_stress()
    print("PASS")
