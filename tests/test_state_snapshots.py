# Recurrent-state snapshots of LinearAttention (prefix caching for hybrid models), bit for bit:
#  - plan_fla_windows cuts at 64-aligned offsets; a snapshot row inside a sequence gets an extra FLA sequence ending at
#    the row, or a window cut when the extra tokens do not fit the window
#  - a snapshot taken at a block boundary inside a prefill batch equals the state after prefilling just that prefix,
#    through extra sequences, window cuts or both
#  - restoring a snapshot into another slot and prefilling the rest gives the same outputs and states as never stopping
#  - a chunk that starts 1-2 tokens before a boundary takes the conv snapshot partly from the previous conv state
#  - a sequence's outputs do not depend on where it sits in the batch (FLA windows used to cut it at unaligned offsets)
#  - plan_fla_windows on random batches: every call within budget, every snapshot served once from the right tokens,
#    and never more FLA calls than cutting at every snapshot row (the previous planner)
# FLA autotuning is pinned (fla_pin.py) so that window sizes do not pick different kernel configs.
# Usage: python tests/test_state_snapshots.py   (needs a GPU)
import random
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from fla_pin import pin_fla_autotune
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.layers.linear_attention import LinearAttention, plan_fla_windows
from nanovllm.utils.context import set_context, reset_context

DEVICE, DTYPE = "cuda", torch.bfloat16
HK, HV, K, V, W = 4, 8, 128, 128, 4
C = 2 * HK * K + HV * V


class Harness:

    def __init__(self, num_slots=16, num_snapshots=16):
        torch.manual_seed(0)
        self.layer = LinearAttention(HK, HV, K, V, W)
        self.layer.conv_state = torch.zeros(num_slots, W - 1, C, dtype=DTYPE, device=DEVICE)
        self.layer.ssm_state = torch.zeros(num_slots, HV, K, V, dtype=torch.float32, device=DEVICE)
        self.layer.conv_snap = torch.zeros(num_snapshots, W - 1, C, dtype=DTYPE, device=DEVICE)
        self.layer.ssm_snap = torch.zeros(num_snapshots, HV, K, V, dtype=torch.float32, device=DEVICE)
        self.weight = torch.randn(C, 1, W, dtype=DTYPE, device=DEVICE) * 0.3
        # a spread of decay rates: heads with exp(A_log) ~ 0.01 remember thousands of tokens, so a wrong state would show
        self.A_log = torch.linspace(-5, 1, HV, device=DEVICE)
        self.dt_bias = torch.randn(HV, device=DEVICE)

    def inputs(self, T):
        x = torch.randn(T, C + 7, dtype=DTYPE, device=DEVICE)[:, :C]    # non-contiguous rows, like the real split
        a, b = torch.randn(T, HV, device=DEVICE), torch.randn(T, HV, device=DEVICE)
        return x, -self.A_log.exp() * F.softplus(a + self.dt_bias), b.sigmoid()

    def prefill(self, chunks, slots, has_initial_state, snapshots=(), window=8192):
        """chunks: per sequence (x, g, beta) of this step; snapshots: (sequence index, boundary offset inside its chunk,
        snapshot id). Returns the output rows of every sequence."""
        lengths = [x.size(0) for x, _, _ in chunks]
        cu = [0]
        for L in lengths:
            cu.append(cu[-1] + L)
        x = torch.empty(cu[-1], C + 7, dtype=DTYPE, device=DEVICE)[:, :C]
        x.copy_(torch.cat([c[0] for c in chunks]))
        g = torch.cat([c[1] for c in chunks]).contiguous()
        beta = torch.cat([c[2] for c in chunks]).contiguous()
        rows = [(i, cu[i] + off, sid) for i, off, sid in snapshots]
        conv_snapshots = None
        if rows:    # the real ModelRunner code, on stand-in sequences that only carry their state slot
            runner = SimpleNamespace(conv_shape=(W - 1, C))
            seqs = [SimpleNamespace(state_slots=[s]) for s in slots]
            conv_snapshots = ModelRunner.prepare_conv_snapshots(runner, seqs, cu, rows)
        set_context(True, cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=DEVICE), max_seqlen_q=max(lengths),
                    state_slots=torch.tensor(slots, dtype=torch.int32, device=DEVICE),
                    has_initial_state=torch.tensor(has_initial_state, dtype=torch.int32, device=DEVICE),
                    fla_windows=plan_fla_windows(cu, window, rows), conv_snapshots=conv_snapshots)
        out = self.layer(x, self.weight, g, beta)
        reset_context()
        return [out[cu[i]:cu[i + 1]] for i in range(len(chunks))]

    def state(self, slot):
        return self.layer.conv_state[slot].clone(), self.layer.ssm_state[slot].clone()

    def snapshot(self, sid):
        return self.layer.conv_snap[sid].clone(), self.layer.ssm_snap[sid].clone()


def cut(seq, start, end):
    return tuple(t[start:end] for t in seq)


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_window_plan():
    # second sequence starts at 3192: the 8192 cut falls 5000 tokens into it and moves back to 4992 = 78 * 64
    windows = plan_fla_windows([0, 3192, 11192], 8192)
    assert [(w[0], w[1]) for w in windows] == [(0, 8184), (8184, 11192)]
    snaps = lambda w: None if w[6] is None else (w[6][0].tolist(), w[6][1].tolist())
    # rows inside a sequence ride along as extra sequences [0, 256) and [0, 768); a row at a sequence end is its state
    (w,) = plan_fla_windows([0, 1000, 1300], 8192, [(0, 256, 5), (0, 768, 6), (1, 1300, 7)])
    assert (w[0], w[1]) == (0, 1300) and w[5].tolist() == [0, 1000, 1300, 1556, 2324]
    assert snaps(w) == ([2, 3, 1], [5, 6, 7])
    assert w[7][0].tolist() == list(range(1300)) + list(range(256)) + list(range(768)) and w[7][1].tolist() == [0, 1, 0, 0]
    # no room for extra tokens in a 1024-token window: the rows cut it instead
    windows = plan_fla_windows([0, 1000, 1300], 1024, [(0, 256, 5), (0, 768, 6), (1, 1300, 7)])
    assert [(w[0], w[1]) for w in windows] == [(0, 256), (256, 768), (768, 1000), (1000, 1300)]
    assert [snaps(w) for w in windows] == [([0], [5]), ([0], [6]), None, ([0], [7])] and all(w[7] is None for w in windows)
    # mixed: [0, 1536) does not fit next to 1700 tokens and cuts; [0, 256) then fits in the window before the cut
    windows = plan_fla_windows([0, 1700], 2048, [(0, 256, 1), (0, 1536, 2)])
    assert [(w[0], w[1]) for w in windows] == [(0, 1536), (1536, 1700)]
    assert snaps(windows[0]) == ([1, 0], [1, 2]) and windows[0][5].tolist() == [0, 1536, 1792]
    assert windows[1][6] is None and windows[1][7] is None
    # one cut per window and round, so cuts make room: 26 prompts of 300 tokens, a row at 256 in each, run in 4 calls
    windows = plan_fla_windows([300 * k for k in range(27)], 8192, [(k, 300 * k + 256, k) for k in range(26)])
    assert [(w[0], w[1]) for w in windows] == [(0, 556), (556, 1756), (1756, 4456), (4456, 7800)]
    assert sum(len(w[6][1]) for w in windows) == 26 and all(w[5][-1] <= 8192 for w in windows)
    # a call with extra sequences stays within the budget (the largest call of the engine's warmup), else the row cuts
    (w,) = plan_fla_windows([0, 2048], 8192, [(0, 1792, 0)])
    assert w[5].tolist() == [0, 2048, 3840]
    windows = plan_fla_windows([0, 2048], 8192, [(0, 1792, 0)], budget=2048)
    assert [(w[0], w[1]) for w in windows] == [(0, 1792), (1792, 2048)] and windows[0][7] is None


def test_window_plan_fuzz(cases=2000, seed=0):
    rng = random.Random(seed)
    calls = cut_calls = extra_tokens = real_tokens = 0
    for _ in range(cases):
        lengths = [rng.choice([rng.randint(1, 300), rng.randint(1, 3000), rng.randint(1, 12000)])
                   for _ in range(rng.randint(1, 16))]
        cu = [0]
        for n in lengths:
            cu.append(cu[-1] + n)
        window = rng.choice([8192, 8192, 4096, 2048, 64 * rng.randint(1, 128)])
        budget = rng.choice([None, None, rng.randint(64, 16384)])
        snapshots = []
        for i, n in enumerate(lengths):    # block boundaries after a cached prefix (the engine's case), some at the end
            cached = rng.choice([0, 0, 256, 512, 16384, rng.randint(0, 1000)])
            rows = [b - cached for b in range(256, cached + n + 1, 256) if b > cached and rng.random() < 0.5]
            if rng.random() < 0.3:
                rows.append(n)
            snapshots += [(i, cu[i] + r, len(snapshots) + k) for k, r in enumerate(sorted(set(rows)))]
        rng.shuffle(snapshots)
        windows = plan_fla_windows(cu, window, snapshots, budget=budget)
        limit = window if budget is None else min(window, budget)
        done = [0] * len(lengths)    # tokens of each sequence that FLA has run, as the layer carries states
        served = {}
        assert windows[0][0] == 0 and windows[-1][1] == cu[-1]
        for k, (start, end, first, last, cu_gpu, bounds, snaps, extra) in enumerate(windows):
            assert start < end and (k == 0 or windows[k - 1][1] == start)
            bounds = bounds.tolist()
            assert cu_gpu.tolist() == bounds and bounds[last - first] == end - start
            if extra is not None:
                assert bounds[-1] <= limit, (bounds[-1], limit)
                rows, seqs = extra[0].tolist(), extra[1].tolist()
                assert rows[:end - start] == list(range(start, end)) and seqs[:last - first] == list(range(first, last))
            else:
                rows, seqs = list(range(start, end)), list(range(first, last))
            states = [done[i] for i in seqs]    # initial states, gathered before the call
            for j, i in enumerate(seqs):        # every FLA sequence continues its sequence from the state it got
                span = rows[bounds[j]:bounds[j + 1]]
                assert span == list(range(cu[i] + states[j], cu[i] + states[j] + len(span))) and span
                states[j] += len(span)
            for j, i in enumerate(range(first, last)):
                done[i] = states[j]
            if snaps is not None:
                for j, sid in zip(*(t.tolist() for t in snaps)):
                    assert sid not in served
                    served[sid] = (seqs[j], states[j])
            calls += 1
            real_tokens += end - start
            extra_tokens += bounds[-1] - (end - start)
        assert done == lengths
        assert served == {sid: (i, row - cu[i]) for i, row, sid in snapshots}    # the state right at each row
        regular = len(plan_fla_windows(cu, window))
        cut = len(set(w[1] for w in plan_fla_windows(cu, window)) | {row for i, row, _ in snapshots if row < cu[i + 1]})
        assert len(windows) <= cut and len(windows) >= regular
        cut_calls += cut
    print(f"window plan fuzz: {cases} batches, {calls} FLA calls (cutting at every row: {cut_calls}), "
          f"extra tokens {100 * extra_tokens / real_tokens:.1f}% of the batch tokens")


def test_snapshots_and_restore(window):
    h = Harness()
    A, B = h.inputs(1000), h.inputs(512)
    rows = [(0, 256, 0), (0, 512, 1), (0, 768, 2), (1, 512, 3)]
    plan = plan_fla_windows([0, 1000, 1512], window, [(i, [0, 1000][i] + off, sid) for i, off, sid in rows])
    # one batch: A snapshotted at 256 / 512 / 768, B at its end (512)
    out_A, _ = h.prefill([A, B], [0, 1], [0, 0], snapshots=rows, window=window)
    full_A = h.state(0)
    for b, sid in ((256, 0), (512, 1), (768, 2)):
        h.prefill([cut(A, 0, b)], [5], [0])
        assert same(h.snapshot(sid), h.state(5)), f"snapshot at {b} differs from the state after prefilling {b} tokens"
    assert same(h.snapshot(3), h.state(1))
    # reference without any cut, alone in the batch
    (ref_A,) = h.prefill([A], [6], [0])
    assert torch.equal(out_A, ref_A) and same(full_A, h.state(6))
    # restore the snapshot at 512 into a fresh slot and prefill the rest
    h.layer.conv_state[7], h.layer.ssm_state[7] = h.layer.conv_snap[1], h.layer.ssm_snap[1]
    (tail,) = h.prefill([cut(A, 512, 1000)], [7], [1])
    assert torch.equal(tail, ref_A[512:]), (tail.float() - ref_A[512:].float()).abs().max()
    assert same(h.state(7), full_A)
    extra = sum(w[7] is not None and (len(w[7][0]) - (w[1] - w[0])) for w in plan)
    print(f"snapshot / restore, window {window}: bit-identical ({len(plan)} FLA calls, {extra} extra tokens)")
    return len(plan), extra


def test_conv_fallback():
    # a chunk starting at 510: the conv snapshot at 512 needs inputs 509 (previous conv state), 510 and 511 (this chunk)
    h = Harness()
    A = h.inputs(1000)
    h.prefill([cut(A, 0, 510)], [0], [0])
    h.prefill([cut(A, 510, 1000)], [0], [1], snapshots=[(0, 2, 0), (0, 258, 1)])
    for b, sid in ((512, 0), (768, 1)):
        h.prefill([cut(A, 0, b)], [5], [0])
        ref_conv, ref_ssm = h.state(5)
        conv, ssm = h.snapshot(sid)
        assert torch.equal(conv, ref_conv), f"conv snapshot at {b}"
        # the chunk starts at 510, so FLA's 64-token chunks are shifted against the reference: close, not bit-identical
        err = ((ssm - ref_ssm).norm() / ref_ssm.norm()).item()
        assert err < 2e-2, err
        print(f"conv fallback at {b}: conv bit-identical, ssm relative error {err:.1e}")


def test_batch_position_invariance():
    h = Harness()
    A, B = h.inputs(3192), h.inputs(6000)
    (alone,) = h.prefill([B], [0], [0])
    _, behind = h.prefill([A, B], [1, 2], [0, 0])
    assert torch.equal(alone, behind) and same(h.state(0), h.state(2))
    print("same outputs alone and behind a 3192-token sequence")


if __name__ == "__main__":
    print(f"pinned {pin_fla_autotune()} FLA autotuners")
    test_window_plan()
    test_window_plan_fuzz()
    # extra sequences only (8192), extra sequences and cuts (2048, 1024), cuts only (512)
    plans = {window: test_snapshots_and_restore(window) for window in (8192, 2048, 1024, 512)}
    assert plans[8192][0] == 1 and plans[512][1] == 0 and all(plans[w][1] for w in (2048, 1024))
    test_conv_fallback()
    test_batch_position_invariance()
    print("PASS")
