# CUDA graph prefill (ModelRunner.run_prefill_graph) against the eager prefill it replaces, in one engine:
#  - first-token logits of the same request through both paths, for single sequences across the token buckets, batches
#    up to and past 16 sequences, the second chunk of a chunked prefill (initial state from the first chunk) and the tail
#    of a prefix-cache hit (state restored from a snapshot). The graph path runs the fused recurrent GDN kernel instead
#    of the chunked one, so logits differ by rounding: required are max |dlogit| <= 1.0 and the same argmax unless the
#    eager top-2 gap is <= 0.5 (a bf16 near-tie), the criteria of test_prefix_cache_e2e.py
#  - isolation: a sequence that is decoding while another one is prefilled through the graph (padding rows, padding
#    sequences on the spare state slot) generates exactly the same tokens as when that prefill runs eagerly
#  - eligibility: prefills that are too long, have too many sequences or save a snapshot stay eager
# Usage: NANOVLLM_PORT=2398 python tests/test_prefill_graph.py [model dir] [--k 1]
import argparse
import os
import random

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="?", default="~/huggingface/Qwen3.5-4B")
parser.add_argument("--k", type=int, default=0)
ARGS = parser.parse_args()
MODEL, K = os.path.expanduser(ARGS.model), ARGS.k
GREEDY = SamplingParams(temperature=1e-5, max_tokens=1)


class Probe:
    """Records the logits of each prefill and whether the step went through the graph."""
    def __init__(self, llm):
        mr = llm.model_runner
        self.mr, self.logits, self.graph_steps = mr, [], 0
        compute_logits, run_graph = mr.model.compute_logits, mr.run_prefill_graph

        def recording_logits(hidden_states):
            out = compute_logits(hidden_states)
            if get_is_prefill():
                self.logits.append(out.float().clone())
            return out

        def counting_graph(*args):
            self.graph_steps += 1
            return run_graph(*args)
        from nanovllm.utils.context import get_context
        get_is_prefill = lambda: get_context().is_prefill
        mr.model.compute_logits = recording_logits
        mr.run_prefill_graph = counting_graph
        self.graphs = mr.prefill_graphs

    def run(self, llm, prompts, graph):
        self.mr.prefill_graphs = self.graphs if graph else {}
        self.logits.clear()
        before = self.graph_steps
        llm.generate(prompts, GREEDY, use_tqdm=False)
        self.mr.prefill_graphs = self.graphs
        # the last prefill step: the one that produces every sequence's first token (a run can have fewer steps than
        # the other when it finds the earlier chunk's blocks in the prefix cache)
        return self.logits[-1], self.graph_steps - before


def compare(name, eager, graph, expect_graph, used):
    assert (used > 0) == expect_graph, f"{name}: graph steps {used}, expected graph={expect_graph}"
    diff = (eager - graph).abs().max().item()
    top2 = eager.topk(2, dim=-1).values
    gap = (top2[:, 0] - top2[:, 1])
    same = eager.argmax(-1) == graph.argmax(-1)
    ok = diff <= 1.0 and bool((same | (gap <= 0.5)).all())
    print(f"{name:44s} graph steps {used:2d}  max|dlogit| {diff:.4f}  argmax equal {int(same.sum())}/{same.numel()}  "
          f"{'ok' if ok else 'FAIL'}")
    assert ok, name


def main():
    llm = LLM(MODEL, max_model_len=4096, max_num_seqs=32, max_num_batched_tokens=2048, gpu_memory_utilization=0.6,
              prefix_cache_policy="branch", num_speculative_tokens=K)
    mr = llm.model_runner
    print("prefill graphs:", sorted(mr.prefill_graphs), f"(MTP k={K})")
    probe = Probe(llm)
    rng = random.Random(0)
    fresh = lambda n: [rng.randint(1000, 100000) for _ in range(n)]

    def check(name, prompts, expect_graph=True):
        # graph first: the second run of the same prompt finds its full blocks cached, and under "branch" saves a
        # snapshot there, which keeps that run eager (snapshot outputs are bit-identical to not saving, see
        # test_state_snapshots.py), so the order does not change what is computed
        graph, used = probe.run(llm, prompts, graph=True)
        eager, _ = probe.run(llm, prompts, graph=False)
        compare(name, eager, graph, expect_graph, used)

    # single sequences across the buckets, up to the largest one, and one past it
    cap = mr.prefill_graph_tokens[-1]
    for n in sorted({1, 5, 16, 17, 63, 64, 65, 200, 511, 512, 513, cap - 1, cap}):
        check(f"one sequence, {n} tokens", [fresh(n)])
    check(f"one sequence, {cap + 1} tokens (eager)", [fresh(cap + 1)], expect_graph=False)
    # batches
    check("3 sequences: 3 + 70 + 250 tokens", [fresh(3), fresh(70), fresh(250)])
    check("16 sequences x 20 tokens", [fresh(20) for _ in range(16)])
    check("16 sequences x 32 tokens (512)", [fresh(32) for _ in range(16)])
    check(f"16 sequences x {cap // 16} tokens ({cap})", [fresh(cap // 16) for _ in range(16)])
    check("17 sequences x 10 tokens (eager)", [fresh(10) for _ in range(17)], expect_graph=False)
    # chunked prefill (max_num_batched_tokens=2048): 2300 tokens = a 2048-token chunk (eager) and a 252-token chunk with
    # initial state (graph)
    check("chunked prefill: 2048 eager + 252 graph", [fresh(2300)])
    # prefix-cache hit: the second request saves a snapshot at the shared boundary (eager), the third hits it and
    # prefills only its tail through the graph
    shared = fresh(1024)
    probe.run(llm, [shared + fresh(40)], graph=False)
    _, used = probe.run(llm, [shared + fresh(40)], graph=True)
    hybrid = llm.scheduler.block_manager.snapshots is not None
    assert used == (0 if hybrid else 1), "a request that saves a snapshot must stay eager"
    print(f"{'second request (hybrid: saves a snapshot, eager)':44s} graph steps {used:2d}  ok")
    tail = shared + fresh(60)
    check("prefix-cache hit: 1024 cached + 60-token tail", [tail])
    assert not hybrid or llm.scheduler.block_manager.snapshots.stats["restored"] >= 2

    # isolation: A decodes while B is prefilled; A's tokens must not depend on how B was prefilled. A is prefilled
    # eagerly in both runs (graphs are enabled only once B is queued) and has no full block, so the second run finds
    # nothing of A cached and A's own computation is the same in both runs
    def interleaved(graph):
        mr.prefill_graphs = {}
        a = Sequence(list(range(2000, 2200)), SamplingParams(temperature=1e-5, max_tokens=48, ignore_eos=True))
        llm.scheduler.add(a)
        for _ in range(6):
            llm.step()
        b = Sequence(list(range(5000, 5100)), SamplingParams(temperature=1e-5, max_tokens=8, ignore_eos=True))
        llm.scheduler.add(b)
        mr.prefill_graphs = probe.graphs if graph else {}
        while not llm.is_finished():
            llm.step()
        mr.prefill_graphs = probe.graphs
        return a.completion_token_ids, b.completion_token_ids
    before = probe.graph_steps
    a_eager, b_eager = interleaved(graph=False)
    a_graph, b_graph = interleaved(graph=True)
    assert probe.graph_steps - before == 1, "only B should have been prefilled through the graph"
    assert a_eager == a_graph, "a graph prefill of another sequence changed A's tokens"
    print(f"{'isolation: A decoding while B is prefilled':44s} A identical ({len(a_graph)} tokens), "
          f"B identical: {b_eager == b_graph}  ok")
    print("PASS")


if __name__ == "__main__":
    main()
