# What hybrid prefix caching costs and saves per request, one request at a time (no queueing):
#   miss  prefill latency of a prompt that shares nothing (fresh random tokens): the policy's snapshot saves cost FLA
#         window cuts (last: 1 per prompt, all: 1 per 256 tokens) and state copies
#   hit   prefill latency of a --len + --tail token prompt sent for the 3rd time (every policy then has a snapshot at
#         its last full block; branch saves it on the 2nd occurrence): restore + prefill of the last partial block
#   short prefill latency of a fresh --tail token prompt: the engine's fixed cost of a prefill step (eager, not a CUDA
#         graph), which bounds a hit from below
#   copy  time of copying one snapshot into a state slot (all linear-attention layers), CUDA events
# Usage: python benchmarks/bench_snapshot_overhead.py <model> --policy branch --len 8000 [--tail 64] [--repeats 5]
import argparse
import json
import os
import random
import statistics
import subprocess
import time

import torch

import nanovllm
from nanovllm import LLM, SamplingParams


def timed_prefill(llm, prompt):
    torch.cuda.synchronize()
    t = time.perf_counter()
    llm.generate([prompt], SamplingParams(temperature=1e-5, max_tokens=1), use_tqdm=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--policy", default="branch")
    parser.add_argument("--len", type=int, default=8000)
    parser.add_argument("--tail", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-state-snapshots", type=int, default=48)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    llm = LLM(os.path.expanduser(args.model), max_model_len=args.len + args.tail + 64, max_num_seqs=16,
              gpu_memory_utilization=args.gpu_memory_utilization, prefix_cache_policy=args.policy,
              num_state_snapshots=args.num_state_snapshots)
    rng = random.Random(0)
    fresh = lambda n: [rng.randint(1000, 100000) for _ in range(n)]
    for n in (args.tail, args.len, args.len + args.tail):    # warmup: compile and autotune
        timed_prefill(llm, fresh(n))

    miss = [timed_prefill(llm, fresh(args.len + args.tail)) for _ in range(args.repeats)]
    short = [timed_prefill(llm, fresh(args.tail)) for _ in range(args.repeats)]
    hit, cached = [], []
    for _ in range(args.repeats):
        prompt = fresh(args.len + args.tail)
        for _ in range(2):    # the 2nd occurrence makes branch save the shared prefix
            timed_prefill(llm, prompt)
        before = llm.scheduler.block_manager.counters["hit_tokens"]
        hit.append(timed_prefill(llm, prompt))
        cached.append(llm.scheduler.block_manager.counters["hit_tokens"] - before)

    copy_ms = None
    mr = llm.model_runner
    if mr.snapshot_bytes:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(20):
            start.record()
            mr.copy_states([0], [mr.config.num_state_groups * mr.query_len], to_snapshots=False)    # into the spare slot
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        copy_ms = statistics.median(times[5:])
    result = dict(policy=args.policy, len=args.len, tail=args.tail, repeats=args.repeats,
                  miss_prefill_ms=statistics.median(miss), miss_all_ms=miss,
                  hit_prefill_ms=statistics.median(hit), hit_all_ms=hit, hit_cached_tokens=cached,
                  short_prefill_ms=statistics.median(short), short_all_ms=short,
                  snapshot_restore_copy_ms=copy_ms, snapshot_bytes_each=getattr(mr, "state_bytes", 0),
                  num_state_snapshots=mr.config.num_state_snapshots, num_kvcache_blocks=mr.config.num_kvcache_blocks,
                  stats=llm.scheduler.block_manager.stats, gpu=torch.cuda.get_device_name(),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    code = os.path.dirname(os.path.abspath(nanovllm.__file__))    # the engine actually imported (PYTHONPATH worktrees)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=code).stdout.strip()
    result.update(nanovllm_path=code, git=git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else ""))
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in result.items()}, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
