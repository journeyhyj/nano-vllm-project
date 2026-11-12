# Prints the FLA call plan (plan_fla_windows) that one prefill of a single prompt gets when the prefix-cache policy
# saves snapshots at the given block boundaries, next to the number of calls the previous planner made (a window cut at
# every snapshot row). The budget is the engine's ModelRunner.fla_budget = min(8192, num_seqs * seq_len) from warmup,
# which is 8192 for both bench_snapshot_overhead.py (max_model_len = len + 128) and bench_prefix_cache.py.
# Usage: python benchmarks/show_fla_plan.py [--budget 8192]   (needs a GPU: the planner pins and uploads cu_seqlens)
import argparse

from nanovllm.layers.linear_attention import plan_fla_windows


def show(name, length, rows, budget):
    snapshots = [(0, r, k) for k, r in enumerate(rows)]
    plan = plan_fla_windows([0, length], snapshots=snapshots, budget=budget)
    cut_calls = len({w[1] for w in plan_fla_windows([0, length])} | set(rows))
    windows = []
    for start, end, first, last, cu_gpu, cu_cpu, snaps, extra in plan:
        extra_tokens = 0 if extra is None else len(extra[0]) - (end - start)
        windows.append(f"({start}, {end})" + (f"+{extra_tokens} extra" if extra_tokens else ""))
    print(f"{name:36s} {len(plan)} FLA calls per layer (cut at every row: {cut_calls}): {', '.join(windows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=8192)
    args = parser.parse_args()
    b = args.budget
    show("last, 2064-token prompt", 2064, [2048], b)
    show("last, 8064-token prompt", 8064, [7936], b)
    show("all, 2064-token prompt", 2064, list(range(256, 2064, 256)), b)
    show("all, 8064-token prompt", 8064, list(range(256, 8064, 256)), b)
    show("all, 8215-token prompt", 8215, list(range(256, 8215, 256)), b)


if __name__ == "__main__":
    main()
