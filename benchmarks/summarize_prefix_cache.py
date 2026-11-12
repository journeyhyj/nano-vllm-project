# Turns bench_prefix_cache.py JSON records into Markdown tables: one row per (run, policy), repeats shown as mean and
# min-max; TTFT speedups are relative to the baseline directory (the engine before hybrid prefix caching).
# Usage: python benchmarks/summarize_prefix_cache.py benchmarks/results/baseline benchmarks/results/after [...] > table.md
import glob
import json
import os
import re
import sys
from collections import defaultdict


def load(dirs):
    groups = defaultdict(list)    # (run name, policy label) -> records
    for d in dirs:
        label = os.path.basename(os.path.normpath(d))
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            m = re.match(r"(.+)_(\w+)_r(\d+)\.json$", os.path.basename(path))
            run, policy = m.group(1), m.group(2)
            rec = json.load(open(path))
            rec["path"] = path
            groups[(run, policy if policy != "default" else label)].append(rec)
    return groups


def fmt(values, digits=0):
    mean = sum(values) / len(values)
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({min(values):.{digits}f}–{max(values):.{digits}f})"


def main():
    dirs = sys.argv[1:]
    groups = load(dirs)
    runs = sorted({run for run, _ in groups}, key=lambda r: (r.split("_")[0].rstrip("0123456789"), int(re.search(r"\d+", r).group()), r))
    baseline_label = os.path.basename(os.path.normpath(dirs[0]))
    order = [baseline_label, "off", "kv_only", "last", "branch", "all"]
    print("| run | policy | n | TTFT p50 ms | TTFT p99 ms | p50 speedup | hit rate | prefill tokens computed | "
          "output tok/s | snapshots (MiB) | KV blocks |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        policies = sorted({p for r, p in groups if r == run}, key=lambda p: order.index(p) if p in order else 99)
        base = groups.get((run, baseline_label))
        base_p50 = sum(r["ttft_p50_ms"] for r in base) / len(base) if base else None
        for policy in policies:
            recs = groups[(run, policy)]
            p50 = [r["ttft_p50_ms"] for r in recs]
            speedup = f"{base_p50 / (sum(p50) / len(p50)):.2f}x" if base_p50 else "–"
            print(f"| {run} | {policy} | {len(recs)} | {fmt(p50)} | {fmt([r['ttft_p99_ms'] for r in recs])} | {speedup} | "
                  f"{fmt([100 * r['hit_rate'] for r in recs], 1)}% | {fmt([r['prefill_tokens_computed'] for r in recs])} | "
                  f"{fmt([r['output_tok_per_s'] for r in recs])} | {recs[0]['snapshot_bytes'] / 2**20:.0f} | "
                  f"{recs[0]['num_kvcache_blocks']} |")
    first = next(iter(groups.values()))[0]
    print(f"\nGPU: {first['gpu']}, torch {first['torch']}, CUDA_VISIBLE_DEVICES={first['cuda_visible_devices']} "
          f"({first['cuda_device_order']}); engine: {json.dumps(first['engine'])}")


if __name__ == "__main__":
    main()
