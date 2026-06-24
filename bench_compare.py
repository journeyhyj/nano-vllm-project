"""Skip-and-Defer 策略对比 Benchmark

用法：
    python bench_compare.py

运行两轮相同参数的 benchmark：
  1. 原版 scheduler（级联抢占）
  2. 当前 scheduler（Skip-and-Defer）
对比 TTFT / Latency / Throughput。
"""

import os
import time
import numpy as np
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import LLMEngine


def benchmark_with_scheduler(scheduler_module, label: str):
    """用指定的 scheduler 类跑一轮 benchmark"""
    import nanovllm.engine.llm_engine as engine_mod

    # monkey-patch LLMEngine 使用指定 scheduler
    orig_scheduler = engine_mod.Scheduler
    engine_mod.Scheduler = scheduler_module

    try:
        seed(0)
        num_seqs = 256
        max_input_len = 1024
        max_ouput_len = 1024

        path = os.path.expanduser("/workspace/huggingface/Qwen3-0.6B/")
        llm = LLM(path, enforce_eager=False, max_model_len=4096, enable_prefix_caching=True)

        prompt_token_ids = [
            [randint(0, 10000) for _ in range(randint(100, max_input_len))]
            for _ in range(num_seqs)
        ]
        sampling_params = [
            SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
            for _ in range(num_seqs)
        ]

        # warmup
        llm.generate(["Benchmark: "], SamplingParams())

        t = time.time()
        outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        t = time.time() - t

        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / t

        ttfts = [o["ttft"] for o in outputs if o["ttft"] is not None]
        latencies = [o["latency"] for o in outputs if o["latency"] is not None]

        result = {
            "label": label,
            "total_tokens": total_tokens,
            "time": t,
            "throughput": throughput,
            "ttft_mean": np.mean(ttfts) * 1000,
            "ttft_median": np.median(ttfts) * 1000,
            "ttft_p99": np.percentile(ttfts, 99) * 1000,
            "latency_mean": np.mean(latencies) * 1000,
            "latency_median": np.median(latencies) * 1000,
            "latency_p99": np.percentile(latencies, 99) * 1000,
        }

        # 清理，避免 OOM
        del llm
        import torch
        torch.cuda.empty_cache()

        return result
    finally:
        engine_mod.Scheduler = orig_scheduler


def main():
    from nanovllm.engine.scheduler_original import SchedulerOriginal
    from nanovllm.engine.scheduler import Scheduler as SchedulerNew

    print("=" * 70)
    print("  Skip-and-Defer 策略对比 Benchmark")
    print("=" * 70)
    print()
    print("参数: 256 seqs, input 100~1024, output 100~1024")
    print("模型: Qwen3-0.6B, prefix_caching=True, CUDA Graph enabled")
    print()

    # ===== 轮次 1: 原版（级联抢占）=====
    print("[1/2] 运行原版 Scheduler（级联抢占）...")
    r1 = benchmark_with_scheduler(SchedulerOriginal, "原版 (Cascade Preempt)")
    print(f"      完成！耗时 {r1['time']:.1f}s\n")

    # ===== 轮次 2: 新版（Skip-and-Defer）=====
    print("[2/2] 运行新版 Scheduler（Skip-and-Defer）...")
    r2 = benchmark_with_scheduler(SchedulerNew, "新版 (Skip-and-Defer)")
    print(f"      完成！耗时 {r2['time']:.1f}s\n")

    # ===== 对比输出 =====
    def fmt_compare(r1_val, r2_val, unit=""):
        delta = r2_val - r1_val
        pct = (delta / r1_val * 100) if r1_val else 0
        sign = "↓" if delta < 0 else "↑"
        return f"{r1_val:.1f}{unit} → {r2_val:.1f}{unit}  ({sign}{abs(pct):.1f}%)"

    print("=" * 70)
    print("  对比结果")
    print("=" * 70)
    print()
    print(f"  {'指标':<18} {'原版 (Cascade)':<20} {'新版 (Skip&Defer)':<20} {'变化'}")
    print(f"  {'-'*18} {'-'*20} {'-'*20} {'-'*25}")
    print(f"  {'Throughput':<18} {r1['throughput']:<20.1f} {r2['throughput']:<20.1f} {fmt_compare(r1['throughput'], r2['throughput'], ' tok/s')}")
    print()
    print(f"  --- TTFT (首Token延迟) ---")
    print(f"  {'  Mean':<18} {r1['ttft_mean']:<20.1f} {r2['ttft_mean']:<20.1f} {fmt_compare(r1['ttft_mean'], r2['ttft_mean'], ' ms')}")
    print(f"  {'  Median':<18} {r1['ttft_median']:<20.1f} {r2['ttft_median']:<20.1f} {fmt_compare(r1['ttft_median'], r2['ttft_median'], ' ms')}")
    print(f"  {'  P99':<18} {r1['ttft_p99']:<20.1f} {r2['ttft_p99']:<20.1f} {fmt_compare(r1['ttft_p99'], r2['ttft_p99'], ' ms')}")
    print()
    print(f"  --- Latency (端到端延迟) ---")
    print(f"  {'  Mean':<18} {r1['latency_mean']:<20.1f} {r2['latency_mean']:<20.1f} {fmt_compare(r1['latency_mean'], r2['latency_mean'], ' ms')}")
    print(f"  {'  Median':<18} {r1['latency_median']:<20.1f} {r2['latency_median']:<20.1f} {fmt_compare(r1['latency_median'], r2['latency_median'], ' ms')}")
    print(f"  {'  P99':<18} {r1['latency_p99']:<20.1f} {r2['latency_p99']:<20.1f} {fmt_compare(r1['latency_p99'], r2['latency_p99'], ' ms')}")
    print()

    # 结论
    ttft_improve = (r1['ttft_p99'] - r2['ttft_p99']) / r1['ttft_p99'] * 100
    lat_improve = (r1['latency_p99'] - r2['latency_p99']) / r1['latency_p99'] * 100
    tp_improve = (r2['throughput'] - r1['throughput']) / r1['throughput'] * 100

    print("  === 总结 ===")
    print(f"  P99 TTFT 改善:   {ttft_improve:+.1f}%")
    print(f"  P99 Latency 改善: {lat_improve:+.1f}%")
    print(f"  Throughput 变化:  {tp_improve:+.1f}%")
    print()


if __name__ == "__main__":
    main()
