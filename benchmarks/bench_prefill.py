# Prefill latency: --bs prompts of --len random tokens, prefilled in one step (only the first token is generated).
# Every run uses fresh tokens so prefix caching (non-hybrid models) never hits; the best of --repeats runs is reported.
# Usage: python benchmarks/bench_prefill.py <model dir> [--len 8192] [--bs 1] [--repeats 3] [--gpu-memory-utilization 0.6]
import argparse
import os
import random
import time
import torch
from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--len", type=int, default=8192)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()

    num_tokens = args.len * args.bs
    llm = LLM(os.path.expanduser(args.model), max_model_len=args.len + 64, max_num_batched_tokens=max(num_tokens, args.len + 64),
              gpu_memory_utilization=args.gpu_memory_utilization)
    sampling_params = SamplingParams(temperature=1e-5, max_tokens=1)
    best = float("inf")
    for i in range(args.repeats + 1):    # the first run is a warmup
        random.seed(i)
        prompts = [[random.randint(100, 10000) for _ in range(args.len)] for _ in range(args.bs)]
        torch.cuda.synchronize()
        t = time.perf_counter()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        if i:
            best = min(best, time.perf_counter() - t)
    print(f"prefill {args.bs} x {args.len} tokens: {best * 1000:.0f} ms ({num_tokens / best:.0f} tok/s)")


if __name__ == "__main__":
    main()
