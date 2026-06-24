import os
import time
import numpy as np
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("/workspace/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096, enable_prefix_caching=True)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t

    ttfts = [o["ttft"] for o in outputs if o["ttft"] is not None]
    latencies = [o["latency"] for o in outputs if o["latency"] is not None]

    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    print(f"TTFT   - mean: {np.mean(ttfts)*1000:.1f}ms, median: {np.median(ttfts)*1000:.1f}ms, p99: {np.percentile(ttfts, 99)*1000:.1f}ms")
    print(f"Latency- mean: {np.mean(latencies)*1000:.1f}ms, median: {np.median(latencies)*1000:.1f}ms, p99: {np.percentile(latencies, 99)*1000:.1f}ms")


if __name__ == "__main__":
    main()
