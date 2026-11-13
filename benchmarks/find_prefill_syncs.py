# Finds host<->device synchronizations inside one prefill forward: torch.cuda.set_sync_debug_mode("warn") counts the
# synchronizing CUDA calls of the forward, then mode "error" stops at the first one and prints where it came from. A
# synchronization makes the host wait for the GPU queue to drain (no overlap of launching and computing) and makes the
# forward impossible to capture in a CUDA graph.
# Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 [FLA_INTRACARD_CP=1] python benchmarks/find_prefill_syncs.py \
#            ~/huggingface/Qwen3.5-4B [--tokens 64]
import argparse
import os
import random
import traceback
import warnings

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    eager = {"prefill_graph_tokens": 0} if "prefill_graph_tokens" in Config.__dataclass_fields__ else {}
    llm = LLM(os.path.expanduser(args.model), max_model_len=4096, max_num_seqs=16, gpu_memory_utilization=0.6,
              prefix_cache_policy="off", **eager)    # the eager forward (engines before prefill graphs have no option)
    mr = llm.model_runner
    rng = random.Random(0)
    prompt = lambda: [rng.randint(1000, 100000) for _ in range(args.tokens)]
    sp = SamplingParams(temperature=1e-5, max_tokens=1)
    for _ in range(2):    # warmup
        llm.generate([prompt()], sp, use_tqdm=False)
    orig = mr.run_model
    result = {}

    def counting(input_ids, positions, is_prefill):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                return orig(input_ids, positions, is_prefill)
            finally:
                torch.cuda.set_sync_debug_mode("default")
                syncs = [w for w in caught if "synchronizing" in str(w.message)]
                result["count"] = len(syncs)
                result["sites"] = sorted({f"{w.filename.split('site-packages/')[-1]}:{w.lineno}" for w in syncs})

    def first(input_ids, positions, is_prefill):
        torch.cuda.set_sync_debug_mode("error")
        try:
            return orig(input_ids, positions, is_prefill)
        except RuntimeError as e:
            result["first"] = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno} {f.name}  |  {f.line}"
                               for f in traceback.extract_tb(e.__traceback__)[-8:]]
            raise
        finally:
            torch.cuda.set_sync_debug_mode("default")

    mr.run_model = counting
    llm.generate([prompt()], sp, use_tqdm=False)
    print(f"FLA_INTRACARD_CP={os.environ.get('FLA_INTRACARD_CP', '(unset)')}: {result['count']} synchronizing CUDA calls "
          f"in one {args.tokens}-token prefill forward, at {result['sites']}", flush=True)
    if result["count"]:
        mr.run_model = first
        try:
            llm.generate([prompt()], sp, use_tqdm=False)
        except RuntimeError:
            pass
        print("first one (sync debug mode 'error'):\n  " + "\n  ".join(result.get("first", ["(not raised)"])), flush=True)
    os._exit(0)    # the aborted step leaves the engine mid-request


if __name__ == "__main__":
    main()
