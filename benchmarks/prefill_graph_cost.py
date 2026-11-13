# Startup cost of prefill CUDA graphs: engine construction time, number of graphs, GPU memory reserved by the process
# and KV cache blocks, with prefill_graph_tokens set as given (0 = no prefill graphs).
# Usage: python benchmarks/prefill_graph_cost.py ~/huggingface/Qwen3.5-4B --prefill-graph-tokens 1024
import argparse
import json
import os
import time

import torch

from nanovllm import LLM

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--prefill-graph-tokens", type=int, default=1024)
parser.add_argument("--k", type=int, default=0)
args = parser.parse_args()
t = time.perf_counter()
llm = LLM(os.path.expanduser(args.model), max_model_len=10240, max_num_seqs=16, max_num_batched_tokens=16384,
          gpu_memory_utilization=0.85, num_speculative_tokens=args.k, prefill_graph_tokens=args.prefill_graph_tokens)
init_s = time.perf_counter() - t
mr = llm.model_runner
print(json.dumps(dict(prefill_graph_tokens=args.prefill_graph_tokens, k=args.k, engine_init_s=round(init_s, 1),
                      prefill_graphs=len(mr.prefill_graphs), decode_graphs=len(mr.graphs),
                      memory_reserved_mib=round(torch.cuda.memory_reserved() / 2**20),
                      num_kvcache_blocks=mr.config.num_kvcache_blocks)), flush=True)
os._exit(0)
