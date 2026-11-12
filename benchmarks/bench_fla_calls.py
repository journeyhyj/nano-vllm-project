# Host cost of FLA chunk_gated_delta_rule calls as the GDN layers issue them in one prefill step (24 layers of
# Qwen3.5-4B: 32 value heads, K = V = 128). Shows why a snapshot that cuts the FLA window is expensive: the cut doubles
# the calls, each costs ~1 ms of host time even for a few tokens, and alternating windows defeat FLA's single-entry
# index cache. Times are wall clock of issuing the calls (host) and until the GPU finished (total).
# Usage: python benchmarks/bench_fla_calls.py [--layers 24] [--reps 7]
import argparse
import json
import time

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

H, K, V = 32, 128, 128


def make(T, dev="cuda"):
    q, k = (torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16) for _ in range(2))
    v = torch.randn(1, T, H, V, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(1, T, H, device=dev, dtype=torch.float32)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.float32)
    return q, k, v, g, beta


def cu(bounds):
    c = torch.tensor(bounds, dtype=torch.int64)
    return c.pin_memory().cuda(non_blocking=True), c


def call(inputs, c, state):
    q, k, v, g, beta = inputs
    return chunk_gated_delta_rule(q, k, v, g, beta, scale=K ** -0.5, initial_state=state, output_final_state=True,
                                  use_qk_l2norm_in_kernel=True, cu_seqlens=c[0], cu_seqlens_cpu=c[1])


def bench(windows, layers, reps):
    for _ in range(3):    # compile and autotune
        for w in windows:
            call(*w)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(layers):
            for w in windows:
                call(*w)
        host = time.perf_counter() - t0
        torch.cuda.synchronize()
        times.append((host * 1e3, (time.perf_counter() - t0) * 1e3))
    host, total = sorted(times)[len(times) // 2]
    return dict(host_ms=host, total_ms=total, calls=layers * len(windows), host_ms_per_call=host / layers / len(windows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--reps", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(0)
    s = torch.zeros(1, H, K, V, device="cuda")
    s2 = torch.zeros(2, H, K, V, device="cuda")
    cases = {
        "1 call, 16 tokens": [(make(16), cu([0, 16]), s)],
        "1 call, 2064 tokens": [(make(2064), cu([0, 2064]), s)],
        "cut at 2048: calls of 2048 + 16 tokens": [(make(2048), cu([0, 2048]), s), (make(16), cu([0, 16]), s)],
        "extra sequence: 1 call of 2064 + 2048 tokens": [(make(4112), cu([0, 2064, 4112]), s2)],
        "2 calls sharing one cu_seqlens object": [(make(1024), cu([0, 1024]), s)] * 2,
    }
    results = {}
    for name, windows in cases.items():
        results[name] = r = bench(windows, args.layers, args.reps)
        print(f"{name:46s} host {r['host_ms']:6.1f} ms  total {r['total_ms']:6.1f} ms  "
              f"({r['calls']} calls, {r['host_ms_per_call']:.2f} ms host each)")
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, layers=args.layers, reps=args.reps,
                          results=results)))


if __name__ == "__main__":
    main()
