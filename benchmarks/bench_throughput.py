# Throughput under load, closed loop: each round submits C requests at once (random-token prompts, fixed output
# length, EOS ignored) and waits for all of them. Mirrors the method of the Chitu RTX 4090 report in
# artifacts/chitu-4090 (input 128 tokens before the chat template = 137-140 after it, output 1024, 3 warmup requests and
# 10 timed rounds per concurrency) so the numbers can be put side by side. Reports per concurrency: output and total
# token throughput (mean +- sample SD over rounds), mean TTFT and TPOT, the most sequences that actually ran at once
# (state slots can cap it below C), and the MTP acceptance.
# Usage: python benchmarks/bench_throughput.py ~/huggingface/Qwen3.5-4B --concurrency 1,2,4,8,16,32,64 [--k 1]
#        [--draft-vocab assets/qwen3_5_draft_vocab_32k.json] --out result.json
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
from nanovllm.engine.sequence import Sequence


def run_round(llm, prompts, sp, stats):
    seqs = [Sequence(p, sp) for p in prompts]
    t0 = time.perf_counter()
    for seq in seqs:
        llm.scheduler.add(seq)
    first, finish = {}, {}
    while not llm.is_finished():
        llm.step()
        now = time.perf_counter()
        stats["max_running"] = max(stats["max_running"], len(llm.scheduler.running))
        for seq in seqs:
            if seq.seq_id not in first and seq.num_completion_tokens > 0:
                first[seq.seq_id] = now
            if seq.seq_id not in finish and seq.is_finished:
                finish[seq.seq_id] = now
    wall = max(finish.values()) - t0
    out_len = sp.max_tokens
    return dict(
        wall_s=wall,
        output_tok_per_s=len(seqs) * out_len / wall,
        total_tok_per_s=sum(len(p) + out_len for p in prompts) / wall,
        ttft_ms=statistics.mean((first[s.seq_id] - t0) * 1e3 for s in seqs),
        tpot_ms=statistics.mean((finish[s.seq_id] - first[s.seq_id]) * 1e3 / (out_len - 1) for s in seqs),
        output_lens=sorted({s.num_completion_tokens for s in seqs}),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    parser.add_argument("--input-len", type=int, default=138)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--max-round-tokens", type=int, default=200_000,
                        help="fewer rounds for large C: rounds = min(--rounds, max(3, this // (C * output_len)))")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=None, help="default: the largest concurrency")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--k", type=int, default=0, help="num_speculative_tokens (MTP)")
    parser.add_argument("--draft-vocab", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    conc = [int(c) for c in args.concurrency.split(",")]

    llm = LLM(os.path.expanduser(args.model), max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs or max(conc), gpu_memory_utilization=args.gpu_memory_utilization,
              num_speculative_tokens=args.k, draft_vocab=args.draft_vocab, prefix_cache_policy="off")
    cfg = llm.model_runner.config
    rng = random.Random(args.seed)
    prompt = lambda: [rng.randint(1000, 100000) for _ in range(args.input_len)]
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.output_len, ignore_eos=True)

    accepted = dict(steps=0, tokens=0)
    if args.k:
        mr_run = llm.model_runner.run

        def counting_run(seqs, is_prefill):
            result = mr_run(seqs, is_prefill)
            if not is_prefill:
                accepted["steps"] += len(seqs)
                accepted["tokens"] += sum(len(t) - 1 for t in result[0])
            return result
        llm.model_runner.run = counting_run

    load_before = os.getloadavg()
    results = []
    for c in conc:
        stats = dict(max_running=0)
        run_round(llm, [prompt() for _ in range(args.warmup)], sp, stats)    # warmup, as the Chitu client does
        stats["max_running"] = 0
        accepted.update(steps=0, tokens=0)
        rounds = min(args.rounds, max(3, args.max_round_tokens // (c * args.output_len)))
        per_round = [run_round(llm, [prompt() for _ in range(c)], sp, stats) for _ in range(rounds)]
        out = [r["output_tok_per_s"] for r in per_round]
        row = dict(
            concurrency=c, rounds=rounds, max_running=stats["max_running"],
            output_tok_per_s=statistics.mean(out), output_tok_per_s_sd=statistics.stdev(out) if len(out) > 1 else 0.0,
            total_tok_per_s=statistics.mean(r["total_tok_per_s"] for r in per_round),
            ttft_ms=statistics.mean(r["ttft_ms"] for r in per_round),
            tpot_ms=statistics.mean(r["tpot_ms"] for r in per_round),
            output_lens=sorted({n for r in per_round for n in r["output_lens"]}),
            mtp_acceptance=accepted["tokens"] / accepted["steps"] if accepted["steps"] else None,
            per_round=per_round,
        )
        results.append(row)
        acc = f", acceptance {row['mtp_acceptance']:.3f}" if row["mtp_acceptance"] is not None else ""
        print(f"C={c:3d} ({rounds} rounds, up to {row['max_running']} running): output {row['output_tok_per_s']:8.1f} "
              f"+- {row['output_tok_per_s_sd']:.1f} tok/s, total {row['total_tok_per_s']:8.1f} tok/s, "
              f"TTFT {row['ttft_ms']:7.1f} ms, TPOT {row['tpot_ms']:6.2f} ms{acc}", flush=True)

    d = os.path.dirname(nanovllm.__file__)
    try:
        git = subprocess.check_output(["git", "-C", d, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.run(["git", "-C", d, "diff", "--quiet", "HEAD", "--", "."]).returncode)
    except Exception:    # noqa: BLE001
        git, dirty = None, None
    record = dict(
        model=os.path.basename(os.path.normpath(args.model)), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
        nanovllm_path=d, git=git, git_dirty=dirty, loadavg_before=load_before, loadavg_after=os.getloadavg(),
        args=vars(args), max_num_seqs_effective=cfg.max_num_seqs, num_kvcache_blocks=cfg.num_kvcache_blocks,
        results=results,
    )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(record, f, indent=1)


if __name__ == "__main__":
    main()
