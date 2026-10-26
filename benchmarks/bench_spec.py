# Decode throughput and acceptance of MTP speculative decoding (greedy, natural EOS, at most 256 new tokens).
# Questions: MT-Bench question.jsonl (first turns, acceptance per category) or an Alpaca-style JSON (its last --tail records).
# Phase 1 answers one question at a time (batch size 1); phase 2 submits all of them with max_num_seqs=--batch.
# Decode throughput only counts decode steps, so prompt length does not affect it.
# Usage: python benchmarks/bench_spec.py <model dir> <questions> [--k 1] [--draft-vocab <ids.json>] [--tail 80] [--batch 16]
import argparse
import json
import os
import time
from collections import defaultdict
from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("questions")
    parser.add_argument("--k", type=int, default=1, help="num_speculative_tokens, 0 disables MTP")
    parser.add_argument("--draft-vocab")
    parser.add_argument("--tail", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    llm = LLM(os.path.expanduser(args.model), max_model_len=4096, gpu_memory_utilization=0.6, max_num_seqs=args.batch,
              num_speculative_tokens=args.k, draft_vocab=args.draft_vocab)
    if args.questions.endswith(".jsonl"):
        questions = [json.loads(line) for line in open(args.questions)]
        texts, categories = [q["turns"][0] for q in questions], [q["category"] for q in questions]
    else:
        records = json.load(open(args.questions))[-args.tail:]
        texts = [r["instruction"] + ("\n\n" + r["input"] if r.get("input") else "") for r in records]
        categories = ["all"] * len(texts)
    prompts = [llm.tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False) for t in texts]
    sampling_params = SamplingParams(temperature=1e-5, max_tokens=256)

    stats = defaultdict(float)
    step, run = llm.step, llm.model_runner.run
    def timed_step():
        t = time.perf_counter()
        outputs, num_tokens = step()
        if num_tokens < 0:    # decode step
            stats["decode_s"] += time.perf_counter() - t
            stats["decode_tokens"] += -num_tokens
        return outputs, num_tokens
    def counting_run(seqs, is_prefill):
        result = run(seqs, is_prefill)
        if args.k and not is_prefill:    # result: (accepted tokens per sequence, next drafts)
            stats["seq_steps"] += len(seqs)
            stats["accepted"] += sum(len(t) - 1 for t in result[0])
        return result
    llm.step, llm.model_runner.run = timed_step, counting_run

    def report(tag):
        line = f"{tag}: decode {stats['decode_tokens'] / stats['decode_s']:.1f} tok/s"
        if args.k:
            line += f", acceptance {stats['accepted'] / stats['seq_steps']:.3f}"
        print(line, flush=True)

    name = f"k={args.k} draft_vocab={llm.model_runner.draft_vocab_size}"
    llm.generate(prompts[:2], sampling_params, use_tqdm=False)    # warmup
    stats.clear()
    per_category = defaultdict(lambda: [0.0, 0.0])
    for category, prompt in zip(categories, prompts):
        accepted, seq_steps = stats["accepted"], stats["seq_steps"]
        llm.generate([prompt], sampling_params, use_tqdm=False)
        per_category[category][0] += stats["accepted"] - accepted
        per_category[category][1] += stats["seq_steps"] - seq_steps
    report(f"[{name}] batch size 1, {len(prompts)} questions")
    if args.k and len(per_category) > 1:
        print("  acceptance per category: " + " ".join(f"{c}={a / s:.2f}" for c, (a, s) in per_category.items()), flush=True)
    stats.clear()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    report(f"[{name}] all {len(prompts)} questions, max_num_seqs={llm.scheduler.max_num_seqs}")


if __name__ == "__main__":
    main()
