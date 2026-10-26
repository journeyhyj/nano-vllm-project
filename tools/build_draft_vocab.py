# Build the draft vocabulary for MTP speculative decoding (LLM(..., draft_vocab=<out>)):
# let the target model answer instruction prompts, count the tokens it generates and keep the most frequent ones.
# Prompt files are JSON lists of strings or of Alpaca-style {"instruction", "input"} records; the first --num of each are used.
# Usage: python tools/build_draft_vocab.py <model dir> <out.json> <prompts.json> [<prompts.json> ...] [--size 32768] [--num 1500]
import argparse
import json
import os
import torch
from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--size", type=int, default=32768)
    parser.add_argument("--num", type=int, default=1500)
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()

    path = os.path.expanduser(args.model)
    llm = LLM(path, max_model_len=4096, gpu_memory_utilization=0.6)
    tokenizer = llm.tokenizer
    prompts = []
    for file in args.prompts:
        with open(file) as f:
            records = json.load(f)[:args.num]
        for r in records:
            text = r if isinstance(r, str) else (r["instruction"] + ("\n\n" + r["input"] if r.get("input") else ""))
            prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=False))
    outputs = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=args.max_tokens))

    vocab_size = llm.model_runner.config.hf_config.vocab_size
    counts = torch.zeros(vocab_size, dtype=torch.int64)
    for o in outputs:
        counts += torch.bincount(torch.tensor(o["token_ids"]), minlength=vocab_size)
    order = counts.argsort(descending=True)
    special = [tokenizer.eos_token_id, tokenizer.pad_token_id]    # end of turn / end of text must stay draftable
    ids = special + [i for i in order[:args.size].tolist() if i not in special]
    ids = ids[:args.size]
    total = counts.sum().item()
    print(f"{total} generated tokens, {(counts > 0).sum().item()} distinct; "
          f"the {len(ids)} kept tokens cover {counts[ids].sum().item() / total:.4%} of them")
    with open(args.out, "w") as f:
        json.dump(ids, f)


if __name__ == "__main__":
    main()
