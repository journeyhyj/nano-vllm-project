# End to end: greedy outputs of Qwen3.5 with hybrid prefix caching vs without it, request by request.
# Scenarios (each phase runs to completion before the next one starts, so later phases can hit earlier ones):
#   miss        unrelated prompts
#   partial     a 1000-token shared prefix with different questions: phases [A], [B], [C, D]
#   full        the same prompt three times
#   multiturn   turn 2 = turn 1 prompt + its 300-token answer + a new question (hits a snapshot taken during decode)
#   preempt     4 requests on a seen prefix with 600-token answers while all but 8 free KV blocks are held back:
#               requests get preempted after a cache hit and re-admitted (hitting again)
#   long        a 3000-token prompt, chunked by max_num_batched_tokens=2048, then a request sharing 2600 tokens of it
# Configurations: off (reference) vs branch / last / all / kv_only, with CUDA graphs; off vs branch with MTP (k=1) and with
# enforce_eager. Noise floor: off again, every request submitted alone (other batch shapes, same arithmetic otherwise).
# Sampling is argmax (also for MTP verification) and FLA autotuning is pinned, so differences come only from the cache.
# For every request: identical tokens or not, the first differing position and the reference's top-1 / top-2 logit gap
# there (recorded at every position), and the largest |logit difference| up to it (full logits kept for the first 32).
# Each configuration runs in its own process. Usage: python tests/test_prefix_cache_e2e.py   (needs a GPU; ~15 min)
import json
import os
import subprocess
import sys
import tempfile

import torch

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/huggingface/Qwen3.5-4B"))
QUESTIONS = "/data/csd/nano-vllm-data/mt_bench_question.jsonl"
DOCS = ["/usr/share/common-licenses/GPL-3", "/usr/share/common-licenses/GFDL-1.3"]
NUM_LOGITS = 32
CONFIGS = {    # name: (engine options, reference, submit requests one by one)
    "off": (dict(prefix_cache_policy="off"), None, False),
    "noise": (dict(prefix_cache_policy="off"), "off", True),
    "branch": (dict(prefix_cache_policy="branch"), "off", False),
    "last": (dict(prefix_cache_policy="last"), "off", False),
    "all": (dict(prefix_cache_policy="all"), "off", False),
    "kv_only": (dict(prefix_cache_policy="kv_only"), "off", False),
    "off_mtp": (dict(prefix_cache_policy="off", num_speculative_tokens=1), None, False),
    "branch_mtp": (dict(prefix_cache_policy="branch", num_speculative_tokens=1), "off_mtp", False),
    "off_eager": (dict(prefix_cache_policy="off", enforce_eager=True), None, False),
    "branch_eager": (dict(prefix_cache_policy="branch", enforce_eager=True), "off_eager", False),
}


def build_scenarios(tokenizer):
    questions = [json.loads(line)["turns"][0] for line in open(QUESTIONS)]
    doc = tokenizer("\n\n".join(open(p).read() for p in DOCS)).input_ids

    def prompt(offset, length, q):
        messages = [{"role": "system", "content": tokenizer.decode(doc[offset:offset + length])}, {"role": "user", "content": questions[q]}]
        return tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)).input_ids

    def turn(q):
        return tokenizer(f"\n<|im_start|>user\n{questions[q]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").input_ids

    # phases: lists of (name, prompt token ids or ("continue", earlier request, question), max_tokens)
    return {
        "miss": [[("m1", prompt(0, 700, 0), 48), ("m2", prompt(900, 500, 1), 48), ("m3", prompt(1500, 300, 2), 48)]],
        "partial": [[("A", prompt(2000, 1000, 3), 48)], [("B", prompt(2000, 1000, 4), 48)],
                    [("C", prompt(2000, 1000, 5), 48), ("D", prompt(2000, 1000, 6), 48)]],
        "full": [[("F1", prompt(3500, 900, 7), 48)], [("F2", prompt(3500, 900, 7), 48)], [("F3", prompt(3500, 900, 7), 48)]],
        "multiturn": [[("T1", prompt(4600, 800, 8), 300)], [("T2", ("continue", "T1", turn(9)), 48)]],
        # 600 forced tokens (no EOS) need 3 more blocks per request: more than the 8 free blocks left, whatever is shared
        "preempt": [[("P0", prompt(5600, 1100, 10), 48)],
                    [(f"P{i}", prompt(5600, 1100, 10 + i), -600) for i in range(1, 5)]],
        "long": [[("L1", prompt(7000, 3000, 15), 48)], [("L2", prompt(7000, 2600, 16), 48)]],
    }


def worker(name, out_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fla_pin import pin_fla_autotune
    pin_fla_autotune()
    from nanovllm import LLM, SamplingParams
    import nanovllm.engine.model_runner as model_runner_module

    options, ref_name, one_by_one = CONFIGS[name]
    llm = LLM(MODEL, max_model_len=4096, max_num_seqs=8, max_num_batched_tokens=2048, gpu_memory_utilization=0.5,
              num_state_snapshots=16, **options)
    mr, scheduler = llm.model_runner, llm.scheduler
    q = 1 + mr.num_speculative_tokens

    # argmax sampling everywhere, including MTP verification and drafting
    mr.sampler = lambda logits, temperatures: logits.float().argmax(dim=-1)
    def greedy_rejection(target_probs, draft_tokens, draft_probs):
        first, bonus = target_probs[:, 0].argmax(-1), target_probs[:, 1].argmax(-1)
        return torch.stack([first, bonus], dim=1), (first == draft_tokens).long()
    model_runner_module.rejection_sample = greedy_rejection
    model_runner_module.sample_from_probs = lambda probs: probs.argmax(dim=-1)

    # record the logits behind every generated token, and cached tokens at every admission
    names, logits_of, gaps_of, admissions, preemptions, hit_then_preempted = {}, {}, {}, {}, [0], []
    compute_logits, run, allocate, preempt = mr.model.compute_logits, mr.run, scheduler.block_manager.allocate, scheduler.preempt
    last = {}
    def recording_compute_logits(h):
        last["logits"] = compute_logits(h)
        return last["logits"]
    def recording_run(seqs, is_prefill):
        result = run(seqs, is_prefill)
        tokens = result[0] if mr.num_speculative_tokens else [[t] for t in result]
        logits = last["logits"]
        for i, seq in enumerate(seqs):
            if is_prefill and seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens:
                continue
            rows = [logits[i]] if is_prefill else list(logits.view(len(seqs), q, -1)[i][:len(tokens[i])])
            for j, row in enumerate(rows):
                pos = seq.num_completion_tokens + j
                top2 = row.float().topk(2).values
                gaps_of.setdefault(names[seq.seq_id], {})[pos] = (top2[0] - top2[1]).item()
                if pos < NUM_LOGITS:
                    logits_of.setdefault(names[seq.seq_id], {})[pos] = row.to("cpu", torch.bfloat16)
        return result
    def recording_allocate(seq, *args):
        allocate(seq, *args)
        admissions.setdefault(names[seq.seq_id], []).append(seq.num_cached_tokens)
    def counting_preempt(seq):
        preemptions[0] += 1
        if seq.num_cached_tokens and any(admissions.get(names[seq.seq_id], [0])):
            hit_then_preempted.append(names[seq.seq_id])
        preempt(seq)
    mr.model.compute_logits, mr.run, scheduler.block_manager.allocate, scheduler.preempt = \
        recording_compute_logits, recording_run, recording_allocate, counting_preempt

    from nanovllm.engine.sequence import Sequence
    outputs, report = {}, {}
    for scenario, phases in build_scenarios(llm.tokenizer).items():
        preempt_before = preemptions[0]
        for phase in phases:
            held = []
            if scenario == "preempt" and len(phase) > 1:
                # hold back all but 8 free KV blocks (the oldest go first; P0's cached blocks are the newest and stay)
                bm = scheduler.block_manager
                while len(bm.free_block_ids) > 8:
                    held.append(bm._allocate_block())
            batches = [[r] for r in phase] if one_by_one else [phase]
            for batch in batches:
                seqs = []
                for rname, ids, max_tokens in batch:
                    if isinstance(ids, tuple):    # ("continue", earlier request, next user turn)
                        _, prev, nxt = ids
                        ids = outputs[prev]["prompt"] + outputs[prev]["tokens"] + nxt
                    # max_tokens < 0: generate exactly -max_tokens tokens, ignoring EOS
                    seq = Sequence(ids, SamplingParams(temperature=1e-5, max_tokens=abs(max_tokens), ignore_eos=max_tokens < 0))
                    names[seq.seq_id] = rname
                    scheduler.add(seq)
                    seqs.append(seq)
                while not llm.is_finished():
                    llm.step()
                for seq in seqs:
                    outputs[names[seq.seq_id]] = dict(prompt=seq.prompt_token_ids, tokens=seq.completion_token_ids)
            for b in held:
                scheduler.block_manager.blocks[b].ref_count = 0
                scheduler.block_manager._deallocate_block(b)
        report[scenario] = dict(preemptions=preemptions[0] - preempt_before, hit_then_preempted=list(hit_then_preempted))
        hit_then_preempted.clear()
    result = dict(outputs=outputs, admissions=admissions, scenarios=report, stats=scheduler.block_manager.stats)
    if ref_name is None:
        torch.save(dict(result, logits=logits_of, gaps=gaps_of), out_path)
    else:
        ref = torch.load(out_path.replace(f"{name}.pt", f"{ref_name}.pt"))
        result["comparison"] = compare(ref, outputs, logits_of)
        torch.save(result, out_path)


def compare(ref, outputs, logits_of):
    rows = {}
    for rname, out in outputs.items():
        ref_tokens, tokens = ref["outputs"][rname]["tokens"], out["tokens"]
        n = min(len(ref_tokens), len(tokens))
        first = next((i for i in range(n) if ref_tokens[i] != tokens[i]), None)
        if first is None and len(ref_tokens) != len(tokens):
            first = n
        upto = NUM_LOGITS if first is None else min(first + 1, NUM_LOGITS)
        max_diff = 0.0
        for i in range(upto):
            a, b = ref["logits"].get(rname, {}).get(i), logits_of.get(rname, {}).get(i)
            if a is not None and b is not None:
                max_diff = max(max_diff, (a.float() - b.float()).abs().max().item())
        gap = ref["gaps"].get(rname, {}).get(first) if first is not None else None
        rows[rname] = dict(identical=first is None, first_diff=first, top2_gap=gap, max_logit_diff=max_diff, length=len(tokens))
    return rows


def main():
    tmp = tempfile.mkdtemp(prefix="prefix_cache_e2e_")
    env = dict(os.environ, NANOVLLM_PORT=os.environ.get("NANOVLLM_PORT", "2399"))
    results = {}
    for name in CONFIGS:
        path = os.path.join(tmp, f"{name}.pt")
        proc = subprocess.run([sys.executable, __file__, "--worker", name, path], env=env, capture_output=True, text=True)
        if proc.returncode:
            print(proc.stdout[-3000:], proc.stderr[-6000:])
            raise SystemExit(f"worker {name} failed")
        results[name] = torch.load(path)
    failures = []
    # last saves where a prompt ends, which includes each question, so different questions never hit it; branch saves
    # the shared prefix when it shows up the second time and hits from the third
    hits_expected = {"branch": {"C", "D", "F3", "T2"}, "last": {"F2", "F3", "T2"},
                     "all": {"B", "C", "D", "F2", "F3", "T2", "L2", "P1", "P2", "P3", "P4"},
                     "branch_mtp": {"C", "D", "F3", "T2"}, "branch_eager": {"C", "D", "F3", "T2"}}
    noise = results["noise"]["comparison"]
    print(f"{'config':13s} {'identical':>9s} {'max|dlogit|':>11s}  hits (cached tokens at admission) / divergences")
    for name, res in results.items():
        if "comparison" not in res:
            continue
        cmp_rows = res["comparison"]
        same = sum(r["identical"] for r in cmp_rows.values())
        max_diff = max(r["max_logit_diff"] for r in cmp_rows.values())
        hits = {k: v for k, v in res["admissions"].items() if any(v)}
        diverged = {k: (r["first_diff"], None if r["top2_gap"] is None else round(r["top2_gap"], 3))
                    for k, r in cmp_rows.items() if not r["identical"]}
        print(f"{name:13s} {same:4d}/{len(cmp_rows):<4d} {max_diff:11.4f}  {hits}  diverged(first, top2 gap)={diverged}")
        for rname in hits_expected.get(name, ()):
            if not any(res["admissions"].get(rname, [0])):
                failures.append(f"{name}: {rname} was expected to hit")
        if name == "kv_only" and any(any(v) for v in res["admissions"].values()):
            failures.append("kv_only must not skip any prefill")
        for rname, r in cmp_rows.items():
            # a cache hit may only change bf16 rounding: outputs may part at a near tie, never at a clear decision
            if r["max_logit_diff"] > 1.0:
                failures.append(f"{name}/{rname}: logits differ by {r['max_logit_diff']:.3f}")
            if not r["identical"] and (r["top2_gap"] is None or r["top2_gap"] > 0.5):
                failures.append(f"{name}/{rname}: diverged at {r['first_diff']} where the top-2 gap is {r['top2_gap']:.3f}")
    preempts = {name: (res["scenarios"]["preempt"]["preemptions"], res["scenarios"]["preempt"]["hit_then_preempted"])
                for name, res in results.items()}
    print(f"preempt scenario (preemptions, requests preempted after a cache hit): {preempts}")
    for name in ("branch", "all", "branch_mtp"):
        if not preempts[name][1]:
            failures.append(f"{name}: no request was preempted after a cache hit")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("logits", "gaps")} for k, v in results.items()},
              open(os.path.join(tmp, "summary.json"), "w"), indent=1, default=str)
    print(f"details: {tmp}/summary.json")
    for path in os.listdir(tmp):
        if path.endswith(".pt"):
            os.remove(os.path.join(tmp, path))
    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        main()
