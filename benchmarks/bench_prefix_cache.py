# Prefix caching under online load: requests arrive over time and TTFT is measured from each request's arrival.
#
# Workload "shared": every request is one shared system prompt (a reference document cut to --prefix tokens) followed by
#   one MT-Bench first-turn question; requests arrive as a Poisson process at --qps.
# Workload "multiturn": --convs conversations start as a Poisson process at --qps. Each has its own document slice
#   (--prefix tokens) as system prompt and asks --turns consecutive MT-Bench questions. A turn is sent as soon as the
#   previous one finishes and appends to the previous prompt plus the generated tokens at the token level (re-applying
#   the chat template would drop the empty <think> block of earlier answers and break the shared prefix).
#
# Reports TTFT p50/p99, prefill tokens actually computed, prefix-cache hits (num_cached_tokens right after admission),
# throughput, and the engine's memory split. Writes a JSON record next to the printed summary.
# Usage: python benchmarks/bench_prefix_cache.py <model dir> --workload shared --prefix 4000 --qps 2 --out result.json
#        [--policy branch|last|all|kv_only|off] [extra engine options, see --help]
import argparse
import json
import os
import random
import subprocess
import time
from dataclasses import fields

import numpy as np
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence

DOCS = ["/usr/share/common-licenses/GPL-3", "/usr/share/common-licenses/GFDL-1.3",
        "/usr/share/common-licenses/Apache-2.0", "/usr/share/common-licenses/MPL-2.0"]


def load_questions(path):
    return [json.loads(line) for line in open(path)]


def doc_tokens(tokenizer, paths):
    return tokenizer("\n\n".join(open(p).read() for p in paths)).input_ids


def chat_prompt(tokenizer, system_ids, question):
    # the system content is given as token ids so that its length is exact; decode -> encode round-trips here
    messages = [{"role": "system", "content": tokenizer.decode(system_ids)}, {"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text).input_ids


def next_turn_ids(tokenizer, question):
    text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return tokenizer(text).input_ids


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def percentile(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


class Driver:
    """Feeds requests into the engine at their arrival times and records per-request timings."""

    def __init__(self, llm):
        self.llm = llm
        self.records = {}    # seq_id -> dict
        self.prefill_tokens = 0
        self.admissions = []    # (seq_id, num_cached_tokens right after allocate, num_tokens)
        bm = llm.scheduler.block_manager
        allocate = bm.allocate

        def recording_allocate(seq, *args, **kwargs):
            out = allocate(seq, *args, **kwargs)
            self.admissions.append((seq.seq_id, seq.num_cached_tokens, seq.num_tokens))
            return out
        bm.allocate = recording_allocate

    def submit(self, token_ids, sampling_params, arrival, tag):
        seq = Sequence(token_ids, sampling_params)
        self.records[seq.seq_id] = dict(seq=seq, arrival=arrival, first=None, finish=None, prompt_len=len(token_ids), **tag)
        self.llm.scheduler.add(seq)
        return seq

    def step(self, now_fn):
        _, num_tokens = self.llm.step()
        if num_tokens > 0:
            self.prefill_tokens += num_tokens
        now = now_fn()
        finished = []
        for rid, r in self.records.items():
            seq = r["seq"]
            if r["first"] is None and seq.num_completion_tokens > 0:
                r["first"] = now
            if r["finish"] is None and seq.is_finished:
                r["finish"] = now
                finished.append(r)
        return finished


def run_shared(llm, args, questions, doc):
    tok = llm.tokenizer
    rng = random.Random(args.seed)
    qs = [q["turns"][0] for q in questions]
    rng.shuffle(qs)
    # cut the document so that the prefix shared by all requests (template + system prompt) is exactly --prefix tokens
    n = args.prefix
    for _ in range(4):
        shared = common_prefix_len(chat_prompt(tok, doc[:n], qs[0]), chat_prompt(tok, doc[:n], qs[1]))
        if shared == args.prefix:
            break
        n += args.prefix - shared
    system_ids = doc[:n]
    prompts = [chat_prompt(tok, system_ids, qs[i % len(qs)]) for i in range(args.num_requests)]
    t, arrivals = 0.0, []
    for _ in prompts:
        arrivals.append(t)
        t += rng.expovariate(args.qps)
    sp = SamplingParams(temperature=1e-5, max_tokens=args.max_tokens)
    driver = Driver(llm)
    start = time.perf_counter()
    now = lambda: time.perf_counter() - start
    pending = list(zip(arrivals, prompts, range(len(prompts))))
    while pending or not llm.is_finished():
        while pending and pending[0][0] <= now():
            arrival, ids, i = pending.pop(0)
            driver.submit(ids, sp, arrival, dict(index=i, turn=1, conv=i))
        if llm.is_finished():
            time.sleep(max(0.0, pending[0][0] - now()))
            continue
        driver.step(now)
    shared = min(common_prefix_len(prompts[0], p) for p in prompts[1:])
    return driver, dict(shared_prefix_tokens=shared, prompt_tokens=[len(p) for p in prompts])


def run_multiturn(llm, args, questions, doc):
    tok = llm.tokenizer
    rng = random.Random(args.seed)
    order = list(range(len(questions)))
    rng.shuffle(order)
    sp = SamplingParams(temperature=1e-5, max_tokens=args.max_tokens)
    convs = []
    t = 0.0
    for c in range(args.num_convs):
        offset = (c * 997) % max(1, len(doc) - args.prefix)    # a different document slice per conversation
        qs = []
        for k in range(args.turns):
            q = questions[order[(c * args.turns + k) % len(order)]]
            qs.append(q["turns"][0])
        first = chat_prompt(tok, doc[offset:offset + args.prefix], qs[0])
        convs.append(dict(start=t, questions=qs, prompt=first))
        t += rng.expovariate(args.qps)
    driver = Driver(llm)
    start = time.perf_counter()
    now = lambda: time.perf_counter() - start
    pending = [(conv["start"], c, 1, conv["prompt"]) for c, conv in enumerate(convs)]    # (arrival, conv, turn, ids)
    while pending or not llm.is_finished():
        pending.sort(key=lambda x: x[0])
        while pending and pending[0][0] <= now():
            arrival, c, turn, ids = pending.pop(0)
            driver.submit(ids, sp, arrival, dict(index=len(driver.records), turn=turn, conv=c))
        if llm.is_finished():
            if pending:
                time.sleep(max(0.0, pending[0][0] - now()))
            continue
        for r in driver.step(now):
            c, turn = r["conv"], r["turn"]
            if turn < args.turns:
                seq = r["seq"]
                ids = seq.prompt_token_ids + seq.completion_token_ids
                if ids[-1] != tok.eos_token_id:    # stopped by max_tokens: close the answer like the template does
                    ids = ids + [tok.eos_token_id]
                ids = ids + tok("\n").input_ids + next_turn_ids(tok, convs[c]["questions"][turn])
                pending.append((r["finish"], c, turn + 1, ids))
    return driver, dict(prompt_tokens=[r["prompt_len"] for r in driver.records.values()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--workload", choices=["shared", "multiturn"], default="shared")
    parser.add_argument("--questions", default="/data/csd/nano-vllm-data/mt_bench_question.jsonl")
    parser.add_argument("--prefix", type=int, default=4000, help="system prompt tokens (document slice)")
    parser.add_argument("--qps", type=float, default=1.0, help="request (shared) or conversation (multiturn) arrival rate")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--num-convs", type=int, default=16)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", default=None, help="hybrid prefix cache policy; omitted = engine default")
    parser.add_argument("--num-state-snapshots", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=10240)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--k", type=int, default=0, help="num_speculative_tokens (MTP)")
    parser.add_argument("--draft-vocab", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_fields = {f.name for f in fields(Config)}
    engine_kwargs = dict(max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
                         max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                         num_speculative_tokens=args.k, draft_vocab=args.draft_vocab, enforce_eager=args.enforce_eager)
    for name, value in (("prefix_cache_policy", args.policy), ("num_state_snapshots", args.num_state_snapshots)):
        if value is not None:
            assert name in config_fields, f"this engine has no {name} option"
            engine_kwargs[name] = value
    load_before = os.getloadavg()
    llm = LLM(os.path.expanduser(args.model), **engine_kwargs)
    questions = load_questions(args.questions)
    doc = doc_tokens(llm.tokenizer, DOCS)
    assert len(doc) >= args.prefix + 1000, f"document has only {len(doc)} tokens"

    # warmup with unrelated random tokens (shares no prefix with the trace): torch.compile specializes on the first
    # batch shapes and FLA autotunes per window size, so cover single and batched prefills and batched decode
    rng = random.Random(1234)
    random_prompt = lambda n: [rng.randint(1000, 100000) for _ in range(n)]
    for n in (64, 300, 2048, 4096, 8192):
        if n + 16 <= args.max_model_len:
            llm.generate([random_prompt(n)], SamplingParams(temperature=1e-5, max_tokens=4), use_tqdm=False)
    for _ in range(2):
        lengths = [rng.randint(50, min(args.prefix + 400, args.max_model_len - 64)) for _ in range(args.max_num_seqs)]
        llm.generate([random_prompt(n) for n in lengths],
                     [SamplingParams(temperature=1e-5, max_tokens=rng.randint(4, 24)) for _ in lengths], use_tqdm=False)
    torch.cuda.synchronize()

    run = run_shared if args.workload == "shared" else run_multiturn
    t0 = time.perf_counter()
    driver, info = run(llm, args, questions, doc)
    wall = time.perf_counter() - t0
    load_after = os.getloadavg()

    recs = sorted(driver.records.values(), key=lambda r: r["arrival"])
    ttft = [r["first"] - r["arrival"] for r in recs]
    later = [r["first"] - r["arrival"] for r in recs[1:]] if args.workload == "shared" else \
            [r["first"] - r["arrival"] for r in recs if r["turn"] >= 2]
    out_tokens = sum(r["seq"].num_completion_tokens for r in recs)
    prompt_tokens = sum(r["prompt_len"] for r in recs)
    trace_ids = {r["seq"].seq_id for r in recs}
    adm = [a for a in driver.admissions if a[0] in trace_ids]
    first_adm, readm = {}, []
    for sid, cached, n in adm:
        if sid in first_adm:
            readm.append((sid, cached, n))
        else:
            first_adm[sid] = (cached, n)
    hit_tokens = sum(c for c, _ in first_adm.values())
    mr = llm.model_runner
    config = mr.config
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    result = dict(
        workload=args.workload, prefix=args.prefix, qps=args.qps, policy=args.policy or "engine default",
        requests=len(recs), wall_s=wall,
        ttft_p50_ms=percentile(ttft, 50) * 1e3, ttft_p99_ms=percentile(ttft, 99) * 1e3,
        ttft_mean_ms=float(np.mean(ttft)) * 1e3,
        ttft_later_p50_ms=percentile(later, 50) * 1e3, ttft_later_p99_ms=percentile(later, 99) * 1e3,
        prompt_tokens=prompt_tokens, prefill_tokens_computed=driver.prefill_tokens,
        hit_tokens_first_admission=hit_tokens, hit_rate=hit_tokens / prompt_tokens,
        readmissions=len(readm), readmission_hit_tokens=sum(c for _, c, _ in readm),
        output_tokens=out_tokens, output_tok_per_s=out_tokens / wall, req_per_s=len(recs) / wall,
        num_kvcache_blocks=config.num_kvcache_blocks, num_state_groups=config.num_state_groups,
        max_num_seqs=config.max_num_seqs, snapshot_bytes=getattr(mr, "snapshot_bytes", 0),
        num_state_snapshots=getattr(config, "num_state_snapshots", 0),
        prefix_cache_stats=getattr(llm.scheduler.block_manager, "stats", None),
        engine=dict(engine_kwargs, kvcache_block_size=config.kvcache_block_size),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
        loadavg_before=load_before, loadavg_after=load_after, git=git + ("-dirty" if dirty else ""),
        max_tokens=args.max_tokens, seed=args.seed, **{k: v for k, v in info.items() if k != "prompt_tokens"},
    )
    if args.workload == "multiturn":
        result.update(num_convs=args.num_convs, turns=args.turns)
    else:
        result.update(num_requests=args.num_requests)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in result.items()}, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(dict(result, per_request=[dict(arrival=r["arrival"], first=r["first"], finish=r["finish"],
                                                     prompt_len=r["prompt_len"], turn=r["turn"], conv=r["conv"],
                                                     output_len=r["seq"].num_completion_tokens) for r in recs]),
                      f, indent=1)


if __name__ == "__main__":
    main()
