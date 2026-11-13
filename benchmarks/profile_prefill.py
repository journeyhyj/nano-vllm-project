# Where does one prefill step of Qwen3.5-4B spend its time? Five measurements, from coarse to fine:
#   A  prefill latency vs prompt length (fresh prompts, whole llm.generate call): the intercept is the fixed cost
#   B  engine phases of one short prefill, each wrapped in synchronized timers (schedule, prepare, forward, logits,
#      sampling, postprocess); the forward is split into "issue" (until the Python call returns) and "wait for GPU"
#   C  host time per model part inside the forward (forward hooks, no synchronization), plus the FLA call and our
#      Triton conv kernels
#   D  torch.profiler of one short prefill: kernel launches per model part, GPU busy time, top CPU ops
#   E  counterfactual: the same forward captured in a CUDA graph and replayed, i.e. the step without host overhead
# Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_prefill.py ~/huggingface/Qwen3.5-4B \
#            --out-dir benchmarks/results/profile_prefill
import argparse
import bisect
import collections
import gzip
import json
import os
import random
import shutil
import statistics
import subprocess
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import nanovllm
import nanovllm.layers.linear_attention as la_mod
from nanovllm import LLM, SamplingParams


def sync():
    torch.cuda.synchronize()


def generate_ms(llm, prompt):
    sync()
    t = time.perf_counter()
    llm.generate([prompt], SamplingParams(temperature=1e-5, max_tokens=1), use_tqdm=False)
    sync()
    return (time.perf_counter() - t) * 1e3


class Phases:
    """Synchronized timers around engine methods: each phase includes the GPU work it issued."""
    def __init__(self):
        self.t = collections.defaultdict(float)
        self.on = False

    def wrap(self, obj, name, label, split=False):
        f = getattr(obj, name)

        def timed(*args, **kwargs):
            if not self.on:
                return f(*args, **kwargs)
            sync()
            t0 = time.perf_counter()
            out = f(*args, **kwargs)
            t1 = time.perf_counter()
            sync()
            t2 = time.perf_counter()
            if split:
                self.t[label + ": issue (until the call returns)"] += t1 - t0
                self.t[label + ": wait for the GPU afterwards"] += t2 - t1
            else:
                self.t[label] += t2 - t0
            return out
        setattr(obj, name, timed)


class Parts:
    """Host time per model part (forward hooks, no synchronization), optionally as profiler ranges."""
    def __init__(self, model):
        self.t = collections.defaultdict(float)
        self.on = False
        self.ranges = False
        self.stack = []
        m = model.model
        self.add(m.embed_tokens, "embedding")
        for layer in m.layers:
            kind = "GDN" if layer.layer_type == "linear_attention" else "attention"
            self.add(layer, f"{kind} layers")
            self.add(layer.input_layernorm, f"{kind}: norms")
            self.add(layer.post_attention_layernorm, f"{kind}: norms")
            mixer = layer.linear_attn if kind == "GDN" else layer.self_attn
            self.add(mixer, f"{kind}: mixer")
            self.add(mixer.attn, "GDN: mixer: conv + recurrence" if kind == "GDN" else "attention: mixer: flash-attn + KV store")
            self.add(layer.mlp, f"{kind}: MLP")
        self.add(m.norm, "final norm")
        self.labels = set()

    def add(self, module, label):
        def pre(*_):
            if self.on:
                ctx = record_function(label) if self.ranges else None
                if ctx is not None:
                    ctx.__enter__()
                self.stack.append((time.perf_counter(), ctx))

        def post(*_):
            if self.on:
                t, ctx = self.stack.pop()
                self.t[label] += time.perf_counter() - t
                if ctx is not None:
                    ctx.__exit__(None, None, None)
        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)


class TimedCall:
    """Host time of a function or a Triton kernel launch (kernel[grid](...)), no synchronization."""
    def __init__(self, f, label, acc, launcher=False):
        self.f, self.label, self.acc, self.launcher = f, label, acc, launcher
        self.on = False

    def _timed(self, g):
        def call(*args, **kwargs):
            if not self.on:
                return g(*args, **kwargs)
            ctx = record_function(self.label)
            ctx.__enter__()
            t = time.perf_counter()
            out = g(*args, **kwargs)
            self.acc[self.label] += time.perf_counter() - t
            ctx.__exit__(None, None, None)
            return out
        return call

    def __call__(self, *args, **kwargs):
        return self._timed(self.f)(*args, **kwargs)

    def __getitem__(self, grid):
        return self._timed(self.f[grid])


def busy_ms(intervals):
    total, end = 0.0, -1.0
    for s, e in sorted(intervals):
        if e <= end:
            continue
        total += e - max(s, end)
        end = e
    return total / 1e3


def analyse_profile(prof, labels):
    events = list(prof.events())
    # device-side events, minus the GPU-timeline copies of our record_function ranges (they span whole parts)
    device = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA and e.name not in labels]
    launches = [e for e in events if e.device_type == torch.autograd.DeviceType.CPU and "LaunchKernel" in e.name]
    step = [e for e in events if e.name == "prefill step"][0]
    ranges = sorted((e for e in events if e.name in labels), key=lambda e: e.time_range.start)
    per_range = collections.Counter()
    for ev in launches:
        inner = None
        for r in ranges:
            if r.time_range.start <= ev.time_range.start and ev.time_range.end <= r.time_range.end:
                if inner is None or r.time_range.elapsed_us() < inner.time_range.elapsed_us():
                    inner = r
        per_range[inner.name if inner else "(outside the model forward)"] += 1
    kernels = collections.defaultdict(lambda: [0, 0.0])
    for e in device:
        kernels[e.name][0] += 1
        kernels[e.name][1] += e.time_range.elapsed_us() / 1e3
    top_kernels = sorted(kernels.items(), key=lambda kv: -kv[1][1])[:15]
    return dict(
        step_wall_ms=step.time_range.elapsed_us() / 1e3,
        gpu_busy_ms=busy_ms([(e.time_range.start, e.time_range.end) for e in device]),
        device_events=len(device),
        kernel_launches=len(launches),
        launch_api_calls=dict(collections.Counter(e.name for e in launches)),
        launches_per_innermost_part=dict(per_range.most_common()),
        top_kernels_by_gpu_ms=[dict(name=n[:120], count=c, gpu_ms=round(ms, 3)) for n, (c, ms) in top_kernels],
    )


class _CaptureFailed(Exception):
    pass


def graph_counterfactual(llm, prompt, reps):
    # replays the model forward of this prefill from a CUDA graph; re-running a fresh sequence's prefill is idempotent
    # (its states are masked by has_initial_state and its KV slots get the same values)
    mr = llm.model_runner
    orig = mr.run_model
    res = {}

    def patched(input_ids, positions, is_prefill):
        out = orig(input_ids, positions, is_prefill)
        if not is_prefill:
            return out
        model = mr.model
        times = []
        for _ in range(reps):
            sync()
            t = time.perf_counter()
            model(input_ids, positions)
            sync()
            times.append((time.perf_counter() - t) * 1e3)
        res["eager_forward_ms"] = statistics.median(times)
        try:
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                model(input_ids, positions)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph):
                graph_out = model(input_ids, positions)
            times = []
            for _ in range(reps):
                sync()
                t = time.perf_counter()
                graph.replay()
                sync()
                times.append((time.perf_counter() - t) * 1e3)
            res["graph_replay_ms"] = statistics.median(times)
            res["graph_output_equals_eager"] = bool(torch.equal(graph_out, model(input_ids, positions)))
            del graph
        except Exception as e:    # noqa: BLE001 - the failure itself is a finding
            res["graph_error"] = f"{type(e).__name__}: {e}"[:600]
            import traceback
            res["graph_error_where"] = [f"{fr.filename}:{fr.lineno} {fr.name}" for fr in traceback.extract_tb(e.__traceback__)][-8:]
            print("E  capture failed:", res["graph_error"], *res["graph_error_where"], sep="\n     ")
            raise _CaptureFailed from e    # the CUDA RNG is left in capture mode: stop before sampling
        return out

    mr.run_model = patched
    try:
        generate_ms(llm, prompt)
    except _CaptureFailed:
        pass
    finally:
        mr.run_model = orig
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--lens", default="16,64,256,1024,2048,4096")
    parser.add_argument("--short", type=int, default=64)
    parser.add_argument("--long", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--experiments", default="ABCDE", help="subset of A-E to run")
    args = parser.parse_args()
    lens = [int(x) for x in args.lens.split(",")]

    llm = LLM(os.path.expanduser(args.model), max_model_len=args.max_model_len, max_num_seqs=16,
              gpu_memory_utilization=args.gpu_memory_utilization, prefix_cache_policy="off")
    rng = random.Random(0)
    fresh = lambda n: [rng.randint(1000, 100000) for _ in range(n)]
    for n in sorted(set(lens + [args.short, args.long])):    # warmup: torch.compile shapes, FLA autotune
        generate_ms(llm, fresh(n))
        generate_ms(llm, fresh(n))
    res = dict(model=os.path.basename(os.path.normpath(args.model)), gpu=torch.cuda.get_device_name(),
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), torch=torch.__version__,
               nanovllm_path=os.path.dirname(nanovllm.__file__), loadavg_before=os.getloadavg(),
               engine=dict(max_model_len=args.max_model_len, max_num_seqs=16,
                           gpu_memory_utilization=args.gpu_memory_utilization, prefix_cache_policy="off"))
    try:
        d = os.path.dirname(nanovllm.__file__)
        res["git"] = subprocess.check_output(["git", "-C", d, "rev-parse", "--short", "HEAD"], text=True).strip()
        res["git_dirty"] = bool(subprocess.run(["git", "-C", d, "diff", "--quiet", "HEAD", "--", "."]).returncode)
    except Exception:    # noqa: BLE001
        res["git"] = None

    run = set(args.experiments)
    # A: latency vs prompt length
    if "A" in run:
      res["A_generate_ms_by_len"] = {n: statistics.median(generate_ms(llm, fresh(n)) for _ in range(args.repeats)) for n in lens}
      print("A  prefill latency (whole generate call, median):", {n: round(v, 1) for n, v in res["A_generate_ms_by_len"].items()})

    # B: engine phases (synchronized)
    mr, sched = llm.model_runner, llm.scheduler
    phases = Phases()
    phases.wrap(llm, "step", "engine step (total)")
    phases.wrap(sched, "schedule", "schedule")
    phases.wrap(mr, "prepare_prefill", "prepare_prefill (inputs, FLA window plan, H2D)")
    phases.wrap(mr, "prepare_sample", "prepare_sample")
    phases.wrap(mr, "run_model", "model forward", split=True)
    phases.wrap(mr.model, "compute_logits", "compute_logits (final norm excluded, lm_head)")
    phases.wrap(mr, "sampler", "sampler")
    phases.wrap(sched, "postprocess", "postprocess")
    res["B_phases_ms"] = {}
    for n in ((args.short, args.long) if "B" in run else ()):
        runs = []
        for _ in range(args.repeats):
            phases.t.clear()
            phases.on = True
            total = generate_ms(llm, fresh(n))
            phases.on = False
            row = {k: v * 1e3 for k, v in phases.t.items()}
            row["generate call (total)"] = total
            runs.append(row)
        res["B_phases_ms"][n] = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
        print(f"B  phases, {n} tokens (median ms):")
        for k, v in res["B_phases_ms"][n].items():
            print(f"     {k:60s} {v:8.2f}")

    # C: host time per model part (no synchronization)
    parts = Parts(mr.model)
    acc = parts.t
    fla = TimedCall(la_mod.chunk_gated_delta_rule, "GDN: FLA chunk_gated_delta_rule call", acc)
    conv = TimedCall(la_mod.causal_conv1d_kernel, "GDN: our Triton kernels (conv1d + conv state)", acc, launcher=True)
    conv_state = TimedCall(la_mod.conv_state_update_kernel, "GDN: our Triton kernels (conv1d + conv state)", acc, launcher=True)
    la_mod.chunk_gated_delta_rule, la_mod.causal_conv1d_kernel, la_mod.conv_state_update_kernel = fla, conv, conv_state
    res["C_host_ms_by_part"] = {}
    for n in ((args.short, args.long) if "C" in run or "D" in run else ()):
        runs = []
        for _ in range(args.repeats):
            acc.clear()
            parts.on = fla.on = conv.on = conv_state.on = True
            generate_ms(llm, fresh(n))
            parts.on = fla.on = conv.on = conv_state.on = False
            runs.append({k: v * 1e3 for k, v in acc.items()})
        res["C_host_ms_by_part"][n] = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
        print(f"C  host time by model part, {n} tokens (median ms, includes ~hook overhead):")
        for k, v in sorted(res["C_host_ms_by_part"][n].items(), key=lambda kv: -kv[1]):
            print(f"     {k:60s} {v:8.2f}")

    # D: torch.profiler of one short prefill, with the parts above as ranges
    labels = set(res["C_host_ms_by_part"].get(args.short, {})) | {"prefill step"}
    parts.ranges = True
    res["D_profile"] = {}
    for n in ((args.short, args.long) if "D" in run else ()):
        prompt = fresh(n)
        parts.on = fla.on = conv.on = conv_state.on = True
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function("prefill step"):
                generate_ms(llm, prompt)
        parts.on = fla.on = conv.on = conv_state.on = False
        d = analyse_profile(prof, labels)
        try:
            table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
        except Exception as e:    # noqa: BLE001
            table = repr(e)
        d["top_cpu_ops_table"] = table
        res["D_profile"][n] = d
        print(f"D  profile, {n} tokens: step {d['step_wall_ms']:.1f} ms wall, GPU busy {d['gpu_busy_ms']:.1f} ms, "
              f"{d['kernel_launches']} kernel launches")
        for k, v in d["launches_per_innermost_part"].items():
            print(f"     launches in {k:56s} {v:5d}")
        print(table)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            trace = os.path.join(args.out_dir, f"trace_{n}.json")
            prof.export_chrome_trace(trace)
            with open(trace, "rb") as fi, gzip.open(trace + ".gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(trace)
    parts.ranges = False
    la_mod.chunk_gated_delta_rule, la_mod.causal_conv1d_kernel, la_mod.conv_state_update_kernel = fla.f, conv.f, conv_state.f

    # E: the forward replayed from a CUDA graph
    res["E_graph"] = {}
    for n in ((args.short, args.long) if "E" in run else ()):
        res["E_graph"][n] = graph_counterfactual(llm, fresh(n), args.repeats * 3)
        if "graph_error" in res["E_graph"][n]:
            break    # a failed capture leaves the CUDA RNG in capture mode; the next generate would fail
    print("E  eager forward vs CUDA graph replay:", json.dumps(res["E_graph"]))

    res["loadavg_after"] = os.getloadavg()
    print(json.dumps({k: v for k, v in res.items() if k != "D_profile"}))
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, f"profile_prefill_{args.experiments}.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
