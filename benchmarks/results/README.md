# 结果目录

本目录存放 hybrid prefix caching 的全部实测记录。`summary.md` 汇总了在线负载的各组结果，`docs/hybrid_prefix_cache.md` 引用的每个数字都能在这里找到出处。

## 在线负载（`baseline/`、`v1_cut/`、`v2_extra/`、`v2_extra_mtp/`、`rerun_shared4000/`）

由 `benchmarks/run_prefix_cache_matrix.sh` 调用 `benchmarks/bench_prefix_cache.py` 生成。每个 `.json` 对应一次运行；同名 `.log` 的开头几行是时间、主机、git 版本、loadavg、`CUDA_VISIBLE_DEVICES`、各 GPU 的占用和完整命令。

公共条件：
- RTX 4090（`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`，PCI 1A:00.0），Qwen3.5-4B（bf16），torch 2.6.0+cu124，fla-core 0.4.2，triton 3.2.0；
- `max_num_seqs=16`、`max_num_batched_tokens=16384`、`max_model_len=10240`、`gpu_memory_utilization=0.85`；
- 开 cache 的组都加了 `--num-state-snapshots 48`。

| 目录 | 引擎版本 | 内容 |
|---|---|---|
| `baseline/` | c5af84b（加入 hybrid prefix caching 之前） | 引擎默认配置，当时混合模型不做 prefix caching |
| `v1_cut/` | 6d8628a（快照通过切开 FLA 窗口获得） | 策略 off / kv_only / last / branch / all |
| `v2_extra/` | 40a3ccf（快照通过附加的 FLA 序列获得） | 策略 off / last / branch / all。kv_only 没有重跑：v2 改动的是窗口规划，没有快照时生成的窗口与 v1 相同（据代码） |
| `v2_extra_mtp/` | 40a3ccf | 开启 MTP（`--k 1`）时 off 与 branch 的对比，JSON 里有接受率 `mtp_acceptance` |
| `rerun_shared4000/{v1_cut,v2_extra}/` | 6d8628a / 40a3ccf | 共享 4000 token、两种 QPS 下 branch 的第 3–5 次运行，v1 与 v2 交替进行，用来判断 v2 前两次运行之间的差别是不是噪声（见 doc 第 4.2 节） |

引擎是怎样选定的：每个版本都冻结在一个 git worktree 里（`/data/csd/nano-vllm-baseline`、`-after`、`-v2`），用 `PYTHONPATH` 指向它，由 `run_prefix_cache_matrix.sh` 启动。各目录记录的完整程度不同：
- `v2_extra/`、`v2_extra_mtp/`、`rerun_shared4000/`：JSON 的 `nanovllm_path` 记录了实际导入的引擎目录，`git` 是该目录的提交。日志头都不是引擎的提交：`v2_extra/`、`v2_extra_mtp/` 的日志头没有 `PYTHONPATH`；`rerun_shared4000/` 的日志头记录了 `PYTHONPATH`，其中的 `git=8ac53f9` 是主仓库的提交，而当时主仓库的 `run_prefix_cache_matrix.sh` 有未提交的改动（`R0` 和记录 `PYTHONPATH`），这些改动后来在 590121f 中提交。
- `baseline/`、`v1_cut/`：运行时在对应 worktree 里执行，并设置了 `PYTHONPATH`，但当时的日志头和 JSON 都没有记录 `PYTHONPATH` 或 `nanovllm_path`。JSON 的 `git` 是 benchmark 脚本所在 worktree 的提交（c5af84b / 6d8628a）。启动命令如下：
  ```bash
  cd /data/csd/nano-vllm-baseline && export PYTHONPATH=/data/csd/nano-vllm-baseline && \
    REPEATS=2 benchmarks/run_prefix_cache_matrix.sh /data/csd/nano-vllm/benchmarks/results/baseline -
  cd /data/csd/nano-vllm-after && export PYTHONPATH=/data/csd/nano-vllm-after && \
    REPEATS=2 EXTRA="--num-state-snapshots 48" benchmarks/run_prefix_cache_matrix.sh /data/csd/nano-vllm/benchmarks/results/after off branch && \
    REPEATS=1 EXTRA="--num-state-snapshots 48" benchmarks/run_prefix_cache_matrix.sh /data/csd/nano-vllm/benchmarks/results/after kv_only last all
  ```
  `v1_cut/` 当时叫 `after/`，所以日志里的 `--out` 写的是 `results/after/`。

`prefix_cache_stats`（快照的保存、淘汰等计数）从引擎启动开始累计，包含每次运行前的 37 个预热请求。

运行开始时 GPU 4 上有其他任务的（日志头 `gpu: 4` 行显示占用）：`baseline/` 中 shared8000_qps0.4 r2、shared8000_qps1 r1、multiturn r1 / r2，`v1_cut/` 中 13 个 off 运行。其余目录的运行开始时 GPU 4 都空闲。影响没有单独测量；对照见 doc 第 5 节。

## 单请求开销（`overhead/`）

`benchmarks/run_snapshot_overhead.sh` 调用 `benchmarks/bench_snapshot_overhead.py`：单个请求独占引擎，测未命中 prefill、命中 prefill、64 token 新 prompt 的 prefill（各 5 次取中位数）和快照拷贝的耗时（CUDA event 计时 20 次，去掉前 5 次，取中位数）。
- 引擎参数与在线负载不同：`max_model_len = len + 128`（2128 或 8128），`max_num_seqs=16`，`--num-state-snapshots` 默认 48（off 为 0），所以 KV block 是 1139 / 1133（off）或 845 / 838（48 份快照）。
- 脚本用主仓库里的版本，引擎由 `PYTHONPATH` 指定（v1：`/data/csd/nano-vllm-after`，v2：`/data/csd/nano-vllm-v2`），JSON 的 `nanovllm_path` 和 `git` 记录了实际导入的引擎。日志头记录 `PYTHONPATH`，不记录 git。

| 目录 | 内容 |
|---|---|
| `overhead/v1_cut/`、`overhead/v2_extra/` | 正式结果。机器上没有其他 GPU 任务时运行，按策略交替运行 v1 和 v2（off、last、branch、all 依次各跑 v1、v2） |
| `overhead/first_run/{v1_cut,v2_extra}/` | 第一次运行，已被上一行取代，保留作记录。两次运行时 GPU 4 上都有其他任务（日志头 `gpu: 4` 行：v1 8 个运行中 6 个、v2 中 3 个开始时 GPU 4 被占用；v2 那次能从文件确认的是同时在跑的 `fla_calls/first_run/bench_fla_calls.log`），而这些测量都受 host 开销影响。v1 那次的日志中，`--out` 写的是改名前的目录 `overhead/cut_6d8628a/` |

## FLA 调用（`fla_calls/`）

不加载引擎，只调用 fla-core 0.4.2。
- `bench_fla_calls.log`：`benchmarks/bench_fla_calls.py`，FLA 调用的 host 开销。在 GPU 0 上单独运行。
- `show_fla_plan.log`：`benchmarks/show_fla_plan.py`，单个 prompt 在各策略下得到的 FLA 调用计划（用主仓库的 `plan_fla_windows`）。
- `first_run/bench_fla_calls.log`：第一次运行，在 GPU 4 上，当时 GPU 0 正在跑 v2 的单请求开销测量，已被取代。

## 正确性（`correctness/`）

单测和端到端测试的输出。
- 590121f 上的四个单测日志和 `test_prefix_cache_e2e_v2.log`：开头记录了时间、git、GPU 和命令，末尾记录了退出码；在 GPU 4（PCI 3E:00.0，另一块 4090）上运行，使用主仓库的引擎（`nanovllm/` 自 40a3ccf 起没有改动）。
- `test_prefix_cache_e2e.log`（v1）：开头有时间、git、GPU 和命令，没有退出码，结尾是 PASS。
- `smoke_f00fed2.log`：只记录了日期；引擎来自 f00fed2 的 worktree（`/data/csd/nano-vllm-f00`，已删除）。
- 端到端测试用的是自己的引擎参数：`max_model_len=4096`、`max_num_seqs=8`、`max_num_batched_tokens=2048`、`gpu_memory_utilization=0.5`、`num_state_snapshots=16`。

| 文件 | 内容 |
|---|---|
| `test_snapshot_pool.log`、`test_state_snapshots.log`、`test_linear_attention.log`、`test_rejection_sampler.log` | 单测 |
| `test_prefix_cache_e2e_v2.log` + `_summary.json` | 端到端测试，当前代码（v2） |
| `test_prefix_cache_e2e.log` + `test_prefix_cache_e2e_summary.json` | 端到端测试，v1（6d8628a 加上当时未提交的测试改动，见日志头） |
| `smoke_f00fed2.log` | 中间提交 f00fed2 的冒烟运行 |
| `first_run/` | 早先的单测日志和 v2 端到端日志，已被上面的文件取代。除 `test_rejection_sampler.log`（日志头：17:29:20，git=6d8628a，GPU 4）外都没有日志头，无法确认是在哪个提交上跑的。v2 端到端的 summary JSON 与 v1、v2 的两份逐字节相同（md5 68f51c77ee99efe2939acf73d14bf89e），为节省空间没有保留 |

## 短 prefill 的定位与 CUDA Graph（`profile_prefill/`、`prefill_fix/`、`prefill_graph/`）

见 `docs/prefill_cuda_graph.md`。计时类测量都在 GPU 0 上单独运行；日志开头记录了时间、提交、`CUDA_VISIBLE_DEVICES`、环境变量（含 `PYTHONPATH`、`FLA_INTRACARD_CP`）、各 GPU 占用和命令，末尾是退出码。

| 目录 | 引擎 | 内容 |
|---|---|---|
| `profile_prefill/before.log`、`after.log`（及同名目录里的 JSON 和 profiler trace） | 524d5de，`FLA_INTRACARD_CP=1`（修复前的行为）/ 默认 | `benchmarks/profile_prefill.py` 实验 A–E |
| `profile_prefill/find_syncs_intracard_{1,0}.log` | 524d5de（`PYTHONPATH=/data/csd/nano-vllm-fix`） | `benchmarks/find_prefill_syncs.py`：前向里的同步次数与位置。在 GPU 4 上运行（不计时） |
| `profile_prefill/exploration/` | 0833ed6 / 524d5de | 定位过程中的探索性运行（含 `FLA_INTRACARD_CP` 开 / 关时 2K–32K token 的长 prompt 对比）。`first_profile_intracard_on.log` 的实验 E 因脚本缺陷在采样处崩溃，D 中的 "GPU busy" 计算有误，均已在后来的脚本中修正 |
| `prefill_fix/correctness/` | 524d5de | 修复一之后的单测与端到端测试；端到端 summary 与修复前逐字节相同 |
| `prefill_fix/throughput_no_warmup_round/` | 524d5de | 第一次满载吞吐测量，没有不计时的预热轮（C=16、32 的第一轮含约 3 s 的一次性编译），已被 `prefill_graph/throughput/fix_*` 取代 |
| `prefill_graph/latency/{graph,eager}.log` | d7e9967（`/data/csd/nano-vllm-g1`） | 同一引擎 prefill Graph 开 / 关时的 prefill 延迟 |
| `prefill_graph/latency/graph_cap2048.log` | 同上 | 上限设为 2048 时的延迟，用来选择上限 1024 |
| `prefill_graph/overhead/` | d7e9967 | `run_snapshot_overhead.sh`，单请求的未命中 / 命中 / 64 token prefill |
| `prefill_graph/online/` | d7e9967 | `run_prefix_cache_matrix.sh`，branch 2 次、off 1 次，参数与 `v2_extra/` 相同 |
| `prefill_graph/throughput/{graph,fix}_*.{json,log}` | d7e9967 / 524d5de（`/data/csd/nano-vllm-fix`） | `bench_throughput.py`，每个并发度先跑 1 轮不计时的预热；脚本都来自 d7e9967 |
| `prefill_graph/correctness/` | e0eb359（引擎代码与 d7e9967 相同） | kernel 精度、prefill Graph 对比（Qwen3.5-4B、MTP、Qwen3-0.6B）、单测、端到端测试；`first_attempt/` 是测试修正前的一次运行（上限改为 1024 后，测试仍按 512 判断哪些长度该走 eager；端到端测试当时把 Graph 路径与 eager 参照直接比较） |
| `prefill_graph/startup_cost.log` | e0eb359 | 启动时间、prefill Graph 数量、显存与 KV block 数（在 GPU 4 上运行，当时 GPU 0 在跑端到端测试，只作粗略参考） |

## 汇总

```bash
python benchmarks/summarize_prefix_cache.py benchmarks/results/baseline benchmarks/results/v1_cut benchmarks/results/v2_extra
python benchmarks/summarize_prefix_cache.py benchmarks/results/v2_extra_mtp
```

`summary.md` 就是这两条命令的输出。
