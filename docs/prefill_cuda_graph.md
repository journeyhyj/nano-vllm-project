# 短 prefill 的定位与优化：去掉 FLA 的同步，短 prefill 走 CUDA Graph

本文的每个数字都来自实际运行，并注明了命令和日志位置（日志在 `benchmarks/results/` 下）。没有测过的项目写"未测"。条件：RTX 4090（PCI 1A:00.0，`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`），Qwen3.5-4B bf16，torch 2.6.0+cu124，fla-core 0.4.2，triton 3.2.0；计时类测量都在机器上没有其他 GPU 任务时进行。

## 1. 问题

做完 hybrid prefix caching（`docs/hybrid_prefix_cache.md`）之后，命中缓存的请求只需 prefill 十几到一百多个 token，TTFT 却仍然停在约 100 ms。单独测 prefill 延迟随 prompt 长度的变化（`benchmarks/profile_prefill.py` 实验 A，修复前，日志 `benchmarks/results/profile_prefill/before.log`）：

| prompt 长度 | 16 | 64 | 256 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| prefill 耗时（ms，含整个 `generate` 调用，7 次中位数） | 94.0 | 94.7 | 94.5 | 95.4 | 128.3 | 240.0 |

16 到 1024 token 几乎一样：耗时由一个与长度无关的固定开销决定。

## 2. 定位过程

`benchmarks/profile_prefill.py` 把一步 prefill 从粗到细拆成五个实验，下面的数字都是 64 token 的 prompt（括号里是 2048 token），日志 `benchmarks/results/profile_prefill/before.log`（修复前，`FLA_INTRACARD_CP=1`）。

**B：时间花在引擎的哪个阶段。** 在调度、准备输入、前向、算 logits、采样、后处理这些函数外面包上同步计时：

| 阶段 | 64 token | 2048 token |
|---|---|---|
| 调度 + 准备输入 + 采样 + 后处理 | 0.7 ms | 1.2 ms |
| 算 logits（lm_head） | 1.5 ms | 1.5 ms |
| 模型前向：Python 发起完所有 kernel 所用的时间 | 91.8 ms | 123.0 ms |
| 模型前向：发起完之后再等 GPU 算完 | 0.03 ms | 3.1 ms |

时间几乎都在模型前向里；而且 64 token 时，Python 发起完最后一个 kernel，GPU 只需再等 0.03 ms 就算完了，说明 GPU 一直在等 CPU：**这一步是 CPU（host）瓶颈**。

**C：前向里哪部分模块最耗 CPU。** 在每层和子模块上挂 forward hook 计 CPU 时间（不同步）：24 个 GDN 层共 85.1 ms，其中 FLA 的 `chunk_gated_delta_rule` 调用 39.8 ms（每次约 1.7 ms），我们自己的两个 Triton kernel 共 4.3 ms（48 次发起，约 90 µs 一次）；8 个全注意力层共 13.6 ms。2048 token 时 FLA 调用变成 65.4 ms：多出来的部分是 CPU 在 FLA 里等 GPU（见 E）。

**D：torch.profiler。** 一步 64 token 的 prefill 发起了 1202 个 kernel，GPU 上的 kernel 总耗时只有 13.2 ms（2048 token 时 109.4 ms）。GDN 层每层约 43 个 kernel，全注意力层约 21 个，FLA 一次调用约 10.6 个。也就是说 CPU 平均每发起一个 kernel 花约 76 µs（91.8 ms ÷ 1202），而 GPU 平均只算约 11 µs。

**E：反事实——如果没有 CPU 开销会多快。** 把同一次前向录成 CUDA Graph 再回放。修复前录制失败（"operation failed due to a previous error during capture"）。用 `torch.cuda.set_sync_debug_mode` 查前向里的同步（`benchmarks/find_prefill_syncs.py`，日志 `benchmarks/results/profile_prefill/find_syncs_intracard_1.log`）：一次前向有 24 次 GPU→CPU 同步，每个 GDN 层一次，全部来自

```
fla/ops/gated_delta_rule/chunk.py:76     chunk_gated_delta_rule_fwd_h(...)       # 没有把 cu_seqlens_cpu 传下去
fla/ops/backends/__init__.py:150         dispatch -> intracard 后端（推理模式 + 变长输入时默认启用）
fla/ops/common/intracard_cp.py:437       cu_seqlens_cpu = cu_seqlens.cpu()      # 同步
```

我们在调用 FLA 时已经给了 `cu_seqlens_cpu`，但 fla-core 0.4.2 只在准备 chunk 索引时用了它，没有传给 `chunk_gated_delta_rule_fwd_h`；而该函数默认被分派到"卡内 context parallel"后端，后端拿不到 CPU 副本，就自己 `.cpu()` 一次。同步的后果有两个：每次都要等 GPU 把队列里的活干完，CPU 和 GPU 无法重叠（2048 token 时前向发起 123.0 ms + 等待 3.1 ms，而 GPU 只需 109.4 ms）；并且整段 prefill 没法录成 CUDA Graph。

## 3. 修复一：关掉 FLA 的卡内并行后端

`nanovllm/layers/linear_attention.py` 在导入 FLA 之前执行 `os.environ.setdefault("FLA_INTRACARD_CP", "0")`（提交 524d5de）。只改环境变量的默认值，不改 FLA，用户显式设置时仍以用户为准。

- **同步消失**：`find_syncs_intracard_0.log` 中，error 模式下前向不再触发同步（warn 模式计到的 1 次是切换调试模式那一行本身）。
- **这个后端在单卡 4090 上没有收益**：它是为长序列在多个 SM 上并行设计的，但实测关掉后长 prompt 不变慢（实验 A，5 次中位数，日志 `benchmarks/results/profile_prefill/exploration/long_prompts_intracard_{on,off}.log`）：

  | prompt | 2048 | 8192 | 16384 | 32768 |
  |---|---|---|---|---|
  | 开（原默认） | 132.1 ms | 490.2 ms | 1042.2 ms | 2332.4 ms |
  | 关 | 115.1 ms | 485.0 ms | 1025.4 ms | 2300.3 ms |

- **效果**（`profile_prefill/after.log` 对 `before.log`）：64 token 94.7 → 91.0 ms，2048 token 128.3 → 114.5 ms；2048 token 时前向变成"发起 88.8 ms，GPU 再算 23.1 ms"，CPU 和 GPU 重叠起来了。
- **输出不变**：修复前后 `tests/test_prefix_cache_e2e.py` 的 summary（每个配置的输出 token、命中长度、logits 比较）逐字节相同（`benchmarks/results/prefill_fix/correctness/`）。
- 修复后录图成功：64 token 的前向，eager 97.2 ms（含 profile 用的 hook），CUDA Graph 回放 11.8 ms，输出逐位相同；2048 token 112.5 → 110.2 ms，几乎不变，因为那时已经是 GPU 瓶颈（`after.log` 实验 E）。这个 11.8 ms 就是短 prefill 能达到的下限。

## 4. 修复二：短 prefill 走 CUDA Graph

### 4.1 为什么 prefill 也要用 CUDA Graph

通常只有 decode 用 CUDA Graph：decode 每步每条序列只有 1 个 token，GPU 很快算完，瓶颈在 CPU 发 kernel；而长 prefill 是 GPU 瓶颈，录图省不了多少。但 Graph 解决的是"CPU 发 kernel"的开销，该看的是这一步 GPU 有多少活，而不是它叫 prefill 还是 decode：上面测出 1K token 以下的 prefill 在这块卡上同样是 CPU 瓶颈。命中 prefix cache 之后的 prefill、短对话的 prompt 都属于这种情况。

### 4.2 做法（提交 0ca4eca，`nanovllm/engine/model_runner.py`）

- **哪些 prefill 走 Graph**：总 token 数不超过 `prefill_graph_tokens`（默认 1024）、不超过 16 条序列、并且这一步不保存状态快照（保存快照需要 FLA chunk kernel 的附加序列，见 hybrid prefix caching 文档）。其余照旧 eager。
- **分档与补齐**：按 token 数（16、32、…、1024）和序列数（1、2、4、8、16）各录一张图，共 35 张，与 decode 的图共用一个显存池。每一步补齐到最近的档位：多出来的 token 行不属于任何序列，写 KV 的槽位设为 −1；多出来的序列长度为 0，状态槽指向 decode 图本来就留着的备用槽。
- **注意力**：Graph 路径里总是通过 block table 读 K/V（本步的新 K/V 在注意力之前已经写入 cache），这样有没有命中 prefix cache 走的是同一段代码。
- **GDN 层换 kernel**：FLA 的 chunk kernel 的网格大小取决于各序列长度（由 CPU 根据 `cu_seqlens` 计算），补齐之后也会变，无法回放。Graph 路径改用 FLA 的 `fused_recurrent_gated_delta_rule`：一次 kernel 调用，网格只取决于序列条数，序列长度从 GPU 上的 `cu_seqlens` 读，长度为 0 的序列直接保留初始状态。
- **代价**：引擎启动多 10 s（录 35 张图），显存多占 176 MiB，KV block 数不变（录图在分配 KV cache 之后，用的是 `gpu_memory_utilization` 之外的余量）。日志 `benchmarks/results/prefill_graph/startup_cost.log`（与另一块卡上的测试同时运行，只作粗略参考）。

### 4.3 上限为什么是 1024

fused recurrent kernel 逐个 token 递推，单条序列越长越慢。同一个引擎分别开、关 prefill Graph（实验 A，7 次中位数，日志 `benchmarks/results/prefill_graph/latency/{graph,eager}.log`）：

| prompt | 1 | 16 | 64 | 128 | 256 | 512 | 768 | 1024 | 1536 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eager | 82.8 | 89.3 | 89.4 | 89.6 | 89.3 | 89.5 | 90.0 | 90.3 | 91.7 | 115.8 | 238.2 |
| Graph（≤1024 时） | 12.0 | 12.2 | 13.6 | 15.6 | 22.1 | 35.9 | 63.1 | 67.1 | 94.6 | 114.9 | 233.3 |

单位 ms。1536 及以上两列都是 eager（Graph 引擎超过上限时也走 eager，两次运行相差 3 ms 以内）。把上限设为 2048 时（日志 `benchmarks/results/prefill_graph/latency/graph_cap2048.log`），1536 token 走 Graph 是 129.7 ms、2048 token 是 141.3 ms，都比 eager（91.7、115.8 ms）慢，所以上限取 1024。

### 4.4 正确性

- **kernel 精度**（`tests/test_gdn_kernel_accuracy.py`）：以 float64 逐 token 递推为参照，bf16 输入、Qwen3.5-4B 的形状，16 到 1024 token：chunk kernel 的输出相对误差 4.1–4.5e-3、最终状态 3.3–3.4e-3；fused recurrent 分别为 1.7e-3、1.9–2.8e-7。Graph 路径用的 kernel 更精确，只是和 eager 路径不再逐位相同。
- **Graph 与 eager 对比**（`tests/test_prefill_graph.py`，同一个引擎里分别走两条路径，比较第一个 token 的 logits）：单条序列 1 到 1025 token、3 条和 16 条序列的 batch、17 条序列（应走 eager）、chunked prefill 的第二段（带初始状态）、prefix cache 命中后的尾部（带恢复的快照），全部满足"logits 最大差 ≤ 1.0，argmax 不同时 top-2 差 ≤ 0.5"。另有隔离性检查：一条序列在 decode 时，另一条序列的 prefill 走 Graph 还是 eager，前者生成的 48 个 token 完全相同——补齐用的行和序列没有写坏别人的 KV 或状态。在 Qwen3.5-4B（含 MTP）和 Qwen3-0.6B 上都通过。日志 `benchmarks/results/prefill_graph/correctness/test_prefill_graph*.log`。
- **端到端**（`tests/test_prefix_cache_e2e.py`，提交 d9b4c26，日志 `benchmarks/results/prefill_graph/correctness/test_prefix_cache_e2e.log` 和 `_summary.json`）：
  - 原有的 10 个配置改为显式关闭 prefill Graph（纯 chunk kernel），结果与加入 Graph 之前逐项相同（与 `benchmarks/results/correctness/test_prefix_cache_e2e_v2_summary.json` 比较）。
  - 新增的 Graph 配置：`off_graph` 对 `off`（两者只差在短 prefill 用哪个 kernel）19 个请求中 18 个完全一致，最大 |Δlogit| 1.69，唯一的分叉在 top-2 差为 0 的平局处。这是"换 kernel"本身带来的差异。
  - `branch_graph` / `last_graph` / `all_graph` 对 `off_graph`：最大 |Δlogit| 0.94 / 1.63 / 1.63，都不超过上一条的 1.69；分叉分别有 6 / 5 / 6 处，都在 top-2 差 ≤ 0.25 的位置；命中情况与 eager 的三组相同，命中后被抢占的场景也出现了。
  - 测试对这几组的 logits 上限是 2.0（其余组仍为 1.0，理由见 kernel 精度一条），分叉规则不变。
  - 第一次运行时测试本身有两处缺陷：按旧的上限 512 判断哪些长度该走 eager；`off_graph` 既是被比较的配置又是参照，它的 logits 没有保存。修正后重跑；失败的日志在 `correctness/first_attempt/`。

实现过程中出过一个错误，被上面的测试发现：FLA 0.4.2 的 `fused_recurrent_gated_delta_rule` 参数顺序是 `(q, k, v, g, gk, gv, beta, ...)`，按位置传 beta 会落到 `gk` 上，造成越界读和 NaN；现在用关键字传参。

## 5. 实验结果

### 5.1 单个请求（命中 prefix cache 之后的下限）

`bench_snapshot_overhead.py`，5 次中位数，单位 ms。"修复前"是 hybrid prefix caching 文档中的 v2（40a3ccf），"现在"是提交 d7e9967（含两项修复）：

| | 命中 2048 token 后 prefill 16 token | 命中 7936 token 后 prefill 128 token | 64 token 的新 prompt | 2064 token 未命中（超过上限，eager） |
|---|---|---|---|---|
| 修复前（branch） | 92.9 | 97.3 | 92.5 | 134.8 |
| 现在（branch） | 13.7 | 20.8 | 13.6 | 120.9 |

日志：`benchmarks/results/overhead/v2_extra/` 与 `benchmarks/results/prefill_graph/overhead/`。

### 5.2 在线负载下的 TTFT

与 hybrid prefix caching 文档相同的负载与参数（`max_num_seqs=16`、48 份快照），branch 各 2 次运行取平均，单位 ms，日志 `benchmarks/results/prefill_graph/online/`：

| 负载 | 改动前（c5af84b）p50 / p99 | hybrid prefix caching v2 p50 / p99 | 现在 p50 / p99 | p50 相对改动前 |
|---|---|---|---|---|
| 共享 2000 token，QPS 1.5 | 139 / 254 | 106 / 185 | 34 / 130 | 4.15× |
| 共享 2000 token，QPS 4 | 148 / 1094 | 106 / 209 | 34 / 130 | 4.29× |
| 共享 2048 token，QPS 4 | 153 / 1196 | 105 / 194 | 22 / 135 | 6.82× |
| 共享 4000 token，QPS 0.75 | 246 / 455 | 113 / 265 | 29 / 256 | 8.45× |
| 共享 4000 token，QPS 2 | 251 / 673 | 111 / 261 | 31 / 252 | 8.05× |
| 共享 8000 token，QPS 0.4 | 491 / 903 | 110 / 541 | 25 / 493 | 19.28× |
| 共享 8000 token，QPS 1 | 494 / 1372 | 106 / 539 | 27 / 490 | 18.31× |
| 多轮对话，0.5 段/s | 184 / 415 | 101 / 188 | 32 / 163 | 5.74× |

- p99 仍由最早的一两个未命中请求决定（共享前缀 2000–8000 token 都超过 1024，未命中时走 eager），所以长前缀时 p99 下降不多。
- 不开 cache（off，1 次运行）时，修复一让 2000 token 的 prefill 变快：共享 2000 token、QPS 4 的 p50 从 150 降到 131 ms，p99 从 1107 降到 829 ms；8000 token 时只快 2%。
- 汇总命令：`python benchmarks/summarize_prefix_cache.py benchmarks/results/baseline benchmarks/results/v2_extra benchmarks/results/prefill_graph/online`。

### 5.3 满载吞吐

`benchmarks/bench_throughput.py`：每轮同时提交 C 个请求，输入 138 个随机 token、输出 1024 个（忽略 EOS），temperature 1.0；每个并发度先跑 3 个请求和 1 轮不计时的预热，再计时 10 轮（并发大时减为 6 或 3 轮），取平均。日志 `benchmarks/results/prefill_graph/throughput/{fix,graph}_{k0,mtp,mtp_vocab32k}.{json,log}`。不开投机解码时：

| 并发 C | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 输出吞吐（tok/s），只有修复一 | 95 | 174 | 338 | 641 | 1181 | 1986 | 2996 |
| 输出吞吐（tok/s），两项修复 | 95 | 175 | 338 | 640 | 1181 | 1986 | 2994 |
| TTFT（ms），只有修复一 | 92 | 91 | 90 | 91 | 135 | 265 | 532 |
| TTFT（ms），两项修复 | 20 | 30 | 55 | 95 | 135 | 265 | 533 |
| TPOT（ms），两项修复 | 10.46 | 11.43 | 11.77 | 12.43 | 13.43 | 15.87 | 20.87 |

- 吞吐不变：满载时时间花在 decode 上，decode 本来就用 CUDA Graph。prefill Graph 改善的是低并发时的 TTFT；C ≥ 8 时这一批 prefill 超过 1024 token，仍走 eager。
- **MTP 在满载时反而更慢**：开 MTP 后每条序列要两个状态槽，同时运行的序列最多 47 条（不开时 64 条）。两项修复下，C=32 时 MTP 2338 tok/s，不开 1986 tok/s（1.18×）；C=64 时 MTP 2319 tok/s，不开 2994 tok/s。
- **32K 草稿词表在 temperature 1.0 下没有收益**：接受率从完整词表的约 0.71 降到约 0.60，吞吐与完整词表相当或更低（例如 C=32：2201 对 2338 tok/s）。README 中 1.55× 的结果是在 MT-Bench、贪心解码下测的，两者条件不同。
- 与 Chitu 在同一块 4090 上的报告（`artifacts/chitu-4090/REPORT_EN.md`，相同的输入输出长度和并发度）相比，输出吞吐：C=1 为 95 对 86.67 tok/s，C=16 为 1181 对 934.81，C=32 为 1986 对 1258.96。两者不完全可比：Chitu 走 HTTP 服务、采样带 top-p / top-k，nano-vllm 是进程内调用、只有 temperature；TTFT 因此不做比较。
- 第一次测量时没有不计时的预热轮，C=16、32 的第一轮因首次遇到新的 batch 形状（编译、autotune）TTFT 约 3 s，拉低了平均值；保留在 `benchmarks/results/prefill_fix/throughput_no_warmup_round/`，本表为加入预热轮后的结果。

## 6. 局限与没有收益的部分

- **超过 1024 token 的 prefill 没有收益**：仍走 eager，固定开销约 90 ms；fused recurrent kernel 逐 token 递推，单条长序列在 Graph 里反而更慢。对长序列也能录图需要能处理变长的 chunk kernel 网格，未实现。
- **保存快照的 prefill 不走 Graph**：branch 策略下第二次遇到某个前缀时的那次 prefill 仍是 eager。
- **数值**：Graph 路径的 GDN 换成 fused recurrent kernel，结果与 eager 不逐位相同（更接近 float64 参照，见 4.4）。
- **MTP**：prefill 之后 MTP 层自己的那次前向仍是 eager（1 层，未单独测量其开销）。
- **启动与显存**：多录 35 张图，启动多约 10 s、显存多 176 MiB（粗测）。
- **测量**：每组 1–2 次运行；满载吞吐只测了随机 token 的 prompt 和 temperature 1.0；与 Chitu 的比较条件不完全相同。精度评测（下游任务）未测。

## 7. 复现

```bash
source /home/csd/anaconda3/etc/profile.d/conda.sh && conda activate nano-vllm
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
# 定位
FLA_INTRACARD_CP=1 python benchmarks/profile_prefill.py ~/huggingface/Qwen3.5-4B --out-dir benchmarks/results/profile_prefill/before
python benchmarks/profile_prefill.py ~/huggingface/Qwen3.5-4B --out-dir benchmarks/results/profile_prefill/after
FLA_INTRACARD_CP=1 python benchmarks/find_prefill_syncs.py ~/huggingface/Qwen3.5-4B
# prefill 延迟，Graph 开 / 关
python benchmarks/profile_prefill.py ~/huggingface/Qwen3.5-4B --experiments A --lens 1,16,64,128,256,512,768,1024,1536,2048,4096
python benchmarks/profile_prefill.py ~/huggingface/Qwen3.5-4B --experiments A --lens 1,16,64,128,256,512,768,1024,1536,2048,4096 --prefill-graph-tokens 0
# 单请求、在线负载、满载吞吐
GPU=0 benchmarks/run_snapshot_overhead.sh benchmarks/results/prefill_graph/overhead off last branch all
REPEATS=2 EXTRA="--num-state-snapshots 48" benchmarks/run_prefix_cache_matrix.sh benchmarks/results/prefill_graph/online branch
python benchmarks/bench_throughput.py ~/huggingface/Qwen3.5-4B --out graph_k0.json
python benchmarks/bench_throughput.py ~/huggingface/Qwen3.5-4B --k 1 --out graph_mtp.json
# 测试
python tests/test_gdn_kernel_accuracy.py
python tests/test_prefill_graph.py && python tests/test_prefill_graph.py --k 1 && python tests/test_prefill_graph.py ~/huggingface/Qwen3-0.6B
NANOVLLM_PORT=2399 python tests/test_prefix_cache_e2e.py
```

实际运行时，计时类测量用冻结的 worktree（`/data/csd/nano-vllm-g1` 在 d7e9967，`/data/csd/nano-vllm-fix` 在 524d5de）并以 `PYTHONPATH` 指定，见各日志开头。
