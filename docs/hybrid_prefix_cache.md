# Qwen3.5 混合架构上的 Prefix Caching（草稿）

本文的每个数字都来自实际运行，并注明了命令和日志位置（日志在 `benchmarks/results/` 下，目录说明见该目录的 README）。没有测过的项目一律写"未测"。

## 1. 问题

Qwen3.5-4B 共 32 层：24 层 Gated DeltaNet（GDN，线性注意力），8 层全注意力，开启 MTP 时还有 1 层全注意力的 MTP 层。

nano-vllm 原有的 prefix caching 以 256 token 为一个 block，对 block 计算链式 hash，复用前缀的 KV。这套机制对混合模型不起作用，本项目此前对混合模型直接关闭了它：
- **KV 里没有 GDN 的状态。** GDN 层要从前缀末尾的递归状态（conv 状态和 SSM 状态）接着往下算。这份状态是前缀所有 token 逐个累积的结果，KV block 里没有它。
- **只复用 KV 节省不了计算。** 第 l 层 GDN 在前缀末尾的状态，由前缀每个 token 在第 l 层的输入决定；这些输入又取决于前面所有层（包括全注意力层）对这些 token 的输出。所以想得到这份状态，就只能把整个前缀在整个网络里重新跑一遍，而全注意力层的 KV 也会在这一遍里顺带算出来。实测（第 4.2 节第一张表的 kv_only 一列）：共享 KV block、GDN 全部重算时，各负载的 TTFT p50 与不开 cache 相差不到 3%。

结论：要跳过前缀的计算，必须在 block 边界处保存 GDN 的状态（下文称为**快照**），命中时与 KV 一起恢复。

## 2. 测量：不开 prefix cache 时的耗时

测试条件：
- RTX 4090（PCI 1A:00.0），Qwen3.5-4B bf16，引擎版本 c5af84b（加入本功能之前）；
- 引擎参数：`max_num_seqs=16`、`max_num_batched_tokens=16384`、`max_model_len=10240`、`gpu_memory_utilization=0.85`，共 1177 个 KV block。

两种负载，均由 `benchmarks/bench_prefix_cache.py` 生成，TTFT 从请求到达的时刻开始计时：
- **共享 system prompt**：
  - 64 个请求，按泊松过程到达；
  - 每个请求由同一段共享前缀和一道 MT-Bench 第一轮问题组成。共享前缀（chat 模板 + system prompt）恰好 N token，system prompt 取自 `/usr/share/common-licenses/` 下的许可证文本；
  - 最多生成 128 个 token，遇到 EOS 提前结束。baseline 中每个请求平均生成 115–117 个。
- **多轮对话**：
  - 16 段对话，每段 6 轮，共 96 个请求；
  - 每段对话的开始按泊松过程到达，之后每一轮都在上一轮结束时立即发出；
  - 下一轮的 prompt 是上一轮的完整上下文（prompt + 回答）加上新问题，按 token 拼接；
  - 每轮最多生成 256 个 token，baseline 中平均 211 个。
- **预热**：每次运行前，先用 37 个由随机 token 组成的 prompt 预热（64 / 300 / 2048 / 4096 / 8192 token 各一个，再加两批各 16 个随机长度的 prompt），让 torch.compile 和 FLA autotune 完成。预热请求不计入 TTFT 和吞吐，但计入快照池的统计（见第 4.2 节）。

| 负载 | QPS | TTFT p50 | TTFT p99 | 输出吞吐 |
|---|---|---|---|---|
| 共享 2000 token | 1.5 | 139 ms | 254 ms | 146 tok/s |
| 共享 2000 token | 4 | 148 ms | 1094 ms | 361 tok/s |
| 共享 2048 token | 4 | 153 ms | 1196 ms | 359 tok/s |
| 共享 4000 token | 0.75 | 246 ms | 455 ms | 74 tok/s |
| 共享 4000 token | 2 | 251 ms | 673 ms | 188 tok/s |
| 共享 8000 token | 0.4 | 491 ms | 903 ms | 40 tok/s |
| 共享 8000 token | 1 | 494 ms | 1372 ms | 96 tok/s |
| 多轮对话（2000 token 起） | 0.5 段/s | 184 ms | 415 ms | 367 tok/s |

每行是 2 次运行的平均。日志：`benchmarks/results/baseline/<负载>_default_r{1,2}.{json,log}`，命令见每个日志的开头。

单个请求独占引擎时的 prefill 耗时用 `bench_snapshot_overhead.py --policy off` 测量，5 次取中位数。这组测量的条件与上面不同：
- 引擎为 v1 / v2 两个版本（6d8628a / 40a3ccf）。off 不存快照，走的是与改动前相同的路径；
- `max_model_len = len + 128`（2128 或 8128），KV block 1139 / 1133 个。

日志在 `benchmarks/results/overhead/{v1_cut,v2_extra}/len{2000,8000}_off.{json,log}`。

| prompt 长度 | prefill 耗时（v1 / v2） |
|---|---|
| 2064 token | 134.9 / 134.6 ms |
| 8064 token | 480.9 / 481.1 ms |
| 64 token（新 prompt，`short_prefill_ms`） | 94.4 和 94.6 / 92.8 和 94.6 ms（`--len 2000` 和 `--len 8000` 两个进程） |

最后一行说明，即使只有 64 个 token，一步 prefill 也要约 93–95 ms。这一下限也就是命中 cache 后 TTFT 能降到的最低值。这段时间由什么组成没有做 profile（未测）。只知道其中一部分：prefill 走 eager 模式；在单独的微基准里，24 层各调用一次 FLA 的 host 时间为 21.9 ms（第 3.4 节）。另外，这个下限在不同进程之间会变：第 3.4 节的 16 个进程里，有一个是 104.7 ms，其余在 92.5–96.2 ms。

## 3. 方法

### 3.1 快照里存什么，占多少显存

每个 GDN 层要存两样东西：
- **conv 状态**：边界前最后 3 个 token 的原始输入，3 × 8192 个 bf16，共 48 KiB；
- **SSM 状态**：32 头 × 128 × 128 的 fp32，共 2 MiB。

24 层合计 51,511,296 字节，即 49.125 MiB。作为对照，一个 256 token 的 KV block（8 层全注意力）是 8 MiB，开启 MTP 时为 9 MiB。所以一份快照约等于 6.1 个 KV block（开启 MTP 时为 5.5 个）。这些大小是按 `~/huggingface/Qwen3.5-4B/config.json` 和 `ModelRunner` 的分配代码算出来的，不是测量值；51,511,296 与单请求开销 JSON 里的 `snapshot_bytes_each` 一致，48 份占 2358 MiB、KV block 剩 882 个见 `benchmarks/results/summary.md`。

快照放在一个固定大小的池里，大小由 `num_state_snapshots` 决定，在分配 KV cache 之前从同一份显存预算中扣除：
- 默认值 −1 表示自动：取 64 和"剩余显存的 25% 能放下的份数"中较小的一个；
- benchmark 中统一设为 48，占用 2358 MiB，KV block 数相应从 1177 降到 882。

### 3.2 索引与命中规则

- **索引**：快照以"边界前那个 block 的链式 hash"为 key，所以沿 KV block 做前缀匹配时，可以顺便查到每个边界有没有快照。
- **命中位置**：H = 已匹配的 KV block 中，带快照的最深那个边界。
  - [0, H) 的 KV block 直接共享；
  - 快照拷贝进序列自己的状态槽；
  - 只 prefill [H, L) 这一段。
- **不复用的部分**：KV 能匹配得更深（到 M）时，[H, M) 之间的 block 目前不共享。这段 token 反正要重算 GDN，重算时全注意力的 KV 也一并写进序列自己的 block。
- **prompt 的最后一个 block 永远不参与匹配**：最后一个 token 的 logits 必须算出来。

### 3.3 什么时候存快照（准入策略）

快照比 KV 贵 6 倍，不能每个边界都存。实现了三种策略，由 `prefix_cache_policy` 选择，另有两个对照组：

| 策略 | prefill 时在哪里存 | 特点 |
|---|---|---|
| `all` | 每个 block 边界 | 命中最早，但显存消耗最大（每缓存一个 block 要 57.1 MiB，是纯 KV 的 7.1 倍），池子很快被挤满 |
| `last` | prompt 最后一个完整 block 的边界，即 ((L−1)//256)·256 | 每个请求存 1 份。只有当这个边界落在共享部分之内时，别的请求才用得上；如果问题把它推过了共享部分，就只有前面这些 token 完全相同的 prompt 才能命中 |
| `branch`（默认） | 只在 KV 匹配位置超过最深快照时，存在 KV 匹配的末端 M 处 | 类似 Marconi：前缀被第二个请求走到时才存，从第三个请求开始命中。prefill 中只为确实被共享过的前缀存快照 |
| `kv_only` | 不存 | 对照组：共享 KV block，但每个 token 都重算，并且不写入共享的 block |
| `off` | 不存 | 对照组：不做 prefix caching |

三种保存策略都会在 **decode 越过 block 边界时**额外存一份快照，多轮对话的下一轮正好从这里接着算。branch 和 last 把它当作这条序列的"尾部快照"：同一条序列存了更新的尾部快照后，旧的那份被移到 LRU 队首，最先淘汰。last 下，prompt 自己的那份快照也算尾部快照。all 不做这种降级（`BlockManager.finish_step`）。

### 3.4 prefill 中怎样拿到边界处的状态

FLA 的 chunk kernel 只输出每条序列的最终状态（fp32），中间各 chunk 的状态是 bf16，精度不够，不能直接当快照用。下面是两种做法，以及实测差别。

**v1：在边界处切开 FLA 窗口**（提交 6d8628a）
- 做法：窗口在边界处结束，这个窗口输出的最终状态就是快照。边界与序列起点的距离是 64 的倍数，所以切开后的计算与不切逐位相同。
- 问题：每切一刀，每个 GDN 层就要多调用一次 FLA。FLA 的一次调用不论 token 多少，host 时间都约 0.9 ms：

  | 调用方式（24 层，每层调用一次或两次） | host 时间 | 每次调用 |
  |---|---|---|
  | 每层 1 次，16 token | 21.9 ms | 0.91 ms |
  | 每层 1 次，2064 token | 21.9 ms | 0.91 ms |
  | 在 2048 处切开：每层 2 次（2048 + 16 token） | 60.8 ms | 1.27 ms |
  | 每层 2 次（1024 + 1024 token），两次共用同一个 `cu_seqlens` 对象 | 44.5 ms | 0.93 ms |
  | v2：每层 1 次，2064 + 附加的 2048 token | 22.2 ms | 0.93 ms |

  - 命令：`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_fla_calls.py`，7 次取中位数。
  - 日志：`benchmarks/results/fla_calls/bench_fla_calls.log`（GPU 0，单独运行，不加载引擎，输入是随机张量）。
  - 切开之后，每次调用从 0.91 ms 涨到 1.27 ms；而两次调用共用同一个 `cu_seqlens` 时只有 0.93 ms。推测原因是：切开后两个窗口的 `cu_seqlens` 在各层之间交替出现，FLA 内部只缓存一项的 chunk 索引每次都不命中。没有直接观察 FLA 缓存的命中情况。

**v2：附加一条序列**（提交 91fe9a1）
- 做法：把"这条序列在本窗口中的起点 → 边界"这段 token 复制一份，作为一条额外的序列，追加进同一次 FLA 调用，并给它同一个初始状态。它的最终状态就是边界处的状态；真实序列照常不切开计算。
- 代价：FLA 的调用次数不变，但多出这些 token 的计算。另外，每层每个窗口要对 x、g、beta 按行 gather 一次（包括窗口自己的行），并 gather 初始状态。
- 回退：一次调用的总 token 数不超过 warmup 跑过的最大调用（KV cache 正是按那次的峰值显存定的大小；本文的两种 benchmark 配置下都是 8192）。放不下的边界退回到切开窗口；每个窗口每轮只切一处，让切开腾出的空间留给附近的边界使用。
- 正确性：
  - 附加序列得到的快照与切开得到的逐位相同，真实序列的输出与不存快照时逐位相同（`tests/test_state_snapshots.py`）。覆盖了三种情况：只用附加序列（窗口 8192）、附加序列与切开混合（窗口 2048、1024）、只切开（窗口 512）。
  - 同一文件里的 `test_window_plan_fuzz` 随机生成 2000 个 batch。它只检查调用计划，不做数值计算：每个带附加序列的调用都不超出预算；每份快照都由恰好走到对应边界的那条 FLA 序列提供；FLA 调用次数从不多于"每个边界都切开"（即 v1）。2000 个 batch 共 52,072 次调用，v1 为 94,140 次。
  - 日志：`benchmarks/results/correctness/test_state_snapshots.log`。

单个 prompt 的调用计划（`python benchmarks/show_fla_plan.py`，日志 `benchmarks/results/fla_calls/show_fla_plan.log`；预算 8192）：

| prompt | 策略要存的边界 | v2 每层调用次数 | v1 每层调用次数 |
|---|---|---|---|
| 2064 token | last：2048 | 1（附加 2048 token） | 2 |
| 8064 token | last：7936 | 2（8064 + 7936 超出预算，退回切开） | 2 |
| 2064 token | all：256, 512, …, 2048（8 个） | 2（在 1792 处切开一次，共附加 5632 token） | 9 |
| 8064 token | all：256, …, 7936（31 个） | 8 | 32 |

单个请求独占引擎时的实测（`bench_snapshot_overhead.py`，5 次取中位数；引擎参数见第 2 节。last / branch / all 另有 48 份快照，KV block 为 845 / 838 个）：

| 策略 | 2064 token 未命中，v1 | 2064 token 未命中，v2 | 8064 token 未命中，v1 | 8064 token 未命中，v2 | 命中，v1（2000 / 8000） | 命中，v2（2000 / 8000） |
|---|---|---|---|---|---|---|
| off | 134.9 | 134.6 | 480.9 | 481.1 | – | – |
| last | 191.9 | 148.0 | 523.3 | 524.0 | 92.8 / 95.8 | 93.2 / 95.7 |
| branch | 135.0 | 134.8 | 481.6 | 481.3 | 92.7 / 95.5 | 92.9 / 97.3 |
| all | 574.6 | 222.7 | 2139.8 | 905.8 | 93.9 / 95.5 | 95.2 / 106.6 |

- 单位 ms。"未命中"指一个全新的 prompt，其中包含该策略要保存的边界：last 存 1 份，all 每 256 token 存 1 份；branch 在第一次出现时不保存。
- "命中"指同一个 prompt 第三次发送，此时命中 2048 或 7936 个 token，只需 prefill 剩下的 16 或 128 个。
- 日志：`benchmarks/results/overhead/{v1_cut,v2_extra}/len{2000,8000}_<策略>.{json,log}`。
- 这些运行在机器上没有其他 GPU 任务时进行，按策略交替运行 v1 和 v2。单次的值在 JSON 的 `*_all_ms` 里：未命中和 64 token prefill 的 5 次测量之间最多相差 2.3%；命中时，branch 的 4 个进程里第一次都比其余几次慢（3.8–8.2%），last 和 all 没有这种现象，所以取中位数。
- 更早的一次运行在 `benchmarks/results/overhead/first_run/`。未命中耗时与本表的差别：off 与 branch 不超过 1.6%，last 为 0.6–3.3%，all 在 v1 2064 token 上差 14.9%（660.2 对 574.6 ms）、在 v2 8064 token 上差 6.0%（851.5 对 905.8 ms）。两次运行都与 GPU 4 上的其他任务重叠：日志头的 nvidia-smi 行显示，v1 那次 8 个运行里有 6 个开始时 GPU 4 占用 8.8–12.6 GB，v2 那次也有 3 个。本表的 16 个运行开始时 GPU 4 都空闲，所以以本表为准。

读这张表：
- **v1 的切开很贵，而且不全是 FLA 调用的 host 开销。** 2064 token 上：
  - last 切 1 刀，多 57.0 ms；all 切 8 刀，多 439.7 ms，相当于每多一次 FLA 调用多约 2.3 ms。
  - 按微基准中切开一刀让 24 层多出 38.9 ms host 时间来估算，host 开销约占七成；其余部分没有分解（未测）。
- **v2 在 2064 token 上把 last 的额外开销从 57.0 ms 降到 13.4 ms，把 all 的从 439.7 ms 降到 88.1 ms。**
  - last 剩下的 13.4 ms 没有分解。在微基准里，24 层各一次调用、每次多附加 2048 token，host 时间合计只多 0.3 ms（22.2 对 21.9 ms），所以推测主要是附加 token 的 GPU 计算和 gather。
  - all 在 v2 中仍有一次回退切开（见上面的调用计划）。
- **8064 token 上，last 在 v1 和 v2 中一样**（523.3 对 524.0 ms，比 off 多 42.4 / 42.9 ms），因为两者的调用计划相同：附加序列放不进预算，退回切开。
- **all 在 8064 token 上从 2139.8 ms 降到 905.8 ms**，但仍比 off 多 88%。
- **命中耗时与同一进程里 64 token 新 prompt 的 prefill 耗时相当**（`short_prefill_ms`，同一行同一进程的两个值相差不到 2 ms），也就是命中之后只剩一步 prefill 的固定开销。
  - 这个固定开销在不同进程之间会变：16 个进程中，15 个在 92.5–96.2 ms，v2 的 all / 8000 那个进程为 104.7 ms，同一进程里的命中也相应慢了约 10 ms。
  - 原因没有查明。FLA 的 autotune 在每个进程里各自进行（triton 3.2.0 不把 autotune 结果存盘），这是一个没有验证过的可能。

**conv 状态**直接从这一批的输入中取边界前 3 行，在 conv 状态被更新之前拷贝出来。如果这一段 prefill 的起点距离边界不到 3 个 token，缺的那几行从序列原来的 conv 状态里补。

### 3.5 decode 中保存快照，以及恢复

- **decode 中保存**：为了支持 MTP 回滚，验证 kernel 本来就会把"处理完第 j 个 token 后的状态"写进第 j 个槽位。所以 decode 越过边界时，只需把对应槽位拷贝进快照池（conv 池和 SSM 池各一次带索引的拷贝，与层数无关）。登记快照前会检查边界上的 token 是否被接受：草稿被拒绝时，这份快照直接释放。
- **恢复**：在 forward 之前把快照拷贝进序列当前的状态槽。实测 0.33–0.36 ms（`snapshot_restore_copy_ms`，见第 3.4 节表格对应的日志）。

### 3.6 生命周期与淘汰

| 阶段 | 做什么 |
|---|---|
| 调度时 reserve | 预留一个快照 id |
| forward 时 | 写入快照内容 |
| 这一步结束后 | 边界已经算到的，register，从此可以被查到；没算到的 release（例如 MTP 草稿被拒绝，边界前的 token 没被接受） |
| 序列被抢占或结束时 | 还没有 register 的预留一并 release |
| 命中时 pin，这一步结束后 unpin | 被 pin 的快照不会被淘汰 |
| 池满时 | 淘汰 LRU 中第一个没被 pin 的快照，并且立刻注销，保证它在被覆盖之前已经查不到 |
| KV block 被重新分配时 | 以它结尾的快照同时作废；如果这时还被 pin 着，等 unpin 时再回收 |

### 3.7 与其他特性的兼容

- **chunked prefill**：只在请求被接纳时匹配一次。每个 chunk 只保存落在本 chunk 范围 (start, end] 内的边界。端到端测试中，长 prompt 分 chunk prefill 之后再被命中的情况，只在 all 下出现（L2 命中 2560）。
- **抢占**：被抢占的序列释放预留的快照、解除 pin。重新接纳时重新匹配，可以再次命中。端到端测试中观察到：P2 先在 1024 命中，被抢占后在 1280 命中（`benchmarks/results/correctness/test_prefix_cache_e2e*.log` 的 branch 行）。
- **CUDA Graph**：decode 走 CUDA Graph，保存快照的拷贝在图执行之后进行；prefill 本来就是 eager 模式。
- **MTP**：
  - 快照只覆盖 GDN 层。MTP 层的 KV 和其他全注意力层一样，按 block 共享。
  - 有一个已知的近似：MTP 层在位置 H−1 的 KV 是用原请求在位置 H 的 token 算出的。新请求在 H 处的 token 可能不同，这时这一项是旧值。按设计这只影响草稿质量（接受率），不影响输出，因为每个草稿都由目标模型验证。实测的接受率见第 4.3 节。

## 4. 实验

### 4.1 正确性

单元测试（GPU 4，提交 590121f；日志在 `benchmarks/results/correctness/`，开头是命令和提交，末尾是退出码）：

| 测试 | 覆盖内容 | 结果 |
|---|---|---|
| `tests/test_snapshot_pool.py` | 各策略的命中长度；快照 id 的生命周期；KV block 被回收时快照作废；MTP 下的 hash。另有 40 组随机压力测试，用一个假的 runner 核对每次恢复的内容是否恰好是 `token_ids[:num_cached]`，共检查 66 次恢复 | PASS |
| `tests/test_state_snapshots.py` | 窗口规划及其随机测试（2000 个 batch，只检查计划）；快照与恢复逐位相同（窗口 8192 只用附加序列，2048 / 1024 混合，512 只切开）；conv 回退路径；序列在 batch 中的位置不影响结果 | PASS |
| `tests/test_linear_attention.py` | 已有的 GDN 层对比测试 | PASS |

端到端测试（`tests/test_prefix_cache_e2e.py`）：Qwen3.5-4B，argmax 采样，每种配置单独起一个进程，与对应的不开 cache 配置（`off`、`off_mtp`、`off_eager`）逐 token 对比。引擎参数是测试自己的：`max_model_len=4096`、`max_num_seqs=8`、`max_num_batched_tokens=2048`、`gpu_memory_utilization=0.5`、`num_state_snapshots=16`。

- 场景：
  - 未命中、部分命中、完全命中；
  - 多轮对话；
  - 命中后被抢占：刻意只留 8 个空闲 block，并让请求生成 600 个 token，不在 EOS 处停止；
  - 长 prompt；
  - 叠加 MTP（`branch_mtp`）；
  - 关闭 CUDA Graph（`branch_eager`）。
- 判定标准：
  - logits 最大误差 ≤ 1.0，比较范围是前 32 个生成位置，并且只比到第一次分叉为止；
  - 只允许在 top-2 logit 差 ≤ 0.5 的位置分叉，即 bf16 下的近似平局；
  - 每个请求在前 32 个位置之后只比较 token 是否相同（多数请求生成 48 个 token，T1 生成 300 个，抢占场景 P1–P4 生成 600 个）。

| 配置 | 19 个请求中完全一致的个数 | 最大 \|Δlogit\| | 命中情况 | 第一次分叉处的 top-2 差 |
|---|---|---|---|---|
| noise（off 再跑一次，batch 组成不同） | 18 | 0.69 | – | 0.125 |
| branch | 13 | 0.44 | C、D、F3 命中 768；T2 命中 1024；P2–P4 被抢占前后都命中 | 0.0 – 0.125 |
| last | 13 | 0.28 | F2、F3、T2、P1–P4 | 0.0 – 0.125 |
| all | 12 | 0.44 | 比 branch 多命中 B、F2、P1、L2（L2 命中 2560） | 0.0 – 0.125 |
| kv_only | 14 | 0.31 | 不跳过任何计算 | 0.0 – 0.125 |
| branch_mtp | 15 | 0.41 | 与 branch 相同，P2 被抢占两次后都再次命中 | 0.0 – 0.125 |
| branch_eager | 14 | 0.63 | 与 branch 相同 | 0.0 – 0.125 |

结论：
- **贪心输出并非完全一致**，但所有分叉都发生在 top-2 差 ≤ 0.125 的位置。
- **logits 误差不超过噪声基线。** 在比较范围内（见判定标准），开 cache 时的最大误差（≤ 0.63）小于只改变 batch 组成的 noise 组（0.69）。
- **不用快照也会分叉。** kv_only 不用快照，也有 5 个请求分叉，同样都在 top-2 差 ≤ 0.125 的位置。但分叉的请求和位置与 branch 不完全相同：两者都在 P3 的第 173 个位置分叉，其余各不相同。推测差异来自 bf16 在不同 batch 形状下的舍入（noise 组只改变 batch 组成，也有 1 个请求分叉），没有单独验证。
- **v1 和 v2 的结果完全相同。** 两次运行的 `_summary.json` 逐字节相同。它包含每个配置的输出 token、每次接纳时的命中长度、与对照组的比较结果（含 logits 误差）和快照池计数。

日志（都在 `benchmarks/results/correctness/`）：
- v2：`test_prefix_cache_e2e_v2.log` 和 `test_prefix_cache_e2e_v2_summary.json`，提交 590121f，GPU 4；
- v1：`test_prefix_cache_e2e.log` 和 `test_prefix_cache_e2e_summary.json`，6d8628a 加上当时未提交的测试改动，见日志头。

### 4.2 在线负载下的性能

条件与第 2 节相同。开 cache 的各组都设 `--num-state-snapshots 48`：last、branch、all 的快照占 2358 MiB，KV block 从 1177 个降到 882 个；kv_only 不分配快照，仍为 1177 个。
- baseline：加入本功能之前的引擎（c5af84b），2 次运行。
- kv_only：v1 引擎（6d8628a），1 次运行。kv_only 不存快照。v2 改动的是窗口规划，没有快照时它生成的窗口和 FLA 调用与 v1 相同（据代码，没有用 v2 重跑）。
- last、branch：v2 引擎（40a3ccf），各 2 次运行，表中是平均。
- all：v2 引擎，1 次运行。

**TTFT p50 / p99（ms）**

| 负载 | QPS | baseline | kv_only | last | branch | all | branch 的 p50 加速 |
|---|---|---|---|---|---|---|---|
| 共享 2000 token | 1.5 | 139 / 254 | 139 / 258 | 108 / 190 | 106 / 185 | 108 / 206 | 1.31× |
| 共享 2000 token | 4 | 148 / 1094 | 146 / 1079 | 108 / 222 | 106 / 209 | 109 / 251 | 1.39× |
| 共享 2048 token | 4 | 153 / 1196 | 154 / 1186 | 105 / 202 | 105 / 194 | 104 / 214 | 1.46× |
| 共享 4000 token | 0.75 | 246 / 455 | 245 / 454 | 107 / 295 | 113 / 265 | 107 / 249 | 2.18× |
| 共享 4000 token | 2 | 251 / 673 | 250 / 679 | 106 / 286 | 111 / 261 | 107 / 265 | 2.26× |
| 共享 8000 token | 0.4 | 491 / 903 | 491 / 902 | 111 / 542 | 110 / 541 | 109 / 426 | 4.46× |
| 共享 8000 token | 1 | 494 / 1372 | 494 / 1373 | 109 / 540 | 106 / 539 | 106 / 446 | 4.64× |
| 多轮对话 | 0.5 段/s | 184 / 415 | 179 / 339 | 101 / 182 | 101 / 188 | 103 / 284 | 1.82× |

**命中率与 prefill 计算量**

命中率 = 请求被接纳时命中的 token 数 / prompt 总 token 数。prefill 计算量 = 引擎实际 prefill 的 token 总数。

| 负载 | 命中率 last / branch / all | prefill 计算量：baseline → branch |
|---|---|---|
| 共享 2000 token | 83.5% / 83.5% / 84.8% | 133,117 → 22,013 |
| 共享 2048 token | 94.7% / 93.2% / 94.7% | 136,189 → 9,213 |
| 共享 4000 token | 91.2% / 91.2% / 92.6% | 261,117 → 23,037 |
| 共享 8000 token | 95.1% / 95.1% / 96.7% | 517,117 → 25,085 |
| 多轮对话 | 81.4% / 78.1% / 79.0% | 270,615 → 59,438 |

同一负载两种 QPS 的命中率相同。快照池的计数（`prefix_cache_stats` 中的保存、淘汰次数）把 37 个预热请求也算了进去，无法与测量部分分开，所以不列在表里。

**v1 → v2：last 和 all 的变化（p50 / p99，ms）**

| 负载 | QPS | last v1 | last v2 | all v1 | all v2 |
|---|---|---|---|---|---|
| 共享 2000 token | 1.5 | 122 / 311 | 108 / 190 | 111 / 410 | 108 / 206 |
| 共享 2000 token | 4 | 171 / 462 | 108 / 222 | 170 / 587 | 109 / 251 |
| 共享 2048 token | 4 | 103 / 198 | 105 / 202 | 109 / 368 | 104 / 214 |
| 共享 4000 token | 0.75 | 106 / 308 | 107 / 295 | 109 / 569 | 107 / 249 |
| 共享 4000 token | 2 | 108 / 303 | 106 / 286 | 104 / 566 | 107 / 265 |
| 共享 8000 token | 0.4 | 108 / 541 | 111 / 542 | 110 / 938 | 109 / 426 |
| 共享 8000 token | 1 | 105 / 539 | 109 / 540 | 109 / 979 | 106 / 446 |
| 多轮对话 | 0.5 段/s | 103 / 274 | 101 / 182 | 165 / 1230 | 103 / 284 |

v1 的 last 和 all 都是 1 次运行，v2 见上。日志：`benchmarks/results/{baseline,v1_cut,v2_extra}/<负载>_<策略>_r<n>.{json,log}`，每个日志开头有完整命令。所有组（包括 v1 的 off 和 branch、每次运行的范围）的完整表格在 `benchmarks/results/summary.md`。

读这几张表：
- **前缀越长，收益越大。** 命中后的 TTFT p50 都在 101–113 ms，比单个请求时 64 token prefill 的 93–96 ms 略高（差别的来源没有分解），所以加速比大致等于"不开 cache 的 TTFT ÷ 这个下限"：8000 token 时 4.46–4.64×，2000 token 时只有 1.31–1.39×。
- **p99 要分两种情况看。**
  - 2000 / 2048 token、QPS 4 时，baseline 的 p99 超过 1 秒，来自一次突发到达：两次运行的到达序列相同（同一个随机种子），14.37–17.41 s 这约 3 秒内到了 16 个请求；TTFT 超过 800 ms 的请求全部在这段时间到达（`per_request` 字段）。cache 去掉了大部分 prefill，这段排队随之消失：p99 从 1094 / 1196 ms 降到 209 / 194 ms。
  - 8000 token 时，p99 由最早到达的一两个请求决定：共享负载每次运行只有 64 个请求，p99 落在最慢的两个之间。branch 下前两个请求都要完整 prefill（QPS 1 的 r1 中分别为 539 和 540 ms，QPS 0.4 为 540 和 544 ms；第 2 个请求还要保存快照），所以 p99 停在 540 ms 左右。all 的 p99 更低（426 / 446 ms），是因为第 2 个请求就能命中，只剩第 1 个请求未命中；但这个请求本身更慢（QPS 1 的 r1 中 914 ms，QPS 0.4 为 917 ms：它的 prompt 有 8215 token，要在 32 个边界上保存快照）。这不说明 all 在稳态下更好。
- **kv_only 没有收益**，每组的 p50 都与 baseline 相差不到 3%。
- **v2 修好了 last 和 all 在压力下的退化。** v1 中两者在共享 2000 token、QPS 4 时比不开 cache 还慢（p50 171 / 170 ms，baseline 148 ms），因为每个要保存快照的请求都要切开 FLA 窗口。这也包括命中的请求：last 存在各自 prompt 的最后一个完整 block（其中含有各自的问题），all 存在命中点之后的每个边界。v2 中两者回到命中的下限（108 / 109 ms）。all 的 p99 在各负载上都降了下来，例如多轮对话从 1230 ms 降到 284 ms。last 的 p99 在共享 2000 / 4000 token 和多轮对话上下降，在共享 2048 / 8000 token 上持平（198 → 202、541 → 542、539 → 540 ms）。
- **branch 的 p50 在两个版本间基本不变。** branch 在 prefill 中只在第二次遇到某个前缀时保存一份快照，其余的保存大多是 decode 中的尾部快照（直接拷贝验证槽位，不经过 FLA），所以 v1 → v2 的改动对它影响很小。多轮对话的 p99 从 230 降到 188 ms（各 2 次，范围 219–240 对 174–202，不重叠）。
- **共享 4000 token 下 branch 的 v2 两次运行相差 10 ms 以上**（QPS 0.75：119.0 和 106.4 ms；QPS 2：116.1 和 105.8 ms）。两次运行的请求序列相同，可以逐个请求比较（`per_request`，去掉前两个未命中的请求）：QPS 0.75 时 62 个请求在 r1 中全部更慢，中位数慢 13.2 ms；QPS 2 时 62 个中有 56 个慢 5 ms 以上，中位数同样是 13.2 ms。所以是整条分布右移，不是个别请求。为了判断这是不是 v2 的问题，又把 v1 和 v2 交替各跑了 3 次（r3–r5，`benchmarks/results/rerun_shared4000/`）：

  | QPS | v1 的 p50，r1–r5（ms） | v2 的 p50，r1–r5（ms） |
  |---|---|---|
  | 0.75 | 104.8、104.6、107.0、103.0、106.1 | **119.0**、106.4、105.8、105.8、106.0 |
  | 2 | 103.4、103.3、105.3、103.1、106.9 | **116.1**、105.8、107.0、105.2、105.6 |

  v2 的 r1 之外，两个版本的 p50 都落在 103–107 ms，r1 的偏移没有再出现。v2 的两次 r1 分别在 20:12 和 20:17 开始，中间 20:15 的 QPS 0.75 r2 是正常的。原因没有查明。一个没有验证过的可能：FLA 的 autotune 在每个进程里各自计时并选择 kernel 配置，不同进程可能选到不同的配置。上面主表仍按 r1、r2 两次的平均列出，没有剔除。
- **输出吞吐。** 共享 system prompt 负载是开环的：请求按固定 QPS 到达，与服务快慢无关，所以各策略的输出吞吐基本相同（例如共享 2000 token、QPS 4 时 baseline 为 361 tok/s，branch 为 365 tok/s）。多轮对话中，每段对话的下一轮在上一轮结束后才发出，回答得越快，下一轮来得越早，所以吞吐从 367 升到 413 tok/s（branch，+12.4%）。引擎在满载时的最大吞吐未测。

### 4.3 MTP 叠加 prefix caching

Qwen3.5-4B 自带的 MTP 层作为草稿模型，每步 1 个草稿 token（`--k 1`），用 v2 引擎，每组只跑了 1 次：

| 负载 | 策略 | TTFT p50 / p99 | 命中率 | MTP 接受率 | 输出吞吐 | KV block |
|---|---|---|---|---|---|---|
| 共享 4000 token，QPS 2 | off | 261 / 702 ms | 0% | 0.810 | 190 tok/s | 925 |
| 共享 4000 token，QPS 2 | branch | 112 / 270 ms | 91.2% | 0.810 | 192 tok/s | 663 |
| 多轮对话，0.5 段/s | off | 189 / 394 ms | 0% | 0.827 | 421 tok/s | 925 |
| 多轮对话，0.5 段/s | branch | 105 / 181 ms | 78.1% | 0.823 | 470 tok/s | 663 |

接受率 = 被接受的草稿 token 数 / 解码步数（按序列计）。日志：`benchmarks/results/v2_extra_mtp/<负载>_<策略>_r1.{json,log}`。

- **TTFT 的变化与不开 MTP 时一致**：共享 4000 token 时 p50 从 261 降到 112 ms，多轮对话时从 189 降到 105 ms。
- **接受率基本不变。** 在共享 4000 token 上为 0.8104 对 0.8098，相差 0.0006；在多轮对话上为 0.8268 对 0.8226，相差 0.0042。第 3.7 节提到的近似（MTP 层在 H−1 处的 KV 可能是旧值）在这两个负载上看不出明显影响；只跑了 1 次，这点差别是否是噪声无法判断。
- 开启 MTP 后，每个 KV block 多出 MTP 层的 1 MiB，所以同样 48 份快照之后 KV block 只剩 663 个（不开快照时 925 个）。
- MTP 下分别有 4 次（共享）和 7 次（多轮）预留的快照没有登记就被释放（`prefix_cache_stats.unused`）；不开 MTP 的各组都是 0。按代码，释放发生在两种情况：边界前的草稿 token 被拒绝，这一步没有走到边界（`BlockManager.finish_step`）；或者序列在保存之前被释放（`BlockManager.deallocate`）。这两种情况没有分开计数。

### 4.4 哪些设计没有收益

按实测结果，以下设计没有带来收益，或者收益小于代价：

1. **只复用 KV（kv_only）**：所有负载的 p50 都与 baseline 相差不到 3%（第 4.2 节第一张表）。原因见第 1 节：GDN 状态需要整个前缀经过整个网络，跳不过任何计算。
2. **在边界处切开 FLA 窗口来取快照（v1）**：在压力下比不开 cache 还慢。共享 2000 token、QPS 4 时，last 和 all 的 p50 分别为 171 和 170 ms，baseline 为 148 ms。单个请求的测量中，all 让 2064 token 的未命中 prefill 从 134.9 ms 变成 574.6 ms。每切一刀，每层就多一次 FLA 调用；按微基准估算，这些调用的 host 开销约占多出时间的七成，其余没有分解（第 3.4 节）。已由 v2 取代。
3. **每个边界都存（all）**，即使用 v2：
   - 未命中仍然更贵：2064 token 为 222.7 ms（off 134.6 ms），8064 token 为 905.8 ms（off 481.1 ms）。
   - 保存次数多：共享 8000 token 时一次运行保存 594 份、淘汰 546 份（池子 48 份）；branch 在同样条件下保存 16 份、不淘汰。但这些计数包含 37 个预热请求，all 在预热的随机 prompt 上也要在每个边界保存，测量部分自身的保存和淘汰次数没有单独统计，所以测量期间池子替换得有多频繁未测。
   - 在线负载下 p50 与 branch 相差不超过 6 ms，p99 有高有低（第 4.2 节已说明 8000 token 时 p99 更低的原因）。
   - 命中率只比 branch 高 1.35–1.53 个百分点（共享负载）。
   - 结论：不作为默认。
4. **last 与 branch**：
   - 在 v2 中两者的 p50 在各负载上相差不超过 6 ms。
   - 前缀对齐或多轮对话时，last 的命中率略高（共享 2048 token 时 94.7% 对 93.2%，多轮对话时 81.4% 对 78.1%），但 p50 没有因此变好。
   - 共享 2000 token（不对齐）时两者命中率相同（83.5%）：这个负载里 64 个请求中有 36 个的最后一个完整 block 边界（1792）落在共享部分之内，别的请求可以用上。
   - last 每个未命中的请求都要付保存的代价（2064 token 多 13.4 ms；8064 token 因为回退切开窗口，多 42.9 ms），存的快照也更多（共享 2000 token 时保存 102 份、淘汰 54 份，branch 保存 41 份、不淘汰；都含预热请求）。
   - 所以默认用 branch。
5. **短前缀的收益有限**：共享 2000 token 时，命中率 83.5%，p50 只快 1.31–1.39×。命中后仍要付一步 prefill 的固定开销（单个请求时约 93–96 ms），而 2064 token 的完整 prefill 本身也只要约 135 ms。

## 5. 局限

- **命中后的下限是一步 prefill 的固定开销**：单个请求时约 93–96 ms，在线负载下命中的 TTFT p50 为 101–113 ms。这段时间的构成没有分解（未测）；已知 prefill 走 eager 模式，单独的微基准里 24 次 FLA 调用的 host 时间为 21.9 ms。把 prefill 做成 CUDA Graph 或合并 FLA 调用可能降低这个下限，但都未实现，也未测。
- **长 prompt 上保存快照会退回到切开窗口。** 一次 FLA 调用的 token 数不能超过 warmup 跑过的最大调用（本文配置下为 8192），所以 prompt 本身加上附加序列超过 8192 时，只能切开。8064 token 的 prompt 用 last 保存一份快照，未命中时多花 42.9 ms，与 v1（42.4 ms）相同（第 3.4 节）。
- **[H, M) 的 KV 不共享**（第 3.2 节）：KV 匹配得比最深快照更深时，中间这段 block 会重新算，并写进序列自己的 block。
- **branch 要第三次遇到同一前缀才命中。** 第二次遇到时才知道这个前缀被共享，并在这时保存快照。
- **只在请求被接纳时匹配一次。** 快照在一步结束后才登记，所以突发到达、在同一步里被接纳的多个请求互相用不上对方的快照，都要各自完整 prefill（未单独测量）。
- **last 能否被共享取决于问题长度。** 只有当最后一个完整 block 的边界落在共享部分之内时，last 存下的快照别人才用得上。本文的 MT-Bench 问题较短，所以 last 的命中率不比 branch 低；问题更长的负载未测。
- **快照很占显存。** 每份 49.125 MiB，约 6 个 KV block。benchmark 用的 48 份占 2358 MiB，KV block 从 1177 降到 882（开 MTP 时从 925 降到 663）。KV block 少了，并发高时更容易被抢占；对这一点没有单独测量。默认的自动大小（64 和"剩余显存的 25%"中较小的那个）没有在 benchmark 中使用，未测。
- **淘汰只用 LRU**，没有考虑一份快照能省掉多少计算（Marconi 的做法是按节省的 FLOP 加权）。未实现。
- **MTP 的近似**：MTP 层在 H−1 处的 KV 可能是旧值。实测两个负载的接受率分别相差 0.0006 和 0.0042（各 1 次运行）。按设计不影响输出。
- **只在单卡上实现和测试。** 张量并行下快照池需要按 rank 切分，未实现，也未测。
- **只测了 Qwen3.5-4B。** Qwen3.5-0.8B 等其他混合模型未测。纯注意力模型（Qwen3-0.6B）只在中间提交 f00fed2 上做过冒烟运行（`benchmarks/results/correctness/smoke_f00fed2.log`）。
- **贪心输出并非逐 token 一致**，分叉都在 top-2 差 ≤ 0.125 的近似平局处，推测来自 bf16 在不同 batch 形状下的舍入，见第 4.1 节。
- **测量本身的局限**：
  - 每组 1–2 次运行，共享负载每次 64 个请求、多轮对话每次 96 个，所以 p99 基本就是最慢的一两个请求；
  - 不同进程之间，一步 prefill 的固定开销可以差 10 ms 以上（第 3.4 节），在线负载中也出现过两次整条分布右移约 13 ms 的运行（v2 共享 4000 token 两种 QPS 的 r1，第 4.2 节），原因都没有查明；
  - 快照池的计数包含预热请求，无法单独给出测量部分的保存和淘汰次数；
  - baseline 和 v1 的在线负载运行，日志头里没有 `PYTHONPATH`，JSON 里也没有 `nanovllm_path`，只能靠当时的启动命令说明导入的是哪份引擎（见 `benchmarks/results/README.md`）。v2 的日志头同样没有 `PYTHONPATH`，靠 JSON 的 `nanovllm_path` 确认；
  - baseline 有 4 次运行（共享 8000 token QPS 0.4 的 r2、QPS 1 的 r1，多轮对话 r1 / r2），v1 的 off 有 13 次运行，开始时 GPU 4 上有其他任务（日志头 `gpu: 4` 行），与单请求开销第一次运行被弃用的原因相同。对照：v2 的 off 运行时 GPU 4 空闲，共享 8000 token 的 p50 为 489 / 494 ms（baseline 490–495 ms），多轮对话 p50 182 ms（baseline 180 / 188 ms），但多轮对话的 p99 为 395 ms（baseline 408 / 423 ms）。影响没有单独测量；
  - 只测了低于处理能力的 QPS，满载时的吞吐未测。

## 6. 复现

环境：
- conda 环境 `nano-vllm`：torch 2.6.0+cu124，fla-core 0.4.2，triton 3.2.0；
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`。GPU 0 是跑 benchmark 的 RTX 4090（PCI 1A:00.0），GPU 4 是跑测试的另一块 4090（PCI 3E:00.0）；
- 模型在 `~/huggingface/Qwen3.5-4B`；
- MT-Bench 问题文件不在仓库里，默认路径是 `/data/csd/nano-vllm-data/mt_bench_question.jsonl`，可以用 `--questions` 指定。它有 80 道题，格式与 README 里下载的 FastChat MT-Bench `question.jsonl` 相同。

计时类的测量（benchmark、`bench_fla_calls.py`）都受 host 开销影响，运行时机器上不要有别的 GPU 任务。

**测试**（在主仓库里运行；日志在 `benchmarks/results/correctness/`，日志头记录了提交和命令）：

```bash
cd /data/csd/nano-vllm
source /home/csd/anaconda3/etc/profile.d/conda.sh && conda activate nano-vllm
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
python tests/test_snapshot_pool.py
python tests/test_state_snapshots.py
python tests/test_linear_attention.py
python tests/test_rejection_sampler.py
NANOVLLM_PORT=2399 python tests/test_prefix_cache_e2e.py
```

**在线负载 benchmark**：每个引擎版本先冻结在一个 git worktree 里，再用 `PYTHONPATH` 指定加载哪一份。下面是 v2 实际运行的命令；baseline 和 v1 的命令见 `benchmarks/results/README.md`。

```bash
git worktree add /data/csd/nano-vllm-v2 40a3ccf
cd /data/csd/nano-vllm-v2
export PYTHONPATH=/data/csd/nano-vllm-v2 GPU=0 R=/data/csd/nano-vllm/benchmarks/results
REPEATS=2 EXTRA="--num-state-snapshots 48" benchmarks/run_prefix_cache_matrix.sh $R/v2_extra branch last
REPEATS=1 EXTRA="--num-state-snapshots 48" benchmarks/run_prefix_cache_matrix.sh $R/v2_extra off all
REPEATS=1 ONLY='shared4000_qps2|multiturn2000_qps0.5' EXTRA="--num-state-snapshots 48 --k 1" \
    benchmarks/run_prefix_cache_matrix.sh $R/v2_extra_mtp off branch
```

共享 4000 token 的追加运行（主仓库的脚本，`R0` 指定重复编号从几开始）：

```bash
cd /data/csd/nano-vllm
for k in 3 4 5; do
  for v in v1_cut:/data/csd/nano-vllm-after v2_extra:/data/csd/nano-vllm-v2; do
    PYTHONPATH=${v#*:} R0=$k REPEATS=1 ONLY='^shared4000_' EXTRA="--num-state-snapshots 48" GPU=0 \
      benchmarks/run_prefix_cache_matrix.sh benchmarks/results/rerun_shared4000/${v%%:*} branch
  done
done
```

**单请求开销、FLA 调用开销和调用计划**（主仓库的脚本；单请求开销按策略交替测 v1 和 v2）：

```bash
cd /data/csd/nano-vllm
for p in off last branch all; do
  PYTHONPATH=/data/csd/nano-vllm-after GPU=0 benchmarks/run_snapshot_overhead.sh benchmarks/results/overhead/v1_cut $p
  PYTHONPATH=/data/csd/nano-vllm-v2 GPU=0 benchmarks/run_snapshot_overhead.sh benchmarks/results/overhead/v2_extra $p
done
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_fla_calls.py
CUDA_VISIBLE_DEVICES=0 python benchmarks/show_fla_plan.py
```

**汇总表**：

```bash
python benchmarks/summarize_prefix_cache.py benchmarks/results/baseline benchmarks/results/v1_cut benchmarks/results/v2_extra
```
