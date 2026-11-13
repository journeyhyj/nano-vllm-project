# nano-vllm：Qwen3.5 混合架构推理 + MTP 投机解码

本项目基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（一个约 1200 行 Python 的 vLLM 极简实现）。在保留原有功能（paged KV cache、prefix caching、chunked prefill、CUDA Graph、张量并行）的基础上，新增了下面五项：

| 功能 | 一句话说明 |
|---|---|
| **Qwen3.5 dense 文本推理** | 支持 Gated DeltaNet 线性注意力与 gated attention 按 3:1 交错的混合架构，引擎负责管理每条序列的递归状态 |
| **MTP 投机解码** | 用模型自带的 MTP 头起草 token，再由目标模型一次验证；拒绝采样保证输出分布与目标模型一致 |
| **小词表起草** | 起草时 lm_head 只计算 32K 个高频 token，这部分耗时降为原来的约 1/7，输出依然无损 |
| **FLA prefill** | 线性注意力的 prefill 改用 flash-linear-attention 的 chunk kernel，长 prompt 的 prefill 快 1.26–1.31× |
| **混合模型的 prefix caching** | 在 block 边界保存 Gated DeltaNet 的状态快照，命中时与 KV 一起恢复。共享 8000 token system prompt 时，TTFT p50 从 494 ms 降到 106 ms |

## 结果

模型为 Qwen3.5-4B（bf16）。所有数字都可以用 `benchmarks/` 下的脚本复现，跑法见下文[测试与复现](#测试与复现)一节。

### Decode：MT-Bench

测试条件：MT-Bench 的 80 道题（只取第一轮），贪心解码，自然结束，最多生成 256 个 token。吞吐只统计 decode 步，不受 prompt 长度影响。

| 配置 | RTX 4090，batch=1 | RTX 3090，batch=1 | RTX 4090，16 条并发 | 接受率（4090 / 3090） |
|---|---|---|---|---|
| 不开投机 | 95.7 tok/s | 76.6 tok/s | 1076 tok/s | – |
| MTP，完整词表（248K） | 137.6 tok/s（1.44×） | 118.2 tok/s（1.54×） | 1567 tok/s | 0.826 / 0.818 |
| **MTP + 32K 草稿词表** | **148.3 tok/s（1.55×）** | **127.6 tok/s（1.67×）** | **1663 tok/s** | 0.787 / 0.791 |

接受率因题型而异（MTP + 32K 词表，RTX 4090）：math 0.93、extraction 0.91、reasoning 0.85、coding 0.83、stem 0.78、humanities 0.72、writing 0.71、roleplay 0.69。

### Prefill：线性注意力换成 FLA

测试条件：RTX 4090，prompt 为随机 token，表中是单次 prefill 的耗时。

| prompt | 改动前（自写递推 kernel） | FLA chunk kernel | 加速 | 对照：Qwen3-4B |
|---|---|---|---|---|
| 8K × 1 | 633 ms | 484 ms | 1.31× | 520 ms |
| 1K × 8 | 473 ms | 462 ms | 1.02× | – |
| 32K × 1 | 2881 ms | 2294 ms | 1.26× | – |

说明：
- "改动前"那一列，是在同一个进程里切换新旧两种实现测得的。
- Qwen3.5 只有 1/4 的层是全注意力层，所以接入 FLA 后，长 prompt 的 prefill 比同尺寸的 Qwen3-4B 还快。
- 1K × 8 这种多条短 prompt 的场景，原来的 kernel 并行度已经够了，所以基本没有收益。

### Prefix caching：共享 system prompt 与多轮对话

测试条件：
- RTX 4090，在线到达，TTFT 从请求到达时开始计时；`max_num_seqs=16`，`max_num_batched_tokens=16384`。
- 共享负载：64 个请求，按泊松过程到达。每个请求都是同一段 N token 的共享前缀（chat 模板 + system prompt）加一道 MT-Bench 问题，最多生成 128 个 token。
- 多轮对话：16 段对话 × 6 轮，每轮在上一轮结束后立即发出，最多生成 256 个 token。
- "改动前"是加入本功能之前的引擎（c5af84b），当时混合模型不做 prefix caching。
- "branch" 是默认策略。benchmark 中把快照池设为 48 份（2358 MiB，`--num-state-snapshots 48`）；默认的自动大小没有测过。

| 负载 | QPS | 改动前 TTFT p50 / p99 | branch TTFT p50 / p99 | p50 加速 | 命中率 |
|---|---|---|---|---|---|
| 共享 2000 token | 4 | 148 / 1094 ms | 106 / 209 ms | 1.39× | 83.5% |
| 共享 4000 token | 2 | 251 / 673 ms | 111 / 261 ms | 2.26× | 91.2% |
| 共享 8000 token | 1 | 494 / 1372 ms | 106 / 539 ms | 4.64× | 95.1% |
| 多轮对话 | 0.5 段/s | 184 / 415 ms | 101 / 188 ms | 1.82× | 78.1% |

说明：
- 每格是 2 次运行的平均。
- 命中后的 TTFT p50 在 101–113 ms。单个请求时，64 token prompt 的一步 prefill 也要 93–96 ms，这是命中后的下限，它的构成没有分解。所以前缀越长，加速越明显。
- 8000 token 时的 p99 来自最早的两个请求：branch 要第三次遇到同一个前缀才命中，前两个请求都得完整 prefill。
- 只共享 KV、GDN 全部重算（kv_only）时，TTFT 与不开 cache 相同。
- 开启 MTP 时同样有效（各 1 次运行）：共享 4000 token、QPS 2 下，p50 从 261 ms 降到 112 ms，接受率都是 0.810。

全部策略、各次运行的范围、开销分解以及没有收益的设计，见 [docs/hybrid_prefix_cache.md](docs/hybrid_prefix_cache.md)。原始 JSON 和日志在 `benchmarks/results/`，汇总表是 `benchmarks/results/summary.md`。

## 设计要点

### 1. 混合架构的状态管理

Qwen3.5-4B 共 32 层，其中 24 层是 Gated DeltaNet，8 层是全注意力。
- **KV cache** 只为 8 层全注意力分配。
- **Gated DeltaNet 层**为每条序列维护一份大小固定的递归状态：conv 状态（最近 3 个输入）和 SSM 状态（fp32，形状为 32 头 × 128 × 128）。4B 模型每条序列约 49 MiB。

具体实现：
- **状态槽位池**：在 KV cache 之前分配，最多占用剩余显存的一半。`BlockManager` 在给序列分配 KV block 时同时分配一个槽位；序列结束或被抢占时归还。
- **prefix caching**：缓存住的 KV block 无法还原出递归状态，所以另外在 block 边界保存状态快照（见第 5 节）。
- **CUDA Graph**：`state_slots` 是图里的静态 buffer；为了凑齐 batch 而填充的条目，都指向一个备用槽位，不会改动任何真实序列的状态。
- **Kernel**：causal conv1d 和状态更新是自己写的 Triton kernel。delta rule 的递推在 prefill 时调用 FLA，在 decode 时用自己写的 kernel（原因见第 4 节）。

### 2. MTP 投机解码（每步起草 1 个 token）

- **验证**：每条序列把 `[上一个 token, 草稿 token]` 一起送进目标模型做一次前向，然后拒绝采样：
  - 以 min(1, p/q) 的概率接受草稿；
  - 被拒绝时，从归一化后的 (p−q)⁺ 中重新采样；
  - 草稿被接受时，再额外采样一个 bonus token。

  这样输出分布严格等于目标模型的分布（Leviathan et al., 2023）。
- **起草**：MTP 头读取"最后一个被接受位置"的隐状态，以及下一个 token 的 embedding，预测再下一个 token。
- **不拷贝的回滚**：
  - 每条序列拥有一组 2 个状态槽位；
  - 验证时，kernel 把"处理完第 j 个 token 之后的状态"写进第 j 个槽位；
  - 调度器只需把被接受位置的槽位轮换到组尾，作为下一步的当前状态，不拷贝任何数据。
- **全程 CUDA Graph**：验证和起草两次前向都被捕获成 CUDA Graph；起草图直接读取验证图的输出 buffer。
- **prefill 阶段**：把这一块 token 整体右移一位，让 MTP 也跑一遍，用来填满 MTP 自己那一层的 KV cache，同时得到第一个草稿。

### 3. 小词表起草

分阶段计时显示：
- 起草一步要读取的权重约为目标模型前向的 18%，其中 84% 来自 24.8 万词表的 lm_head；
- 在 4090 上 batch=1 时，起草用的 lm_head 耗时 1.31 ms，是 MTP 层本身（0.41 ms）的 3 倍多。

做法：
- **起草词表**：用 `tools/build_draft_vocab.py` 生成，即让目标模型回答 Alpaca 中英文指令，统计它生成的 token 的频率。3000 条 prompt 共生成 73.7 万个 token，其中不同的只有 48,449 个；取频率最高的 32K 个。
- **计算**：加载权重后复制出这 32K 行 lm_head，起草时只算这部分 logits，lm_head 耗时从 1.31 ms 降到 0.18 ms。
- **依然无损**：草稿分布在词表外为 0。拒绝采样之前用 `index_copy_` 把它补回完整词表；拒绝采样对任意草稿分布都成立，所以输出分布不变，代价只是接受率略降。

两个踩过的坑：
- **必须按词频选**：如果直接取 token id 最小的 32K 个，中文接受率会从 0.79 掉到 0.32，因为 Qwen3.5 的中文 token 大多在 id 靠后的位置。
- **显存碎片**：Qwen3.5-4B 的 lm_head 和 embedding 共享权重，被替换掉的那份 1.18 GiB lm_head 权重会留在 PyTorch 的缓存里。复制出的 168 MB 子矩阵如果切在这个缓存块里，约 1 GiB 显存就再也释放不掉。解决办法是复制前先调用一次 `empty_cache()`。

### 4. FLA prefill

- **问题**：原来自己写的递推 kernel 按 token 串行计算。8K 单条 prompt 时，GPU 上只有 128 个 program，每个都要串行走 8192 步，Gated DeltaNet 占了 prefill 时间的 35%。
- **做法**：换成 FLA 的 chunk 并行算法（WY 表示），大部分计算变成矩阵乘，可以用上 tensor core。接入时处理了三件事：
  - 调用前从槽位池取出这批序列的状态，调用后写回；
  - FLA 要求 q、k 的头数与 v 相同，所以先对 q、k 做 `repeat_interleave`；
  - `cu_seqlens` 每个 batch 只构造一次，并同时提供一份 CPU 副本，避免 24 层每层都触发一次 GPU→CPU 同步。
- **显存**：FLA 的中间张量每个 token 约占 64 KB。不加处理的话，预热时一次处理 16K 个 token，峰值多出 1.04 GiB，开启 MTP 时的最大并发会从 19 降到 14。现在把一次 prefill 切成最多 8192 个 token 的窗口，序列的状态在窗口之间接力，结果严格等价；额外显存降到 0.44 GiB，最大并发为 17。
- **decode 仍用自己的 kernel**：FLA 公开的 decode 接口不能按槽位原地更新状态，也不输出每个 token 之后的中间状态，没法支持投机解码的回滚。vLLM 也是自己维护了一份改过的 FLA。

### 5. 混合模型的 prefix caching

- **为什么只复用 KV 不够**：第 l 层 GDN 在前缀末尾的状态，取决于前缀每个 token 在前面所有层的输出。所以要得到它，只能把整个前缀在整个网络里重算一遍。实测只共享 KV 时 TTFT 不变。
- **快照**：在 256 token 的 block 边界处，保存 24 层 GDN 的 conv 状态和 SSM 状态，每份 49.125 MiB，约等于 6 个 KV block。
  - 快照放在一个固定大小的池里，从 KV cache 的显存预算中扣除；
  - 以边界前那个 block 的链式 hash 为 key，前缀匹配时顺带查找；
  - 命中最深的那个快照后，前面的 KV block 直接共享，快照拷进序列自己的状态槽，只 prefill 剩下的部分。
- **什么时候存**（`prefix_cache_policy`）：
  - `branch`（默认）：只在某个前缀第二次被走到时，在 KV 匹配的末端存一份，从第三次开始命中；
  - `last`：每个请求存在最后一个完整 block 的边界；
  - `all`：每个边界都存。
  - 三种策略都会在 decode 越过边界时存一份快照，供多轮对话的下一轮使用。
  - 池满时淘汰 LRU 中没有被 pin 的快照。
- **prefill 中怎样拿到边界处的状态**：
  - FLA 只输出每条序列的最终状态。最直接的做法是在边界处切开窗口，但每切一刀，每层就多一次 FLA 调用，而单次调用的 host 时间约 0.9 ms，与 token 数无关。实测 all 策略（2064 token 的 prompt 存 8 份快照）下，未命中的 prefill 从 134.9 ms 变成 574.6 ms。
  - 现在的做法是把"序列起点 → 边界"这段 token 作为一条额外的序列，追加进同一次 FLA 调用，它的最终状态就是快照。这样调用次数不变，结果与切开逐位相同；同样的 prefill 降到 222.7 ms（其中还剩一次回退切开）。
  - 放不下时（总 token 数超过 warmup 跑过的最大调用）退回切开。
- **decode 中保存**：MTP 验证 kernel 本来就会写出每个 token 之后的状态，越过边界时直接拷贝；草稿被拒绝时这份快照作废。
- **兼容性**：支持 chunked prefill、抢占（重新接纳后可以再次命中）、CUDA Graph 和 MTP。端到端测试中，贪心输出只在 bf16 近似平局处分叉（top-2 logit 差 ≤ 0.125）；在前 32 个生成位置内，最大 logit 误差 0.63，小于只改变 batch 组成时的 0.69。

## 使用

```bash
pip install -e .    # 依赖 flash-attn 和 fla-core 0.4.x；fla-core 0.5 起要求 torch>=2.7、triton>=3.3
```

```python
from nanovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3.5-4B",
          num_speculative_tokens=1,                                # 开启 MTP 投机解码
          draft_vocab="assets/qwen3_5_draft_vocab_32k.json")       # 可选：小词表起草
outputs = llm.generate(["Hello, Nano-vLLM."], SamplingParams(temperature=0.7, max_tokens=256))
```

混合模型的 prefix caching 默认开启，有两个参数：
- `prefix_cache_policy`：`"branch"`（默认）、`"last"`、`"all"`；另有两个对照组，`"kv_only"`（只共享 KV）和 `"off"`。
- `num_state_snapshots`：快照池的大小。默认 −1，表示取 64 和"剩余显存的 25% 能放下的份数"中较小的一个。

例如 `LLM(path, prefix_cache_policy="branch", num_state_snapshots=48)`。

`assets/qwen3_5_draft_vocab_32k.json` 适用于整个 Qwen3.5 系列（它们共用同一个 tokenizer）。如果要为其他模型或数据重新生成：

```bash
python tools/build_draft_vocab.py ~/huggingface/Qwen3.5-4B out.json alpaca_en.json alpaca_zh.json --size 32768
```

## 测试与复现

```bash
# 单测（需要 GPU）
python tests/test_linear_attention.py   # 与 transformers 的 PyTorch 参考实现对比：变长 prefill（含 48 token 小窗口）、续算、decode、验证 + 回滚
python tests/test_rejection_sampler.py  # 统计检验：完整词表和小词表起草下，输出都服从目标分布

# MT-Bench 题目
curl -o question.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl

# decode 吞吐与接受率
python benchmarks/bench_spec.py ~/huggingface/Qwen3.5-4B question.jsonl --k 0
python benchmarks/bench_spec.py ~/huggingface/Qwen3.5-4B question.jsonl --k 1
python benchmarks/bench_spec.py ~/huggingface/Qwen3.5-4B question.jsonl --k 1 --draft-vocab assets/qwen3_5_draft_vocab_32k.json

# prefill 耗时
python benchmarks/bench_prefill.py ~/huggingface/Qwen3.5-4B --len 8192 --bs 1

# prefix caching：单测与端到端测试（与 off 逐 token 对比，覆盖命中、命中后被抢占、MTP、关闭 CUDA Graph）
python tests/test_snapshot_pool.py
python tests/test_state_snapshots.py
python tests/test_prefix_cache_e2e.py

# prefix caching：在线负载矩阵、单请求开销、FLA 调用开销，以及汇总
# （结果目录里用 git worktree 冻结了每个引擎版本，并用 PYTHONPATH 指定，见 benchmarks/results/README.md）
EXTRA="--num-state-snapshots 48" benchmarks/run_prefix_cache_matrix.sh benchmarks/results/mine branch off
benchmarks/run_snapshot_overhead.sh benchmarks/results/mine_overhead off last branch all
python benchmarks/bench_fla_calls.py
python benchmarks/summarize_prefix_cache.py benchmarks/results/baseline benchmarks/results/mine
```

`bench_prefix_cache.py` 默认从 `/data/csd/nano-vllm-data/mt_bench_question.jsonl` 读 MT-Bench 问题，可以用 `--questions question.jsonl` 换成上面下载的文件。

开发过程中还做过以下检查：
- **与 HF transformers 逐 token 对比**：贪心输出只在 bf16 近似平局处分叉（这些位置前两名的 logit 差 ≤ 0.25）；
- **压力测试**：在投机解码叠加强制抢占的条件下，所有请求都正确完成，状态槽位全部归还；
- **回归**：Qwen3-0.6B 的 C-Eval（908 题）中，有 899 题答案与改动前相同；作为对照，同一版本跑两次也只有 902 题相同。

## 局限

- MTP 每步只起草 1 个 token，只支持单卡，只支持温度采样（不支持 top-p / top-k）。
- Qwen3.5 还不支持张量并行。
- 混合模型的 prefix caching：
  - 命中后的 TTFT 不会低于一步 prefill 的固定开销（单个请求时 93–96 ms）；
  - `branch` 要第三次遇到同一前缀才命中；
  - 同一步被接纳的请求互相用不上对方的快照；
  - 长 prompt 上保存快照会退回到切开 FLA 窗口；
  - 只在 Qwen3.5-4B、单卡上测过。
  - 详见 [docs/hybrid_prefix_cache.md](docs/hybrid_prefix_cache.md) 第 5 节。
- SSM 状态用 fp32 存储，开启 MTP 后每条序列要占两份。所以在 24 GB 显卡上，开启 MTP 的最大并发约为不开时的一半。

## 致谢

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)：本项目的基础
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)：提供 Gated DeltaNet 的 chunk kernel
- [vLLM](https://github.com/vllm-project/vllm)：Qwen3.5 / MTP 实现的参考
