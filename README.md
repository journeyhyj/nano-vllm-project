# nano-vllm：Qwen3.5 混合架构推理 + MTP 投机解码

本项目基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（一个约 1200 行 Python 的 vLLM 极简实现）。在保留原有功能（paged KV cache、prefix caching、chunked prefill、CUDA Graph、张量并行）的基础上，新增了下面四项：

| 功能 | 一句话说明 |
|---|---|
| **Qwen3.5 dense 文本推理** | 支持 Gated DeltaNet 线性注意力与 gated attention 按 3:1 交错的混合架构，引擎负责管理每条序列的递归状态 |
| **MTP 投机解码** | 用模型自带的 MTP 头起草 token，再由目标模型一次验证；拒绝采样保证输出分布与目标模型一致 |
| **小词表起草** | 起草时 lm_head 只计算 32K 个高频 token，这部分耗时降为原来的约 1/7，输出依然无损 |
| **FLA prefill** | 线性注意力的 prefill 改用 flash-linear-attention 的 chunk kernel，长 prompt 的 prefill 快 1.26–1.31× |

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

## 设计要点

### 1. 混合架构的状态管理

Qwen3.5-4B 共 32 层，其中 24 层是 Gated DeltaNet，8 层是全注意力。
- **KV cache** 只为 8 层全注意力分配。
- **Gated DeltaNet 层**为每条序列维护一份大小固定的递归状态：conv 状态（最近 3 个输入）和 SSM 状态（fp32，形状为 32 头 × 128 × 128）。4B 模型每条序列约 49 MiB。

具体实现：
- **状态槽位池**：在 KV cache 之前分配，最多占用剩余显存的一半。`BlockManager` 在给序列分配 KV block 时同时分配一个槽位；序列结束或被抢占时归还。
- **prefix caching 对混合模型关闭**：缓存住的 KV block 无法还原出递归状态。
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
```

开发过程中还做过以下检查：
- **与 HF transformers 逐 token 对比**：贪心输出只在 bf16 近似平局处分叉（这些位置前两名的 logit 差 ≤ 0.25）；
- **压力测试**：在投机解码叠加强制抢占的条件下，所有请求都正确完成，状态槽位全部归还；
- **回归**：Qwen3-0.6B 的 C-Eval（908 题）中，有 899 题答案与改动前相同；作为对照，同一版本跑两次也只有 902 题相同。

## 局限

- MTP 每步只起草 1 个 token，只支持单卡，只支持温度采样（不支持 top-p / top-k）。
- Qwen3.5 还不支持张量并行；混合模型不支持 prefix caching。
- SSM 状态用 fp32 存储，开启 MTP 后每条序列要占两份。所以在 24 GB 显卡上，开启 MTP 的最大并发约为不开时的一半。

## 致谢

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)：本项目的基础
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)：提供 Gated DeltaNet 的 chunk kernel
- [vLLM](https://github.com/vllm-project/vllm)：Qwen3.5 / MTP 实现的参考
