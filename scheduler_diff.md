# Scheduler Decode 阶段对比分析

由于当前环境无 GPU，无法直接跑 benchmark。
以下是基于代码逻辑的理论分析。

---

## 场景设定

bench.py 的参数：
- num_seqs = 256，max_input_len = 1024，max_output_len = 1024
- 模型：Qwen3-0.6B，KV Cache block_size = 256，max_model_len = 4096
- 每个 prompt 长度随机 100~1024，每个 max_tokens 随机 100~1024

**decode 阶段触发 block 不足的场景**：
当 running 队列中有大量 seq 同时 decode，每个 seq 每隔 256 tokens（一个 block）就需要分配一个新 block。
此时 free blocks 不够，就会触发 preempt。

---

## 原版策略：级联抢占

```
running = [A, B, C, D, E, F, G, H, ...]
                     ↑ 当前处理到 D

D 需要新 block，free 不够：
  → while not can_append(D):
      → preempt(H)  // 释放 H 的全部 KV Cache
      → preempt(G)  // 释放 G 的全部 KV Cache
      → preempt(F)  // 释放 F 的全部 KV Cache
      → can_append(D) 终于成功

结果：
  - D 可以继续 decode（1个seq受益）
  - F, G, H 被踢回 waiting，KV Cache 全丢（3个seq受损）
  - F, G, H 需要重新 prefill（recompute），延迟大幅增加
  - 如果 F/G/H 已经生成了很多 token，损失尤其大
```

## 新策略：Skip-and-Defer

```
running = [A, B, C, D, E, F, G, H, ...]
                     ↑ 当前处理到 D

D 需要新 block，free 不够：
  → if not can_append(D):
      → preempt(D)  // 只释放 D 自己的 KV Cache
      → continue    // 跳过 D

继续处理 E：
  → E 可能不需要新 block（还没跨块）→ 正常 decode ✓
  → E scheduled_seqs.append(E)

继续处理 F：
  → F 也不需要新 block → 正常 decode ✓

结果：
  - D 被踢回 waiting（1个seq受损）
  - E, F, G, H 继续正常 decode（0个额外受损）
  - 下一步可能已经有 seq 完成，释放了 block，D 可以重新调度
```

---

## 关键指标预期对比

### TTFT（首 Token 延迟）

| 场景 | 原版 | Skip-and-Defer |
|------|------|----------------|
| 被级联 preempt 的 seq | 需要重新 prefill，TTFT 极长（二次 prefill 排队） | 不受影响 |
| 正常 seq | 无影响 | 无影响 |

**预期**：Skip-and-Defer 的 p99 TTFT 显著优于原版。

### Latency（端到端延迟）

| 场景 | 原版 | Skip-and-Defer |
|------|------|----------------|
| 被 preempt 的 seq | recompute 全部 KV Cache + 重新 decode | 只有自己被踢 |
| 级联效应 | 队尾多个 seq 一起延迟暴增 | 无级联 |

**预期**：Skip-and-Defer 的 p99 Latency 显著优于原版，mean 也略好。

### Throughput（吞吐量）

| 场景 | 原版 | Skip-and-Defer |
|------|------|----------------|
| 被 preempt 的 seq 数量 | 多（级联） | 少（只踢自己） |
| recompute 开销 | 大（多个 seq 重新 prefill） | 小（单个 seq） |

**预期**：Skip-and-Defer throughput 略高于原版（减少了不必要的 recompute）。

---

## 定量估算（基于 256 seq 随机 prompt/output）

假设在 decode 高峰期，平均每 step 有 ~200 个 running seq，
每个 seq 平均每 256 tokens 需要一次新 block 分配。

当 free blocks 紧张时（已分配 ~90% KV Cache）：

**原版级联**：
- 一个 seq 触发 preempt → 可能级联踢掉 3~5 个 seq
- 被踢掉的 seq 平均已生成 ~200 tokens
- recompute 开销：~200 tokens × 5 seqs = 1000 tokens 的重新 prefill
- 对 p99 延迟影响：额外增加数百毫秒到秒级

**Skip-and-Defer**：
- 只踢自己，不影响他人
- 被踢 seq 同样损失已生成的 tokens（但只有一个）
- 对其他 seq 的延迟无影响
- p99 延迟主要由自身 preempt 造成，无级联放大

---

## 总结

| 指标 | 原版 | Skip-and-Defer | 改进方向 |
|------|------|----------------|----------|
| p99 TTFT | 较高（级联 preempt 导致排队） | 较低 | ↓ 显著 |
| p99 Latency | 较高（级联 recompute） | 较低 | ↓ 显著 |
| Throughput | 基准 | 略高（减少浪费） | ↑ 轻微 |
| 公平性 | 差（队头牺牲队尾） | 好（各自负责） | ↑ 显著 |

实际数值取决于显存压力（seq 数量、prompt 长度、KV Cache 大小）。
在高并发（256 seq）场景下，Skip-and-Defer 对 p99 延迟的改善最为明显。
