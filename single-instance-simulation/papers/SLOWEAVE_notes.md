# SLOWEAVE 阅读笔记

## 核心问题
Prefill Chunk Size 影响 Prefill 推进速度，也会占用当前 iteration 的执行时间并干扰 Decode。固定 Chunk 无法直接根据当前 Decode 请求的剩余时间预算调整 Prefill 工作量。

## 核心思想
以 Decode deadline/slack 作为约束，估计执行一段 Prefill chunk 对当前 iteration 的时间影响，再选择满足 deadline 约束的最大 chunk。若预测成本对 chunk 单调，可通过二分搜索快速求解。

## 抽象形式
令当前时刻为 (t)，活跃 Decode 请求的下一 token deadline 为 (d_i)，则最紧迫 slack 可写为：

`B_t = max(0, min_i(d_i - t))`

基于预测成本 `T_hat(n,c,h)`，选择：

`c* = max c, subject to T_hat(n,c,h) × (1+rho) <= 1000 B_t`

其中时间单位需统一；本项目 slack 以秒计、成本模型以毫秒计。

## 本项目实现边界
当前 `deadline_adaptive` 仅是 SLOWEAVE-inspired simplified baseline：使用合成成本函数、固定 safety margin 和二分搜索。它未实现论文完整的 profiling predictor、残差反馈、安全余量自适应或所有边界处理，不能称为论文原算法复现。

## 后续真实化
1. 在 2×3090、TP=2 下测量 Decode-only、Prefill-only、Mixed iteration。
2. 对每个 mesh point 重复采样，保留 P50/P99。
3. 建立保守的 P99 cost predictor，并检查对 chunk/context/batch 的单调性。
4. 使用真实 deadline slack 做在线 chunk 选择。
5. 评估 deadline miss、TTFT、TPOT、Goodput 和 starvation。
