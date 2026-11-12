# 原始单实例推理工作方案摘录

本文件根据用户提供的两页《单实例推理仿真 modeling》方案整理，保留原方案的术语与研究拆分。

## 1. 核心指标

- Request-level：TTFT、TBT。
- System-level：吞吐量、Goodput。

## 2. 调度层次

### 2.1 Request-level Schedule：优先处理哪个 Request

状态参量：
- real-time TTFT
- real-time TPOT
- 输入 Prompt 长度
- 剩余 Decode 长度
- 已生成 Token 数量
- 已完成 Prefill

原方案提出：
- 当前实时 TTFT 和 TPOT 哪个更失调，就优先优化哪个（参考 SLOA）。
- 当前推理与下一次 iteration 的参数更新解耦，避免 online 计算、计算和通信形成额外 overhead（参考 DRLServe）。
- Waiting Queue 中每个 Request 计算 Priority 分数，按优先级决定调度顺序（参考 SLOA、RAD&SLAI）。
- 优化 TTFT 时约束 TPOT SLO；优化 TPOT 时约束 TTFT SLO（参考 SLOA）。

### 2.2 System-level Schedule：最大化 GPU 资源利用与吞吐

状态参量：
- KV Cache 显存占用
- running / waiting queue
- 输出长度分布

原方案提出：
- 结合 GPU 利用率动态调整 Chunk Size 和 Token Budget（参考 RAD&SLAI）。
- GPU 利用率低于阈值 α 时增大 Prefill Chunk Size，否则减小。
- 已按此前 Chunk Size 分割的 Request 不回滚，在下一 iteration 更新参数（参考 DRLServe）。
- 系统目标为最大化 Goodput（参考 RAD&SLAI）。

## 3. 运行平台与模型

- 运行平台：2×3090，TP=2。
- 量化模型：Qwen3-32B。

## 4. Profiling 方案

分别测试两张 3090 环境下的 Decode-only、Prefill-only 和 Mixed 模式算子运行情况。每个 mesh point 记录 P99、P50 用时，profiling 至少三次并进行一致性检验；后续插值、拟合或随机森林建模使用 P99 数据（参考 SLOWEAVE）。

## 5. 建模计算

原方案将建模计算参考指向 LLMServingSim 2.0 与 SLOA。
