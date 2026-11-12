# DRLServe 阅读笔记

## 研究问题
单实例 LLM serving 中，Prefill 与 Decode 共享 GPU 时存在资源竞争。固定 Prefill Chunk Size 难以同时适应请求长度、Decode batch 和实时负载变化：Chunk 太大可能阻塞 Decode、恶化 TPOT；Chunk 太小则增加调度开销、降低 Prefill 效率。

## 核心思路
DRLServe 将动态 Prefill Chunk 决策建模为强化学习问题。调度器观察系统状态，根据策略输出 Chunk/资源决策，并通过服务质量与系统效率相关的奖励信号训练策略。论文的关键研究价值在于：把 chunking 从静态超参数转变为随负载变化的在线决策。

## 阅读时重点关注
- State：请求队列、Prefill/Decode 状态、资源状态及历史执行信息。
- Action：Prefill chunk 或相关调度控制量。
- Reward：吞吐与延迟/SLO 之间的权衡。
- Training：策略优化、探索稳定性、训练/推理开销。
- Evaluation：与固定 chunk、规则策略的比较，以及不同负载下的泛化能力。

## 与本项目的关系
本项目的 `gpu_threshold` 和 `deadline_adaptive` 是可解释的规则基线；后续可将 DRL policy 作为新的 System-level controller，与 FCFS/Priority Request Selection 组合。

## 复现边界
当前仿真代码没有实现论文完整的 RL state/action/reward、训练过程或论文算法复现。此笔记用于定位阅读与实验接口，不应将本项目的规则策略称为 DRLServe reproduction。

## 建议消融
1. 固定 Chunk vs rule-based adaptive vs DRL adaptive。
2. 去掉 SLO reward 项，观察吞吐/违约变化。
3. 去掉 GPU/KV state，测试状态信息贡献。
4. 测量策略推理开销与策略更新开销。
