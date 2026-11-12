# 单实例 LLM 推理建模与仿真

本目录包含单实例 LLM Serving 离散事件仿真基线、合成负载结果、论文阅读笔记与复现实验说明。

## 重要声明

- 仿真目标配置：2×RTX 3090、TP=2、Qwen3-32B（作为方案目标配置）。
- 当前 `SyntheticCostModel` 是解析式合成成本模型，**不是 GPU 实测或经真实 profiling 校准的模型**。
- CSV 和曲线均为模型生成的合成数据，不能作为真实硬件性能结论或论文实验数据。
- 要进行真实预测，需要替换成本模型为目标硬件、模型、并行配置下的 P99 profiling lookup table。

## 目录

- `single_instance_sim.py`：离散事件仿真主程序。
- `requirements.txt`：Python 依赖。
- `results/`：实验汇总、运行级 CSV、成本面 CSV 与关键结果。
- `papers/`：DRLServe、RAD&SLAI、SLOWEAVE 阅读与复核笔记。
- `reference/`：原始方案附件（若已获授权并适合公开，可自行补入）。
- `UPSTREAM.md`：LLMServingSim 上游项目说明。

## 环境与运行

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python single_instance_sim.py --horizon 45 --seeds 3 --out results
```

默认测试四种负载（low / medium / high / overload）与五种策略：

1. `fixed256`
2. `fixed1024`
3. `gpu_threshold`
4. `deadline_adaptive`
5. `priority_deadline`

输出包括 P50/P95/P99 TTFT、P50/P95/P99 TPOT、完成请求吞吐、Goodput、平均 Chunk、Iteration 耗时、GPU 利用率代理量，以及合成成本面。

## 建模概要

每轮迭代的成本由 Decode、Prefill 和并发竞争项组成：

```text
T_iter(n, c, L) = T_decode(n, L) + T_prefill(c, L) + T_contention(n, c)
```

其中：
- `n`：当前活跃 Decode 请求数；
- `c`：本轮 Prefill Chunk；
- `L`：缓存上下文长度。

Deadline-adaptive 策略依据活跃 Decode 请求最早的 TPOT deadline 计算 Slack，并通过二分搜索选择满足安全余量约束的最大 Chunk。当前代码使用 1.25 的固定安全系数作为示意。

## 指标口径

- TTFT：请求到达至首 token 的时间。
- TPOT：生成阶段 token 间隔的请求级均值；代码将其作为请求级 SLO 判定量。
- Request throughput：单位仿真时间内完成的请求数。
- Goodput：同时满足该负载 TTFT 与 TPOT 门限的完成请求数 / 仿真时间。
- GPU utilization proxy、KV peak proxy：仅为模型代理量，不是 GPU telemetry。

## 从仿真走向真实 profiling

1. 固定 GPU 型号、驱动、CUDA、Runtime 版本、模型权重、精度和 TP 配置。
2. 分别测量 Decode-only、Prefill-only、Mixed iteration。
3. 对 `(active_decode, chunk_tokens, cached_context)` 网格采样，每点重复多次，记录 P50/P90/P99。
4. 对预测成本做单调性检查；必要时沿 Chunk 维构造单调保守上包络。
5. 用真实 P99 lookup table 替换 `SyntheticCostModel.predict`。
6. 在独立到达 trace 上评估 TTFT、TPOT、Goodput、队列长度和 deadline violation。
7. 将预测误差和安全余量作为单独消融变量。

## 相关论文笔记

- [DRLServe](papers/README_DRLSERVE_REVISED.md)
- [RAD & SLAI](papers/README_RAD_SLAI_reviewed.md)
- [SLOWEAVE](papers/SLOWEAVE_README_revised.md)

## 上游仿真框架

本项目打包时包含 LLMServingSim 源码快照及 profiler 数据。由于其中含大型数据文件，未将整个上游仓库复制到本目录。请查看 [UPSTREAM.md](UPSTREAM.md) 获取上游仓库地址、复现建议与版本说明。

## 可复现性与限制

随机种子固定为代码中的 `SEED`，但不同依赖版本、平台与浮点实现仍可能产生细微差异。当前仿真器为研究原型，适合比较策略机制与构建实验流程；不应将合成成本模型外推为真实 GPU 性能。
