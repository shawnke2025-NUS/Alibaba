# Alibaba — LLM Inference & Serving Research

当前仓库包含单实例 LLM 推理调度建模与仿真研究材料。

## 单实例推理仿真

- [项目总 README](single-instance-simulation/README.md)
- [Python 仿真源码](single-instance-simulation/single_instance_sim.py)
- [原始方案整理](single-instance-simulation/ORIGINAL_PLAN.md)
- [实验汇总 CSV](single-instance-simulation/results/summary.csv)
- [关键实验结论](single-instance-simulation/results/KEY_RESULTS.md)

### 曲线图

- [Goodput](single-instance-simulation/results/goodput_rps.svg)
- [P99 TTFT](single-instance-simulation/results/p99_ttft_ms.svg)
- [P99 TPOT](single-instance-simulation/results/p99_tpot_ms.svg)
- [Request throughput](single-instance-simulation/results/request_throughput_rps.svg)
- [Mean chunk size](single-instance-simulation/results/mean_chunk_tokens.svg)
- [GPU utilization proxy](single-instance-simulation/results/gpu_util_proxy.svg)

### 论文阅读笔记

- [DRLServe](single-instance-simulation/papers/DRLServe_notes.md)
- [RAD & SLAI](single-instance-simulation/papers/RAD_SLAI_notes.md)
- [SLOWEAVE](single-instance-simulation/papers/SLOWEAVE_notes.md)

## 第三方源码

LLMServingSim 通过 Git submodule 固定到上游 commit。请使用：

```bash
git clone --recurse-submodules https://github.com/shawnke2025-NUS/Alibaba.git
```

## 数据声明

仿真曲线与 CSV 来自合成请求 Trace 和解析式成本模型，不是 2×RTX 3090 / Qwen3-32B 的真实实测或 profiling。真实硬件结论需等待目标环境 profiling 校准。
