# 单实例 LLM 推理建模与仿真 — V2

V2 在保留 V1 论文笔记、原始结果和 LLMServingSim submodule 的基础上，优先完成 Phase 1（仿真正确性）与 Phase 2（Profiling-driven Runtime Modeling）。

> **数据口径：** 默认 `SyntheticCostModel` 是解析式合成模型，不是 RTX 3090 / Qwen3-32B 的真实 Profiling。仓库没有伪造硬件实测数据。只有按 `PROFILING_V2.md` 采集并填入的测量 CSV，才能用于 profile-driven predictor。

## V2 改动

### Phase 1 — Simulator Correctness

- 修正 Token 生命周期：Prefill 完成时产生首 Token；每个 Decode iteration 恰好产生一个后续 Token。
- 修正 ITL / TPOT 口径：ITL 是相邻输出 Token 时间差；TPOT 是单请求 ITL 均值；单 Token 输出不参与 TPOT 分位数。
- 所有策略在相同 `(workload, seed)` 下使用完全相同的 Request Trace（到达时间、Prompt Length、Output Length）。
- 通过独立 Request 副本避免策略之间状态污染。
- 明确 Throughput / Goodput 使用固定 offered-load window 作为分母，Drain 时间只用于完成已到达请求。
- 输出 unfinished requests、queue wait、P99 ITL、P99 TPOT、P99 TTFT、iteration latency 等指标。
- `gpu_threshold` 明确标注为队列压力代理量，不冒充真实 GPU utilization。

### Phase 2 — Runtime Modeling

- `SyntheticCostModel`：透明的非线性解析式模型，仅用于机制验证。
- `generate_profiling_mesh`：生成 Prefill / Decode / Mixed Profiling 参数网格模板。
- `load_profile_csv`：校验真实测量 CSV 的列、数值和单位。
- `summarize_profile`：聚合 P50 / P95 / P99、均值、标准差、样本数和 CV。
- `P99LookupPredictor`：精确查表 + log-feature inverse-distance interpolation。
- `RandomForestRuntimePredictor`：可选的非线性预测基线（需要 scikit-learn）。
- 按 `mode` 隔离 profiling 数据，避免混用 prefill-only / decode-only / mixed 样本。

## 项目结构

```text
v2/
  runtime.py                 # 成本模型、Profiling 数据处理、预测器
  simulator.py               # 请求生命周期、调度、指标和实验入口
scripts/
  generate_profiling_mesh.py # 生成硬件 Profiling CSV 模板
tests/
  test_v2.py                 # 单元测试
```

## 环境安装

```bash
cd single-instance-simulation
pip install -r requirements.txt
# 如需 Random Forest：
pip install -r requirements-ml.txt
```

## 运行合成仿真

```bash
python single_instance_sim.py --horizon 45 --seeds 5 --out results_v2
```

输出：`runs_v2.csv`（逐 seed 结果）和 `summary_v2.csv`（跨 seed 聚合结果）。

## 运行自动化测试

```bash
python -m unittest discover -s tests -v
```

## 生成 Profiling Mesh

```bash
python scripts/generate_profiling_mesh.py \
  --decode-batches 0,1,2,4,8,16,32 \
  --chunks 0,128,256,512,1024,2048,4096 \
  --contexts 512,2048,4096,8192,16384,32768 \
  --kv-cache-tokens 0,8192,32768 \
  --repeats 5 --mode mixed \
  --out profiling_mixed.csv
```

模板中的 `iteration_ms` 留空，需要通过目标硬件和实际推理框架测量后填写。完整字段、测量建议、训练/验证划分和误差分析见 [PROFILING_V2.md](PROFILING_V2.md)。

## 使用实测 P99 Lookup Predictor

```bash
python single_instance_sim.py \
  --horizon 45 --seeds 5 --out results_profiled \
  --profile-csv profiling_mixed.csv \
  --predictor p99_lookup
```

## 使用 Random Forest Predictor

```bash
python single_instance_sim.py \
  --horizon 45 --seeds 5 --out results_rf \
  --profile-csv profiling_mixed.csv \
  --predictor random_forest
```

## 当前模型边界

- 当前抽象每轮最多执行一个等待请求的 Prefill Chunk，并对所有 Active Decode 请求执行一次 Decode；尚未模拟真实引擎的多请求 Prefill Packing、抢占、Speculative Decoding 和精确 Paged-KV Block 管理。
- `kv_cache_peak_tokens_proxy` 是 Token 数量代理量，不是显存字节或真实 KV block allocator。
- `priority_deadline` 是简化 EDF-like 基线，不是 SLAI 原算法。
- `deadline_adaptive` 是基于 Runtime Predictor 的 slack-aware Chunk 规则，不是 SLOWEAVE 原算法复现。
- 预测器不能保证测量网格之外的外推精度；必须使用 held-out mesh points 验证。

请先阅读 [MODELING_V2.md](MODELING_V2.md) 理解请求生命周期、指标和策略定义，再阅读 [PROFILING_V2.md](PROFILING_V2.md) 采集真实运行时数据。
