# Alibaba — 单实例推理调度研究

本仓库当前新增研究目录：

- [单实例推理建模与仿真](single-instance-simulation/README.md)
  - [离散事件仿真源码](single-instance-simulation/single_instance_sim.py)
  - [Python 依赖](single-instance-simulation/requirements.txt)
  - [合成仿真汇总数据](single-instance-simulation/results/summary.csv)
  - [关键结果与数据说明](single-instance-simulation/results/KEY_RESULTS.md)
  - [LLMServingSim 上游源码说明](single-instance-simulation/UPSTREAM.md)

**数据声明：**当前单实例仿真使用合成解析成本模型，不是 RTX 3090 / Qwen3-32B 实测。过载场景中的极端 TTFT 反映当前模型下到达率远高于服务能力，不应作为真实服务性能结论。

## 快速运行

```bash
cd single-instance-simulation
pip install -r requirements.txt
python single_instance_sim.py --horizon 45 --seeds 3 --out results
```
