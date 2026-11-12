# LLMServingSim 上游源码说明

本地研究包中包含一个 LLMServingSim 源码快照（约 53 MB 压缩），解压后的完整项目及 profiler / workload 数据更大。为避免把大型第三方源码快照和实验数据误当成本项目原创代码，本仓库不重复托管该归档。

请从上游项目获取源码，并遵循其 LICENSE 与数据集/模型各自的许可条款：

- LLMServingSim：请以原始上传归档内的 Git 元数据、README 或项目主页为准。
- 本仓库的 `single_instance_sim.py` 是本项目的独立合成离散事件仿真脚本，不是 LLMServingSim 的修改版。

若需将 LLMServingSim 作为子模块引入，请先核对原始归档中的 `.gitmodules` 和 upstream remote，再执行：

```bash
git submodule add <verified-upstream-url> third_party/LLMServingSim
```

不要在未核验来源和许可前直接公开上传整个第三方快照。
