# LLMServingSim 上游源码

本项目已在 `single-instance-simulation/third_party/LLMServingSim` 配置 Git submodule，指向：

- 上游仓库：https://github.com/casys-kaist/LLMServingSim
- 固定 commit：`49df63f2af61230da63dd2136044a2dbf9bd4161`

完整克隆（包括上游源码）：

```bash
git clone --recurse-submodules https://github.com/shawnke2025-NUS/Alibaba.git
```

如果已经 clone 主仓库但没有拉取子模块：

```bash
git submodule update --init --recursive
```

说明：
- 使用 submodule 是为了保留上游作者和许可边界，避免把第三方项目误标为本项目原创。
- 用户提供的压缩包内还包含一个 LLMServingSim 源码快照归档；该快照并未作为本仓库的独立二进制文件重复上传。此处 submodule 固定到上游仓库的一个明确 commit，便于 clone 和复现，但不保证与用户上传的 zip 快照逐字节一致。
- 使用上游源码前请遵循其 LICENSE，以及模型、trace、profiler 数据各自的使用许可。
