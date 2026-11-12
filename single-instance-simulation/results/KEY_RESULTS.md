# 本轮合成仿真关键结果（5 seeds）

**Synthetic only，不是 GPU 实测。** TTFT/TBT 单位 ms，吞吐和 Goodput 单位 req/s。

| workload | policy | p99_ttft_ms | p99_tpot_ms | request_throughput_rps | goodput_rps |
|---|---|---:|---:|---:|---:|
| high | deadline_adaptive | 970896 | 123.217 | 0.175 | 0.004 |
| high | fixed1024 | 1.22903e+06 | 588.878 | 0.132 | 0.001 |
| high | fixed256 | 847958 | 163.649 | 0.197 | 0.005 |
| high | gpu_threshold | 804314 | 180.569 | 0.194 | 0.005 |
| high | priority_deadline | 1.08641e+06 | 343.315 | 0.160 | 0.002 |
| low | deadline_adaptive | 63.76 | 9.89 | 0.371 | 0.371 |
| low | fixed1024 | 79.159 | 10.877 | 0.478 | 0.478 |
| low | fixed256 | 166.408 | 10.764 | 0.396 | 0.396 |
| low | gpu_threshold | 51.07 | 9.855 | 0.355 | 0.355 |
| low | priority_deadline | 70.146 | 9.676 | 0.427 | 0.427 |
| medium | deadline_adaptive | 4175.61 | 68.364 | 1.258 | 1.149 |
| medium | fixed1024 | 3975.72 | 150.234 | 1.047 | 0.799 |
| medium | fixed256 | 12324.5 | 83.385 | 1.058 | 0.632 |
| medium | gpu_threshold | 3130.69 | 40.468 | 1.228 | 1.109 |
| medium | priority_deadline | 9658.3 | 123.144 | 1.092 | 0.521 |
| overload | deadline_adaptive | 9.50981e+06 | 178.69 | 0.058 | 0 |
| overload | fixed1024 | 1.08142e+07 | 809.833 | 0.050 | 0 |
| overload | fixed256 | 9.86651e+06 | 251.982 | 0.056 | 0 |
| overload | gpu_threshold | 9.64355e+06 | 261.248 | 0.057 | 0 |
| overload | priority_deadline | 8.0832e+06 | 368.198 | 0.069 | 0 |

Overload 的 P99 TTFT 达到数千秒量级，表示该合成到达强度远超模型服务能力；Goodput 接近 0 是过载信号，不是正常工作区间内的性能结论。
