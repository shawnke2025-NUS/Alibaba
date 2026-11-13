# V2 Synthetic Simulation Results

> SyntheticCostModel only. These results are not measured RTX 3090/Qwen3-32B performance.

## Experiment setup

- 4 workloads x 5 policies x 5 paired seeds = 100 runs
- 45-second offered-load window per run; bounded drain period
- Same request trace for all policies within each workload/seed pair
- Summary reports mean, standard deviation, and Student-t 95% CI half-width

## Files

- runs_v2.csv: per-seed raw results
- summary_v2.csv: aggregate results and confidence intervals
- figures/: metric curves
- RUN_MANIFEST.json: reproducibility metadata

## Interpretation caveat

Do not compare these values directly with V1 or paper results unless request traces, SLOs, runtime model, measurement window, and metric denominators are made identical.
