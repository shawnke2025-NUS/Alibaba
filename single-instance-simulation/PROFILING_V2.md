# V2 Runtime Profiling Protocol

## Purpose

Replace the V1 hand-written latency expression with a predictor fitted to measured per-iteration latency. No GPU measurements are bundled in V2; the repository contains the schema, mesh generator, ingestion, aggregation, and predictor interfaces only.

## Profiling dimensions

Default mesh dimensions:

- Active Decode Batch: `0, 1, 2, 4, 8, 16, 32, 64`
- Prefill Chunk: `0, 128, 256, 512, 1024, 2048, 4096, 8192`
- Maximum Context: `512, 2048, 4096, 8192, 16384, 32768, 65536`
- KV Cache Tokens: `0, 8192, 32768, 65536`
- Repeats: 5 per mesh point
- Modes: separate `prefill`, `decode`, and `mixed` sweeps

The default mixed-mode mesh has 6,860 valid rows (7 non-zero decode batches x 7 non-zero chunks x 7 contexts x 4 KV settings x 5 repeats). Prefill-only and decode-only sweeps generate their corresponding boundary configurations separately. Start with a reduced mesh to validate the benchmark harness.

## Generate a template

```bash
cd single-instance-simulation
python scripts/generate_profiling_mesh.py \
  --decode-batches 0,1,2,4,8,16,32 \
  --chunks 0,128,256,512,1024,2048,4096 \
  --contexts 512,2048,4096,8192,16384,32768 \
  --kv-cache-tokens 0,8192,32768 \
  --repeats 5 --mode mixed \
  --out profiling_mixed.csv
```

The generated file is a **template**: `iteration_ms` is intentionally blank. Fill it with measured per-iteration latency before training a predictor. Do not fill it with values from `SyntheticCostModel` and then label it hardware profiling.

## Required CSV schema

| Column | Meaning |
|---|---|
| `active_decode_batch` | Number of active decode sequences |
| `prefill_chunk_tokens` | Number of prefill tokens scheduled in the iteration |
| `max_context_tokens` | Maximum effective context length in the measured batch |
| `kv_cache_tokens` | Total resident KV tokens (or a consistently defined proxy) |
| `iteration_ms` | Measured iteration latency in milliseconds; strictly positive |
| `mode` | `prefill`, `decode`, or `mixed`; optional, defaults to `mixed` |
| `repeat` | Repeat index; optional |
| `seed` | Benchmark seed/run identifier; optional |

The mesh generator enforces physically meaningful configurations: prefill-only rows have decode batch zero and a positive prefill chunk; decode-only rows have a positive decode batch and zero prefill chunk; mixed rows have both non-zero. Keep execution modes separate.

## Measurement recommendations

1. Pin model revision, quantization, framework version, CUDA/PyTorch versions, TP size, GPU model, clock/power mode, and KV block configuration in a companion metadata file.
2. Warm up each configuration before collecting samples. Exclude compilation, model loading, and graph-capture setup from steady-state latency unless explicitly studying cold start.
3. Measure Prefill-only, Decode-only, and Mixed workloads separately. For mixed runs, record the active decode batch and actual prefill chunk executed in the same iteration.
4. Repeat each point at least five times; preferably collect enough steady-state iterations to estimate a stable P99. Five individual observations do not provide a reliable tail estimate.
5. Record raw per-iteration samples, not only a precomputed percentile. Preserve outliers and annotate preemption, synchronization, cache misses, or background activity.
6. Validate the harness on a small set of repeated configurations before sweeping the full mesh.
7. Re-profile after changing hardware, model, quantization, framework version, or parallelism.

## Build and use predictors

```python
from v2.runtime import load_profile_csv, summarize_profile
from v2.runtime import P99LookupPredictor, RandomForestRuntimePredictor

raw = load_profile_csv("profiling_mixed.csv")
summary = summarize_profile(raw)
lookup = P99LookupPredictor(summary)
latency_ms = lookup.predict_ms(8, 512, 8192, 32768, mode="mixed")

# Optional dependency: pip install -r requirements-ml.txt
forest = RandomForestRuntimePredictor(summary, mode="mixed")
latency_ms_rf = forest.predict_ms(8, 512, 8192, 32768, mode="mixed")
```

`P99LookupPredictor` uses exact mesh matches when available and inverse-distance interpolation in log-feature space otherwise. It rejects requests for execution modes absent from the profile table. `RandomForestRuntimePredictor` is trained only on the selected mode; at inference it snaps each feature to the nearest measured axis value in log space and caches predictions to control scheduling overhead. It is a mesh-based surrogate, not a guarantee of continuous interpolation. Neither predictor guarantees accurate extrapolation outside the measured mesh; validate on held-out mesh points.

## Validation before scheduling claims

- Split validation by mesh points, not random repeated rows, to avoid leakage.
- Report MAE, RMSE, MAPE (with care near zero), and P95/P99 absolute prediction error.
- Compare P50 and P99 targets; a P99 predictor should not be judged only by mean absolute error.
- Inspect residuals by decode batch, chunk size, context length, KV occupancy, and execution mode.
- Check local monotonicity as a diagnostic only. Real measurements can be non-monotonic due to kernel selection, graph capture, cache behavior, and measurement noise.
- Use a conservative predictor for deadline-aware scheduling and measure deadline misses on held-out online traces.
