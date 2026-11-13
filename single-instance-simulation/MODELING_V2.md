# V2 Modeling Contract

## Scope

V2 implements a correctness-first discrete-event simulator and a pluggable runtime predictor. It does **not** claim to reproduce vLLM, SGLang, TensorRT-LLM, DRLServe, RAD/SLAI, or SLOWEAVE in full. The simulator is a controlled research abstraction for paired-policy experiments.

## Request lifecycle

Each request follows:

`arrival -> waiting/prefill chunks -> first output token -> decode iterations -> completion`.

- `TTFT = first_token_time - arrival_time`.
- `ITL[k] = token_time[k] - token_time[k-1]` for output tokens after the first.
- Per-request `TPOT = mean(ITL)`; it is undefined for a one-token output and excluded from TPOT percentiles.
- Each active decode request emits exactly one token per simulated iteration.
- The first output token is emitted when its prompt prefill completes.

## Scheduling abstraction

| Policy | Waiting-request selection | Prefill chunk |
|---|---|---|
| `fixed256` | FCFS | Up to 256 tokens |
| `fixed1024` | FCFS | Up to 1024 tokens |
| `gpu_threshold` | FCFS | 256/1024 based on a queue-pressure heuristic |
| `deadline_adaptive` | FCFS | Largest chunk predicted to fit active decode TPOT slack |
| `priority_deadline` | Earliest absolute TTFT deadline; prompt length breaks ties | Up to 512 tokens |

The `gpu_threshold` policy uses a synthetic queue-pressure proxy, **not measured GPU utilization**. The `priority_deadline` strategy is a simplified EDF-like baseline, not an implementation of SLAI or another paper's algorithm.

The current abstraction schedules one waiting request's prefill chunk per iteration and batches all active decode requests. It does not yet model continuous-batching internals, chunked-prefill packing across multiple waiting requests, preemption, speculative decoding, or exact paged-KV block allocation.

## Metric contract

- The offered-load measurement window is `[0, horizon_s)`.
- Requests arriving during that window may complete during a subsequent drain period.
- `request_throughput_rps = completed / horizon_s`.
- `goodput_rps = SLO-compliant completed requests / horizon_s`.
- Goodput counts a request if TTFT is within its workload TTFT SLO and its per-request mean TPOT is within its TPOT SLO. For one-token outputs, the TPOT condition is vacuous.
- `p99_itl_ms` is computed over all observed token-to-token intervals; `p99_tpot_ms` is computed over per-request mean TPOT values. These are different statistics.
- Unfinished requests remain in the denominator of completion ratio and are not silently dropped from the arrival count.

All percentiles are computed over completed requests or observed token intervals. Very small sample counts make P99 statistically unstable; use longer horizons and multiple paired seeds for conclusions.

## Paired experiments

For a given `(workload, seed)`, all policies receive the same arrival times, prompt lengths, and output lengths. Each policy gets a fresh copy of the trace so request state cannot leak between runs. Use the same seed set across all policies and report per-seed results, means, standard deviations, and confidence intervals.

## Runtime predictor contract

The simulator calls:

```python
predict_ms(
    active_decode_batch,
    prefill_chunk_tokens,
    max_context_tokens,
    kv_cache_tokens=0,
    mode="mixed",
)
```

The return value is one iteration's predicted latency in milliseconds. The simulator rejects non-finite or non-positive predictions.

`SyntheticCostModel` is an illustrative analytical model. Its coefficients are not hardware-calibrated and its output must not be described as measured RTX 3090 performance. `P99LookupPredictor` and `RandomForestRuntimePredictor` consume measured profiling tables using the schema in `PROFILING_V2.md`.
