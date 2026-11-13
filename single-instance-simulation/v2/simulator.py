"""Correctness-first single-instance continuous-time serving simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .runtime import RuntimePredictor, SyntheticCostModel


@dataclass(frozen=True)
class WorkloadConfig:
    arrival_rate_rps: float
    prompt_median: int
    prompt_sigma: float
    output_median: int
    output_sigma: float
    burst: bool = False


WORKLOADS = {
    "low": WorkloadConfig(0.35, 512, 0.8, 64, 0.6),
    "medium": WorkloadConfig(1.5, 2048, 1.0, 128, 0.8),
    "high": WorkloadConfig(4.0, 8192, 1.25, 256, 1.0),
    "overload": WorkloadConfig(8.0, 16384, 1.5, 512, 1.2, burst=True),
}

SLO_MS = {
    "low": (1500.0, 80.0),
    "medium": (2500.0, 100.0),
    "high": (6000.0, 150.0),
    "overload": (10000.0, 220.0),
}

POLICIES = ("fixed256", "fixed1024", "gpu_threshold", "deadline_adaptive", "priority_deadline")


@dataclass
class Request:
    request_id: int
    arrival_time_s: float
    prompt_tokens: int
    output_tokens: int
    prefill_done_tokens: int = 0
    generated_tokens: int = 0
    first_token_time_s: float | None = None
    completion_time_s: float | None = None
    prefill_start_time_s: float | None = None
    last_token_time_s: float | None = None
    itl_ms: list[float] = field(default_factory=list)

    @property
    def remaining_prefill_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.prefill_done_tokens)

    @property
    def is_complete(self) -> bool:
        return self.completion_time_s is not None

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_time_s is None:
            return None
        return (self.first_token_time_s - self.arrival_time_s) * 1000.0

    @property
    def tpot_ms(self) -> float | None:
        # TPOT is undefined for one-token outputs; exclude it from TPOT percentiles.
        if not self.itl_ms:
            return None
        return float(np.mean(self.itl_ms))


@dataclass
class SimulationConfig:
    horizon_s: float = 45.0
    max_drain_s: float = 1800.0
    max_iterations: int = 2_000_000
    deadline_safety_factor: float = 1.15
    max_adaptive_chunk: int = 8192
    seed: int = 20261111


def make_trace(workload: str, horizon_s: float, seed: int) -> list[Request]:
    """Generate a deterministic request trace for paired-policy comparisons."""
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload: {workload}")
    if horizon_s <= 0:
        raise ValueError("horizon_s must be positive")
    cfg = WORKLOADS[workload]
    rng = np.random.default_rng(seed)
    requests: list[Request] = []
    t = 0.0
    while t < horizon_s:
        if cfg.burst:
            hot = (int(t // 8) % 2 == 0)
            rate = cfg.arrival_rate_rps * (3.0 if hot else 0.18)
        else:
            rate = cfg.arrival_rate_rps
        t += float(rng.exponential(1.0 / max(rate, 1e-9)))
        if t >= horizon_s:
            break
        prompt = int(np.clip(rng.lognormal(np.log(cfg.prompt_median), cfg.prompt_sigma), 128, 100000))
        output = int(np.clip(rng.lognormal(np.log(cfg.output_median), cfg.output_sigma), 1, 4096))
        requests.append(Request(len(requests), t, prompt, output))
    return requests


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def _t_critical_95(df: int) -> float:
    # Two-sided 95% Student-t critical values for small seed counts.
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(max(1, df), 1.96)


def _request_order(
    waiting: list[Request],
    policy: str,
    now_s: float,
    ttft_slo_ms: float,
    active_count: int,
    kv_cache_tokens: int,
    predictor: RuntimePredictor,
) -> list[Request]:
    if policy != "priority_deadline":
        return sorted(waiting, key=lambda r: (r.arrival_time_s, r.request_id))

    # Least-slack-time-first: deadline slack = TTFT deadline - now - estimated
    # remaining prefill service time. This differs from FCFS even with one SLO.
    def slack(request: Request) -> tuple[float, float, int]:
        remaining = request.remaining_prefill_tokens
        chunk = min(512, remaining)
        context = request.prefill_done_tokens + chunk
        one_chunk_ms = predictor.predict_ms(
            active_count, chunk, context, kv_cache_tokens + chunk, mode="mixed"
        )
        estimated_chunks = math.ceil(remaining / 512)
        estimated_service_ms = estimated_chunks * one_chunk_ms
        deadline_s = request.arrival_time_s + ttft_slo_ms / 1000.0
        slack_ms = (deadline_s - now_s) * 1000.0 - estimated_service_ms
        return slack_ms, request.arrival_time_s, request.request_id

    return sorted(waiting, key=slack)


def _choose_chunk(
    policy: str,
    waiting: list[Request],
    active: list[Request],
    now_s: float,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    predictor: RuntimePredictor,
    cfg: SimulationConfig,
    kv_cache_tokens: int,
) -> int:
    if not waiting:
        return 0
    request = waiting[0]
    remaining = request.remaining_prefill_tokens
    if remaining <= 0:
        return 0
    if policy == "fixed256":
        return min(256, remaining)
    if policy == "fixed1024":
        return min(1024, remaining)
    if policy == "gpu_threshold":
        # This is a scheduling heuristic, not a measured GPU utilization signal.
        pressure_proxy = 0.12 + 0.075 * len(active) + 0.025 * len(waiting)
        return min(1024 if pressure_proxy < 0.55 else 256, remaining)
    if policy == "priority_deadline":
        return min(512, remaining)
    if policy == "deadline_adaptive":
        upper = min(cfg.max_adaptive_chunk, remaining)
        if not active:
            return upper
        slacks = [
            (r.last_token_time_s + tpot_slo_ms / 1000.0 - now_s) * 1000.0
            for r in active
            if r.last_token_time_s is not None
        ]
        if not slacks:
            return upper
        slack_ms = min(slacks)
        if slack_ms <= 0:
            return 0
        context = max(
            [r.prefill_done_tokens + r.generated_tokens for r in active]
            + [request.prefill_done_tokens],
            default=0,
        )
        lo, hi = 0, upper
        while lo < hi:
            mid = (lo + hi + 1) // 2
            predicted = predictor.predict_ms(
                len(active), mid, context, kv_cache_tokens, mode="mixed"
            )
            if predicted * cfg.deadline_safety_factor <= slack_ms:
                lo = mid
            else:
                hi = mid - 1
        return lo
    raise ValueError(f"unknown policy: {policy}")


def simulate(
    workload: str,
    policy: str,
    seed: int,
    config: SimulationConfig | None = None,
    predictor: RuntimePredictor | None = None,
    trace: list[Request] | None = None,
) -> dict:
    """Run one workload/policy pair and return request- and system-level metrics.

    The measurement window is [0, horizon_s); after that, the simulator drains
    already-arrived requests for up to max_drain_s. Throughput and Goodput use
    the fixed offered-load window as denominator.
    """
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload: {workload}")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    config = config or SimulationConfig(seed=seed)
    predictor = predictor or SyntheticCostModel()
    source = trace if trace is not None else make_trace(workload, config.horizon_s, seed)
    # A fresh copy prevents policy runs from mutating the shared paired trace.
    requests = [Request(r.request_id, r.arrival_time_s, r.prompt_tokens, r.output_tokens) for r in source]
    ttft_slo_ms, tpot_slo_ms = SLO_MS[workload]

    now_s = 0.0
    next_arrival = 0
    waiting: list[Request] = []
    active: list[Request] = []
    iteration_ms: list[float] = []
    selected_chunks: list[int] = []
    kv_peak_tokens = 0
    stopped_by_guard = False
    max_time_s = config.horizon_s + config.max_drain_s

    for _iteration in range(config.max_iterations):
        while next_arrival < len(requests) and requests[next_arrival].arrival_time_s <= now_s + 1e-12:
            waiting.append(requests[next_arrival])
            next_arrival += 1

        if not waiting and not active:
            if next_arrival >= len(requests):
                break
            now_s = max(now_s, requests[next_arrival].arrival_time_s)
            if now_s > max_time_s:
                break
            continue

        kv_current = sum(
            r.prefill_done_tokens + r.generated_tokens
            for r in (waiting + active)
            if (r.prefill_done_tokens > 0 or r.generated_tokens > 0) and not r.is_complete
        )
        waiting = _request_order(
            waiting, policy, now_s, ttft_slo_ms, len(active), kv_current, predictor
        )
        chunk = _choose_chunk(
            policy, waiting, active, now_s, ttft_slo_ms, tpot_slo_ms,
            predictor, config, kv_current
        )
        prefill_request = waiting[0] if waiting and chunk > 0 else None
        context_candidates = [r.prefill_done_tokens + r.generated_tokens for r in active]
        if prefill_request is not None:
            context_candidates.append(prefill_request.prefill_done_tokens + chunk)
        max_context = max(context_candidates, default=0)
        kv_for_predictor = kv_current + chunk
        decode_batch = list(active)
        dt_ms = predictor.predict_ms(
            len(decode_batch), chunk, max_context, kv_for_predictor, mode="mixed"
        )
        if not math.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError(f"runtime predictor returned invalid iteration time: {dt_ms}")

        end_s = now_s + dt_ms / 1000.0
        if prefill_request is not None:
            if prefill_request.prefill_start_time_s is None:
                prefill_request.prefill_start_time_s = now_s
            prefill_request.prefill_done_tokens += chunk
            selected_chunks.append(chunk)

        newly_active: list[Request] = []
        if prefill_request is not None and prefill_request.prefill_done_tokens >= prefill_request.prompt_tokens:
            waiting.remove(prefill_request)
            prefill_request.generated_tokens = 1
            prefill_request.first_token_time_s = end_s
            prefill_request.last_token_time_s = end_s
            if prefill_request.output_tokens <= 1:
                prefill_request.completion_time_s = end_s
            else:
                newly_active.append(prefill_request)

        finished: list[Request] = []
        for request in decode_batch:
            # Each active request emits exactly one token per decode iteration.
            if request.last_token_time_s is not None:
                request.itl_ms.append((end_s - request.last_token_time_s) * 1000.0)
            request.generated_tokens += 1
            request.last_token_time_s = end_s
            if request.generated_tokens >= request.output_tokens:
                request.completion_time_s = end_s
                finished.append(request)
        for request in finished:
            active.remove(request)
        active.extend(newly_active)

        kv_now = sum(
            r.prefill_done_tokens + r.generated_tokens
            for r in (waiting + active)
            if (r.prefill_done_tokens > 0 or r.generated_tokens > 0) and not r.is_complete
        )
        kv_peak_tokens = max(kv_peak_tokens, kv_now)
        iteration_ms.append(float(dt_ms))
        now_s = end_s
        if now_s > max_time_s:
            stopped_by_guard = True
            break
    else:
        stopped_by_guard = True

    completed = [r for r in requests if r.is_complete and r.first_token_time_s is not None]
    ttft_values = [r.ttft_ms for r in completed if r.ttft_ms is not None]
    tpot_values = [r.tpot_ms for r in completed if r.tpot_ms is not None]
    itl_values = [x for r in completed for x in r.itl_ms]
    queue_wait_values = [
        (r.prefill_start_time_s - r.arrival_time_s) * 1000.0
        for r in completed
        if r.prefill_start_time_s is not None
    ]
    good = 0
    ttft_pass = 0
    tpot_pass = 0
    for request in completed:
        ttft_ok = request.ttft_ms is not None and request.ttft_ms <= ttft_slo_ms
        tpot_ok = request.tpot_ms is None or request.tpot_ms <= tpot_slo_ms
        ttft_pass += int(ttft_ok)
        tpot_pass += int(tpot_ok)
        if ttft_ok and tpot_ok:
            good += 1

    arrivals = len(requests)
    horizon = max(config.horizon_s, 1e-9)
    return {
        "workload": workload,
        "policy": policy,
        "seed": seed,
        "offered_rps": arrivals / horizon,
        "arrivals": arrivals,
        "completed": len(completed),
        "unfinished": arrivals - len(completed),
        "completion_ratio": len(completed) / max(1, arrivals),
        "p50_ttft_ms": _percentile(ttft_values, 50),
        "p95_ttft_ms": _percentile(ttft_values, 95),
        "p99_ttft_ms": _percentile(ttft_values, 99),
        "p50_tpot_ms": _percentile(tpot_values, 50),
        "p95_tpot_ms": _percentile(tpot_values, 95),
        "p99_tpot_ms": _percentile(tpot_values, 99),
        "p99_itl_ms": _percentile(itl_values, 99),
        "mean_queue_wait_ms": float(np.mean(queue_wait_values)) if queue_wait_values else float("nan"),
        "p99_queue_wait_ms": _percentile(queue_wait_values, 99),
        "request_throughput_rps": len(completed) / horizon,
        "goodput_rps": good / horizon,
        "goodput_attainment": good / max(1, arrivals),
        "completed_slo_attainment": good / max(1, len(completed)),
        "ttft_pass_ratio_all_arrivals": ttft_pass / max(1, arrivals),
        "tpot_pass_ratio_all_arrivals": tpot_pass / max(1, arrivals),
        "slo_miss_rate_all_arrivals": 1.0 - good / max(1, arrivals),
        "mean_chunk_tokens": float(np.mean(selected_chunks)) if selected_chunks else 0.0,
        "chunk_count": len(selected_chunks),
        "mean_iteration_ms": float(np.mean(iteration_ms)) if iteration_ms else float("nan"),
        "p99_iteration_ms": _percentile(iteration_ms, 99),
        "kv_cache_peak_tokens_proxy": kv_peak_tokens,
        "drain_time_s": max(0.0, now_s - config.horizon_s),
        "stopped_by_guard": stopped_by_guard,
        "runtime_model": type(predictor).__name__,
        "runtime_measurement_status": "synthetic_or_user_supplied_predictor",
    }


def run_experiments(
    horizon_s: float = 45.0,
    seeds: int = 5,
    output_dir: str = "results_v2",
    predictor: RuntimePredictor | None = None,
) -> tuple:
    """Run paired-trace experiments and return raw and aggregated DataFrames."""
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    config = SimulationConfig(horizon_s=horizon_s)
    rows = []
    for workload_index, workload in enumerate(WORKLOADS):
        for seed_index in range(seeds):
            seed = config.seed + workload_index * 1000 + seed_index
            shared_trace = make_trace(workload, horizon_s, seed)
            for policy in POLICIES:
                rows.append(
                    simulate(
                        workload, policy, seed, config=config, predictor=predictor, trace=shared_trace
                    )
                )
    raw = pd.DataFrame(rows)
    numeric = [c for c in raw.select_dtypes(include=[np.number]).columns if c != "seed"]
    grouped = raw.groupby(["workload", "policy"])[numeric]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std(ddof=1).fillna(0.0).add_suffix("_std")
    counts = grouped.count().clip(lower=1)
    std_values = grouped.std(ddof=1).fillna(0.0)
    critical = counts.apply(lambda col: col.map(lambda n: _t_critical_95(int(n) - 1)))
    ci95 = (critical * std_values / np.sqrt(counts)).add_suffix("_ci95_halfwidth")
    aggregate = pd.concat([means, stds, ci95], axis=1).reset_index()
    raw.to_csv(os.path.join(output_dir, "runs_v2.csv"), index=False)
    aggregate.to_csv(os.path.join(output_dir, "summary_v2.csv"), index=False)
    return raw, aggregate
