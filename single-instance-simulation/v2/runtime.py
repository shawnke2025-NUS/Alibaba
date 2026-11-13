"""Runtime cost models and profiling-data utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol
import math

import numpy as np
import pandas as pd

PROFILE_KEYS = [
    "active_decode_batch",
    "prefill_chunk_tokens",
    "max_context_tokens",
    "kv_cache_tokens",
]
PROFILE_REQUIRED = PROFILE_KEYS + ["iteration_ms"]


class RuntimePredictor(Protocol):
    def predict_ms(
        self,
        active_decode_batch: int,
        prefill_chunk_tokens: int,
        max_context_tokens: int,
        kv_cache_tokens: int = 0,
        mode: str = "mixed",
    ) -> float: ...


@dataclass
class SyntheticCostModel:
    """Transparent analytical stress model; not calibrated to any GPU."""

    safety_floor_ms: float = 0.25

    def predict_ms(
        self,
        active_decode_batch: int,
        prefill_chunk_tokens: int,
        max_context_tokens: int,
        kv_cache_tokens: int = 0,
        mode: str = "mixed",
    ) -> float:
        n = max(0, int(active_decode_batch))
        c = max(0, int(prefill_chunk_tokens))
        h = max(0, int(max_context_tokens))
        kv = max(0, int(kv_cache_tokens))
        decode_ms = 0.0
        if n:
            decode_ms = 7.5 + 0.82*n + 0.000075*n*h + 0.00000000035*n*h*h
        prefill_ms = 0.0
        if c:
            prefill_ms = 2.4 + 0.0135*c + 0.00000065*c*h + 0.00000000022*c*h*h
        interference_ms = 0.0
        if n and c:
            interference_ms = 1.0 + 0.0022*c*math.sqrt(n) + 0.035*n*math.log1p(h/1024.0)
        cache_pressure_ms = 0.000015*kv if n or c else 0.0
        return max(self.safety_floor_ms, decode_ms + prefill_ms + interference_ms + cache_pressure_ms)

    def predict(self, n_active: int, chunk: int, cached_context: int, mode: str = "mixed") -> float:
        return self.predict_ms(n_active, chunk, cached_context, 0, mode)


@dataclass
class P99LookupPredictor:
    """P99 lookup with inverse-distance interpolation over measured mesh data."""

    profile_table: pd.DataFrame
    distance_power: float = 2.0
    _by_mode: dict | None = None

    def __post_init__(self) -> None:
        required = PROFILE_KEYS + ["p99_ms"]
        missing = [c for c in required if c not in self.profile_table.columns]
        if missing:
            raise ValueError(f"profile table missing columns: {missing}")
        table = self.profile_table.dropna(subset=required).copy()
        if table.empty:
            raise ValueError("profile table contains no valid P99 rows")
        if np.any(table["p99_ms"].to_numpy(dtype=float) <= 0):
            raise ValueError("all p99_ms values must be positive")
        if "mode" not in table:
            table["mode"] = "mixed"
        self._by_mode = {}
        for mode_name, group in table.groupby("mode", dropna=False):
            features = group[PROFILE_KEYS].to_numpy(dtype=float)
            log_features = np.log1p(features)
            scale = np.maximum(np.ptp(log_features, axis=0), 1e-9)
            self._by_mode[str(mode_name)] = (
                group["p99_ms"].to_numpy(dtype=float),
                log_features,
                scale,
            )

    def predict_ms(
        self,
        active_decode_batch: int,
        prefill_chunk_tokens: int,
        max_context_tokens: int,
        kv_cache_tokens: int = 0,
        mode: str = "mixed",
    ) -> float:
        point = np.array(
            [active_decode_batch, prefill_chunk_tokens, max_context_tokens, kv_cache_tokens],
            dtype=float,
        )
        if np.any(point < 0):
            raise ValueError("runtime predictor inputs must be non-negative")
        if str(mode) not in self._by_mode:
            raise ValueError(f"profile table has no mode={mode!r}; do not substitute a different execution mode")
        targets, log_features, scale = self._by_mode[str(mode)]
        log_point = np.log1p(point)
        distances = np.linalg.norm((log_features - log_point) / scale, axis=1)
        exact = np.flatnonzero(distances < 1e-12)
        if len(exact):
            return float(np.median(targets[exact]))
        k = min(8, len(distances))
        nearest = np.argpartition(distances, k - 1)[:k]
        d = np.maximum(distances[nearest], 1e-9)
        weights = 1.0 / np.power(d, self.distance_power)
        return float(np.dot(weights, targets[nearest]) / weights.sum())


class RandomForestRuntimePredictor:
    """Optional Random Forest P99 predictor, snapped to measured mesh axes."""

    def __init__(self, profile_table: pd.DataFrame, random_state: int = 42, mode: str = "mixed") -> None:
        required = PROFILE_KEYS + ["p99_ms"]
        missing = [c for c in required if c not in profile_table.columns]
        if missing:
            raise ValueError(f"profile table missing columns: {missing}")
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise ImportError(
                "RandomForestRuntimePredictor requires scikit-learn; install requirements-ml.txt"
            ) from exc
        table = profile_table.dropna(subset=required).copy()
        self.mode = mode
        if "mode" in table.columns:
            table = table[table["mode"].astype(str) == mode]
            if table.empty:
                raise ValueError(f"profiling table has no mode={mode!r}; collect that execution mode separately")
        if len(table) < 4:
            raise ValueError("at least four valid profiling mesh rows are required")
        self._feature_axes = [np.unique(table[col].to_numpy(dtype=float)) for col in PROFILE_KEYS]
        self.model = RandomForestRegressor(
            n_estimators=64,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=random_state,
            n_jobs=1,
        )
        self.model.fit(table[PROFILE_KEYS].to_numpy(dtype=float), table["p99_ms"].to_numpy(dtype=float))
        self._prediction_cache: dict[tuple[int, int, int, int], float] = {}

    def predict_ms(
        self,
        active_decode_batch: int,
        prefill_chunk_tokens: int,
        max_context_tokens: int,
        kv_cache_tokens: int = 0,
        mode: str = "mixed",
    ) -> float:
        if str(mode) != self.mode:
            raise ValueError(f"predictor was trained for mode={self.mode!r}, not mode={mode!r}")
        raw = (active_decode_batch, prefill_chunk_tokens, max_context_tokens, kv_cache_tokens)
        if any(float(value) < 0 for value in raw):
            raise ValueError("runtime predictor inputs must be non-negative")
        # Snap to nearest measured axis in log space. Cache reduces online overhead.
        snapped = []
        for axis, value in zip(self._feature_axes, raw):
            nearest_index = int(np.argmin(np.abs(np.log1p(axis) - np.log1p(float(value)))))
            snapped.append(int(axis[nearest_index]))
        key = tuple(snapped)
        if key not in self._prediction_cache:
            self._prediction_cache[key] = max(
                0.01, float(self.model.predict(np.array([key], dtype=float))[0])
            )
        return self._prediction_cache[key]


def load_profile_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in PROFILE_REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"profiling CSV missing columns: {missing}")
    for col in PROFILE_REQUIRED:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[PROFILE_REQUIRED].isna().any().any():
        bad = frame[PROFILE_REQUIRED].isna().any(axis=1).sum()
        raise ValueError(f"profiling CSV contains {bad} rows with missing/non-numeric required values")
    if (frame[PROFILE_KEYS] < 0).any().any():
        raise ValueError("profiling feature values must be non-negative")
    if (frame["iteration_ms"] <= 0).any():
        raise ValueError("iteration_ms must be positive")
    if "mode" not in frame:
        frame["mode"] = "mixed"
    if "repeat" not in frame:
        frame["repeat"] = 0
    if "seed" not in frame:
        frame["seed"] = 0
    return frame


def summarize_profile(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PROFILE_REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"profiling frame missing columns: {missing}")
    group_keys = PROFILE_KEYS + (["mode"] if "mode" in frame.columns else [])
    summary = (
        frame.groupby(group_keys, dropna=False)["iteration_ms"]
        .agg(
            sample_count="count",
            mean_ms="mean",
            std_ms="std",
            p50_ms=lambda x: x.quantile(0.50),
            p95_ms=lambda x: x.quantile(0.95),
            p99_ms=lambda x: x.quantile(0.99),
        )
        .reset_index()
    )
    summary["cv"] = summary["std_ms"].fillna(0.0) / summary["mean_ms"].clip(lower=1e-9)
    return summary


def check_profile_consistency(summary: pd.DataFrame, cv_warning_threshold: float = 0.15) -> list[str]:
    messages: list[str] = []
    if "sample_count" in summary:
        sparse = summary[summary["sample_count"] < 3]
        if not sparse.empty:
            messages.append(f"{len(sparse)} mesh points have fewer than 3 measurements")
    if "cv" in summary:
        noisy = summary[summary["cv"] > cv_warning_threshold]
        if not noisy.empty:
            messages.append(
                f"{len(noisy)} mesh points have coefficient of variation > {cv_warning_threshold:.2f}"
            )
    return messages


def generate_profiling_mesh(
    decode_batches: Iterable[int] = (0, 1, 2, 4, 8, 16, 32, 64),
    chunks: Iterable[int] = (0, 128, 256, 512, 1024, 2048, 4096, 8192),
    contexts: Iterable[int] = (512, 2048, 4096, 8192, 16384, 32768, 65536),
    kv_cache_tokens: Iterable[int] = (0, 8192, 32768, 65536),
    repeats: int = 5,
    mode: str = "mixed",
) -> pd.DataFrame:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if mode not in {"prefill", "decode", "mixed"}:
        raise ValueError("mode must be one of: prefill, decode, mixed")
    rows = []
    for n in decode_batches:
        for c in chunks:
            if mode == "prefill" and (int(n) != 0 or int(c) <= 0):
                continue
            if mode == "decode" and (int(n) <= 0 or int(c) != 0):
                continue
            if mode == "mixed" and (int(n) <= 0 or int(c) <= 0):
                continue
            for h in contexts:
                for kv in kv_cache_tokens:
                    for repeat in range(repeats):
                        rows.append({
                            "active_decode_batch": int(n),
                            "prefill_chunk_tokens": int(c),
                            "max_context_tokens": int(h),
                            "kv_cache_tokens": int(kv),
                            "mode": mode,
                            "repeat": repeat,
                            "seed": 20261113 + repeat,
                            "iteration_ms": np.nan,
                        })
    if not rows:
        raise ValueError(f"mesh has no valid configurations for mode={mode!r}; include zero/non-zero boundary values")
    return pd.DataFrame(rows)


def build_p99_table(raw: pd.DataFrame) -> pd.DataFrame:
    return summarize_profile(raw)


def warn_if_non_monotonic(summary: pd.DataFrame, metric: str = "p99_ms") -> list[str]:
    messages: list[str] = []
    if metric not in summary:
        return [f"metric {metric!r} is absent; monotonicity check skipped"]
    for feature in ("active_decode_batch", "prefill_chunk_tokens", "max_context_tokens"):
        other = [c for c in PROFILE_KEYS if c != feature]
        if "mode" in summary.columns:
            other.append("mode")
        violations = 0
        for _, group in summary.groupby(other, dropna=False):
            values = group.sort_values(feature)[metric].to_numpy(dtype=float)
            if len(values) > 1:
                violations += int(np.sum(np.diff(values) < -1e-6))
        if violations:
            messages.append(
                f"{feature}: {violations} local decreases found; inspect noise/cache effects, "
                "do not force monotonic fitting blindly"
            )
    return messages
