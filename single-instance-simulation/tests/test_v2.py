from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
import math

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v2.runtime import (
    SyntheticCostModel,
    P99LookupPredictor,
    generate_profiling_mesh,
    summarize_profile,
    load_profile_csv,
)
from v2.simulator import SimulationConfig, Request, make_trace, simulate


class FixedRuntime:
    def predict_ms(self, active_decode_batch, prefill_chunk_tokens, max_context_tokens, kv_cache_tokens=0, mode="mixed"):
        return 10.0


class SimulatorTests(unittest.TestCase):
    def test_trace_is_deterministic(self):
        a = make_trace("low", 10.0, 123)
        b = make_trace("low", 10.0, 123)
        self.assertEqual(
            [(r.arrival_time_s, r.prompt_tokens, r.output_tokens) for r in a],
            [(r.arrival_time_s, r.prompt_tokens, r.output_tokens) for r in b],
        )

    def test_one_output_token_completes_without_decode_iteration(self):
        trace = [Request(0, 0.0, 128, 1)]
        result = simulate(
            "low", "fixed256", 1,
            config=SimulationConfig(horizon_s=1.0, max_drain_s=5.0),
            predictor=FixedRuntime(), trace=trace,
        )
        self.assertEqual(result["completed"], 1)
        self.assertTrue(math.isnan(result["p99_itl_ms"]))
        self.assertEqual(result["chunk_count"], 1)

    def test_two_output_tokens_have_one_itl(self):
        trace = [Request(0, 0.0, 128, 2)]
        result = simulate(
            "low", "fixed256", 1,
            config=SimulationConfig(horizon_s=1.0, max_drain_s=5.0),
            predictor=FixedRuntime(), trace=trace,
        )
        self.assertEqual(result["completed"], 1)
        self.assertAlmostEqual(result["p99_itl_ms"], 10.0)
        self.assertAlmostEqual(result["p99_tpot_ms"], 10.0)

    def test_paired_trace_gives_same_arrival_count(self):
        trace = make_trace("medium", 10.0, 999)
        config = SimulationConfig(horizon_s=10.0, max_drain_s=60.0)
        a = simulate("medium", "fixed256", 999, config, FixedRuntime(), trace)
        b = simulate("medium", "fixed1024", 999, config, FixedRuntime(), trace)
        self.assertEqual(a["arrivals"], b["arrivals"])
        self.assertEqual(a["offered_rps"], b["offered_rps"])

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            simulate("low", "not-a-policy", 1)

    def test_priority_deadline_policy_uses_request_slack(self):
        trace = [Request(0, 0.0, 4096, 8), Request(1, 0.0, 512, 8)]
        config = SimulationConfig(horizon_s=1.0, max_drain_s=30.0)
        result = simulate("low", "priority_deadline", 1, config, FixedRuntime(), trace)
        self.assertEqual(result["arrivals"], 2)
        self.assertEqual(result["completed"], 2)


class RuntimeTests(unittest.TestCase):
    def test_synthetic_cost_is_positive_and_monotonic_in_chunk(self):
        model = SyntheticCostModel()
        values = [model.predict_ms(8, c, 4096, 10000) for c in (0, 128, 256, 512, 1024)]
        self.assertTrue(all(v > 0 for v in values))
        self.assertEqual(values, sorted(values))

    def test_profile_summary_and_lookup_predictor(self):
        rows = []
        for n in (0, 4):
            for c in (0, 256):
                for repeat in range(5):
                    rows.append({
                        "active_decode_batch": n,
                        "prefill_chunk_tokens": c,
                        "max_context_tokens": 1024,
                        "kv_cache_tokens": 0,
                        "mode": "mixed",
                        "repeat": repeat,
                        "iteration_ms": 10 + n + c / 256 + repeat * 0.01,
                    })
        summary = summarize_profile(pd.DataFrame(rows))
        self.assertIn("p99_ms", summary.columns)
        predictor = P99LookupPredictor(summary)
        self.assertGreater(predictor.predict_ms(4, 256, 1024, 0), 0)
        self.assertGreater(predictor.predict_ms(2, 128, 1024, 0), 0)

    def test_profile_csv_validation(self):
        frame = pd.DataFrame([{
            "active_decode_batch": 1,
            "prefill_chunk_tokens": 256,
            "max_context_tokens": 1024,
            "kv_cache_tokens": 0,
            "iteration_ms": 12.3,
        }])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.csv"
            frame.to_csv(path, index=False)
            loaded = load_profile_csv(path)
            self.assertEqual(loaded.iloc[0]["mode"], "mixed")

    def test_profiling_mesh_size(self):
        mesh = generate_profiling_mesh((0, 1), (0, 256), (1024,), (0,), repeats=3)
        self.assertEqual(len(mesh), 12)
        self.assertTrue(mesh["iteration_ms"].isna().all())


if __name__ == "__main__":
    unittest.main()
