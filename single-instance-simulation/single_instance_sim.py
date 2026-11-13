#!/usr/bin/env python3
"""V2 command-line entry point; V1 is preserved on the main branch."""
from __future__ import annotations

import argparse

from v2.runtime import load_profile_csv, summarize_profile, P99LookupPredictor, RandomForestRuntimePredictor
from v2.simulator import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-instance LLM serving V2 simulator")
    parser.add_argument("--horizon", type=float, default=45.0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default="results_v2")
    parser.add_argument("--profile-csv", default=None, help="Measured profiling CSV using the documented schema")
    parser.add_argument("--predictor", choices=("synthetic", "p99_lookup", "random_forest"), default="synthetic")
    args = parser.parse_args()

    predictor = None
    if args.predictor != "synthetic":
        if not args.profile_csv:
            parser.error("--profile-csv is required when --predictor is not synthetic")
        raw_profile = load_profile_csv(args.profile_csv)
        profile_table = summarize_profile(raw_profile)
        if args.predictor == "p99_lookup":
            predictor = P99LookupPredictor(profile_table)
        else:
            predictor = RandomForestRuntimePredictor(profile_table)

    _, aggregate = run_experiments(args.horizon, args.seeds, args.out, predictor)
    columns = [
        "workload", "policy", "p99_ttft_ms", "p99_tpot_ms", "p99_itl_ms",
        "request_throughput_rps", "goodput_rps", "unfinished",
    ]
    print(aggregate[columns].to_string(index=False))
    print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
