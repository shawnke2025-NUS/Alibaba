#!/usr/bin/env python3
"""Generate a repeatable CSV template for real GPU runtime profiling."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2.runtime import generate_profiling_mesh


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-batches", default="0,1,2,4,8,16,32,64")
    parser.add_argument("--chunks", default="0,128,256,512,1024,2048,4096,8192")
    parser.add_argument("--contexts", default="512,2048,4096,8192,16384,32768,65536")
    parser.add_argument("--kv-cache-tokens", default="0,8192,32768,65536")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mode", choices=("prefill", "decode", "mixed"), default="mixed")
    parser.add_argument("--out", default="profiling_mesh_template.csv")
    args = parser.parse_args()
    frame = generate_profiling_mesh(
        parse_ints(args.decode_batches),
        parse_ints(args.chunks),
        parse_ints(args.contexts),
        parse_ints(args.kv_cache_tokens),
        args.repeats,
        args.mode,
    )
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} profiling rows to {args.out}")
    print("Fill iteration_ms with measured per-iteration latency; do not leave template NaNs in a training CSV.")


if __name__ == "__main__":
    main()
