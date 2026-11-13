# Alibaba — LLM Inference & Serving Research

This repository contains research notes and simulation work for single-instance LLM serving. The original V1 implementation remains on `main`; V2 development is isolated on `v2-phase1-phase2`.

## V2: Phase 1 + Phase 2

V2 focuses on simulator correctness and profiling-driven runtime modeling.

- [V2 simulation README](single-instance-simulation/README.md)
- [Modeling contract and metric definitions](single-instance-simulation/MODELING_V2.md)
- [Profiling schema and protocol](single-instance-simulation/PROFILING_V2.md)
- [V2 simulator entry point](single-instance-simulation/single_instance_sim.py)
- [Automated tests](single-instance-simulation/tests/test_v2.py)

### Data provenance

The default `SyntheticCostModel` is an illustrative analytical stress model, not measured GPU performance. No real RTX 3090 / Qwen3-32B profiling data is included. To use measured runtime data, fill a profiling CSV using the documented schema, then run with `--predictor p99_lookup` or `--predictor random_forest`.

## Existing V1 material

V1 research notes, paper summaries, and original result curves are preserved on the `main` branch. The V2 branch retains the original paper-note directory and upstream LLMServingSim submodule.

## Repository layout

```text
single-instance-simulation/
├── papers/                     # V1 paper-reading notes
├── results/                    # V1 synthetic result snapshots
├── third_party/LLMServingSim/   # upstream submodule
├── v2/
│   ├── runtime.py               # synthetic/profile-driven predictors
│   └── simulator.py             # corrected discrete-event simulator
├── scripts/generate_profiling_mesh.py
├── tests/test_v2.py
├── single_instance_sim.py       # V2 CLI
├── MODELING_V2.md
└── PROFILING_V2.md
```

Clone with submodules:

```bash
git clone --recurse-submodules --branch v2-phase1-phase2 \
  https://github.com/shawnke2025-NUS/Alibaba.git
```
