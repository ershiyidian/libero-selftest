# LIBERO-SelfTest

Official code and dataset-generation repository for:

**Boundary Fragility in Chunked Robot Manipulation and Post-Training**  
*Qiancheng Zhou* (School of Future Technology, Shanghai University)

---

## Overview

Foundation models for robot manipulation execute multi-step **action chunks** open-loop between slow policy queries. We show that chunked execution does not degrade uniformly over time; instead, failures concentrate sharply across two kinds of boundaries:

1. **Physical Contact Boundaries**: Observation lag induces periodic sawtooth overshoot that pushes curved-rim grasps outside friction cones, while initial elastic finger-pad compression creates a post-contact deadband that masks hidden joint resistance until rigid closure risks backstop slam.
2. **Decision Branching Boundaries**: Supervising an unmasked action chunk across a post-contact feedback instant collapses deterministic regression onto the majority branch, while outcome-based reinforcement post-training (PPO / GRPO) narrows multi-attempt retries by collapsing opening branch choice while leaving within-branch execution intact.

This repository provides the **LIBERO-SelfTest** dataset-generation pipeline, core policy/memory modules, and the full suite of physics-boundary, imitation-bifurcation, and post-training audit scripts.

## Repository Structure

```text
libero-selftest/
├── dataset/                         # LIBERO-SelfTest dataset construction & annotation
│   ├── catalog/                     # 96-task x 10-method x 20-condition taxonomy & source binding
│   ├── pipeline/                    # MuJoCo physical probing engine, adapters & scale-500 batch generator
│   └── annotation/                  # Event segmentation, language annotation & SFT dataset export
├── experiments/                     # Paper experiments (§III–§VI & Appendices A–C)
│   ├── physics_boundaries/          # Lagged approach, Floquet bounds, curved self-locking & probing deadband
│   ├── imitation_bifurcation/       # Cross-boundary chunk truncation, QueST / SmolVLA / VLA-Adapter audits
│   └── post_training_audit/         # 7B VLA opening-branch coverage, GRPO subgoal retry & Pass@k scaling
├── selftest/                        # Core recurrent memory, action-history, branching & GRPO modules
├── figures/                         # Publication figure reproduction script & compact summary tables
│   ├── generate_all_figures.py      # Reproduces all 7 paper figures (fig01–fig07)
│   └── source_data/                 # Lightweight CSV/JSON summaries used by generate_all_figures.py
├── patches/                         # Optional runtime compatibility patch for VLA-Adapter inference
├── tests/                           # Unit tests for memory, action history, rollout queue & GRPO math
├── requirements.txt
└── LICENSE
```

## Installation

```bash
conda create -n libero-selftest python=3.10 -y
conda activate libero-selftest
pip install -r requirements.txt
```

For MuJoCo / LIBERO headless rendering on Linux servers, set:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

External model backbones (LIBERO, VLA-Adapter, SmolVLA, QueST, OpenVLA-OFT) and checkpoints are intentionally excluded from this repository; point each experiment's CLI path arguments or environment variables to your local dataset and checkpoint directories.

## Quick Start

### 1. Reproduce All 7 Paper Figures

All seven vector figures (`fig01_boundary_teaser` through `fig07_app_countdown`) can be regenerated directly from the included summary tables:

```bash
python3 figures/generate_all_figures.py
```

Outputs are written to `figures/generated/` as both `.pdf` and `.png`.

### 2. Run Unit Tests

```bash
PYTHONPATH=. pytest -q tests/
```

### 3. Generate LIBERO-SelfTest Trajectories

See [`dataset/README.md`](dataset/README.md) for the full three-stage workflow:
- **Catalog & source coverage**: `dataset/catalog/build_catalogue.py` and `validate_catalogue.py`
- **MuJoCo physical probing generation**: `dataset/pipeline/run_pipeline.py` (or `dataset/pipeline/scale500/generate_500.py` for batch runs)
- **Event segmentation & SFT export**: `dataset/annotation/run.py` and `dataset/annotation/export_sft.py`

### 4. Run Core Paper Experiments

- **Physics & numerical boundaries (§III–§IV, App. A & C)**: scripts in `experiments/physics_boundaries/` (`exp1_delayed_approach.py` through `exp9_drawer_hold_vs_withdraw.py`).
- **Imitation & cross-boundary truncation (§V)**: scripts in `experiments/imitation_bifurcation/` (e.g., `run_bifurcation_table4.py`, `eval_causal_bifurcation_3seeds.py`, `train_smolvla_selftest.py`, `eval_smolvla_selftest.py`).
- **Post-training narrowing & retry audits (§VI, App. C)**: scripts in `experiments/post_training_audit/` (e.g., `eval_7b_blocked_corridor_libero_plus.py`, `eval_grpo_subgoal_retry.py`, `eval_quest_passk.py`).

## Citation

If you find this codebase or **LIBERO-SelfTest** useful, please cite:

```bibtex
@article{zhou2026boundary,
  title   = {Boundary Fragility in Chunked Robot Manipulation and Post-Training},
  author  = {Zhou, Qiancheng},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
