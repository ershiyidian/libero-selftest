# LIBERO-SelfTest Dataset Pipeline

This directory contains the clean source scripts for constructing, executing, auditing, and annotating the **LIBERO-SelfTest** physical probing dataset.

## Modules

- **`catalog/`**: Defines the 96-task taxonomy, 10 physical self-test/recovery protocols, and 20 hidden physical perturbation conditions (`design_library.py`, `condition_library.py`, `build_catalogue.py`, `validate_catalogue.py`, `collect_sources.py`, `bind_sources.py`).
- **`pipeline/`**: Executes candidate probing and recovery trajectories in MuJoCo (`engine.py`, `run_pipeline.py`, `verify_replay.py`, `audit_dataset.py`, `render_robot.py`, `build_delivery.py`).
  - `pipeline/adapters/`: Contact-mode adapters (`actuated.py`, `holding.py`, `push.py`).
  - `pipeline/scale500/`: Batch generation, planning, gallery preview, and publishing utilities for large-scale 500+ trajectory builds.
- **`annotation/`**: Segments physical contact/probing events (`events.py`), formats structured annotation prompts (`prompts.py`, `run.py`), and exports verified SFT datasets (`sft_dataset.py`, `export_sft.py`, `verify_export.py`).

## Design Principles

1. **Physics-grounded trajectories only**: Every action sequence and object state is executed through MuJoCo physics rather than offline array splicing.
2. **Separate probing outcome from task outcome**: Failed or inconclusive physical probes are retained alongside successful recoveries so policies can learn post-contact branching.
3. **No raw media or checkpoints in Git**: Generated HDF5/Parquet/MP4 outputs are ignored by `.gitignore` and stored in your configured output directory.
