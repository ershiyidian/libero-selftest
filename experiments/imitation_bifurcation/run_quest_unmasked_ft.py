#!/usr/bin/env python3
"""Build unmasked (m_{t,j} = 1) full-trajectory cache (including perturbation/failure-setup prefix chunks
before cut_step=128 + post-boundary teacher recovery chunks on the exact same 18 demonstrations),
train 3 seeds (0, 1, 2) with identical optimizer hyperparameters as burst_teacher_end / burst_cropped4,
and evaluate closed-loop on the identical held-out Spatial states (8..15, 2 repeats, nominal & 100ms delay).
"""
from __future__ import annotations
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path("/data/guest/vla_branch_lab")
sys.path.insert(0, str(ROOT / "src"))

def build_unmasked_cache(masked_cache_dir: Path, burst_dir: Path, out_cache_dir: Path):
    from vla_branch_lab.continuity.cache import read_cache, write_cache
    saved, teacher_rows, anchors, states = read_cache(masked_cache_dir)
    prefixes = {
        r["prefix_id"]: r
        for r in [json.loads(x) for x in (burst_dir / "prefixes.jsonl").read_text().splitlines() if x.strip()]
    }
    unmasked_rows = []
    total_pre_chunks = 0
    total_suf_chunks = 0
    for row in teacher_rows:
        pr = prefixes[row["prefix_id"]]
        src_npz = Path(pr["source_run"]) / "trajectories" / f"{pr['source_episode']}.npz"
        with np.load(src_npz, allow_pickle=False) as z:
            keep_pre = z["chunk_steps"] < pr["cut_step"]
            pre_ctx = z["student_contexts"][keep_pre].copy()
            pre_tok = z["chunk_tokens"][keep_pre].copy()
        suf_ctx = row["context"].copy()
        suf_tok = row["tokens"].copy()
        full_ctx = np.concatenate([pre_ctx, suf_ctx], axis=0)
        full_tok = np.concatenate([pre_tok, suf_tok], axis=0)
        total_pre_chunks += len(pre_tok)
        total_suf_chunks += len(suf_tok)
        new_row = dict(row)
        new_row["context"] = full_ctx
        new_row["tokens"] = full_tok
        unmasked_rows.append(new_row)
    saved_new = dict(saved)
    saved_new["mask_mode"] = "unmasked_full_trajectory_m_eq_1"
    write_cache(out_cache_dir, saved_new, unmasked_rows, anchors, states)
    print(f"Built {out_cache_dir.name}: {len(unmasked_rows)} demos, "
          f"pre_chunks={total_pre_chunks}, suf_chunks={total_suf_chunks}, total={total_pre_chunks+total_suf_chunks}", flush=True)

def main():
    py = "/home/guest/miniconda3/envs/vla-branch/bin/python"
    train_cache = ROOT / "outputs/v8/cache_burst_spatial_unmasked_all_train"
    val_cache = ROOT / "outputs/v8/cache_burst_spatial_unmasked_all_validation"
    build_unmasked_cache(
        ROOT / "outputs/v8/cache_burst_spatial_teacher_end_train",
        ROOT / "outputs/v8/bursts_spatial_train",
        train_cache,
    )
    build_unmasked_cache(
        ROOT / "outputs/v8/cache_burst_spatial_teacher_end_validation",
        ROOT / "outputs/v8/bursts_spatial_validation",
        val_cache,
    )

    spec_path = ROOT / "configs/v8/fit_spatial_burst_unmasked_all.yaml"
    fit_dir = ROOT / "outputs/v8/fit_spatial_burst_unmasked_all"
    spec_data = {
        "train_root": "../../outputs/v8/cache_burst_spatial_unmasked_all_train",
        "validation_root": "../../outputs/v8/cache_burst_spatial_unmasked_all_validation",
        "output_dir": "outputs/v8/fit_spatial_burst_unmasked_all",
        "optimizer": {
            "updates": 200,
            "batch_size": 8,
            "learning_rate": 1.0e-5,
            "retention": 0.1,
            "weighting": "uniform",
            "grad_clip": 1.0,
            "save_every": 50,
        },
    }
    spec_path.write_text(yaml.safe_dump(spec_data, sort_keys=False), encoding="utf-8")

    # Train 3 seeds in parallel on GPUs 0, 1, 2
    procs = []
    for seed in (0, 1, 2):
        seed_out = fit_dir / f"seed_{seed:03d}" / "final.pth"
        if seed_out.is_file():
            print(f"Seed {seed} already trained.", flush=True)
            continue
        cmd = [py, "-m", "vla_branch_lab", "recovery-fit", "--spec", str(spec_path), "--gpu", str(seed), "--seed", str(seed)]
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT)))
    for p in procs:
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Training failed with rc={rc}")
    print("All 3 seeds trained cleanly.", flush=True)

    # Evaluate on GPUs 0, 1, 2 across identical held-out Spatial states (8..15, 2 repeats)
    eval_cfg = ROOT / "configs/generated_v9/eval_spatial_unmasked_all.yaml"
    eval_root = ROOT / "outputs/v8/eval_spatial_burst_unmasked_all"
    eval_cfg.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        py, "-m", "vla_branch_lab", "recovery-eval-config",
        "--template", str(ROOT / "configs/v8/eval_fixed_spatial.yaml"),
        "--fit-root", str(fit_dir),
        "--output", str(eval_cfg),
        "--evaluation-root", str(eval_root),
        "--models-only",
    ], cwd=str(ROOT), check=True)
    subprocess.run([
        py, "-m", "vla_branch_lab", "switch-run",
        "--spec", str(eval_cfg),
        "--gpus", "0", "1", "2",
    ], cwd=str(ROOT), check=True)

    # Summarize episode outcomes across seeds 0, 1, 2 at deadline 600
    ep_files = sorted((eval_root / "episodes").glob("*.json"))
    rows = [json.loads(p.read_text()) for p in ep_files]
    summary = {"total_episodes": len(rows), "by_condition": {}}
    for cond in ("delay_two", "nominal"):
        c_rows = [r for r in rows if r["condition"] == cond]
        succ_600 = sum(1 for r in c_rows if r["success"] and (r["first_task_success_step"] or 9999) <= 600)
        per_seed = {}
        for s in (0, 1, 2):
            s_rows = [r for r in c_rows if r["method"] == f"distilled_{s}"]
            s_succ = sum(1 for r in s_rows if r["success"] and (r["first_task_success_step"] or 9999) <= 600)
            per_seed[f"seed_{s}"] = {"n": len(s_rows), "succ": s_succ, "pct": round(100.0 * s_succ / len(s_rows), 2)}
        summary["by_condition"][cond] = {
            "n": len(c_rows),
            "succ_600": succ_600,
            "rate_pct": round(100.0 * succ_600 / len(c_rows), 2),
            "per_seed": per_seed,
        }
    out_json = ROOT / "VLA-SelfTest/paper/source_data/rework_quest_unmasked_ft.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print("UNMASKED_FT_SUMMARY:", json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
