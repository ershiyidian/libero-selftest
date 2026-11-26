#!/usr/bin/env python3
"""Evaluate multi-sample pass@k (k in 1, 2, 4, 8, 16) on real QueST across the 16 held-out
LIBERO-Spatial evaluation states (tasks 0 and 2, state_ids 8..15) at 100 ms delay (delay_steps=2),
comparing:
  1. QueST-RIPT control (collapsed entrance prior + RIPT closed-loop execution)
  2. Entrance-reopened QueST-RIPT (SFT opening skill prior z_0 ~ p_SFT + RIPT closed-loop execution)
  3. Causal-masked recovery-FT QueST (distilled_0 from burst_teacher_end)
Each condition runs N = 16 independent closed-loop rollouts per state (16 states x 16 = 256 rollouts/method, 768 total).
"""
from __future__ import annotations
import csv
import gc
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
sys.path.insert(0, str(ROOT / "src"))

def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def worker_fn(gpu_id: int, method_name: str, jobs: list[tuple[int, int, int]], out_path: Path):
    os.chdir(ROOT)
    from dataclasses import replace
    import torch
    from vla_branch_lab.runtime import select_gpu, configure_torch
    from vla_branch_lab.latency.resources import prepare_runtime
    from vla_branch_lab.config import load_config, Condition, Sampling
    from vla_branch_lab.assets import load_language
    from vla_branch_lab.envs.libero import LiberoEnvironment
    from vla_branch_lab.envs.perturbations import ActionDelay
    from vla_branch_lab.branching.core import execute
    from vla_branch_lab.policies.quest import QuestPolicy
    from vla_branch_lab.plan import seed_for

    prepare_runtime(ROOT)
    select_gpu(gpu_id)
    base = load_config("configs/base.yaml")
    ecfg = replace(base, suite="libero_spatial", checkpoint_suite="libero_spatial",
                   horizon=600, settle_steps=10, image_size=128, control_hz=20, torch_threads=1)
    configure_torch(ecfg, 5719)
    torch.set_num_threads(1)
    lang = load_language(ecfg)

    if method_name == "ript_ctrl":
        pol = QuestPolicy(ecfg, str(ROOT / "checkpoints/LIBERO_SPATIAL_ript.pth"), lang, device="cuda:0")
    elif method_name == "sft_entry_ript":
        pol = QuestPolicy(ecfg, str(ROOT / "checkpoints/LIBERO_SPATIAL_ript.pth"), lang, device="cuda:0")
        sft_donor = QuestPolicy(ecfg, str(ROOT / "checkpoints/LIBERO_SPATIAL_sft.pth"), lang, device="cuda:0")
        pol.entry_policy = sft_donor
    elif method_name == "masked_ft_ript":
        pol = QuestPolicy(ecfg, str(ROOT / "outputs/v8/fit_spatial_burst_teacher_end/seed_000/final.pth"), lang, device="cuda:0")
    else:
        raise ValueError(method_name)

    env = None
    cur_tid = None
    rows = []
    for idx, (tid, sid, rep) in enumerate(jobs, 1):
        if tid != cur_tid:
            if env is not None:
                env.close()
            env = LiberoEnvironment(replace(ecfg, task_ids=(tid,)), tid)
            cur_tid = tid
        env_seed = seed_for(5719, tid, sid, 0, 101)  # fixed initial state per (tid, sid) across all k=1..16 samples!
        pol_seed = seed_for(5719, tid, sid, rep, 909)
        obs, _ = env.reset(sid, env_seed, Condition("delay_two", delay_steps=2))
        queue = ActionDelay(2)
        pol.reset(pol_seed, Sampling(temperature=1.0, top_k=0))
        ok_done = 0
        first_step = None
        first_tok = None
        for step in range(600):
            act, chunk = pol.act(obs, tid, step)
            if chunk is not None and first_tok is None:
                first_tok = int(chunk.tokens[0])
            obs, ok, _, _, _ = execute(env, queue, act)
            if ok:
                ok_done = 1
                first_step = step + 1
                break
        rows.append({
            "method": method_name,
            "task_id": tid,
            "state_id": sid,
            "sample_id": rep,
            "success": ok_done,
            "first_success_step": first_step,
            "opening_token": first_tok,
        })
        if idx % 32 == 0 or idx == len(jobs):
            print(f"[GPU {gpu_id} | {method_name}] {idx}/{len(jobs)} succ_so_far={sum(r['success'] for r in rows)}", flush=True)
    if env is not None:
        env.close()
    out_path.write_text(json.dumps(rows))

def main():
    mp.set_start_method("spawn", force=True)
    all_jobs = [(tid, sid, rep) for tid in (0, 2) for sid in range(8, 16) for rep in range(16)]
    assert len(all_jobs) == 256
    tmp_dir = Path("/tmp/quest_passk_workers")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        (0, "ript_ctrl", all_jobs, tmp_dir / "ript_ctrl.json"),
        (1, "sft_entry_ript", all_jobs, tmp_dir / "sft_entry_ript.json"),
        (2, "masked_ft_ript", all_jobs, tmp_dir / "masked_ft_ript.json"),
    ]
    procs = [mp.Process(target=worker_fn, args=sp) for sp in specs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker failed: {p.exitcode}")

    all_rows = []
    for _, _, _, pth in specs:
        all_rows.extend(json.loads(pth.read_text()))

    ks = [1, 2, 4, 8, 16]
    summary = {"n_states": 16, "samples_per_state": 16, "delay_ms": 100, "methods": {}}
    csv_rows = []
    for m_name in ("ript_ctrl", "sft_entry_ript", "masked_ft_ript"):
        m_rows = [r for r in all_rows if r["method"] == m_name]
        state_succs = []
        distinct_openings = []
        for tid in (0, 2):
            for sid in range(8, 16):
                sr = [r for r in m_rows if r["task_id"] == tid and r["state_id"] == sid]
                assert len(sr) == 16
                state_succs.append(sum(int(r["success"]) for r in sr))
                distinct_openings.append(len({r["opening_token"] for r in sr}))
        pass_k = {}
        for k in ks:
            val = float(np.mean([unbiased_pass_at_k(16, c, k) for c in state_succs]))
            pass_k[f"pass@{k}"] = round(val, 4)
            csv_rows.append({
                "method": m_name,
                "k": k,
                "pass_at_k": round(val, 4),
                "mean_distinct_openings_16": round(float(np.mean(distinct_openings)), 2),
            })
        summary["methods"][m_name] = {
            "pass_at_k": pass_k,
            "mean_distinct_openings_per_state": round(float(np.mean(distinct_openings)), 2),
            "state_success_counts_out_of_16": state_succs,
        }
    paper_data = ROOT / "VLA-SelfTest/paper/source_data"
    (paper_data / "rework_quest_passk.json").write_text(json.dumps(summary, indent=2))
    with (paper_data / "rework_quest_passk.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "k", "pass_at_k", "mean_distinct_openings_16"])
        w.writeheader()
        w.writerows(csv_rows)
    print("QUEST_PASSK_SUMMARY:", json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
