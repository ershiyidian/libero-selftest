#!/usr/bin/env python3
"""Audit exact SFT vs RIPT entrance branch narrowing (both discrete approach-skill branches and physical rim angles)
on equal sample sizes (n=64 SFT vs n=64 RIPT) and evaluate closed-loop RIPT completion when forced onto
SFT-populated branches that RIPT abandoned (<1.5% probability).
"""
from __future__ import annotations

import csv
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
sys.path.insert(0, str(ROOT / "src"))


def main():
    os.chdir(ROOT)
    from dataclasses import replace
    import torch
    from vla_branch_lab.runtime import select_gpu, configure_torch
    from vla_branch_lab.latency.resources import prepare_runtime
    from vla_branch_lab.config import load_config, Condition, Sampling
    from vla_branch_lab.assets import load_language
    from vla_branch_lab.envs.libero import LiberoEnvironment
    from vla_branch_lab.feedback.revision_conditioning import ExecutionPolicy
    from vla_branch_lab.plan import seed_for

    prepare_runtime(ROOT)
    select_gpu(0)
    base = load_config("configs/base.yaml")
    ecfg = replace(base, suite="libero_spatial", checkpoint_suite="libero_spatial",
                   horizon=300, settle_steps=10, image_size=128, control_hz=20, torch_threads=2)
    configure_torch(ecfg, 2026100501)
    lang = load_language(ecfg)

    sft = ExecutionPolicy(ecfg, str(ROOT / "checkpoints/LIBERO_SPATIAL_sft.pth"), lang, device="cuda:0")
    ript = ExecutionPolicy(ecfg, str(ROOT / "checkpoints/LIBERO_SPATIAL_ript.pth"), lang, device="cuda:0")

    # 1. Screen all 10 Spatial tasks (state 0..7) for initial entrance distribution & top branches
    suite_audit = {}
    for tid in range(10):
        env = LiberoEnvironment(replace(ecfg, task_ids=(tid,)), tid)
        sft_dists, ript_dists = [], []
        for sid in range(8):
            obs, _ = env.reset(sid, seed_for(2026100501, tid, sid, 101), Condition("nominal", delay_steps=0))
            sft_dists.append(sft.first_distribution(obs, tid))
            ript_dists.append(ript.first_distribution(obs, tid))
        env.close()
        p_sft = np.mean(sft_dists, axis=0)
        p_ript = np.mean(ript_dists, axis=0)
        h_sft = float(-np.sum(p_sft[p_sft > 1e-30] * np.log(p_sft[p_sft > 1e-30])))
        h_ript = float(-np.sum(p_ript[p_ript > 1e-30] * np.log(p_ript[p_ript > 1e-30])))
        top6_sft = np.argsort(-p_sft)[:6]
        suite_audit[str(tid)] = {
            "H_sft": round(h_sft, 3),
            "H_ript": round(h_ript, 3),
            "active_ge_2pct_sft": int(np.sum(p_sft >= 0.02)),
            "active_ge_2pct_ript": int(np.sum(p_ript >= 0.02)),
            "top_sft_tokens": [
                {"token": int(tok), "sft_pct": round(100.0 * float(p_sft[tok]), 2), "ript_pct": round(100.0 * float(p_ript[tok]), 2)}
                for tok in top6_sft
            ],
        }
        print(f"Task {tid}: H_sft={h_sft:.3f} ({suite_audit[str(tid)]['active_ge_2pct_sft']} branches>=2%) -> "
              f"H_ript={h_ript:.3f} ({suite_audit[str(tid)]['active_ge_2pct_ript']} branches>=2%)", flush=True)

    # 2. On Task 0 (Layout A) and Task 2 (Layout B), take the top 4 valid entrance skill branches ranked by SFT prior
    #    (or RIPT's #1 dominant branch + top 3 SFT-populated branches suppressed by RIPT!),
    #    verify equal-n=64 empirical sampling counts AND run closed-loop RIPT rollouts with first_override=token!
    branch_eval = {}
    csv_rows = []
    for tid, label in [(0, "Layout_A"), (2, "Layout_B")]:
        env = LiberoEnvironment(replace(ecfg, task_ids=(tid,)), tid)
        bowl_name = next(n for n in env.objects if "bowl" in n)
        sft_dists, ript_dists, init_obs = [], [], []
        for sid in range(8):
            obs, _ = env.reset(sid, seed_for(2026100501, tid, sid, 101), Condition("nominal", delay_steps=0))
            sft_dists.append(sft.first_distribution(obs, tid))
            ript_dists.append(ript.first_distribution(obs, tid))

        p_sft = np.mean(sft_dists, axis=0)
        p_ript = np.mean(ript_dists, axis=0)

        # Select 4 distinct major entrance branches:
        # Branch 1 = RIPT's dominant branch; Branches 2, 3, 4 = top 3 SFT branches where RIPT suppressed mass (< 5%)
        dom_tok = int(np.argmax(p_ript))
        suppressed_cands = [int(t) for t in np.argsort(-p_sft) if int(t) != dom_tok and p_ript[t] < 0.05]
        selected_tokens = [dom_tok] + suppressed_cands[:3]

        # Empirical n=64 sampling counts for SFT vs RIPT on exact same (8 states x 8 seeds)
        emp_counts = {"sft": {}, "ript": {}}
        for pol_name, pol_obj in [("sft", sft), ("ript", ript)]:
            sampled_first = []
            for sid in range(8):
                obs, _ = env.reset(sid, seed_for(2026100501, tid, sid, 101), Condition("nominal", delay_steps=0))
                for rep in range(8):
                    pol_obj.reset(seed_for(2026100501, tid, sid, rep, 202), Sampling(temperature=1.0, top_k=0))
                    _, chunk = pol_obj.act(obs, tid, 0)
                    sampled_first.append(int(chunk.tokens[0]))
            distinct_visited = len(set(sampled_first))
            vals, cnts = np.unique(sampled_first, return_counts=True)
            p_emp = cnts / cnts.sum()
            h_emp = float(-np.sum(p_emp * np.log(p_emp)))
            emp_counts[pol_name] = {
                "n": 64,
                "distinct_branches_visited": distinct_visited,
                "empirical_entropy_nats": round(h_emp, 3),
                "token_counts": {str(int(v)): int(c) for v, c in zip(vals, cnts)},
            }

        # Evaluate closed-loop RIPT completion when initialized with each of the 4 entrance branches (8 states x 2 repeats = 16 each, 64 total)
        branch_results = []
        for b_rank, tok in enumerate(selected_tokens, 1):
            succs = 0
            total = 0
            contact_radii = []
            for sid in range(8):
                for rep in range(2):
                    obs, _ = env.reset(sid, seed_for(2026100501, tid, sid, 101), Condition("nominal", delay_steps=0))
                    ript.reset(seed_for(2026100501, tid, sid, rep, 303), Sampling(temperature=1.0, top_k=0),
                               first_override=int(tok))
                    applied_hist = []
                    done_ok = 0
                    c_rad = float("nan")
                    for step in range(300):
                        ript.set_execution_context(applied_hist, [])
                        act, _ = ript.act(obs, tid, step)
                        clipped = env.clip_action(act)
                        obs, ok, _ = env.step(clipped)
                        applied_hist.append(clipped)
                        if math.isnan(c_rad):
                            if any(c["object"] == bowl_name and c["constraint_active"] for c in env.contacts()):
                                eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                                bpos = env.object_poses()[bowl_name][:3]
                                c_rad = float(np.linalg.norm((eef - bpos)[:2]) * 1000.0)
                        if ok:
                            done_ok = 1
                            break
                    succs += done_ok
                    total += 1
                    if not math.isnan(c_rad):
                        contact_radii.append(c_rad)
            row = {
                "task_id": tid,
                "layout": label,
                "branch_rank": b_rank,
                "skill_token": int(tok),
                "role": "dominant" if b_rank == 1 else "suppressed",
                "sft_prob_pct": round(100.0 * float(p_sft[tok]), 1),
                "ript_prob_pct": round(100.0 * float(p_ript[tok]), 1),
                "sft_emp_64": emp_counts["sft"]["token_counts"].get(str(int(tok)), 0),
                "ript_emp_64": emp_counts["ript"]["token_counts"].get(str(int(tok)), 0),
                "ript_takeover_n": total,
                "ript_takeover_succ": succs,
                "ript_takeover_pct": round(100.0 * succs / total, 1),
                "mean_contact_radius_mm": round(float(np.mean(contact_radii)), 1) if contact_radii else None,
            }
            branch_results.append(row)
            csv_rows.append(row)
            print(f"[{label}] B{b_rank} (tok {tok}, {row['role']}): SFT={row['sft_prob_pct']}% ({row['sft_emp_64']}/64) -> "
                  f"RIPT={row['ript_prob_pct']}% ({row['ript_emp_64']}/64) | RIPT takeover={succs}/{total} ({row['ript_takeover_pct']}%)", flush=True)

        branch_eval[label] = {
            "empirical_n64": emp_counts,
            "branches": branch_results,
        }
        env.close()

    paper_data = ROOT / "VLA-SelfTest/paper/source_data"
    out_json = {"suite_audit": suite_audit, "branch_eval": branch_eval}
    (paper_data / "rework_quest_branch_collapse.json").write_text(json.dumps(out_json, indent=2))
    with (paper_data / "rework_quest_branch_collapse.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print("BRANCH_COLLAPSE_DONE", flush=True)


if __name__ == "__main__":
    main()
