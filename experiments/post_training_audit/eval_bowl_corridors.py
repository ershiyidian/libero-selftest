#!/usr/bin/env python3
"""Equal-sample (n=64 SFT vs n=64 RIPT) real QueST reachable bowl-rim corridor audit
and real QueST RIPT closed-loop takeover from reachable corridors (including RIPT-abandoned corridors).
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

CORRIDORS = [
    ("Left-flank", -90.0, -45.0, -67.5),
    ("Front-left", -45.0, 0.0, -22.5),
    ("Front-right", 0.0, 45.0, 22.5),
    ("Right-flank", 45.0, 90.0, 67.5),
]


def get_bowl_name(env) -> str:
    for name in env.objects:
        if "bowl" in name:
            return name
    raise ValueError(f"No bowl found in {env.objects}")


def robot_facing_frame(env, bowl_pos: np.ndarray):
    model = env.env.sim.model
    data = env.env.sim.data
    base_id = model.body_name2id("robot0_base")
    base_pos = np.asarray(data.body_xpos[base_id], dtype=np.float64)[:3]
    vec_xy = (base_pos - bowl_pos)[:2]
    norm = float(np.linalg.norm(vec_xy))
    u_front = vec_xy / max(norm, 1e-8)
    u_lat = np.array([-u_front[1], u_front[0]], dtype=np.float64)
    return u_front, u_lat


def angle_to_corridor(psi_deg: float) -> int:
    if psi_deg < -45.0:
        return 0
    elif psi_deg < 0.0:
        return 1
    elif psi_deg < 45.0:
        return 2
    else:
        return 3


def bowl_contact_active(env, bowl_name: str) -> bool:
    return any(c["object"] == bowl_name and c["constraint_active"] for c in env.contacts())


def run_spontaneous_episode(env, policy, obs, task_id: int, p_seed: int, bowl_name: str, horizon: int = 300):
    from vla_branch_lab.config import Sampling
    policy.reset(p_seed, Sampling(temperature=1.0, top_k=0))
    first_dist = policy.first_distribution(obs, task_id)
    p_pos = first_dist[first_dist > 1e-30]
    token_entropy = float(-np.sum(p_pos * np.log(p_pos)))
    token_top1 = float(np.max(first_dist))

    bowl_init = env.object_poses()[bowl_name][:3].copy()
    u_front, u_lat = robot_facing_frame(env, bowl_init)

    first_contact = None
    first_success = None
    applied_history = []
    for step in range(horizon):
        policy.set_execution_context(applied_history, [])
        action, _ = policy.act(obs, task_id, step)
        clipped = env.clip_action(action)
        obs, success, _ = env.step(clipped)
        applied_history.append(clipped)
        if first_contact is None and bowl_contact_active(env, bowl_name):
            eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            bpos = env.object_poses()[bowl_name][:3]
            rel = (eef - bpos) * 1000.0
            proj_f = float(np.dot(rel[:2], u_front))
            proj_l = float(np.dot(rel[:2], u_lat))
            psi_deg = float(math.degrees(math.atan2(proj_l, proj_f)))
            rad_mm = float(np.linalg.norm(rel[:2]))
            first_contact = {
                "contact_step": step + 1,
                "psi_deg": round(psi_deg, 2),
                "in_frontal_180": int(-90.0 <= psi_deg <= 90.0),
                "corridor_idx": angle_to_corridor(psi_deg),
                "radial_mm": round(rad_mm, 2),
                "rel_x_mm": round(float(rel[0]), 2),
                "rel_y_mm": round(float(rel[1]), 2),
                "rel_z_mm": round(float(rel[2]), 2),
            }
        if success and first_success is None:
            first_success = step + 1
            break

    if first_contact is None:
        first_contact = {
            "contact_step": -1, "psi_deg": float("nan"), "in_frontal_180": 0,
            "corridor_idx": -1, "radial_mm": float("nan"),
            "rel_x_mm": float("nan"), "rel_y_mm": float("nan"), "rel_z_mm": float("nan"),
        }
    return {
        "success": int(first_success is not None),
        "first_success_step": -1 if first_success is None else first_success,
        "token_entropy_nats": round(token_entropy, 4),
        "token_top1": round(token_top1, 4),
        **first_contact,
    }


def run_guided_ript_takeover(env, policy, obs, task_id: int, p_seed: int, bowl_name: str,
                             corridor_idx: int, target_psi_deg: float, horizon: int = 300):
    from vla_branch_lab.config import Sampling
    bowl_init = env.object_poses()[bowl_name][:3].copy()
    u_front, u_lat = robot_facing_frame(env, bowl_init)
    psi_rad = math.radians(target_psi_deg)
    rim_r = 0.040
    offset_xy = rim_r * (math.cos(psi_rad) * u_front + math.sin(psi_rad) * u_lat)
    waypoint = np.array([bowl_init[0] + offset_xy[0], bowl_init[1] + offset_xy[1], bowl_init[2] + 0.058], dtype=np.float64)

    pre_contact_violation = 0
    guide_steps = 0
    for _ in range(18):
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        err = waypoint - eef
        if float(np.linalg.norm(err)) < 0.010:
            break
        cmd = np.zeros(7, dtype=np.float32)
        cmd[:3] = np.clip(err * 15.0, -1.0, 1.0)
        cmd[6] = -1.0
        obs, _, _ = env.step(env.clip_action(cmd))
        guide_steps += 1
        if bowl_contact_active(env, bowl_name):
            pre_contact_violation = 1
            break

    policy.reset(p_seed, Sampling(temperature=1.0, top_k=0))
    first_contact = None
    first_success = None
    applied_history = []
    for step in range(guide_steps, horizon):
        policy.set_execution_context(applied_history, [])
        action, _ = policy.act(obs, task_id, step)
        clipped = env.clip_action(action)
        obs, success, _ = env.step(clipped)
        applied_history.append(clipped)
        if first_contact is None and bowl_contact_active(env, bowl_name):
            eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            bpos = env.object_poses()[bowl_name][:3]
            rel = (eef - bpos) * 1000.0
            proj_f = float(np.dot(rel[:2], u_front))
            proj_l = float(np.dot(rel[:2], u_lat))
            psi_deg = float(math.degrees(math.atan2(proj_l, proj_f)))
            first_contact = {
                "contact_step": step + 1,
                "actual_psi_deg": round(psi_deg, 2),
                "actual_corridor_idx": angle_to_corridor(psi_deg),
                "radial_mm": round(float(np.linalg.norm(rel[:2])), 2),
            }
        if success and first_success is None:
            first_success = step + 1
            break

    if first_contact is None:
        first_contact = {"contact_step": -1, "actual_psi_deg": float("nan"), "actual_corridor_idx": -1, "radial_mm": float("nan")}
    return {
        "target_corridor_idx": corridor_idx,
        "target_corridor_name": CORRIDORS[corridor_idx][0],
        "target_psi_deg": target_psi_deg,
        "guide_steps": guide_steps,
        "pre_contact_violation": pre_contact_violation,
        "success": int(first_success is not None),
        "first_success_step": -1 if first_success is None else first_success,
        **first_contact,
    }


def worker_fn(gpu_id: int, jobs: list[dict], out_file: Path):
    os.chdir(ROOT)
    from vla_branch_lab.runtime import select_gpu, configure_torch
    from vla_branch_lab.latency.resources import prepare_runtime
    from vla_branch_lab.config import load_config, Condition
    from vla_branch_lab.assets import load_language
    from vla_branch_lab.envs.libero import LiberoEnvironment
    from vla_branch_lab.feedback.revision_conditioning import ExecutionPolicy
    from vla_branch_lab.plan import seed_for
    from dataclasses import replace
    import torch

    prepare_runtime(ROOT)
    select_gpu(gpu_id)
    base = load_config("configs/base.yaml")
    ecfg = replace(base, suite="libero_spatial", checkpoint_suite="libero_spatial",
                   horizon=300, settle_steps=10, image_size=128, control_hz=20, torch_threads=1)
    configure_torch(ecfg, 2026100501)
    torch.set_num_threads(1)
    lang = load_language(ecfg)

    policies = {
        "sft": ExecutionPolicy(ecfg, str(ROOT / "checkpoints/LIBERO_SPATIAL_sft.pth"), lang, device="cuda:0"),
        "ript": ExecutionPolicy(ecfg, str(ROOT / "checkpoints/LIBERO_SPATIAL_ript.pth"), lang, device="cuda:0"),
    }
    envs = {}
    rows = []
    try:
        for idx, job in enumerate(jobs, 1):
            tid = job["task_id"]
            if tid not in envs:
                envs[tid] = LiberoEnvironment(replace(ecfg, task_ids=(tid,)), tid)
            env = envs[tid]
            bowl_name = get_bowl_name(env)
            env_seed = seed_for(2026100501, tid, job["state_id"], 101)
            p_seed = seed_for(2026100501, tid, job["state_id"], job["repeat"], 202)
            obs, _ = env.reset(job["state_id"], env_seed, Condition("nominal", delay_steps=0))
            if job["kind"] == "spontaneous":
                res = run_spontaneous_episode(env, policies[job["policy"]], obs, tid, p_seed, bowl_name, horizon=300)
            else:
                cidx = job["corridor_idx"]
                res = run_guided_ript_takeover(env, policies["ript"], obs, tid, p_seed, bowl_name,
                                               cidx, CORRIDORS[cidx][3], horizon=300)
            rec = {**job, **res}
            rows.append(rec)
            if idx % 32 == 0 or idx == len(jobs):
                print(f"[GPU {gpu_id}] {idx}/{len(jobs)} kind={job['kind']} succ={res['success']}", flush=True)
    finally:
        for e in envs.values():
            e.close()
        del policies
        gc.collect()

    out_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def main():
    mp.set_start_method("spawn", force=True)
    out_dir = ROOT / "outputs/rework_quest_bowl_corridors"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for tid in (0, 2):
        for pol in ("sft", "ript"):
            for sid in range(8):
                for rep in range(8):
                    jobs.append({"kind": "spontaneous", "task_id": tid, "policy": pol, "state_id": sid, "repeat": rep})
        for cidx in range(4):
            for sid in range(8):
                for rep in range(4):
                    jobs.append({"kind": "guided_ript", "task_id": tid, "policy": "ript", "corridor_idx": cidx,
                                 "state_id": sid, "repeat": rep})

    print(f"Total QueST corridor jobs: {len(jobs)} across 4 GPUs", flush=True)
    t0 = time.time()
    procs = []
    for gpu_id in range(4):
        shard = jobs[gpu_id::4]
        p = mp.Process(target=worker_fn, args=(gpu_id, shard, out_dir / f"shard_{gpu_id}.jsonl"))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker {p} failed with {p.exitcode}")

    all_rows = []
    for gpu_id in range(4):
        for line in (out_dir / f"shard_{gpu_id}.jsonl").read_text().splitlines():
            if line.strip():
                all_rows.append(json.loads(line))

    spont = [r for r in all_rows if r["kind"] == "spontaneous"]
    guided = [r for r in all_rows if r["kind"] == "guided_ript"]

    def summarize_scope(s_rows, g_rows):
        res = {"spontaneous": {}, "guided_ript_by_corridor": {}}
        for pol in ("sft", "ript"):
            sub = [r for r in s_rows if r["policy"] == pol]
            contacted = [r for r in sub if r["contact_step"] > 0]
            counts = [sum(1 for r in contacted if r["corridor_idx"] == c) for c in range(4)]
            tot_c = max(sum(counts), 1)
            shares = [round(100.0 * c / tot_c, 1) for c in counts]
            p = np.array([c / tot_c for c in counts if c > 0], dtype=np.float64)
            ent = float(-np.sum(p * np.log(p))) if len(p) else 0.0
            res["spontaneous"][pol] = {
                "n": len(sub),
                "contacted_n": len(contacted),
                "frontal_180_pct": round(100.0 * sum(r["in_frontal_180"] for r in contacted) / max(len(contacted), 1), 1),
                "successes": sum(r["success"] for r in sub),
                "success_pct": round(100.0 * sum(r["success"] for r in sub) / len(sub), 1),
                "mean_token_entropy_nats": round(float(np.mean([r["token_entropy_nats"] for r in sub])), 3),
                "mean_token_top1": round(float(np.mean([r["token_top1"] for r in sub])), 3),
                "corridor_counts": counts,
                "corridor_shares_pct": shares,
                "corridor_entropy_nats": round(ent, 3),
                "psi_min_deg": round(float(min(r["psi_deg"] for r in contacted)), 1) if contacted else None,
                "psi_max_deg": round(float(max(r["psi_deg"] for r in contacted)), 1) if contacted else None,
                "psi_std_deg": round(float(np.std([r["psi_deg"] for r in contacted])), 1) if contacted else None,
            }
        for cidx, (cname, lo, hi, mid) in enumerate(CORRIDORS):
            gsub = [r for r in g_rows if r["target_corridor_idx"] == cidx]
            res["guided_ript_by_corridor"][cname] = {
                "corridor_idx": cidx,
                "range_deg": [lo, hi],
                "n": len(gsub),
                "successes": sum(r["success"] for r in gsub),
                "success_pct": round(100.0 * sum(r["success"] for r in gsub) / len(gsub), 1),
                "pre_contact_violations": sum(r["pre_contact_violation"] for r in gsub),
                "landed_in_target_corridor_pct": round(
                    100.0 * sum(1 for r in gsub if r["actual_corridor_idx"] == cidx) / max(len(gsub), 1), 1
                ),
            }
        return res

    summary = {
        "elapsed_s": round(time.time() - t0, 2),
        "task_0": summarize_scope([r for r in spont if r["task_id"] == 0], [r for r in guided if r["task_id"] == 0]),
        "task_2": summarize_scope([r for r in spont if r["task_id"] == 2], [r for r in guided if r["task_id"] == 2]),
        "pooled_128": summarize_scope(spont, guided),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    paper_data = ROOT / "VLA-SelfTest/paper/source_data"
    (paper_data / "rework_quest_bowl_corridors_summary.json").write_text(json.dumps(summary, indent=2))

    with (paper_data / "rework_quest_bowl_spontaneous.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["task_id", "policy", "state_id", "repeat", "success", "first_success_step",
                  "token_entropy_nats", "token_top1", "contact_step", "psi_deg", "in_frontal_180",
                  "corridor_idx", "radial_mm", "rel_x_mm", "rel_y_mm", "rel_z_mm"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in spont:
            w.writerow({k: r[k] for k in fields})

    with (paper_data / "rework_quest_bowl_guided.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["task_id", "policy", "corridor_idx", "target_corridor_name", "target_psi_deg",
                  "state_id", "repeat", "guide_steps", "pre_contact_violation",
                  "success", "first_success_step", "contact_step", "actual_psi_deg",
                  "actual_corridor_idx", "radial_mm"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in guided:
            w.writerow({k: r[k] for k in fields})

    print("QUEST_BOWL_SUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
