#!/usr/bin/env python3
"""
Exp 2A: Corrected 1st-Order Projection Blend w_n* on QueST + Real First-Contact Radial Offset Audit
Compares: raw, issued, pref_align, old_blend, proj_blend across Goal/Spatial x RIPT/SFT at 0/100/200ms.
Logs first-contact radial offset r_first (mm) on LIBERO-Spatial bowls to quantify RL geometric embrittlement
and Floquet pullback inside r_crit = 52.3 mm.
"""
from __future__ import annotations
import argparse
from collections import deque
from functools import partial
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
OUT_DIR = ROOT / "outputs/v13_supplement/exp2a_quest_proj"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHA_C = 0.2549
DT = 0.050
GAMMA_STAB = {0: 1.0, 2: 1.0, 4: 0.51}

def _set_env(gpu_id: int):
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "cache/libero_config")
    os.environ["HF_HOME"] = str(ROOT / "cache/hf_home")
    os.environ["TORCH_HOME"] = str(ROOT / "cache/torch_home")
    os.environ["TMPDIR"] = str(ROOT / "cache/tmp")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache/xdg_cache")
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(ROOT / "src"))

def find_bowl_body_id(env) -> int | None:
    model = env.env.sim.model
    for name in model.body_names:
        if "bowl" in name.lower():
            return int(model.body_name2id(name))
    return None

def run_episode(env, pol, obs0, task_id: int, delay: int, horizon: int, mode: str, gamma_stab: float, p_seed: int, bowl_bid: int | None):
    from robosuite.utils.control_utils import set_goal_position, set_goal_orientation
    from vla_branch_lab.policies.quest import Sampling

    controller = env.env.robots[0].controller
    obs = obs0
    pol.reset(p_seed, Sampling(temperature=1.0, top_k=0))
    last_ori = controller.goal_ori.copy()

    ee_hist = deque(maxlen=max(1, delay) + 1)
    cmd_q = deque([np.zeros(7, np.float32) for _ in range(delay)])
    gdot_mean_q = deque([np.zeros(3, np.float64) for _ in range(delay)])
    gdot_step_q = deque([np.zeros(3, np.float64) for _ in range(delay)])

    step = 0
    first_succ = 0 if bool(env.env.check_success()) else None
    cur_fa = None
    cur_j = 0
    cur_gdot_mean = np.zeros(3, np.float64)
    first_contact_r_mm = None

    while step < horizon and first_succ is None:
        controller.update()
        ee_now = controller.ee_pos.copy()
        ee_hist.append(ee_now)

        act, chunk = pol.act(obs, task_id, step)
        if chunk is not None:
            cur_fa = np.asarray(chunk.full_actions, dtype=np.float32)
            cur_j = 0
            c_use = min(8, len(cur_fa))
            deltas = np.array([controller.scale_action(env.clip_action(cur_fa[m])[:6])[:3] for m in range(c_use)])
            cur_gdot_mean = np.mean(deltas, axis=0) / DT

        raw = env.clip_action(act.copy())
        if delay == 0 or mode == "raw":
            if delay == 0:
                obs, ok, _ = env.step(raw)
            else:
                cmd_q.append(raw)
                d_raw = cmd_q.popleft()
                obs, ok, _ = env.step(np.zeros(7, np.float32) if step < delay else d_raw)
        elif mode == "pref_align":
            if cur_fa is not None:
                idx_al = min(cur_j + delay, len(cur_fa) - 1)
                raw_al = env.clip_action(cur_fa[idx_al].copy())
            else:
                raw_al = raw
            cmd_q.append(raw_al)
            d_raw = cmd_q.popleft()
            obs, ok, _ = env.step(np.zeros(7, np.float32) if step < delay else d_raw)
        else:
            step_delta = controller.scale_action(raw[:6])[:3]
            cmd_q.append(raw)
            gdot_mean_q.append(cur_gdot_mean.copy())
            gdot_step_q.append((step_delta / DT).copy())
            d_raw = cmd_q.popleft()
            d_gdot_mean = gdot_mean_q.popleft()
            d_gdot_step = gdot_step_q.popleft()
            if step < delay:
                obs, ok, _ = env.step(np.zeros(7, np.float32))
            else:
                ee_enq = ee_hist[0]
                x_dot_hat = (ee_now - ee_enq) / (delay * DT)
                nx2 = float(np.dot(x_dot_hat, x_dot_hat))
                if mode == "issued":
                    w_star = 1.0
                elif mode == "old_blend":
                    ng2 = float(np.dot(d_gdot_mean, d_gdot_mean))
                    w_star = nx2 / (nx2 + ng2 + 1e-8)
                elif mode == "proj_blend":
                    # Exact 1st-order MSE-minimizing projection weight:
                    # w* = clip(1 - <v_obs, v_cmd_0> / (||v_obs||^2 + eps), 0, 1)
                    dot_xg = float(np.dot(x_dot_hat, d_gdot_step))
                    w_star = float(np.clip(1.0 - dot_xg / (nx2 + 1e-6), 0.0, 1.0))
                else:
                    raise ValueError(mode)

                delta = controller.scale_action(d_raw[:6])
                eff_gain = (1.0 - w_star) * gamma_stab + w_star * 1.0
                ref_pos = (1.0 - w_star) * ee_now + w_star * ee_enq
                pos = set_goal_position(eff_gain * delta[:3], ref_pos, position_limit=controller.position_limits)
                if any(not math.isclose(float(v), 0.0) for v in delta[3:]):
                    last_ori = set_goal_orientation(eff_gain * delta[3:], controller.ee_ori_mat, orientation_limit=controller.orientation_limits)
                orig = controller.set_goal
                controller.set_goal = partial(orig, set_pos=pos, set_ori=last_ori)
                try:
                    obs, ok, _ = env.step(np.r_[np.zeros(6), float(d_raw[6])])
                finally:
                    controller.set_goal = orig

        # Record first-contact radial offset to bowl center when gripper starts closing near bowl
        if first_contact_r_mm is None and bowl_bid is not None and float(raw[6]) > 0.2:
            bpos = env.env.sim.data.body_xpos[bowl_bid]
            r_xy = float(np.linalg.norm(ee_now[:2] - bpos[:2])) * 1000.0
            dz = abs(float(ee_now[2] - bpos[2])) * 1000.0
            if dz < 95.0 and r_xy < 140.0:
                first_contact_r_mm = round(r_xy, 2)

        cur_j += 1
        step += 1
        if ok and first_succ is None:
            first_succ = step

    return {"success": int(first_succ is not None), "steps": step, "first_contact_r_mm": first_contact_r_mm}

def worker_fn(gpu_id: int, jobs: list[dict], out_file: Path):
    _set_env(gpu_id)
    from vla_branch_lab.latency.resources import prepare_runtime
    from vla_branch_lab.latency.initial_states import environment_config, reset_study_state
    from vla_branch_lab.envs.libero import LiberoEnvironment
    from vla_branch_lab.policies.quest import QuestPolicy
    from vla_branch_lab.runtime import configure_torch
    from vla_branch_lab.assets import load_language

    prepare_runtime(ROOT)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    pol_cache = {}
    env = None
    cur_ek = None
    bowl_bid = None
    with out_file.open("w", encoding="utf-8") as fh:
        for idx, job in enumerate(jobs, 1):
            cfg, task, role, delay, mode = job["cfg"], job["task"], job["policy"], job["delay"], job["mode"]
            suite = task["suite"]
            suite_up = suite.removeprefix("libero_").upper()
            ecfg = environment_config(cfg, task)
            if (suite, role) not in pol_cache:
                configure_torch(ecfg, cfg["seed"])
                lang = load_language(ecfg)
                pol_cache[(suite, role)] = QuestPolicy(ecfg, str(ROOT / f"checkpoints/LIBERO_{suite_up}_{role}.pth"), lang, device="cuda:0")
            pol = pol_cache[(suite, role)]
            ek = (suite, task["task_id"])
            if cur_ek != ek:
                if env is not None:
                    env.close()
                env = LiberoEnvironment(ecfg, task["task_id"])
                cur_ek = ek
                bowl_bid = find_bowl_body_id(env) if suite == "libero_spatial" else None
            obs, _ = reset_study_state(env, task, job["state"], 0, delay)
            for _ in range(cfg["settle_steps"]):
                obs, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            p_seed = 20260925 + task["task_id"] * 100 + job["state"]["state_id"]
            res = run_episode(env, pol, obs, task["task_id"], delay, job["horizon"], mode, GAMMA_STAB[delay], p_seed, bowl_bid)
            rec = {"suite": suite, "policy": role, "task_id": task["task_id"], "state_id": job["state"]["state_id"], "delay": delay, "mode": mode, **res}
            fh.write(json.dumps(rec) + "\n")
            if idx % 25 == 0 or idx == len(jobs):
                fh.flush()
                print(f"[Exp2A GPU{gpu_id}] {idx}/{len(jobs)}", flush=True)
    if env is not None:
        env.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--states-per-task", type=int, default=20)
    args = ap.parse_args()

    _set_env(0)
    from vla_branch_lab.latency.study_config import load_study
    t0 = time.time()
    cfg = load_study(str(ROOT / "configs/latency_next/quest.yaml"))
    all_tasks = {
        "libero_spatial": [{"suite": "libero_spatial", "task_id": tid, "role": "primary"} for tid in range(10)],
        "libero_goal": [{"suite": "libero_goal", "task_id": tid, "role": "primary"} for tid in range(10)],
    }
    jobs = []
    if args.smoke:
        task = all_tasks["libero_spatial"][0]
        for mode in ("raw", "issued", "old_blend", "proj_blend"):
            jobs.append({"cfg": cfg, "task": task, "policy": "ript", "delay": 2, "mode": mode, "horizon": 60,
                         "state": {"state_origin": "benchmark", "state_id": 0, "split": "benchmark"}})
        out_f = OUT_DIR / "smoke.jsonl"
        worker_fn(0, jobs, out_f)
        rows = [json.loads(l) for l in out_f.read_text().splitlines() if l.strip()]
        print("[Exp2A SMOKE OK]", json.dumps(rows, indent=2), f"elapsed={time.time()-t0:.1f}s", flush=True)
        return

    # Full run: all 4 rows (Goal/Spatial x RIPT/SFT), 0ms raw + 100ms/200ms (raw, issued, pref_align, old_blend, proj_blend)
    for suite, tasks in all_tasks.items():
        for role in ("ript", "sft"):
            for task in tasks:
                for sid in range(args.states_per_task):
                    st = {"state_origin": "benchmark", "state_id": sid, "split": "benchmark"}
                    jobs.append({"cfg": cfg, "task": task, "policy": role, "delay": 0, "mode": "raw", "horizon": cfg["horizon"], "state": st})
                    for delay in (2, 4):
                        for mode in ("raw", "issued", "pref_align", "old_blend", "proj_blend"):
                            jobs.append({"cfg": cfg, "task": task, "policy": role, "delay": delay, "mode": mode, "horizon": cfg["horizon"], "state": st})

    n_workers = 8  # 2 workers per GPU across 4 GPUs (QueST is tiny ~20M params!)
    buckets = [jobs[i::n_workers] for i in range(n_workers)]
    ctx = mp.get_context("spawn")
    procs = []
    for wid, b in enumerate(buckets):
        p = ctx.Process(target=worker_fn, args=(wid % 4, b, OUT_DIR / f"worker_{wid}.jsonl"))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Exp2A worker failed: {p.exitcode}")

    all_rows = []
    for wid in range(n_workers):
        for l in (OUT_DIR / f"worker_{wid}.jsonl").read_text().splitlines():
            if l.strip():
                all_rows.append(json.loads(l))
    summary = {}
    for suite in ("libero_goal", "libero_spatial"):
        for role in ("ript", "sft"):
            for delay in (0, 2, 4):
                for mode in (("raw",) if delay == 0 else ("raw", "issued", "pref_align", "old_blend", "proj_blend")):
                    sub = [r for r in all_rows if r["suite"] == suite and r["policy"] == role and r["delay"] == delay and r["mode"] == mode]
                    rs = [r["first_contact_r_mm"] for r in sub if r["first_contact_r_mm"] is not None]
                    k = f"{suite}|{role}|d{delay}|{mode}"
                    summary[k] = {
                        "n": len(sub),
                        "sr_pct": round(100.0 * sum(r["success"] for r in sub) / max(1, len(sub)), 2),
                        "median_r_mm": round(float(np.median(rs)), 2) if rs else None,
                    }
    (OUT_DIR / "summary.json").write_text(json.dumps({"summary": summary, "elapsed_s": round(time.time()-t0, 1)}, indent=2))
    print("[Exp2A FULL DONE]", json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
