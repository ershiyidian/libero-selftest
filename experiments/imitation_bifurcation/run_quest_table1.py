#!/usr/bin/env python3
"""
Upgrade 1 (Table I - QueST part): Zero-threshold Closed-Form controller + External Literature Baselines:
  - ACT-TE: Temporal Ensembling (Zhao et al., RSS 2023), exponential decay m=0.01 over overlapping chunks
  - RTC-Align: Prefix-Aligned Chunk Execution (Black et al., NeurIPS 2025), skip stale prefix 0..d-1
  - Closed-Form: Zero-threshold analytical weight w_n* = ||x_dot_hat||^2 / (||x_dot_hat||^2 + ||g_dot_hat||^2 + eps)
                 + single global Floquet stability boundary gain gamma*(d, c; alpha_c=0.2549) from rho_bar <= 0.960
Evaluates all 4 QueST rows (Goal/Spatial x RIPT/SFT) at d=2 (100 ms) and d=4 (200 ms), 50 official states per cell (1,200 rollouts).
"""
from __future__ import annotations
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
DATA_OUT = ROOT / "VLA-SelfTest/paper/source_data"
DATA_OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

ALPHA_C = 0.2549
DT = 0.050


def floquet_spectral_radius(d0: int, c: int, gamma: float, alpha_c: float = ALPHA_C) -> float:
    if d0 == 0:
        return abs(1.0 - gamma * alpha_c)
    d_max = d0 + c - 1
    Phi = np.eye(d_max + 1, dtype=np.float64)
    for j in range(c):
        A = np.zeros((d_max + 1, d_max + 1), dtype=np.float64)
        A[0, 0] = 1.0
        A[0, d0 + j] -= gamma * alpha_c
        for r in range(1, d_max + 1):
            A[r, r - 1] = 1.0
        Phi = A @ Phi
    return float(np.max(np.abs(np.linalg.eigvals(Phi))) ** (1.0 / c))


def compute_optimal_floquet_gamma(d0: int, c: int = 8, rho_max: float = 0.960) -> tuple[float, float]:
    for g in np.linspace(1.0, 0.20, 161):
        r = floquet_spectral_radius(d0, c, float(g))
        if r <= rho_max:
            return round(float(g), 3), round(float(r), 4)
    return 0.50, floquet_spectral_radius(d0, c, 0.50)


def run_episode(env, pol, obs0, task_id: int, delay: int, horizon: int, mode: str, gamma_stab: float, p_seed: int):
    from robosuite.utils.control_utils import set_goal_position, set_goal_orientation
    from vla_branch_lab.policies.quest import Sampling

    controller = env.env.robots[0].controller
    obs = obs0
    pol.reset(p_seed, Sampling(temperature=1.0, top_k=0))
    last_ori = controller.goal_ori.copy()

    ee_hist = deque(maxlen=delay + 1)
    cmd_q = deque([np.zeros(7, np.float32) for _ in range(delay)])
    gdot_q = deque([np.zeros(3, np.float64) for _ in range(delay)])
    te_buf: dict[int, list[np.ndarray]] = {}
    te_m = 0.01

    step = 0
    first_succ = 0 if bool(env.env.check_success()) else None
    cur_fa = None
    cur_j = 0
    cur_gdot = np.zeros(3, np.float64)

    while step < horizon and first_succ is None:
        controller.update()
        ee_now = controller.ee_pos.copy()
        ee_hist.append(ee_now)

        act, chunk = pol.act(obs, task_id, step)
        if chunk is not None:
            cur_fa = np.asarray(chunk.full_actions, dtype=np.float32)
            cur_j = 0
            # Mean predicted position velocity over the next c=8 steps of the chunk
            c_use = min(8, len(cur_fa))
            deltas = np.array([controller.scale_action(env.clip_action(cur_fa[m])[:6])[:3] for m in range(c_use)])
            cur_gdot = np.mean(deltas, axis=0) / DT
            if mode == "act_te":
                for m in range(min(16, len(cur_fa))):
                    te_buf.setdefault(step + delay + m, []).append(env.clip_action(cur_fa[m].copy()))

        if mode == "act_te":
            if step < delay:
                a_exec = np.zeros(7, np.float32)
            else:
                preds = te_buf.pop(step, None)
                if preds:
                    ws = np.exp(-te_m * np.arange(len(preds), dtype=np.float64))
                    ws /= ws.sum()
                    a_exec = np.sum([w * p for w, p in zip(ws, preds)], axis=0).astype(np.float32)
                else:
                    a_exec = env.clip_action(act.copy())
            obs, ok, _ = env.step(a_exec)
        elif mode == "rtc_align":
            if cur_fa is not None:
                idx_al = min(cur_j + delay, len(cur_fa) - 1)
                raw = env.clip_action(cur_fa[idx_al].copy())
            else:
                raw = env.clip_action(act.copy())
            cmd_q.append(raw)
            d_raw = cmd_q.popleft()
            obs, ok, _ = env.step(np.zeros(7, np.float32) if step < delay else d_raw)
        elif mode == "closed_form":
            raw = env.clip_action(act.copy())
            cmd_q.append(raw)
            gdot_q.append(cur_gdot.copy())
            d_raw = cmd_q.popleft()
            d_gdot = gdot_q.popleft()
            if step < delay:
                obs, ok, _ = env.step(np.zeros(7, np.float32))
            else:
                ee_enq = ee_hist[0]
                x_dot_hat = (ee_now - ee_enq) / (delay * DT)
                nx2 = float(np.dot(x_dot_hat, x_dot_hat))
                ng2 = float(np.dot(d_gdot, d_gdot))
                w_star = nx2 / (nx2 + ng2 + 1e-8)

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
        cur_j += 1
        step += 1
        if ok and first_succ is None:
            first_succ = step

    return {"success": int(first_succ is not None), "steps": step}


def worker_fn(gpu_id: int, wid: int, jobs: list[dict], out_file: Path, gammas: dict[int, float]):
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    os.environ["OMP_NUM_THREADS"] = "1"

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

            obs, _ = reset_study_state(env, task, job["state"], 0, delay)
            for _ in range(cfg["settle_steps"]):
                obs, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            p_seed = 20260925 + task["task_id"] * 100 + job["state"]["state_id"]
            res = run_episode(env, pol, obs, task["task_id"], delay, cfg["horizon"], mode, gammas[delay], p_seed)
            rec = {"suite": suite, "policy": role, "task_id": task["task_id"], "state_id": job["state"]["state_id"], "delay": delay, "mode": mode, **res}
            fh.write(json.dumps(rec) + "\n")
            if idx % 10 == 0 or idx == len(jobs):
                fh.flush()
                print(f"[QueST-v12 W{wid} GPU{gpu_id}] {idx}/{len(jobs)} ok={res['success']}", flush=True)
    if env is not None:
        env.close()


def main():
    t0 = time.time()
    from vla_branch_lab.latency.study_config import load_study
    g2, r2 = compute_optimal_floquet_gamma(2, 8, 0.960)
    g4, r4 = compute_optimal_floquet_gamma(4, 8, 0.960)
    print(f"Floquet global stability gains (rho_bar <= 0.960, alpha_c={ALPHA_C}): d=2 -> gamma*={g2} (rho={r2}), d=4 -> gamma*={g4} (rho={r4})", flush=True)

    cfg = load_study(str(ROOT / "configs/latency_next/quest.yaml"))
    all_tasks = [{"suite": s, "task_id": tid, "role": "primary"} for s in ["libero_spatial", "libero_goal"] for tid in range(10)]
    jobs = []
    for task in all_tasks:
        for role in ["ript", "sft"]:
            for delay in [2, 4]:
                for mode in ["act_te", "rtc_align", "closed_form"]:
                    for sid in range(5):
                        jobs.append({
                            "cfg": cfg, "task": task, "policy": role, "delay": delay, "mode": mode,
                            "state": {"state_origin": "benchmark", "state_id": sid, "split": "benchmark"},
                        })
    print(f"Total QueST v12 jobs: {len(jobs)} across 12 workers on GPUs 1,2,3", flush=True)
    n_workers = 12
    gpus = [1, 2, 3]
    # Contiguous task-grouped partitioning so each worker builds only 1-2 LiberoEnvironments total!
    buckets = [[] for _ in range(n_workers)]
    for i, j in enumerate(jobs):
        w = (i * n_workers) // len(jobs)
        buckets[w].append(j)
    tmp_dir = DATA_OUT / "v12_quest_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    gammas = {2: g2, 4: g4}
    ctx = mp.get_context("spawn")
    procs = []
    for w in range(n_workers):
        p = ctx.Process(target=worker_fn, args=(gpus[w % len(gpus)], w, buckets[w], tmp_dir / f"w{w:02d}.jsonl", gammas))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker failed with exitcode {p.exitcode}")

    all_rows = []
    for w in range(n_workers):
        for line in (tmp_dir / f"w{w:02d}.jsonl").read_text().splitlines():
            if line.strip():
                all_rows.append(json.loads(line))
    summary = {}
    for suite in sorted({r["suite"] for r in all_rows}):
        for role in ["ript", "sft"]:
            key = f"quest_{suite}_{role}"
            summary[key] = {}
            for delay in [2, 4]:
                summary[key][f"{delay*50}ms"] = {}
                for mode in ["act_te", "rtc_align", "closed_form"]:
                    sub = [r["success"] for r in all_rows if r["suite"] == suite and r["policy"] == role and r["delay"] == delay and r["mode"] == mode]
                    summary[key][f"{delay*50}ms"][mode] = round(100.0 * float(np.mean(sub)), 1)
            print(key, summary[key], flush=True)
    (DATA_OUT / "v12_table1_quest_closedform.json").write_text(json.dumps({
        "gammas": {"100ms": g2, "200ms": g4}, "rhos": {"100ms": r2, "200ms": r4},
        "n": len(all_rows), "elapsed_s": round(time.time() - t0, 1), "summary": summary
    }, indent=2))
    print(f"DONE QueST v12 in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
