"""
Deep Mathematical & Pure-CPU Physical Extensions (exp6):
  1. Floquet Monodromy Theory of Periodic Sawtooth Observation Age d_n = d_0 + (n mod c):
     - Audits all 256 real async_wallclock NPZ trajectories (Spatial0 = L001 & Spatial2 = L003).
     - Computes exact Floquet spectral radius rho(M(d_0, c, gamma))^(1/c) across chunk strides c in {1..8}.
     - Runs multi-core CPU MuJoCo sawtooth-delay rollouts on L001 & L003 comparing raw c=8,
       receding-horizon stride c=2/4, Floquet damping gamma=0.55, and dynamic ring-buffer calibration m_n = d_n.
  2. Wald SPRT Adaptive Probe Stopping, Nonsmooth Stiction-Endstop Paradox & Multi-Fixture Transfer (L031 + L059):
     - Evaluates hidden-resistance probing & closure-triggered unloading on second articulated fixture L059.
     - Maps the nonsmooth stiction-stall vs endstop-slam bifurcation across anticipatory taper thresholds.
     - Compares online Wald SPRT adaptive stopping tau_SPRT against fixed probe horizons Tp=1..8 across
       continuous resistance theta in [0.1, 35.0] (including critical boundary regimes).
  3. 3D Curved-Boundary Ghost-Cell Harmonic Solver (Delta_h psi(x,y,z) = 0, O(h^2)) & Yaw-Torque Grasp Basin:
     - Proves O(h^2) 3D Shortley-Weller Ghost-Cell convergence vs O(h^1) staircase on off-grid 3D spherical shells.
     - Maps 11x11 = 121 CPU MuJoCo (delta_r, delta_yaw) grasp-wrench basin & one-finger contact asymmetry on L001.
  4. Full-Corpus Causal Boundary & Failure-Mechanism Audit across all 1,040 SelfTest-1040 HDF5 trajectories.
  5. Global Temperature / Noise vs Entrance-Gated Intervention Pareto Frontier (Analytical + Countdown/MuJoCo).
"""
import collections
import concurrent.futures
import csv
import glob
import json
import math
import multiprocessing
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import h5py
import mujoco
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

PAPER_ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = PAPER_ROOT / "source_data"
SPINE_ROOT = Path("/data/guest/vla_branch_lab/VLA_SelfTest_ExpertSpine_v41_20261014")
sys.path.insert(0, str(SPINE_ROOT))
BASE = SPINE_ROOT / "code"
sys.path[:0] = [str(BASE / p) for p in ("pipeline", "planner", "physics")]
os.environ["LIBERO_CANONICAL_CATALOGUE"] = str(BASE / "planner/libero130/official130_catalogue.json")
os.environ["LIBERO_ACCEPTANCE_RULES"] = str(BASE / "planner/libero130/acceptance_rules")

from real_task_v6 import RealTaskEpisode
from source_control_targets_v28 import source_control_targets
from natural_grasp_v34 import native_grasp_reference
from validate_fixture_pair_v44 import create

_EP_CACHE = {}


def _worker_init():
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _get_ep(task_id, demo_id=0):
    key = (task_id, int(demo_id))
    if key not in _EP_CACHE:
        _EP_CACHE[key] = RealTaskEpisode(
            task_id, int(demo_id), 1.12, condition_id="C01", target_outcome="first_pass",
            planner_enabled=False, online_guard=False, record_images=False
        )
    ep = _EP_CACHE[key]
    ep.env.sim.set_state_from_flattened(ep.expert_states[0])
    ep.env.sim.forward()
    ctrl = ep.env.robots[0].controller
    ctrl.update()
    ctrl.goal_pos = ctrl.ee_pos.copy()
    ctrl.goal_ori = ctrl.ee_ori_mat.copy()
    return ep


# ===========================================================================
# MODULE 1: Floquet Monodromy Theory of Sawtooth Delay & Spatial0 + Spatial2
# ===========================================================================
def floquet_companion_matrix(d_step, d_max, gamma, alpha_c=0.2549):
    """Companion operator for eps_{n+1} = eps_n - gamma * alpha_c * eps_{n - d_step}."""
    dim = d_max + 1
    A = np.zeros((dim, dim), dtype=float)
    A[0, 0] += 1.0
    A[0, d_step] -= gamma * alpha_c
    for r in range(1, dim):
        A[r, r - 1] = 1.0
    return A


def compute_floquet_sawtooth_map(alpha_c=0.2549):
    """
    Exact Floquet per-step spectral radius rho_Floq = rho(A_{d0+c-1} ... A_{d0})^(1/c)
    for base inference latency d0 = 2 (100 ms) and chunk execution stride c in {1, 2, 3, 4, 6, 8}.
    """
    d0 = 2
    strides = [1, 2, 3, 4, 6, 8]
    gammas = np.round(np.linspace(0.10, 1.80, 69), 4)
    rows = []
    crit_summary = {}
    for c in strides:
        d_max = d0 + c - 1
        best_rho, best_g = 9.9, None
        g_crit = None
        for g in gammas:
            M = np.eye(d_max + 1)
            for j in range(c):
                M = floquet_companion_matrix(d0 + j, d_max, float(g), alpha_c) @ M
            eigs = np.linalg.eigvals(M)
            rho_step = float(np.max(np.abs(eigs)) ** (1.0 / c))
            if rho_step < best_rho:
                best_rho, best_g = rho_step, float(g)
            if g_crit is None and rho_step >= 1.0:
                g_crit = float(g)
            rows.append({
                "d0": d0,
                "chunk_stride_c": c,
                "mean_age_steps": d0 + 0.5 * (c - 1),
                "max_age_steps": d_max,
                "gamma": float(g),
                "floquet_rho_per_step": round(rho_step, 5),
            })
        crit_summary[f"c={c}"] = {
            "mean_age_ms": (d0 + 0.5 * (c - 1)) * 50.0,
            "max_age_ms": d_max * 50.0,
            "gamma_opt_floquet": best_g,
            "min_rho_per_step": round(best_rho, 4),
            "gamma_crit_floquet": g_crit,
        }
    with (DATA_OUT / "math_exp6_floquet_sawtooth.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return crit_summary


def audit_all_256_wallclock_npzs():
    pairs_csv = Path("/data/guest/vla_branch_lab/outputs/latency_tolerance/cpu_physical_skill_audit/contact_aligned_pairs.csv")
    traj_dir = Path("/data/guest/vla_branch_lab/outputs/latency_tolerance/async_wallclock/runs/trajectories")
    pairs = list(csv.DictReader(pairs_csv.open()))
    age_counts = collections.Counter()
    task_stats = {}
    for t_str, t_name in [("0", "Spatial0_L001"), ("2", "Spatial2_L003")]:
        sub = [r for r in pairs if r["task"] == t_str]
        raw_succ = sum(1 for r in sub if r["outcome"] == "both_success")
        rescued = sum(1 for r in sub if r["outcome"] == "gain_rescue")
        raw_rads, g06_rads = [], []
        raw_prog, g06_prog = [], []
        for r in sub:
            for mode, c_col, fc_col, pr_col, rad_list, pr_list in [
                ("raw", "raw_case", "raw_first_contact", "raw_after_contact32_plate_progress_mm", raw_rads, raw_prog),
                ("gain06", "gain_case", "gain_first_contact", "gain_after_contact32_plate_progress_mm", g06_rads, g06_prog),
            ]:
                npz_f = traj_dir / f"{r[c_col]}.npz"
                if npz_f.exists():
                    arr = np.load(npz_f, allow_pickle=True)
                    age_counts.update(arr["action_observation_ages"].tolist())
                    if r[fc_col]:
                        tfc = int(float(r[fc_col]))
                        bowl_xy = arr["state_history"][tfc, 10:12]
                        eef_xy = arr["eef_history"][tfc, :2]
                        rad = float(np.linalg.norm(eef_xy - bowl_xy) * 1000.0)
                        if rad < 150.0:
                            rad_list.append(rad)
                if r[pr_col]:
                    pr_list.append(float(r[pr_col]))
        task_stats[t_name] = {
            "n_pairs": len(sub),
            "raw_successes": raw_succ,
            "gain06_successes": raw_succ + rescued,
            "raw_median_contact_radius_mm": round(float(np.median(raw_rads)), 2),
            "gain06_median_contact_radius_mm": round(float(np.median(g06_rads)), 2),
            "raw_median_post32_progress_mm": round(float(np.median(raw_prog)), 2),
            "gain06_median_post32_progress_mm": round(float(np.median(g06_prog)), 2),
        }
    total_steps = sum(age_counts.values())
    mean_age = sum(k * v for k, v in age_counts.items()) / max(total_steps, 1)
    return {
        "age_histogram": dict(sorted(age_counts.items())),
        "mean_observation_age_steps": round(mean_age, 3),
        "mean_observation_age_ms": round(mean_age * 50.0, 1),
        "task_breakdown": task_stats,
    }


def _sawtooth_mujoco_job(args):
    task_id, demo_id, stride_c, gamma, dynamic_calib = args
    ep = _get_ep(task_id, demo_id)
    pos_targets, rot_targets = source_control_targets(ep)
    xyz_ref = ep.source_xyz.copy()
    _, contact_step, _ = native_grasp_reference(ep)
    n_exp = len(ep.expert_actions)

    d0 = 2
    d_max = d0 + 8
    hist = deque([ep.eef().copy() for _ in range(d_max + 2)], maxlen=d_max + 2)
    s_cursor = 0.0
    overshoot_mm = 0.0
    max_steps = 155

    for t in range(max_steps):
        age_d = d0 + (t % stride_c)
        obs_delayed = hist[-1 - age_d]
        idx = min(int(round(s_cursor)), n_exp - 1)
        # Policy at (t - age_d) emitted relative increment wanting to reach pos_targets[idx] from obs_delayed
        delta_cmd = pos_targets[idx] - obs_delayed
        if dynamic_calib:
            # Age-matched ring-buffer target reconstruction: g_t = x_{t - d_t} + delta_cmd = pos_targets[idx]
            applied_goal = obs_delayed + gamma * delta_cmd
        else:
            # Raw relative application to current displaced EEF
            applied_goal = ep.eef() + gamma * delta_cmd
        grip = float(ep.expert_actions[idx, 6])
        ep.env.step(ep.command_to(applied_goal, rot_targets[idx], grip))
        cur_eef = ep.eef().copy()
        hist.append(cur_eef)
        if idx <= contact_step:
            overshoot_mm = max(overshoot_mm, float(np.linalg.norm(cur_eef - xyz_ref[idx]) * 1000.0))
        step_adv = 1.0 if dynamic_calib else (min(1.08, 0.32 + 1.05 * gamma) if gamma < 0.95 else 1.08)
        s_cursor += step_adv
        if ep.env.check_success():
            break
    return {
        "task": task_id,
        "demo": demo_id,
        "stride_c": stride_c,
        "gamma": gamma,
        "dynamic_calib": int(dynamic_calib),
        "success": int(bool(ep.env.check_success())),
        "overshoot_mm": overshoot_mm,
    }


def run_sawtooth_mujoco_experiments():
    configs = [
        (8, 1.00, False, "sawtooth_c8_raw_g1.0"),
        (8, 0.55, False, "sawtooth_c8_damped_g0.55"),
        (4, 1.00, False, "sawtooth_c4_raw_g1.0"),
        (2, 1.00, False, "sawtooth_c2_raw_g1.0"),
        (8, 1.00, True, "sawtooth_c8_age_calibrated"),
    ]
    jobs = []
    for task_id in ("L001", "L003"):
        for c, g, cal, _ in configs:
            for dm in range(6):
                jobs.append((task_id, dm, c, g, cal))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=20, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        res = list(pool.map(_sawtooth_mujoco_job, jobs, chunksize=3))

    rows = []
    for task_id in ("L001", "L003"):
        for c, g, cal, label in configs:
            sub = [r for r in res if r["task"] == task_id and r["stride_c"] == c
                   and abs(r["gamma"] - g) < 1e-6 and r["dynamic_calib"] == int(cal)]
            rows.append({
                "task": task_id,
                "protocol": label,
                "stride_c": c,
                "gamma": g,
                "dynamic_calib": int(cal),
                "n": len(sub),
                "success_rate": round(100.0 * float(np.mean([r["success"] for r in sub])), 1),
                "mean_overshoot_mm": round(float(np.mean([r["overshoot_mm"] for r in sub])), 2),
            })
    with (DATA_OUT / "math_exp6_sawtooth_mujoco.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


# ===========================================================================
# MODULE 2: Wald SPRT Adaptive Probing, Nonsmooth Taper Stall & L059 Transfer
# ===========================================================================
def _fixture_unload_and_trace_job(args):
    task_id, demo_id, theta, mode, taper_frac = args
    ep = create(task_id, int(demo_id), float(theta))
    try:
        model, data = ep.model._model, ep.env.sim.data._data
        qidx = int(model.jnt_qposadr[ep.joint])
        source = ep.expert_states[:, 1 + qidx]
        delta = np.abs(source - source[0])
        moving = int(np.flatnonzero(delta > 0.001)[0])
        split = int(np.flatnonzero(delta > 0.010)[0])
        begin = max(0, moving - 3)
        direction = ep.source_xyz[min(moving + 8, len(ep.source_xyz) - 1)] - ep.source_xyz[moving]
        direction /= max(np.linalg.norm(direction), 1e-8)
        targets, turns = source_control_targets(ep)
        q_span = abs(float(source[-1] - source[0]))

        if mode == "probe_trace":
            eef_tr = []
            for i in range(min(begin + 10, len(ep.expert_actions))):
                if i >= begin:
                    eef_tr.append(ep.eef().copy().tolist())
                ep.execute(ep.command_to(targets[i], turns[i], float(ep.expert_actions[i, 6])), "probe")
            eef_tr.append(ep.eef().copy().tolist())
            return {"task": task_id, "demo": demo_id, "theta": theta, "eef": eef_tr}

        target_body = int(model.jnt_bodyid[ep.joint])
        fixture = {target_body}
        for b in range(target_body + 1, model.nbody):
            if int(model.body_parentid[b]) in fixture:
                fixture.add(b)
        force = np.zeros(6)
        peak = [0.0]
        imp = [0.0]
        steps = [0]
        orig_step = ep.env.sim.step

        def watched():
            orig_step()
            for j in range(data.ncon):
                c = data.contact[j]
                a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, z) or "" for z in (a, b)]
                if ((a in fixture) != (b in fixture)) and any(n.startswith(("robot0", "gripper0")) for n in names):
                    mujoco.mj_contactForce(model, data, j, force)
                    v = abs(float(force[0]))
                    peak[0] = max(peak[0], v)
                    imp[0] += v * float(model.opt.timestep)

        ep.env.sim.step = watched
        triggered = False
        for i, act in enumerate(ep.expert_actions):
            reps = 2 if i >= split else 1
            for _ in range(reps):
                steps[0] += 1
                ep.execute(ep.command_to(targets[i], turns[i], float(act[6])), "suffix")
                prog_frac = abs(float(ep.joint_pos() - source[0])) / max(q_span, 1e-6)
                if mode == "event_unload" and ep.env.check_success():
                    triggered = True
                    break
                if mode == "anticipatory_taper" and prog_frac >= taper_frac:
                    triggered = True
                    break
            if triggered:
                break

        if triggered:
            p, r = ep.eef(), ep.rot()
            for w_s in range(1, 9):
                steps[0] += 1
                ep.execute(ep.command_to(p - direction * (0.002 * w_s), r, -1.0), "withdraw")
        p, r = ep.eef(), ep.rot()
        for _ in range(12):
            steps[0] += 1
            ep.execute(ep.command_to(p, r, -1.0), "hold")
        ep.env.sim.step = orig_step
        return {
            "task": task_id,
            "demo": demo_id,
            "theta": theta,
            "mode": mode,
            "taper_frac": taper_frac,
            "success": int(bool(ep.env.check_success())),
            "peak_force_N": round(peak[0], 2),
            "impulse_Ns": round(imp[0], 3),
            "steps": steps[0],
        }
    finally:
        ep.close()


def run_sprt_and_multifixture_study():
    thetas = [0.1, 3.0, 6.0, 8.5, 10.5, 12.5, 15.0, 20.0, 25.0, 30.0]
    jobs = []
    # 1. Probe traces across continuous theta on BOTH L031 and L059
    for t_id in ("L031", "L059"):
        for th in thetas:
            jobs.append((t_id, 0, th, "probe_trace", 1.0))
    # 2. Multi-fixture hold vs event_unload across L031 (demos 0,1) & L059 (demos 0,1)
    for t_id in ("L031", "L059"):
        for dm in (0, 1):
            jobs.append((t_id, dm, 30.0, "hold_no_unload", 9.9))
            jobs.append((t_id, dm, 30.0, "event_unload", 1.0))
    # 3. Nonsmooth stiction-endstop anticipatory taper sweep on L031 demo 0
    for tf in (0.70, 0.80, 0.90, 0.96):
        jobs.append(("L031", 0, 30.0, "anticipatory_taper", tf))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=20, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        raw = list(pool.map(_fixture_unload_and_trace_job, jobs))

    traces_by_task = {}
    for t_id in ("L031", "L059"):
        tr = sorted([r for r in raw if r.get("eef") and r["task"] == t_id], key=lambda x: x["theta"])
        traces_by_task[t_id] = tr

    # Evaluate Fixed Horizon Tp in 1..8 vs Online Wald SPRT across all 10 theta levels (including critical boundary ~10.5!)
    # Critical regime boundary: theta > 10.0 requires slow 2x push (110 steps vs 83 steps; misclassifying high->fast adds 54 retry steps)
    rng = np.random.default_rng(20261113)
    sigma_mm = 0.85
    sprt_rows = []
    for t_id, tr in traces_by_task.items():
        eef_low = np.asarray(tr[0]["eef"]) * 1000.0   # theta = 0.1 reference
        eef_high = np.asarray([x for x in tr if x["theta"] == 20.0][0]["eef"]) * 1000.0  # high prototype
        n_mc = 1500

        # Fixed horizons Tp = 1..8
        for k_fix in range(1, 9):
            errs = 0
            tot_cost = 0.0
            for item in tr:
                is_high_true = (item["theta"] > 10.0)
                mu_seq = np.asarray(item["eef"])[:k_fix + 1] * 1000.0
                noise = rng.normal(0.0, sigma_mm, size=(n_mc, k_fix, 3))
                y_obs = mu_seq[None, 1:k_fix + 1, :] + noise
                d_low = np.sum((y_obs - eef_low[None, 1:k_fix + 1, :]) ** 2, axis=(1, 2))
                d_high = np.sum((y_obs - eef_high[None, 1:k_fix + 1, :]) ** 2, axis=(1, 2))
                pred_high = (d_high < d_low)
                wrong = (pred_high != is_high_true)
                errs += int(np.sum(wrong))
                extra = max(0.0, k_fix - 4) * 1.5
                if is_high_true:
                    c_arr = np.where(wrong, 83.0 + 54.0 + extra, 110.0 + extra)
                else:
                    c_arr = np.where(wrong, 110.0 + extra, 83.0 + extra)
                tot_cost += float(np.mean(c_arr))
            sprt_rows.append({
                "task": t_id,
                "policy": f"fixed_Tp_{k_fix}",
                "mean_probe_steps": float(k_fix),
                "overall_error_pct": round(100.0 * errs / (len(tr) * n_mc), 3),
                "expected_episode_steps": round(tot_cost / len(tr), 2),
            })

        # Online Wald SPRT with thresholds B in {2.5, 4.0, 6.0} (log-likelihood ratio bounds, max cap K_max = 8)
        for B_thr in (2.5, 4.5, 6.5):
            errs = 0
            tot_cost = 0.0
            probe_steps_used = []
            boundary_steps = []
            easy_steps = []
            K_max = 8
            for item in tr:
                is_high_true = (item["theta"] > 10.0)
                mu_seq = np.asarray(item["eef"])[:K_max + 1] * 1000.0
                noise = rng.normal(0.0, sigma_mm, size=(n_mc, K_max, 3))
                y_obs = mu_seq[None, 1:K_max + 1, :] + noise
                # Step-wise log-likelihood ratio increment
                llr_step = (
                    np.sum((y_obs - eef_low[None, 1:K_max + 1, :]) ** 2, axis=2)
                    - np.sum((y_obs - eef_high[None, 1:K_max + 1, :]) ** 2, axis=2)
                ) / (2.0 * sigma_mm ** 2)
                cum_llr = np.cumsum(llr_step, axis=1)
                # Do not stop during known 2-step compliance deadband (k >= 3)
                crossed = np.abs(cum_llr[:, 2:]) >= B_thr
                has_cross = np.any(crossed, axis=1)
                stop_idx = np.where(has_cross, np.argmax(crossed, axis=1) + 3, K_max)
                final_llr = cum_llr[np.arange(n_mc), stop_idx - 1]
                pred_high = (final_llr > 0.0)
                wrong = (pred_high != is_high_true)
                errs += int(np.sum(wrong))
                extra = np.maximum(0.0, stop_idx - 4.0) * 1.5
                if is_high_true:
                    c_arr = np.where(wrong, 83.0 + 54.0 + extra, 110.0 + extra)
                else:
                    c_arr = np.where(wrong, 110.0 + extra, 83.0 + extra)
                tot_cost += float(np.mean(c_arr))
                probe_steps_used.append(float(np.mean(stop_idx)))
                if 8.0 <= item["theta"] <= 13.0:
                    boundary_steps.append(float(np.mean(stop_idx)))
                elif item["theta"] in (0.1, 30.0):
                    easy_steps.append(float(np.mean(stop_idx)))
            sprt_rows.append({
                "task": t_id,
                "policy": f"wald_sprt_B{B_thr}",
                "mean_probe_steps": round(float(np.mean(probe_steps_used)), 2),
                "easy_probe_steps": round(float(np.mean(easy_steps)), 2),
                "boundary_probe_steps": round(float(np.mean(boundary_steps)), 2),
                "overall_error_pct": round(100.0 * errs / (len(tr) * n_mc), 3),
                "expected_episode_steps": round(tot_cost / len(tr), 2),
            })

    with (DATA_OUT / "math_exp6_sprt_probing.csv").open("w", newline="") as f:
        keys = ["task", "policy", "mean_probe_steps", "easy_probe_steps", "boundary_probe_steps",
                "overall_error_pct", "expected_episode_steps"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sprt_rows)

    unload_rows = [r for r in raw if "eef" not in r]
    with (DATA_OUT / "math_exp6_multifixture_unload.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(unload_rows[0].keys()))
        w.writeheader()
        w.writerows(unload_rows)
    return {"sprt": sprt_rows, "multifixture_and_taper": unload_rows}


# ===========================================================================
# MODULE 3: 3D Ghost-Cell Laplace Solver O(h^2) & Yaw-Torque Grasp Basin
# ===========================================================================
def run_3d_ghost_cell_convergence():
    """
    3D Laplace equation Delta_h psi(x,y,z) = 0 on off-grid spherical shell R1 < r < R2:
    Exact harmonic solution psi*(r) = (1/r - 1/R2) / (1/R1 - 1/R2).
    Compares 3D Staircase Dirichlet clamping (O(h^1)) vs 3D Shortley-Weller Ghost-Cell (O(h^2)).
    """
    cx, cy, cz = 0.013, -0.009, 0.011
    R1, R2 = 0.032, 0.078
    rows = []
    prev = {}

    for N in (16, 24, 32, 48, 64):
        grid = np.linspace(-0.095, 0.095, N + 1)
        h = float(grid[1] - grid[0])
        X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
        PHI_inner = R - R1
        PHI = np.minimum(R - R1, R2 - R)  # > 0 inside spherical shell
        U_ex = np.where(PHI > 0, (1.0 / np.maximum(R, 1e-6) - 1.0 / R2) / (1.0 / R1 - 1.0 / R2), 0.0)

        # 3D Level-Set central-difference normal error on interior shell
        gx = (PHI_inner[2:, 1:-1, 1:-1] - PHI_inner[:-2, 1:-1, 1:-1]) / (2.0 * h)
        gy = (PHI_inner[1:-1, 2:, 1:-1] - PHI_inner[1:-1, :-2, 1:-1]) / (2.0 * h)
        gz = (PHI_inner[1:-1, 1:-1, 2:] - PHI_inner[1:-1, 1:-1, :-2]) / (2.0 * h)
        gn = np.maximum(np.sqrt(gx ** 2 + gy ** 2 + gz ** 2), 1e-12)
        Ri = R[1:-1, 1:-1, 1:-1]
        band = (Ri > R1) & (Ri < R2)
        nx_ex = (X[1:-1, 1:-1, 1:-1] - cx) / Ri
        ny_ex = (Y[1:-1, 1:-1, 1:-1] - cy) / Ri
        nz_ex = (Z[1:-1, 1:-1, 1:-1] - cz) / Ri
        norm_l2 = float(np.sqrt(np.mean((gx[band] / gn[band] - nx_ex[band]) ** 2
                                        + (gy[band] / gn[band] - ny_ex[band]) ** 2
                                        + (gz[band] / gn[band] - nz_ex[band]) ** 2)))

        active = PHI > 0.0
        pts = np.argwhere(active)
        n_unk = len(pts)
        idx_map = -np.ones((N + 1, N + 1, N + 1), dtype=int)
        idx_map[active] = np.arange(n_unk)

        def bval_3d(px, py, pz):
            rr = math.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2)
            return 1.0 if abs(rr - R1) < abs(rr - R2) else 0.0

        row = {"N": N, "h_mm": round(h * 1000.0, 3), "unknowns_3d": n_unk, "normal_3d_l2": norm_l2}
        dirs = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

        for mode in ("staircase", "ghost"):
            r_idx, c_idx, v_arr = [], [], []
            rhs = np.zeros(n_unk, dtype=float)
            if mode == "staircase":
                for k, (i, j, m_i) in enumerate(pts):
                    r_idx.append(k)
                    c_idx.append(k)
                    v_arr.append(6.0)
                    for di, dj, dm in dirs:
                        ni, nj, nm = i + di, j + dj, m_i + dm
                        if PHI[ni, nj, nm] > 0.0:
                            r_idx.append(k)
                            c_idx.append(idx_map[ni, nj, nm])
                            v_arr.append(-1.0)
                        else:
                            rhs[k] += bval_3d(grid[ni], grid[nj], grid[nm])
            else:
                for k, (i, j, m_i) in enumerate(pts):
                    pP = float(PHI[i, j, m_i])
                    diag = 0.0
                    for (di1, dj1, dm1), (di2, dj2, dm2) in [(dirs[0], dirs[1]), (dirs[2], dirs[3]), (dirs[4], dirs[5])]:
                        pM = float(PHI[i + di1, j + dj1, m_i + dm1])
                        pPl = float(PHI[i + di2, j + dj2, m_i + dm2])
                        thM = 1.0 if pM > 0.0 else max(pP / (pP - pM), 1e-4)
                        thP = 1.0 if pPl > 0.0 else max(pP / (pP - pPl), 1e-4)
                        scale = 2.0 / (thM + thP)
                        diag += scale * (thM + thP) / (thM * thP)
                        if pM > 0.0:
                            r_idx.append(k)
                            c_idx.append(idx_map[i + di1, j + dj1, m_i + dm1])
                            v_arr.append(-scale / thM)
                        else:
                            xb = grid[i] + di1 * thM * h
                            yb = grid[j] + dj1 * thM * h
                            zb = grid[m_i] + dm1 * thM * h
                            rhs[k] += (scale / thM) * bval_3d(xb, yb, zb)
                        if pPl > 0.0:
                            r_idx.append(k)
                            c_idx.append(idx_map[i + di2, j + dj2, m_i + dm2])
                            v_arr.append(-scale / thP)
                        else:
                            xb = grid[i] + di2 * thP * h
                            yb = grid[j] + dj2 * thP * h
                            zb = grid[m_i] + dm2 * thP * h
                            rhs[k] += (scale / thP) * bval_3d(xb, yb, zb)
                    r_idx.append(k)
                    c_idx.append(k)
                    v_arr.append(diag)

            A_mat = sp.csr_matrix((v_arr, (r_idx, c_idx)), shape=(n_unk, n_unk))
            try:
                sol, _ = spla.bicgstab(A_mat, rhs, tol=1e-10, atol=1e-12)
            except TypeError:
                sol, _ = spla.bicgstab(A_mat, rhs, rtol=1e-10, atol=1e-12)
            l2 = float(np.sqrt(np.mean((sol - U_ex[active]) ** 2)))
            row[f"{mode}_3d_l2"] = l2
            row[f"{mode}_3d_order"] = (
                round(math.log(prev[mode][0] / l2) / math.log(h / prev[mode][1]) * -1.0, 3)
                if mode in prev else None
            )
            prev[mode] = (l2, h)
        rows.append(row)

    with (DATA_OUT / "math_exp6_3d_ghost_convergence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def _yaw_basin_job(args):
    dy_mm, yaw_deg = args
    ep = _get_ep("L001", 0)
    pos_targets, rot_targets = source_control_targets(ep)
    close_step, _, _ = native_grasp_reference(ep)
    init_bowl = ep.target_pos().copy()
    th = math.radians(yaw_deg)
    Rz = np.array([[math.cos(th), -math.sin(th), 0.0],
                   [math.sin(th), math.cos(th), 0.0],
                   [0.0, 0.0, 1.0]])
    offset_m = np.array([0.0, dy_mm / 1000.0, 0.0])
    first_any = None
    first_both = None
    max_lift_mm = 0.0

    for i, act in enumerate(ep.expert_actions):
        ramp = min(1.0, i / max(close_step - 6, 1))
        th_i = ramp * th
        Rz_i = np.array([[math.cos(th_i), -math.sin(th_i), 0.0],
                         [math.sin(th_i), math.cos(th_i), 0.0],
                         [0.0, 0.0, 1.0]])
        ep.env.step(ep.command_to(pos_targets[i] + ramp * offset_m, Rz_i @ rot_targets[i], float(act[6])))
        cur_bowl = ep.target_pos().copy()
        disp_mm = float(np.linalg.norm(cur_bowl - init_bowl) * 1000.0)
        max_lift_mm = max(max_lift_mm, (float(cur_bowl[2]) - float(init_bowl[2])) * 1000.0)
        if first_any is None and disp_mm > 1.0:
            first_any = i
        if first_both is None and ep.has_both_contacts():
            first_both = i

    asym_steps = (first_both - first_any) if (first_any is not None and first_both is not None) else (12 if first_any is not None else 0)
    return {
        "radial_dy_mm": round(float(dy_mm), 1),
        "yaw_deg": round(float(yaw_deg), 1),
        "success": int(bool(ep.env.check_success())),
        "max_lift_mm": round(max_lift_mm, 2),
        "one_finger_asymmetry_steps": int(max(0, asym_steps)),
    }


def run_yaw_wrench_basin():
    dys = np.linspace(-30.0, 30.0, 11)
    yaws = np.linspace(-40.0, 40.0, 11)
    jobs = [(float(dy), float(yw)) for yw in yaws for dy in dys]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=22, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        rows = list(pool.map(_yaw_basin_job, jobs, chunksize=6))
    with (DATA_OUT / "math_exp6_yaw_wrench_basin.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    zero_yaw_width = sum(r["success"] for r in rows if abs(r["yaw_deg"]) < 1e-3) * 6.0
    deg32_yaw_width = sum(r["success"] for r in rows if abs(abs(r["yaw_deg"]) - 32.0) < 1e-3) * 3.0
    return {"n_cells": len(rows), "zero_yaw_radial_span_mm": zero_yaw_width, "yaw32_radial_span_mm": deg32_yaw_width}


# ===========================================================================
# MODULE 4: Full-Corpus SelfTest-1040 Boundary & Failure-Mechanism Audit
# ===========================================================================
def _audit_one_h5(rec):
    base = Path("/data/guest/vla_branch_lab/VLA_SelfTest_Curated130_v46_20261015/outputs/curated130_exact8_final_v56")
    h5_p = base / rec["trajectory_file"]
    mask_p = base / rec["loss_mask_file"]
    mask = np.load(mask_p)
    with h5py.File(h5_p, "r") as f:
        phases = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in f["phase"][:]] if "phase" in f else ["exec"] * len(mask)
        cboth = f["observations/contact_both"][:] if ("observations" in f and "contact_both" in f["observations"]) else None
    T = len(mask)
    # Count 8-step chunks whose window [t, t+8) crosses a phase/role transition or probe completion boundary
    H = 8
    n_win = max(0, T - H + 1)
    cross_cnt = 0
    for t in range(n_win):
        w_m = mask[t:t + H]
        w_p = phases[t:t + H]
        if (np.any(w_m == 0) and np.any(w_m > 0)) or ("probe" in w_p and w_p[-1] != "probe"):
            cross_cnt += 1
    tid_num = int(rec["task_id"][1:])
    if tid_num <= 10:
        suite = "Spatial"
    elif tid_num <= 20:
        suite = "Object"
    elif tid_num <= 30:
        suite = "Goal"
    elif tid_num <= 120:
        suite = "Long-90"
    else:
        suite = "Long-10"
    return {
        "suite": suite,
        "mode": rec["mode"],
        "mechanism": rec.get("actual_failure_mechanism") or rec.get("mechanism") or "first_pass",
        "steps": T,
        "supervised_steps": int(np.sum(mask > 0)),
        "masked_setup_steps": int(np.sum(mask == 0)),
        "total_8step_windows": n_win,
        "cross_boundary_windows": cross_cnt,
        "two_finger_ratio": float(np.mean(cboth)) if cboth is not None else None,
    }


def run_full_1040_corpus_audit():
    sel_p = Path("/data/guest/vla_branch_lab/VLA_SelfTest_Curated130_v46_20261015/outputs/curated130_exact8_final_v56/selection.json")
    dsel = json.loads(sel_p.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        items = list(pool.map(_audit_one_h5, dsel))

    suite_rows = []
    for s_name in ("Spatial", "Object", "Goal", "Long-90", "Long-10"):
        sub = [x for x in items if x["suite"] == s_name]
        rec_sub = [x for x in sub if x["mode"] != "probe_success"]
        tot_w = sum(x["total_8step_windows"] for x in rec_sub)
        crs_w = sum(x["cross_boundary_windows"] for x in rec_sub)
        tf_vals = [x["two_finger_ratio"] for x in sub if x["two_finger_ratio"] is not None]
        suite_rows.append({
            "suite": s_name,
            "episodes": len(sub),
            "recovery_episodes": len(rec_sub),
            "median_steps": round(float(np.median([x["steps"] for x in sub])), 1),
            "recovery_cross_boundary_window_pct": round(100.0 * crs_w / max(tot_w, 1), 2),
            "mean_two_finger_contact_pct": round(100.0 * float(np.mean(tf_vals)), 2) if tf_vals else None,
        })
    with (DATA_OUT / "math_exp6_corpus1040_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(suite_rows[0].keys()))
        w.writeheader()
        w.writerows(suite_rows)
    return suite_rows


# ===========================================================================
# MODULE 5: Global Temperature / Noise vs Entrance-Gated Intervention Pareto
# ===========================================================================
def run_pareto_temperature_vs_entrance():
    """
    Exact L-step sequential decision model (B=4 valid entrances, L=16 continuation steps):
    Collapsed post-RL logits: entrance z_ent = [5.4, 0.2, -0.3, -0.8, -2.5] (mode 1 has ~98.6% share),
    while each continuation step l=1..L-1 has sharp correct-token margin Delta_z_step = 3.25 over V_eff = 12 distractors.
    Compare:
      (A) Global temperature scaling T in [0.7, 2.8] applied to BOTH entrance AND all L-1 continuation steps.
      (B) Entrance-gated scaling T_ent in [0.7, 4.5] applied ONLY at t=0 while keeping T_body = 1.0!
    """
    z_ent = np.array([5.4, 0.2, -0.3, -0.8, -2.5])  # index 0..3 valid modes, index 4 invalid
    u_modes = np.array([0.62, 0.54, 0.50, 0.46, 0.0])
    L_body = 15
    dz_body = 6.20
    V_distract = 12

    rows = []
    for family, T_grid in [("global_temperature", np.linspace(0.70, 2.60, 25)),
                           ("entrance_only_gate", np.linspace(0.70, 4.80, 25))]:
        for param in T_grid:
            T_e = float(param)
            T_b = float(param) if family == "global_temperature" else 1.0
            logits_e = z_ent / T_e
            logits_e -= np.max(logits_e)
            q = np.exp(logits_e) / np.sum(np.exp(logits_e))
            # Continuation survival probability across L_body sequential steps at temperature T_b
            p_step = 1.0 / (1.0 + V_distract * math.exp(-dz_body / T_b))
            w_body = p_step ** L_body
            r_modes = u_modes * w_body
            pass1 = float(np.sum(q[:4] * r_modes[:4]))
            p_any = float(np.sum(q[:4] * r_modes[:4]))
            pass64 = 1.0 - (1.0 - p_any) ** 64
            cov64 = float(np.mean([1.0 - (1.0 - q[b] * r_modes[b]) ** 64 for b in range(4)]))
            rows.append({
                "family": family,
                "temperature_param": round(float(param), 3),
                "w_continuation": round(w_body, 4),
                "pass1": round(pass1, 4),
                "pass64": round(pass64, 4),
                "cov64": round(cov64, 4),
            })
    with (DATA_OUT / "math_exp6_pareto_gate_vs_temp.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return {
        "global_peak_cov64": round(max(r["cov64"] for r in rows if r["family"] == "global_temperature"), 4),
        "entrance_peak_cov64": round(max(r["cov64"] for r in rows if r["family"] == "entrance_only_gate"), 4),
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    print("[1/5] Floquet sawtooth monodromy + 256 wall-clock audit + L001/L003 MuJoCo...", flush=True)
    floq = compute_floquet_sawtooth_map()
    wc256 = audit_all_256_wallclock_npzs()
    if (DATA_OUT / "math_exp6_sawtooth_mujoco.csv").exists():
        saw_mj = list(csv.DictReader((DATA_OUT / "math_exp6_sawtooth_mujoco.csv").open()))
    else:
        saw_mj = run_sawtooth_mujoco_experiments()
    print(f"      Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("[2/5] Wald SPRT adaptive probing + nonsmooth taper stall + L059 multi-fixture...", flush=True)
    if (DATA_OUT / "math_exp6_sprt_probing.csv").exists() and (DATA_OUT / "math_exp6_multifixture_unload.csv").exists():
        sprt_res = {
            "sprt": list(csv.DictReader((DATA_OUT / "math_exp6_sprt_probing.csv").open())),
            "multifixture_and_taper": list(csv.DictReader((DATA_OUT / "math_exp6_multifixture_unload.csv").open())),
        }
    else:
        sprt_res = run_sprt_and_multifixture_study()
    print(f"      Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("[3/5] 3D spherical Ghost-Cell Laplace O(h^2) + L001 yaw-torque grasp basin...", flush=True)
    ghost3d = run_3d_ghost_cell_convergence()
    if (DATA_OUT / "math_exp6_yaw_wrench_basin.csv").exists():
        yrows = list(csv.DictReader((DATA_OUT / "math_exp6_yaw_wrench_basin.csv").open()))
        yaw_res = {
            "n_cells": len(yrows),
            "yaw_0deg_successes_of_11": sum(int(r["success"]) for r in yrows if abs(float(r["yaw_deg"])) < 1e-3),
            "yaw_minus24deg_successes_of_11": sum(int(r["success"]) for r in yrows if abs(float(r["yaw_deg"]) + 24.0) < 1e-3),
            "yaw_minus40deg_successes_of_11": sum(int(r["success"]) for r in yrows if abs(float(r["yaw_deg"]) + 40.0) < 1e-3),
        }
    else:
        yaw_res = run_yaw_wrench_basin()
    print(f"      Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("[4/5] Full SelfTest-1040 corpus boundary & contact audit...", flush=True)
    corp = (
        list(csv.DictReader((DATA_OUT / "math_exp6_corpus1040_audit.csv").open()))
        if (DATA_OUT / "math_exp6_corpus1040_audit.csv").exists()
        else run_full_1040_corpus_audit()
    )
    print(f"      Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("[5/5] Global temperature vs entrance-gated Pareto frontier...", flush=True)
    pareto = run_pareto_temperature_vs_entrance()
    print(f"      Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    out_summary = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "floquet_sawtooth": floq,
        "wallclock_256_audit": wc256,
        "sawtooth_mujoco": saw_mj,
        "sprt_and_multifixture": sprt_res,
        "ghost_3d_convergence": ghost3d,
        "yaw_wrench_basin": yaw_res,
        "corpus_1040_audit": corp,
        "pareto_gate_vs_temp": pareto,
    }
    (DATA_OUT / "math_exp6_summary.json").write_text(json.dumps(out_summary, indent=2))
    print(json.dumps(out_summary, indent=2))
