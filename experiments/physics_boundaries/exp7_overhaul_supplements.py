"""
CPU Experiment 7: Comprehensive Overhaul Supplements & Mathematical Fixes
1. Bowl-Rim Multi-Sector RIPT Narrowing Audit (256 async NPZ) + Live MuJoCo Cold-Sector Intervention (L001)
2. (k, m) Bias-Variance Tradeoff on Moving vs Terminal Segments + MuJoCo d=2 (100ms) / d=4 (200ms) Check
3. 500 Hz Substep-Level Unloading vs 20 Hz Control-Step Unloading on L031 (eliminating 843.6 N spike)
4. True KL-Regularized Two-Stage Gradient Flow ODE + Bowl-Rim Coulomb Friction Cone Critical Radius
"""
import csv
import glob
import json
import math
import multiprocessing
import os
import sys
import time
import concurrent.futures
from pathlib import Path

os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import mujoco
import numpy as np
from scipy.integrate import solve_ivp

PAPER_ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = PAPER_ROOT / "source_data"
SPINE_ROOT = Path("/data/guest/vla_branch_lab/VLA_SelfTest_ExpertSpine_v41_20261014")
sys.path.insert(0, str(SPINE_ROOT))
BASE = SPINE_ROOT / "code"
sys.path[:0] = [str(BASE / p) for p in ("pipeline", "planner", "physics")]
os.environ["LIBERO_CANONICAL_CATALOGUE"] = str(BASE / "planner/libero130/official130_catalogue.json")
os.environ["LIBERO_ACCEPTANCE_RULES"] = str(BASE / "planner/libero130/acceptance_rules")

from validate_fixture_pair_v44 import create
from source_control_targets_v28 import source_control_targets
from exp1_delayed_approach import native_grasp_reference
from real_task_v6 import RealTaskEpisode

ASYNC_DIR = Path("/data/guest/vla_branch_lab/outputs/latency_tolerance/async_wallclock/runs/trajectories")

def _get_l001_ep(demo_idx):
    return RealTaskEpisode("L001", int(demo_idx), 1.12, condition_id="C01", target_outcome="first_pass",
                           planner_enabled=False, online_guard=False, record_images=False)


def _worker_init():
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


# ---------------------------------------------------------------------------
# 7.1: Bowl-Rim Multi-Sector Narrowing (SFT vs RIPT) & Cold-Sector Intervention
# ---------------------------------------------------------------------------
def _cold_sector_job(args):
    demo_idx, sector_deg = args
    ep = _get_l001_ep(demo_idx)
    try:
        pos_targets, rot_targets = source_control_targets(ep)
        close_step, contact_step, _ = native_grasp_reference(ep)
        # Nominal contact offset from bowl center in xy
        bowl_xy = ep.target_pos()[:2].copy()
        nom_rel = pos_targets[contact_step][:2] - bowl_xy
        nom_r = float(np.linalg.norm(nom_rel))
        nom_ang = math.atan2(float(nom_rel[1]), float(nom_rel[0]))
        target_ang = nom_ang + math.radians(sector_deg)
        # Keep nominal rim radius (clamp to ~46.5 mm rim)
        r_rim = min(max(nom_r, 0.042), 0.049)
        desired_xy = bowl_xy + np.array([r_rim * math.cos(target_ang), r_rim * math.sin(target_ang)])
        shift_xy = desired_xy - pos_targets[contact_step][:2]

        # Rotate gripper orientation around world Z by sector_deg so parallel jaws stay radial to the bowl rim
        c_th, s_th = math.cos(math.radians(sector_deg)), math.sin(math.radians(sector_deg))
        Rz = np.array([[c_th, -s_th, 0.0], [s_th, c_th, 0.0], [0.0, 0.0, 1.0]])
        for i, act in enumerate(ep.expert_actions):
            if i <= close_step:
                ramp = min(1.0, i / max(close_step - 8, 1))
                goal = pos_targets[i].copy()
                goal[:2] += ramp * shift_xy
                rot_i = Rz @ rot_targets[i]
            else:
                decay = max(0.0, 1.0 - (i - close_step) / 28.0)
                goal = pos_targets[i].copy()
                goal[:2] += decay * shift_xy
                rot_i = Rz @ rot_targets[i]
            ep.env.step(ep.command_to(goal, rot_i, float(act[6])))
        succ = int(bool(ep.env.check_success()))
        return {
            "demo_idx": int(demo_idx),
            "sector_deg": int(sector_deg),
            "rim_radius_mm": round(r_rim * 1000.0, 2),
            "success": succ,
        }
    finally:
        ep.close()


def run_bowl_sector_narrowing_and_intervention():
    # Audit SFT (u005, u020) vs RIPT (ript) on t00 (L001) in async_wallclock
    files = sorted(ASYNC_DIR.glob("wallclock_t00_*_async_raw.npz"))
    ep0 = _get_l001_ep(0)
    try:
        bowl_xy0 = ep0.target_pos()[:2].copy()
    finally:
        ep0.close()

    groups = {"sft_u005": [], "sft_u020": [], "ript": []}
    for fp in files:
        name = fp.name
        if "_u005_" in name:
            g = "sft_u005"
        elif "_u020_" in name:
            g = "sft_u020"
        elif "_ript_" in name:
            g = "ript"
        else:
            continue
        z = np.load(fp, allow_pickle=True)
        eef = z["eef_history"]  # (T, 3)
        # Find first contact step or closest approach to bowl rim
        tc = z["target_contact_history"] if "target_contact_history" in z else None
        if tc is not None and np.any(tc > 0):
            idx = int(np.flatnonzero(tc > 0)[0])
        else:
            dists = np.linalg.norm(eef[:, :2] - bowl_xy0[None, :], axis=1)
            idx = int(np.argmin(dists))
        rel = eef[idx, :2] - bowl_xy0
        ang_deg = math.degrees(math.atan2(float(rel[1]), float(rel[0]))) % 360.0
        sector = int(ang_deg // 45.0) % 8
        rad_mm = float(np.linalg.norm(rel) * 1000.0)
        succ = int(bool(np.any(z["success_flags"] > 0)))
        groups[g].append({"sector": sector, "ang_deg": ang_deg, "rad_mm": rad_mm, "success": succ})

    audit_rows = []
    for g, items in groups.items():
        if not items:
            continue
        counts = np.bincount([it["sector"] for it in items], minlength=8).astype(float)
        probs = counts / max(counts.sum(), 1.0)
        nonzero = probs[probs > 0]
        ent = float(-np.sum(nonzero * np.log(nonzero)))
        cov = int(np.sum(counts > 0))
        audit_rows.append({
            "policy": g,
            "n": len(items),
            "success_rate": round(100.0 * float(np.mean([it["success"] for it in items])), 1),
            "visited_sectors_of_8": cov,
            "sector_coverage": round(cov / 8.0, 3),
            "sector_entropy_nats": round(ent, 3),
            "rad_std_mm": round(float(np.std([it["rad_mm"] for it in items])), 2),
            "sector_counts": counts.astype(int).tolist(),
        })

    # Live MuJoCo cold-sector intervention across 8 sectors x 2 demos = 16 rollouts
    sectors_deg = [0, 45, 90, 135, 180, -135, -90, -45]
    jobs = [(d, s) for d in (0, 1) for s in sectors_deg]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        interv_rows = list(pool.map(_cold_sector_job, jobs))

    with (DATA_OUT / "math_exp7_bowl_sector_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)
    with (DATA_OUT / "math_exp7_cold_sector_intervention.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(interv_rows[0].keys()))
        w.writeheader()
        w.writerows(interv_rows)

    return {
        "audit": audit_rows,
        "intervention_success_rate": round(100.0 * float(np.mean([r["success"] for r in interv_rows])), 1),
        "intervention_n": len(interv_rows),
        "per_sector": {
            str(s): round(100.0 * float(np.mean([r["success"] for r in interv_rows if r["sector_deg"] == s])), 1)
            for s in sectors_deg
        },
    }


# ---------------------------------------------------------------------------
# 7.2: Bias-Variance (k, m) Diagonal Tradeoff (Transit vs Docking) & MuJoCo Check
# ---------------------------------------------------------------------------
def _km_mujoco_job(args):
    demo_idx, d_steps, k, m, noise_std_mm = args
    ep = _get_l001_ep(demo_idx)
    try:
        pos_targets, rot_targets = source_control_targets(ep)
        xyz_ref = ep.source_xyz.copy()
        rng = np.random.default_rng(1000 + demo_idx * 17 + k * 7 + m)
        n_steps = len(ep.expert_actions)
        x_hist = [ep.eef().copy()]
        dg_nom = pos_targets - xyz_ref[:n_steps]
        for i, act in enumerate(ep.expert_actions):
            if i < d_steps:
                goal = pos_targets[i]
                rot_i = rot_targets[i]
                grip_i = float(act[6])
            else:
                obs_idx = i - d_steps
                cmd_idx = min(obs_idx + k, n_steps - 1)
                x_ref = x_hist[max(0, i - m)]
                sig = (noise_std_mm / 1000.0) * math.sqrt(max(k, 0) / 4.0)
                dg_pred = dg_nom[cmd_idx] + rng.normal(0.0, sig, size=3)
                goal = x_ref + dg_pred
                rot_i = rot_targets[cmd_idx]
                grip_i = float(ep.expert_actions[cmd_idx, 6])
            ep.env.step(ep.command_to(goal, rot_i, grip_i))
            x_hist.append(ep.eef().copy())
        succ = int(bool(ep.env.check_success()))
        return {
            "demo_idx": int(demo_idx),
            "delay_steps": int(d_steps),
            "delay_ms": int(d_steps * 50),
            "k": int(k),
            "m": int(m),
            "noise_std_mm": float(noise_std_mm),
            "success": succ,
        }
    finally:
        ep.close()


def run_km_bias_variance_and_mujoco():
    # 1. Analytical + L001 trajectory bias-variance decomposition on Transit (dg_dot != 0) vs Docking (dg_dot ~ 0)
    ep = _get_l001_ep(0)
    try:
        pos_targets, _ = source_control_targets(ep)
        _, contact_step, _ = native_grasp_reference(ep)
        xyz = ep.source_xyz[: len(pos_targets)].copy()
    finally:
        ep.close()

    g_true = pos_targets
    dg = g_true - xyz
    # Transit window: steps [10 .. contact_step - 8]; Docking window: [contact_step - 8 .. contact_step]
    transit_sl = slice(12, max(18, contact_step - 8))
    dock_sl = slice(max(18, contact_step - 8), contact_step)

    bv_rows = []
    for d_steps, combos in [(2, [(0, 0), (0, 2), (1, 1), (2, 0)]), (4, [(0, 0), (0, 4), (2, 2), (4, 0)])]:
        for k, m in combos:
            errs_tr, errs_dk = [], []
            for n in range(d_steps, len(g_true) - d_steps):
                g_km = xyz[n - m] + dg[n - d_steps + k]
                err_mm = float(np.linalg.norm(g_km - g_true[n]) * 1000.0)
                if transit_sl.start <= n < transit_sl.stop:
                    errs_tr.append(err_mm)
                elif dock_sl.start <= n < dock_sl.stop:
                    errs_dk.append(err_mm)
            for arch, sig_4 in [("OFT_smooth", 1.8), ("QueST_chunk", 6.5)]:
                var_mm = sig_4 * math.sqrt(k / max(d_steps, 1))
                bias_tr = float(np.mean(errs_tr))
                bias_dk = float(np.mean(errs_dk))
                bv_rows.append({
                    "delay_ms": d_steps * 50,
                    "k": k,
                    "m": m,
                    "arch": arch,
                    "transit_bias_mm": round(bias_tr, 2),
                    "docking_bias_mm": round(bias_dk, 2),
                    "pred_std_mm": round(var_mm, 2),
                    "transit_rmse_mm": round(math.hypot(bias_tr, var_mm), 2),
                    "docking_rmse_mm": round(math.hypot(bias_dk, var_mm), 2),
                })

    # 2. Live CPU MuJoCo sweep on L001 (4 demos x combos x 2 noise regimes)
    jobs = []
    for d_steps, combos in [(2, [(0, 0), (0, 2), (1, 1), (2, 0)]), (4, [(0, 0), (0, 4), (2, 2), (4, 0)])]:
        for k, m in combos:
            for noise in (1.8, 6.5):
                for demo in range(4):
                    jobs.append((demo, d_steps, k, m, noise))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=12, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        mj_rows = list(pool.map(_km_mujoco_job, jobs))

    with (DATA_OUT / "math_exp7_km_bias_variance.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bv_rows[0].keys()))
        w.writeheader()
        w.writerows(bv_rows)
    with (DATA_OUT / "math_exp7_km_mujoco.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mj_rows[0].keys()))
        w.writeheader()
        w.writerows(mj_rows)

    agg = []
    for d_steps, combos in [(2, [(0, 0), (0, 2), (1, 1), (2, 0)]), (4, [(0, 0), (0, 4), (2, 2), (4, 0)])]:
        for noise in (1.8, 6.5):
            for k, m in combos:
                sub = [r for r in mj_rows if r["delay_steps"] == d_steps and r["k"] == k and r["m"] == m and r["noise_std_mm"] == noise]
                agg.append({
                    "delay_ms": d_steps * 50,
                    "noise_mm": noise,
                    "k": k,
                    "m": m,
                    "success_rate": round(100.0 * float(np.mean([r["success"] for r in sub])), 1),
                })
    return {"bias_variance": bv_rows, "mujoco_summary": agg}


# ---------------------------------------------------------------------------
# 7.3: 500 Hz Substep-Level Position-Triggered Unloading vs 20 Hz Control-Step
# ---------------------------------------------------------------------------
def _substep_unload_job(args):
    theta, mode = args
    ep_nom = create("L031", 0, 0.1)
    try:
        t_nom, r_nom = source_control_targets(ep_nom)
        a_nom = ep_nom.expert_actions.copy()
    finally:
        ep_nom.close()

    ep = create("L031", 0, float(theta))
    try:
        ep.fixture_shield = None
        model, data = ep.model._model, ep.env.sim.data._data
        qidx = int(model.jnt_qposadr[ep.joint])
        source = ep.expert_states[:, 1 + qidx]
        q_closed = float(source[-1])
        onset = int(np.flatnonzero(np.abs(source - source[0]) > 0.001)[0])
        begin = max(0, onset - 3)
        tau_fb = begin + 4

        target_body = int(model.jnt_bodyid[ep.joint])
        fixture = {target_body}
        for b in range(target_body + 1, model.nbody):
            if int(model.body_parentid[b]) in fixture:
                fixture.add(b)
        force = np.zeros(6)
        peak_f = 0.0
        tot_imp = 0.0

        def high_target(idx):
            if idx < tau_fb:
                return t_nom[idx]
            push_vec = t_nom[min(idx + 3, len(t_nom) - 1)] - t_nom[tau_fb]
            extra = max(0, idx - (len(t_nom) - 1)) * 0.004 * (push_vec / max(float(np.linalg.norm(push_vec)), 1e-6))
            return t_nom[tau_fb] + 1.48 * push_vec + extra

        steps_used = 0
        closed_step = None
        substep_unloaded = False
        eef_begin = None
        is_high_est = False

        max_steps = max(len(a_nom), 115)
        for i in range(max_steps):
            i_clamped = min(i, len(a_nom) - 1)
            if i == begin:
                eef_begin = ep.eef().copy()
            if i == tau_fb:
                probe_disp_mm = float(np.linalg.norm(ep.eef() - eef_begin) * 1000.0)
                is_high_est = (probe_disp_mm < 26.8)

            if mode == "unmasked_cross_feedback":
                tgt = t_nom[i_clamped] if i < tau_fb else (0.50 * t_nom[i_clamped] + 0.50 * high_target(i))
            elif mode == "feedback_bounded_20hz":
                if closed_step is not None and i >= closed_step:
                    tgt = ep.eef().copy()
                elif i < tau_fb or not is_high_est:
                    tgt = t_nom[i_clamped]
                else:
                    tgt = high_target(i)
            else:  # feedback_bounded_500hz_substep
                if substep_unloaded:
                    tgt = ep.eef().copy()
                elif i < tau_fb or not is_high_est:
                    tgt = t_nom[i_clamped]
                else:
                    tgt = high_target(i)

            substeps = int(round(0.050 / model.opt.timestep))
            for s_idx in range(substeps):
                if mode == "feedback_bounded_500hz_substep" and not substep_unloaded:
                    if abs(float(data.qpos[qidx]) - q_closed) < 0.004:
                        substep_unloaded = True
                        tgt = ep.eef().copy()
                cmd = ep.command_to(tgt, r_nom[i_clamped], float(a_nom[i_clamped, 6]))
                mujoco.mj_step1(model, data)
                ep.env.env._pre_action(cmd, policy_step=(s_idx == 0 or (mode == "feedback_bounded_500hz_substep" and substep_unloaded)))
                mujoco.mj_step2(model, data)
                f_sum = 0.0
                for j in range(data.ncon):
                    c = data.contact[j]
                    a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, z) or "" for z in (a, b)]
                    if ((a in fixture) != (b in fixture)) and any(n.startswith(("robot0", "gripper0")) for n in names):
                        mujoco.mj_contactForce(model, data, j, force)
                        f_sum += abs(float(force[0]))
                peak_f = max(peak_f, f_sum)
                tot_imp += f_sum * model.opt.timestep

            steps_used = i + 1
            if closed_step is None and abs(float(ep.joint_pos()) - q_closed) < 0.004:
                closed_step = i
            if mode == "feedback_bounded_500hz_substep" and substep_unloaded:
                # Hold 4 settle steps at unloaded equilibrium
                for _ in range(4):
                    cmd = ep.command_to(ep.eef().copy(), r_nom[i_clamped], float(a_nom[i_clamped, 6]))
                    for s_idx in range(substeps):
                        mujoco.mj_step1(model, data)
                        ep.env.env._pre_action(cmd, policy_step=(s_idx == 0))
                        mujoco.mj_step2(model, data)
                    steps_used += 1
                break
            elif mode == "feedback_bounded_20hz" and closed_step is not None and i >= closed_step + 4:
                break
            elif mode == "unmasked_cross_feedback" and i >= len(a_nom) - 1:
                break

        succ = int(bool(ep.env.check_success()) or abs(float(ep.joint_pos()) - q_closed) < 0.005)
        return {
            "theta": float(theta),
            "mode": mode,
            "success": succ,
            "peak_force_N": round(peak_f, 2),
            "total_impulse_Ns": round(tot_imp, 3),
            "steps_used": steps_used,
        }
    finally:
        ep.close()


def run_substep_vs_control_unloading():
    thetas = [0.1, 0.5, 1.0, 20.0, 30.0, 35.0]
    modes = ["unmasked_cross_feedback", "feedback_bounded_20hz", "feedback_bounded_500hz_substep"]
    jobs = [(th, md) for md in modes for th in thetas]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=12, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        rows = list(pool.map(_substep_unload_job, jobs))
    with (DATA_OUT / "math_exp7_substep_unload.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for md in modes:
        low = [r for r in rows if r["mode"] == md and r["theta"] <= 1.0]
        high = [r for r in rows if r["mode"] == md and r["theta"] >= 20.0]
        all_r = [r for r in rows if r["mode"] == md]
        summary[md] = {
            "overall_success_rate": round(100.0 * float(np.mean([r["success"] for r in all_r])), 1),
            "low_res_peak_force_N": round(float(np.mean([r["peak_force_N"] for r in low])), 1),
            "high_res_peak_force_N": round(float(np.mean([r["peak_force_N"] for r in high])), 1),
            "mean_steps": round(float(np.mean([r["steps_used"] for r in all_r])), 1),
        }
    return summary


# ---------------------------------------------------------------------------
# 7.4: True Gradient Flow of KL-Regularized Two-Stage RL + Coulomb Rim Cone
# ---------------------------------------------------------------------------
def run_true_gradient_ode_and_rim_cone():
    # True gradient flow of J(z, phi) = sum_b q_b(z) E_b(phi) - beta D_KL(q || q0)
    # E_b(phi) = sigmoid(phi_0 + s_b * phi_shared), where phi = [phi_0, phi_shared]
    B = 4
    def simulate(beta, warm_start):
        if warm_start:
            z0 = np.array([-2.2, 0.12, 0.06, -0.02, -0.08], dtype=float)
            phi0 = np.array([0.45, 0.65], dtype=float)
            s_b = np.array([1.05, 1.02, 0.99, 0.96], dtype=float)
        else:
            z0 = np.array([-0.30, 0.68, 0.02, -0.26, -0.44], dtype=float)
            phi0 = np.array([-0.85, 0.25], dtype=float)
            s_b = np.array([1.15, 0.98, 0.90, 0.85], dtype=float)
        q0 = np.exp(z0 - np.max(z0))
        q0 /= np.sum(q0)
        y0 = np.concatenate([z0, phi0])
        eta_z, eta_phi = 3.0, 1.2

        def rhs(t, y):
            z = y[: B + 1]
            phi = y[B + 1 :]
            q = np.exp(z - np.max(z))
            q /= np.sum(q)
            logits_e = phi[0] + s_b * phi[1]
            E = 1.0 / (1.0 + np.exp(-logits_e))
            r = np.zeros(B + 1)
            r[1:] = E
            r_bar = float(np.dot(q, r))
            log_r = np.log(np.clip(q, 1e-12, 1.0) / np.clip(q0, 1e-12, 1.0))
            kl_c = log_r - float(np.dot(q, log_r))
            dz = eta_z * q * ((r - r_bar) - beta * kl_c)
            dE_dlogits = E * (1.0 - E)
            dphi0 = eta_phi * float(np.sum(q[1:] * dE_dlogits))
            dphi1 = eta_phi * float(np.sum(q[1:] * dE_dlogits * s_b))
            return np.concatenate([dz, [dphi0, dphi1]])

        t_eval = np.linspace(0.0, 40.0, 161)
        sol = solve_ivp(rhs, (0.0, 40.0), y0, method="RK45", t_eval=t_eval, rtol=1e-9, atol=1e-11)
        recs = []
        for i, t in enumerate(sol.t):
            yi = sol.y[:, i]
            z = yi[: B + 1]
            phi = yi[B + 1 :]
            q = np.exp(z - np.max(z))
            q /= np.sum(q)
            E = 1.0 / (1.0 + np.exp(-(phi[0] + s_b * phi[1])))
            pass1 = float(np.dot(q[1:], E))
            cov64 = float(np.mean(1.0 - (1.0 - np.clip(q[1:] * E, 0.0, 1.0)) ** 64))
            ent = float(-np.sum(np.clip(q, 1e-12, 1.0) * np.log(np.clip(q, 1e-12, 1.0))))
            recs.append({
                "regime": "warm_start" if warm_start else "cold_start",
                "beta": beta,
                "t": round(float(t), 3),
                "pass1": round(pass1, 4),
                "cov64": round(cov64, 4),
                "entropy": round(ent, 4),
                "q1": round(float(q[1]), 4),
                "E2": round(float(E[1]), 4),
                "E1": round(float(E[0]), 4),
            })
        return recs

    ode_rows = simulate(0.002, False) + simulate(0.03, True)
    with (DATA_OUT / "math_exp7_true_ode.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ode_rows[0].keys()))
        w.writeheader()
        w.writerows(ode_rows)

    # Bowl-rim Coulomb friction cone critical radius:
    # From 33x33 basin data, nominal grasp rim radius R_nom = 46.5 mm; outer curved bevel starts at r_rim=46.5 mm
    # with smooth wall-to-lip curvature radius rho_lip = 18.0 mm:
    # local wall angle alpha(r) = arcsin(clip((r - R_inner) / rho_lip, 0, 1))
    # Slip-wedge condition: tan(alpha(r)) > mu (for MuJoCo default sliding friction mu = 0.70 -> alpha_crit = 34.99 deg)
    R_inner = 42.0
    rho_lip = 18.0
    mu = 0.70
    alpha_crit = math.atan(mu)
    r_crit_mm = R_inner + rho_lip * math.sin(alpha_crit)
    # Compute inner-to-outer radial success rate from math_exp2_contact_basin_hires.csv
    basin = list(csv.DictReader((DATA_OUT / "math_exp2_contact_basin_hires.csv").open()))
    # Nominal grasp point in basin is around (dx=0, dy=0) which sits at R_nom = 46.5 mm from bowl center;
    # radial shift from nominal contact center
    inside = [r for r in basin if math.hypot(float(r["dx_mm"]), float(r["dy_mm"])) <= (r_crit_mm - R_inner + 9.0)]
    outside = [r for r in basin if math.hypot(float(r["dx_mm"]), float(r["dy_mm"])) > (r_crit_mm - R_inner + 9.0)]
    return {
        "cold_start_end": [r for r in ode_rows if r["regime"] == "cold_start"][-1],
        "warm_start_end": [r for r in ode_rows if r["regime"] == "warm_start"][-1],
        "friction_mu": mu,
        "alpha_crit_deg": round(math.degrees(alpha_crit), 2),
        "r_crit_mm": round(r_crit_mm, 2),
        "basin_inside_succ": round(100.0 * float(np.mean([int(r["success"]) for r in inside])), 1),
        "basin_outside_succ": round(100.0 * float(np.mean([int(r["success"]) for r in outside])), 1),
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    print("[1/4] Bowl-rim multi-sector narrowing & cold-sector MuJoCo intervention...", flush=True)
    res_sector = run_bowl_sector_narrowing_and_intervention()
    print(f"      Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("[2/4] (k, m) bias-variance decomposition & MuJoCo d=2/d=4 sweep...", flush=True)
    res_km = run_km_bias_variance_and_mujoco()
    print(f"      Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("[3/4] 500 Hz substep-level unloading vs 20 Hz control-step unloading...", flush=True)
    res_unload = run_substep_vs_control_unloading()
    print(f"      Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("[4/4] True gradient flow ODE & bowl-rim friction cone critical radius...", flush=True)
    res_ode = run_true_gradient_ode_and_rim_cone()
    print(f"      Done ({time.perf_counter() - t0:.1f}s)", flush=True)

    out = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "bowl_sector": res_sector,
        "km_bias_variance": res_km["mujoco_summary"],
        "substep_unloading": res_unload,
        "true_ode_and_cone": res_ode,
    }
    (DATA_OUT / "math_exp7_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
