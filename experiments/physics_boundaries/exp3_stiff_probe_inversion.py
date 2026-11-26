"""
CPU Experiment 3: Stiff Contact Substep Integration (dt), Hidden Resistance Inversion & Probe Horizon Tp
3A. Stiff contact timestep convergence on L031 (demo 0) in CPU MuJoCo:
    Sweeps physics timestep dt in {0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0} ms at fixed 20 Hz control (50 ms),
    quantifying contact impulse quadrature error, peak force distortion, and 20 Hz control-rate aliasing.
3B. Multi-core CPU MuJoCo hidden-resistance sweep theta in [0.1, 35.0] across probe horizons Tp in 1..10 steps:
    Computes finite-difference parameter sensitivity d||x_eef||/d(theta), posterior inversion error under
    proprioceptive noise, binary regime classification accuracy, and U-shaped total execution step cost,
    plus endpoint-unloading trigger timing sweep around mechanical closure.
"""
import concurrent.futures
import csv
import json
import math
import multiprocessing
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import mujoco
import numpy as np

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


def _worker_init():
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _substep_trial(args):
    resistance, dt_ms = args
    dt = dt_ms / 1000.0
    substeps = max(1, int(round(0.050 / dt)))
    actual_dt = 0.050 / substeps

    ep = create("L031", 0, resistance)
    try:
        ep.fixture_shield = None
        model, data = ep.model._model, ep.env.sim.data._data
        qidx = int(model.jnt_qposadr[ep.joint])
        source = ep.expert_states[:, 1 + qidx]
        delta = np.abs(source - source[0])
        onset = int(np.flatnonzero(delta > 0.001)[0])
        split = int(np.flatnonzero(delta > 0.010)[0])
        begin = max(0, onset - 3)
        targets, turns = source_control_targets(ep)

        # Advance nominal 2.0 ms simulation to shared pre-probe state at `begin`
        for i in range(begin):
            cmd = ep.command_to(targets[i], turns[i], float(ep.expert_actions[i, 6]))
            ep.env.step(cmd)

        # Fix contact impedance timeconst in physical seconds (0.006 s) and set physics substep actual_dt
        model.opt.timestep = actual_dt
        for g in range(model.ngeom):
            if model.geom_solref[g, 0] > 0:
                model.geom_solref[g, 0] = max(model.geom_solref[g, 0], 0.006)
        ep.env.env.model_timestep = actual_dt
        ep.env.env.control_timestep = 0.050

        target_body = int(model.jnt_bodyid[ep.joint])
        fixture = {target_body}
        for b in range(target_body + 1, model.nbody):
            if int(model.body_parentid[b]) in fixture:
                fixture.add(b)

        force = np.zeros(6)
        probe_impulse = [0.0]
        peak_force = [0.0]
        active_substeps = [0]
        boundary_sampled_impulse = [0.0]

        def measure_contact_force():
            f_sum = 0.0
            for j in range(data.ncon):
                c = data.contact[j]
                a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, z) or "" for z in (a, b)]
                if ((a in fixture) != (b in fixture)) and any(n.startswith(("robot0", "gripper0")) for n in names):
                    mujoco.mj_contactForce(model, data, j, force)
                    f_sum += abs(float(force[0]))
            return f_sum

        unstable = False
        end_win = min(begin + 8, len(ep.expert_actions))
        for i in range(begin, end_win):
            cmd = ep.command_to(targets[i], turns[i], float(ep.expert_actions[i, 6]))
            for s_idx in range(substeps):
                mujoco.mj_step1(model, data)
                ep.env.env._pre_action(cmd, policy_step=(s_idx == 0))
                mujoco.mj_step2(model, data)
                if not np.all(np.isfinite(data.qpos)) or np.max(np.abs(data.qvel)) > 200.0:
                    unstable = True
                    break
                f_val = measure_contact_force()
                if f_val > 1e-5:
                    peak_force[0] = max(peak_force[0], f_val)
                    probe_impulse[0] += f_val * actual_dt
                    active_substeps[0] += 1
            if unstable:
                unstable = True
                break
            mujoco.mj_step1(model, data)
            f_bound = measure_contact_force()
            boundary_sampled_impulse[0] += f_bound * 0.050

        eef_end = ep.eef().copy() if not unstable else np.full(3, np.nan)
        return {
            "resistance": resistance,
            "dt_ms": round(actual_dt * 1000.0, 4),
            "substeps_per_ctrl": substeps,
            "unstable": int(unstable),
            "probe_impulse_Ns": round(probe_impulse[0], 5),
            "boundary_20Hz_impulse_Ns": round(boundary_sampled_impulse[0], 5),
            "peak_force_N": round(peak_force[0], 2),
            "probe_active_substeps": active_substeps[0],
            "end_eef_x": round(float(eef_end[0]), 6) if not unstable else None,
            "end_eef_y": round(float(eef_end[1]), 6) if not unstable else None,
            "end_eef_z": round(float(eef_end[2]), 6) if not unstable else None,
            "final_joint": round(float(ep.joint_pos()), 5) if not unstable else None,
        }
    finally:
        ep.close()


def run_substep_study():
    dts_ms = [0.125, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 25.00, 50.00]
    jobs = [(res, dt) for res in (0.1, 30.0) for dt in dts_ms]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=9, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        rows = list(pool.map(_substep_trial, jobs))

    for res in (0.1, 30.0):
        ref_row = next(r for r in rows if r["resistance"] == res and abs(r["dt_ms"] - 0.125) < 1e-3)
        j_ref = max(ref_row["probe_impulse_Ns"], 1e-8)
        p_ref = np.array([ref_row["end_eef_x"], ref_row["end_eef_y"], ref_row["end_eef_z"]])
        for r in rows:
            if r["resistance"] == res:
                if r["unstable"]:
                    r["probe_impulse_rel_err"] = None
                    r["eef_pos_err_mm"] = None
                else:
                    r["probe_impulse_rel_err"] = round(abs(r["probe_impulse_Ns"] - j_ref) / j_ref, 5)
                    p_cur = np.array([r["end_eef_x"], r["end_eef_y"], r["end_eef_z"]])
                    r["eef_pos_err_mm"] = round(float(np.linalg.norm(p_cur - p_ref) * 1000.0), 4)
                r["alias_20Hz_rel_err"] = round(abs(r["boundary_20Hz_impulse_Ns"] - j_ref) / j_ref, 5)

    with (DATA_OUT / "math_exp3_substep_convergence.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _resistance_trace_trial(theta):
    ep = create("L031", 0, float(theta))
    try:
        model = ep.model._model
        qidx = int(model.jnt_qposadr[ep.joint])
        source = ep.expert_states[:, 1 + qidx]
        delta = np.abs(source - source[0])
        onset = int(np.flatnonzero(delta > 0.001)[0])
        begin = max(0, onset - 3)  # step 40 in L031/demo0!
        targets, turns = source_control_targets(ep)
        eef_trace = []
        joint_trace = []
        for i in range(min(begin + 12, len(ep.expert_actions))):
            if i >= begin:
                eef_trace.append(ep.eef().copy().tolist())
                joint_trace.append(float(ep.joint_pos()))
            cmd = ep.command_to(targets[i], turns[i], float(ep.expert_actions[i, 6]))
            ep.execute(cmd, "probe")
        eef_trace.append(ep.eef().copy().tolist())
        joint_trace.append(float(ep.joint_pos()))
        return {"theta": float(theta), "begin_step": begin, "eef": eef_trace, "joint": joint_trace}
    finally:
        ep.close()


def _unload_timing_trial(unload_delay_steps):
    """
    Run high-resistance L031/demo0 with extended dwell (repeats=2), varying how many steps AFTER
    closure detection (unload_delay_steps) the controller switches to unloading withdrawal.
    unload_delay_steps == -1 means NO unloading (hold through full expert suffix, baseline 345.7 N / 110 steps).
    """
    ep = create("L031", 0, 30.0)
    try:
        model, data = ep.model._model, ep.env.sim.data._data
        qidx = int(model.jnt_qposadr[ep.joint])
        source = ep.expert_states[:, 1 + qidx]
        delta = np.abs(source - source[0])
        split = int(np.flatnonzero(delta > 0.010)[0])
        moving = int(np.flatnonzero(delta > 0.001)[0])
        direction = ep.source_xyz[min(moving + 8, len(ep.source_xyz) - 1)] - ep.source_xyz[moving]
        direction /= max(np.linalg.norm(direction), 1e-8)
        targets, turns = source_control_targets(ep)

        target_body = int(model.jnt_bodyid[ep.joint])
        fixture = {target_body}
        for b in range(target_body + 1, model.nbody):
            if int(model.body_parentid[b]) in fixture:
                fixture.add(b)

        force = np.zeros(6)
        peak = [0.0]
        impulse = [0.0]
        total = [0]
        orig_step = ep.env.sim.step

        def watched_step():
            orig_step()
            for j in range(data.ncon):
                c = data.contact[j]
                a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, z) or "" for z in (a, b)]
                if ((a in fixture) != (b in fixture)) and any(n.startswith(("robot0", "gripper0")) for n in names):
                    mujoco.mj_contactForce(model, data, j, force)
                    val = abs(float(force[0]))
                    peak[0] = max(peak[0], val)
                    impulse[0] += val * float(model.opt.timestep)

        ep.env.sim.step = watched_step
        first_goal = None
        stopped = False
        for i, act in enumerate(ep.expert_actions):
            for _ in range(2 if i >= split else 1):
                total[0] += 1
                ep.execute(ep.command_to(targets[i], turns[i], float(act[6])), "suffix")
                if ep.env.check_success() and first_goal is None:
                    first_goal = total[0]
                if unload_delay_steps >= 0 and first_goal is not None and (total[0] - first_goal) >= unload_delay_steps:
                    stopped = True
                    break
            if stopped:
                break

        if stopped:
            p, r = ep.eef(), ep.rot()
            for step in range(1, 9):
                total[0] += 1
                ep.execute(ep.command_to(p - direction * (0.002 * step), r, -1.0), "withdraw")
        p, r = ep.eef(), ep.rot()
        for _ in range(12):
            total[0] += 1
            ep.execute(ep.command_to(p, r, -1.0), "final_hold")
        ep.env.sim.step = orig_step
        return {
            "unload_delay_steps": unload_delay_steps,
            "mode": "hold_no_unload" if unload_delay_steps < 0 else f"unload_plus_{unload_delay_steps}",
            "first_goal_step": first_goal,
            "total_steps": total[0],
            "peak_force_N": round(peak[0], 2),
            "total_impulse_Ns": round(impulse[0], 3),
            "final_success": int(bool(ep.env.check_success())),
        }
    finally:
        ep.close()


def run_inversion_and_unload_study():
    thetas = [0.1, 2.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 35.0]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=10, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        traces = list(pool.map(_resistance_trace_trial, thetas))

    traces.sort(key=lambda x: x["theta"])
    nom_eef = np.asarray(traces[0]["eef"])  # theta = 0.1 nominal
    high_eef = np.asarray([t for t in traces if t["theta"] == 30.0][0]["eef"])

    # Evaluate probe horizon Tp in 0..10 steps (0.00..0.50 s)
    # Proprioceptive noise sigma_x = 0.85 mm per step: compute SNR, binary classification error between
    # normal (theta=0.1) and high (theta=30.0), parameter inversion sensitivity, and expected total steps!
    sigma_mm = 0.85
    from scipy.special import erfc

    horizon_rows = []
    for k in range(0, 12):
        gap_mm = float(np.linalg.norm(nom_eef[k] - high_eef[k]) * 1000.0)
        # Cumulative multi-step L2 evidence across steps 1..k
        cum_signal = float(np.sqrt(np.sum(np.linalg.norm(nom_eef[: k + 1] - high_eef[: k + 1], axis=1) ** 2)) * 1000.0)
        snr = cum_signal / (2.0 * sigma_mm)
        # Optimal likelihood-ratio classifier error P_err = 0.5 * erfc(SNR / sqrt(2))
        p_err = 0.5 * float(erfc(snr / math.sqrt(2.0))) if k > 0 else 0.5
        # Decision model matching L031 protocol:
        # Normal fast continuation after 4-step baseline takes 83 steps; extra probe dwell beyond 4 adds (k - 4) overhead,
        # whereas stopping probe early at k < 4 incurs misclassification p_err:
        #   - If high world misclassified as normal: 83-step attempt fails + requires 54-step recovery suffix!
        #   - If normal world misclassified as high: takes 110 conservative steps instead of 83 (+27 steps)!
        extra_probe = max(0, k - 4) * 1.6
        cost_normal = (1.0 - p_err) * (83.0 + extra_probe) + p_err * (110.0 + extra_probe)
        cost_high = (1.0 - p_err) * (110.0 + extra_probe) + p_err * (83.0 + 54.0 + extra_probe)
        expected_steps = 0.5 * (cost_normal + cost_high)

        # Finite-difference sensitivity d(gap_mm)/d(theta) across mid-range theta in [5, 20]
        e5 = np.asarray([t for t in traces if t["theta"] == 5.0][0]["eef"])[k]
        e20 = np.asarray([t for t in traces if t["theta"] == 20.0][0]["eef"])[k]
        sens_mm_per_N = float(np.linalg.norm(e20 - e5) * 1000.0 / 15.0)
        # Posterior entropy reduction (bits) on uniform discrete theta grid under Gaussian likelihood
        pos_k = np.array([np.asarray(t["eef"])[k] * 1000.0 for t in traces])  # (M, 3)
        # Expected conditional entropy H(Theta | Y_k) via Gauss-Hermite / Monte Carlo
        rng = np.random.default_rng(42 + k)
        M = len(thetas)
        mc_samples = 400
        ent_sum = 0.0
        for m_idx in range(M):
            y_sim = pos_k[m_idx][None, :] + rng.normal(0.0, sigma_mm, size=(mc_samples, 3))
            d2 = np.sum((y_sim[:, None, :] - pos_k[None, :, :]) ** 2, axis=2) / (2.0 * sigma_mm ** 2)
            d2 -= np.min(d2, axis=1, keepdims=True)
            post = np.exp(-d2)
            post /= np.sum(post, axis=1, keepdims=True)
            ent_sum += float(-np.mean(np.sum(post * np.log2(np.clip(post, 1e-12, 1.0)), axis=1)))
        cond_entropy_bits = ent_sum / M

        horizon_rows.append({
            "probe_steps": k,
            "probe_time_s": round(k * 0.05, 3),
            "eef_gap_30_vs_01_mm": round(gap_mm, 4),
            "sensitivity_mm_per_unit": round(sens_mm_per_N, 4),
            "misclassification_pct": round(100.0 * p_err, 3),
            "posterior_entropy_bits": round(cond_entropy_bits, 4),
            "expected_total_steps": round(expected_steps, 2),
        })

    with (DATA_OUT / "math_exp3_probe_horizon.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(horizon_rows[0].keys()))
        writer.writeheader()
        writer.writerows(horizon_rows)

    # Save full theta response curves for plotting
    theta_grid_rows = []
    for t in traces:
        th = t["theta"]
        eef_arr = np.asarray(t["eef"])
        j_arr = np.asarray(t["joint"])
        for k in range(len(eef_arr)):
            theta_grid_rows.append({
                "theta": th,
                "probe_steps": k,
                "probe_time_s": round(k * 0.05, 3),
                "eef_gap_to_nom_mm": round(float(np.linalg.norm(eef_arr[k] - nom_eef[k]) * 1000.0), 4),
                "joint_disp_mm": round(float(abs(j_arr[k] - j_arr[0]) * 1000.0), 4),
            })
    with (DATA_OUT / "math_exp3_theta_curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(theta_grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(theta_grid_rows)

    # Endpoint unloading timing sweep
    unload_delays = [0, 2, 4, 8, 14, -1]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=6, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        unload_rows = list(pool.map(_unload_timing_trial, unload_delays))
    with (DATA_OUT / "math_exp3_unload_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(unload_rows[0].keys()))
        writer.writeheader()
        writer.writerows(unload_rows)

    return horizon_rows, unload_rows


if __name__ == "__main__":
    t0 = time.perf_counter()
    sub_rows = run_substep_study()
    if (DATA_OUT / "math_exp3_probe_horizon.csv").exists() and (DATA_OUT / "math_exp3_unload_sweep.csv").exists():
        hor_rows = list(csv.DictReader((DATA_OUT / "math_exp3_probe_horizon.csv").open()))
        unl_rows = list(csv.DictReader((DATA_OUT / "math_exp3_unload_sweep.csv").open()))
    else:
        hor_rows, unl_rows = run_inversion_and_unload_study()
    summary = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "substep_study": sub_rows,
        "probe_horizon": hor_rows,
        "unload_sweep": unl_rows,
    }
    (DATA_OUT / "math_exp3_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
