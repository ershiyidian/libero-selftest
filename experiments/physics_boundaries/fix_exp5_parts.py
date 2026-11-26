"""
Fast (~6s) exact refinement of:
  - math_exp5_curvature_km.csv: Causal observation-time chunk extrapolation from t_n - d*dt
    (no oracle future knowledge after t_n - d*dt), comparing (0,0), (2,2), (4,0), (0,4).
  - math_exp5_unmasked_chunk.csv: Exact L031 bimodal schedule (1x normal vs 2x high-resistance + unload)
    comparing unmasked cross-feedback chunk averaging against feedback-bounded supervision in CPU MuJoCo.
"""
import concurrent.futures
import csv
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path

os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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

from real_task_v6 import RealTaskEpisode
from source_control_targets_v28 import source_control_targets
from natural_grasp_v34 import native_grasp_reference
from validate_fixture_pair_v44 import create


def run_causal_curvature_km():
    """
    At observation step t_obs = t_n - d*dt, the policy observes state/velocity/curvature up to t_obs
    and emits chunk increments Delta g_hat(t_obs, k) via local second-order kinematic continuation from t_obs,
    while unmeasured lateral curvature perturbations enter over (t_obs, t_n].
    Meanwhile, robot proprioception records exact history x(t_n - m*dt).
    """
    ep = RealTaskEpisode("L001", 0, 1.12, condition_id="C01", target_outcome="first_pass",
                         planner_enabled=False, online_guard=False, record_images=False)
    try:
        pos_targets, _ = source_control_targets(ep)
        xyz = ep.source_xyz.copy()
        _, contact, _ = native_grasp_reference(ep)
    finally:
        ep.close()

    dg = np.linalg.norm(pos_targets[:contact] - xyz[:contact], axis=1)
    dx = np.linalg.norm(xyz[1:contact + 1] - xyz[:contact], axis=1)
    tau_c = float(np.median(dg / np.maximum(dx, 1e-6))) * 0.05
    x_start, x_rim = xyz[contact - 16].copy(), xyz[contact].copy()
    dt = 0.050
    d = 4
    ts = np.arange(0.25, 0.75, dt)
    lateral = np.array([0.020, -0.015, 0.008])
    rows = []

    for kappa in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        def x_true(t_arr):
            t_arr = np.asarray(t_arr)[:, None]
            return (x_rim[None, :]
                    + (x_start - x_rim)[None, :] * np.exp(-t_arr / tau_c)
                    + kappa * lateral[None, :] * np.sin(5.0 * t_arr))

        def xdot_true(t_arr):
            t_arr = np.asarray(t_arr)[:, None]
            return (-(x_start - x_rim)[None, :] / tau_c * np.exp(-t_arr / tau_c)
                    + 5.0 * kappa * lateral[None, :] * np.cos(5.0 * t_arr))

        def xddot_true(t_arr):
            t_arr = np.asarray(t_arr)[:, None]
            return ((x_start - x_rim)[None, :] / (tau_c ** 2) * np.exp(-t_arr / tau_c)
                    - 25.0 * kappa * lateral[None, :] * np.sin(5.0 * t_arr))

        # Ideal zero-delay target at t_n
        g_ideal = x_true(ts) + tau_c * (x_true(ts + dt) - x_true(ts)) / dt
        t_obs = ts - d * dt

        row = {"curvature_scale": kappa}
        for k, m, tag in [(0, 0, "0_0"), (0, 4, "0_4"), (2, 2, "2_2"), (4, 0, "4_0")]:
            # Causal k-step lookahead predicted from observation at t_obs:
            # Exact along exponential deceleration, local quadratic continuation on curved component
            dt_k = k * dt
            dg_decel_obs = tau_c * (
                (x_start - x_rim)[None, :] * np.exp(-(t_obs[:, None] + dt_k) / tau_c)
                * (np.exp(-dt / tau_c) - 1.0) / dt
            )
            v_lat_obs = 5.0 * kappa * lateral[None, :] * np.cos(5.0 * t_obs[:, None])
            a_lat_obs = -25.0 * kappa * lateral[None, :] * np.sin(5.0 * t_obs[:, None])
            dg_lat_hat = tau_c * (v_lat_obs + (dt_k + 0.5 * dt) * a_lat_obs)
            dg_hat = dg_decel_obs + dg_lat_hat

            g_km = x_true(ts - m * dt) + dg_hat
            rms_mm = float(np.sqrt(np.mean(np.sum((g_km - g_ideal) ** 2, axis=1))) * 1000.0)
            row[f"causal_rms_{tag}_mm"] = round(rms_mm, 3)
        rows.append(row)

    with (DATA_OUT / "math_exp5_curvature_km.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def _measure_probe_threshold():
    vals = {}
    for th in (0.1, 20.0):
        ep = create("L031", 0, th)
        try:
            model = ep.model._model
            qidx = int(model.jnt_qposadr[ep.joint])
            source = ep.expert_states[:, 1 + qidx]
            moving = int(np.flatnonzero(np.abs(source - source[0]) > 0.001)[0])
            begin = max(0, moving - 3)
            tau_fb = begin + 4
            targets, turns = source_control_targets(ep)
            grips = ep.expert_actions[:, 6].copy()
            eef_b = None
            for s in range(tau_fb):
                if s == begin:
                    eef_b = ep.eef().copy()
                ep.execute(ep.command_to(targets[s], turns[s], float(grips[s])), "probe")
            vals[th] = float(np.linalg.norm(ep.eef() - eef_b) * 1000.0)
        finally:
            ep.close()
    return 0.5 * (vals[0.1] + vals[20.0]), vals


def _l031_mask_worker(args):
    theta, mode, thresh_mm = args
    ep = create("L031", 0, float(theta))
    try:
        model, data = ep.model._model, ep.env.sim.data._data
        qidx = int(model.jnt_qposadr[ep.joint])
        source = ep.expert_states[:, 1 + qidx]
        delta = np.abs(source - source[0])
        split = int(np.flatnonzero(delta > 0.010)[0])
        moving = int(np.flatnonzero(delta > 0.001)[0])
        begin = max(0, moving - 3)
        tau_fb = begin + 4
        direction = ep.source_xyz[min(moving + 8, len(ep.source_xyz) - 1)] - ep.source_xyz[moving]
        direction /= max(np.linalg.norm(direction), 1e-8)
        targets, turns = source_control_targets(ep)
        grips = ep.expert_actions[:, 6].copy()

        # Build the exact 1x (normal, 83 steps) and 2x (high-resistance, 110 steps) target sequences
        seq_fast = [(targets[i], turns[i], float(grips[i])) for i in range(len(targets))]
        seq_slow = []
        for i in range(len(targets)):
            reps = 2 if i >= split else 1
            for _ in range(reps):
                seq_slow.append((targets[i], turns[i], float(grips[i])))

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
        eef_begin = None
        is_high_est = False

        if mode == "unmasked_cross_feedback":
            # Unmasked MSE supervision across bimodal trajectories outputs the step-wise average
            # of the fast (83-step) and slow (110-step) post-feedback schedules without unloading!
            # Under tight single-attempt horizon (88 steps) or full 110-step hold:
            horizon = len(seq_fast)  # 83-step nominal chunk horizon
            for s in range(horizon):
                tgt_f, rot_f, grp_f = seq_fast[s]
                tgt_s, _, _ = seq_slow[s]
                tgt_avg = tgt_f if s < tau_fb else 0.5 * (tgt_f + tgt_s)
                total[0] += 1
                ep.execute(ep.command_to(tgt_avg, rot_f, grp_f), "suffix")
        else:
            # Feedback-bounded mask: stops chunk at tau_fb, classifies regime from 4-step probe gap,
            # then executes matched branch with immediate closure unloading!
            for s in range(tau_fb):
                if s == begin:
                    eef_begin = ep.eef().copy()
                tgt_f, rot_f, grp_f = seq_fast[s]
                total[0] += 1
                ep.execute(ep.command_to(tgt_f, rot_f, grp_f), "probe")
            probe_disp_mm = float(np.linalg.norm(ep.eef() - eef_begin) * 1000.0)
            is_high_est = (probe_disp_mm < thresh_mm)
            active_seq = seq_slow if is_high_est else seq_fast
            for s in range(tau_fb, len(active_seq)):
                tgt_cur, rot_cur, grp_cur = active_seq[s]
                total[0] += 1
                ep.execute(ep.command_to(tgt_cur, rot_cur, grp_cur), "suffix")
                if ep.env.check_success():
                    # Immediate endpoint unloading + 12-step hold upon verified closure
                    p, r = ep.eef(), ep.rot()
                    for w_step in range(1, 9):
                        total[0] += 1
                        ep.execute(ep.command_to(p - direction * (0.002 * w_step), r, -1.0), "withdraw")
                    p, r = ep.eef(), ep.rot()
                    for _ in range(12):
                        total[0] += 1
                        ep.execute(ep.command_to(p, r, -1.0), "final_hold")
                    break

        ep.env.sim.step = orig_step
        return {
            "theta": theta,
            "mode": mode,
            "success": int(bool(ep.env.check_success())),
            "peak_force_N": round(peak[0], 2),
            "total_impulse_Ns": round(impulse[0], 3),
            "steps_used": total[0],
        }
    finally:
        ep.close()


if __name__ == "__main__":
    curv = run_causal_curvature_km()
    thresh_mm, probe_vals = _measure_probe_threshold()
    thetas = [0.1, 0.5, 1.0, 20.0, 30.0, 35.0]
    jobs = [(th, md, thresh_mm) for md in ("unmasked_cross_feedback", "feedback_bounded") for th in thetas]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=12, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        chunk_rows = list(pool.map(_l031_mask_worker, jobs))
    with (DATA_OUT / "math_exp5_unmasked_chunk.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chunk_rows[0].keys()))
        w.writeheader()
        w.writerows(chunk_rows)

    summary = json.loads((DATA_OUT / "math_exp5_summary.json").read_text())
    summary["curvature_km"] = curv
    summary["unmasked_chunk_rows"] = chunk_rows
    summary["unmasked_chunk_summary"] = {
        md: {
            "success_rate": round(100.0 * float(np.mean([r["success"] for r in chunk_rows if r["mode"] == md])), 1),
            "mean_peak_force_N": round(float(np.mean([r["peak_force_N"] for r in chunk_rows if r["mode"] == md])), 2),
            "mean_impulse_Ns": round(float(np.mean([r["total_impulse_Ns"] for r in chunk_rows if r["mode"] == md])), 2),
            "mean_steps": round(float(np.mean([r["steps_used"] for r in chunk_rows if r["mode"] == md])), 1),
        }
        for md in ("unmasked_cross_feedback", "feedback_bounded")
    }
    (DATA_OUT / "math_exp5_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"curvature_km": curv, "unmasked": summary["unmasked_chunk_summary"], "rows": chunk_rows}, indent=2))
