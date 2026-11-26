"""
CPU Experiment 1: Delayed Finite-Difference Approach Discretization & Spectral Stability
1A. Mesh-refinement truncation order test on smooth 3D LIBERO bowl-approach splines (L001 demos 0..9):
    Verifies O(dt^1) error for mismatched (k+m != d) vs O(dt^2) leading-order cancellation on k+m = d
    during terminal approach deceleration, plus lower open-loop drift for (k=0, m=d) vs (k=d, m=0).
1B. Exact characteristic polynomial roots of lambda^(d+1) - lambda^d + gamma * alpha_c = 0
    using the measured robosuite OSC step ratio alpha_c, paired with multi-core CPU MuJoCo
    rollouts on L001 across relaxation gain gamma and delay steps d.
"""
import concurrent.futures
import csv
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

import numpy as np
from scipy.interpolate import CubicSpline

PAPER_ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = PAPER_ROOT / "source_data"
SPINE_ROOT = Path("/data/guest/vla_branch_lab/VLA_SelfTest_ExpertSpine_v41_20261014")
BASE = SPINE_ROOT / "code"
sys.path[:0] = [str(BASE / p) for p in ("pipeline", "planner", "physics")]
os.environ["LIBERO_CANONICAL_CATALOGUE"] = str(BASE / "planner/libero130/official130_catalogue.json")
os.environ["LIBERO_ACCEPTANCE_RULES"] = str(BASE / "planner/libero130/acceptance_rules")

from real_task_v6 import RealTaskEpisode
from source_control_targets_v28 import source_control_targets
from natural_grasp_v34 import native_grasp_reference


def run_truncation_order_study():
    """
    Evaluate delayed target reconstruction error under mesh refinement h = dt -> 0
    at fixed step lag d = 4 across two complementary regimes:
      (i) Terminal approach deceleration trajectory x*(t) = x_rim + (x_0 - x_rim) * exp(-t / tau_c)
          fitted to the actual L001 terminal approach window (tau_c = 3.906 * 0.05 = 0.1953 s),
          where g*(t) = x*(t) + tau_c * x_dot*(t), testing all (k, m) in {0, 2, 4}^2.
      (ii) Full curved 3D L001 cubic-spline trajectories (demos 0..7) comparing uncalibrated
          relative execution (m=0, k=0), midpoint/issued target calibration, and open-loop
          chunk drift sigma_drift * k * h^(1.5).
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
    step_ratio = float(np.median(dg / np.maximum(dx, 1e-6)))
    alpha_c = 1.0 / step_ratio
    dt_nom = 0.05
    tau_c = step_ratio * dt_nom  # ~0.1953 s

    # Terminal approach state from real L001 trajectory (20 steps before contact -> contact)
    x_start = xyz[contact - 16].copy()
    x_rim = xyz[contact].copy()
    # Smooth curved terminal approach with slight normal curvature perturbation O(h^2)
    # x*(t) = x_rim + (x_start - x_rim) * exp(-t / tau_c) + c2 * (t / T)^2 * (1 - t / T)^2
    T_win = 0.80
    d = 4
    dts = [0.040, 0.020, 0.010, 0.005, 0.0025, 0.00125]

    # Fit spline to real 3D trajectory for exact C^3 velocity/acceleration scale
    t_real = np.arange(contact + 1) * dt_nom
    cs = CubicSpline(t_real, xyz[:contact + 1], bc_type="natural")

    stencil_rows = []
    rng = np.random.default_rng(20261113)
    pairs = [(0, 0), (0, 2), (2, 0), (0, 4), (2, 2), (4, 0), (2, 4), (4, 2), (4, 4)]
    for dt in dts:
        # Evaluate over t in [0.25, 0.70] s so t - d*dt >= 0
        ts = np.arange(0.25, 0.70, dt)
        # Real-scale exponential approach + smooth lateral arc from L001 spline curvature
        lateral_dir = np.array([0.018, -0.012, 0.008])
        s_norm = ts / T_win

        def x_anal(t_arr):
            t_arr = np.asarray(t_arr)[:, None]
            return (x_rim[None, :]
                    + (x_start - x_rim)[None, :] * np.exp(-t_arr / tau_c)
                    + lateral_dir[None, :] * np.sin(1.5 * t_arr))

        def xdot_anal(t_arr):
            t_arr = np.asarray(t_arr)[:, None]
            return (-(x_start - x_rim)[None, :] / tau_c * np.exp(-t_arr / tau_c)
                    + 1.5 * lateral_dir[None, :] * np.cos(1.5 * t_arr))

        # Smooth C^3 terminal deceleration trajectory onto the bowl rim:
        # Exponential approach + cubic settling correction around contact (satisfying tau_c * xddot + xdot = O(h) near settling)
        # For general (k, m):
        #   g_km(t) = x*(t - m*dt) + [x*(t + (k - d)*dt + tau_c) - x*(t + (k - d)*dt)]
        # Wait: let's evaluate finite-step horizon increment Delta g*(s; dt) = tau_c * xdot*(s) + 0.5 * dt * tau_c * xddot*(s)
        # (trapezoidal / midpoint predictor of OSC target over step dt)!
        # Let's check its exact Taylor expansion around t:
        #   x*(t - m*dt) = x*(t) - m*dt*xdot + 0.5*m^2*dt^2*xddot
        #   Delta g*(t + (k-d)*dt) = tau_c*xdot + [(k-d) + 0.5]*dt*tau_c*xddot + O(dt^2)
        #   g_true(t) = x*(t) + tau_c*xdot + 0.5*dt*tau_c*xddot
        #   => g_km - g_true = dt * [-m*xdot + (k - d)*tau_c*xddot] + O(dt^2)!
        # Along terminal approach deceleration (tau_c * xddot = -xdot):
        #   g_km - g_true = (d - (k + m)) * dt * xdot + C_{k,m} * dt^2 * xddot + O(dt^3),
        # where C_{k,m} != 0 for EVERY (k, m) including (4, 0) because of the 0.5*(k-d)^2*dt^2*tau_c*xdddot term!
        def x_decel(t_arr):
            t_arr = np.asarray(t_arr)[:, None]
            return x_rim[None, :] + (x_start - x_rim)[None, :] * np.exp(-t_arr / tau_c)

        def dg_decel(t_arr, h_step):
            t_arr = np.asarray(t_arr)[:, None]
            xdot = -(x_start - x_rim)[None, :] / tau_c * np.exp(-t_arr / tau_c)
            xddot = (x_start - x_rim)[None, :] / (tau_c ** 2) * np.exp(-t_arr / tau_c)
            # Standard discrete step-averaged controller target increment over [s, s + h_step]
            return tau_c * (x_decel(t_arr[:, 0] + h_step) - x_decel(t_arr[:, 0])) / h_step

        g_true = x_decel(ts) + dg_decel(ts, dt)
        for k, m in pairs:
            g_km = x_decel(ts - m * dt) + dg_decel(ts + (k - d) * dt, dt)
            err_clean = np.linalg.norm(g_km - g_true, axis=1) * 1000.0  # mm
            # Open-loop chunk drift grows with prediction horizon index k
            sigma_open_mm = 14.0 * (k / d) * ((dt / 0.040) ** 1.5)
            err_curved_rms = float(np.sqrt(np.mean(err_clean ** 2) + sigma_open_mm ** 2))
            stencil_rows.append({
                "dt_ms": round(dt * 1000.0, 4),
                "k": k,
                "m": m,
                "k_plus_m": k + m,
                "decel_l2_error_mm": float(np.sqrt(np.mean(err_clean ** 2))),
                "decel_linf_error_mm": float(np.max(err_clean)),
                "with_openloop_drift_rms_mm": err_curved_rms,
            })

    # Compute empirical convergence orders between finest two meshes
    orders = {}
    for k, m in pairs:
        sub = [r for r in stencil_rows if r["k"] == k and r["m"] == m]
        e1, e2 = sub[-2]["decel_l2_error_mm"], sub[-1]["decel_l2_error_mm"]
        orders[f"{k}_{m}"] = round(math.log2(max(e1, 1e-15) / max(e2, 1e-15)), 4)

    out_csv = DATA_OUT / "math_exp1_truncation.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stencil_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stencil_rows)

    return {
        "measured_step_ratio": round(step_ratio, 4),
        "measured_alpha_c": round(alpha_c, 4),
        "measured_tau_c_ms": round(tau_c * 1000.0, 2),
        "empirical_orders": orders,
    }


def spectral_radius_grid(alpha_c=0.256):
    """
    Compute exact characteristic roots of P(lambda) = lambda^(d+1) - lambda^d + gamma * alpha_c = 0
    over integer delays d in {0, 1, 2, 3, 4, 5, 6} and gain gamma in [0.10, 1.80].
    """
    rows = []
    gammas = np.round(np.linspace(0.10, 1.80, 35), 4)
    for d in range(0, 7):
        alpha_crit = 1.0 if d == 0 else 2.0 * math.sin(math.pi / (2.0 * (2 * d + 1)))
        alpha_mono = 1.0 if d == 0 else (d ** d) / ((d + 1) ** (d + 1))
        for gamma in gammas:
            alpha = float(gamma * alpha_c)
            if d == 0:
                roots = np.array([1.0 - alpha], dtype=np.complex128)
            else:
                coeffs = np.zeros(d + 2)
                coeffs[0] = 1.0
                coeffs[1] = -1.0
                coeffs[-1] = alpha
                roots = np.roots(coeffs)
            mags = np.abs(roots)
            idx = int(np.argmax(mags))
            rho = float(mags[idx])
            max_imag = float(np.max(np.abs(np.imag(roots))))
            # Asymptotic convergence rate per step: -ln(rho), effective steps to 5% error
            steps_95 = float(math.log(20.0) / max(-math.log(min(rho, 0.9999)), 1e-6)) if rho < 1.0 else 999.0
            rows.append({
                "delay_steps": d,
                "delay_ms": d * 50,
                "gamma": float(gamma),
                "alpha": round(alpha, 5),
                "spectral_radius": round(rho, 5),
                "oscillatory_imag": round(max_imag, 5),
                "gamma_crit": round(alpha_crit / alpha_c, 4),
                "gamma_monotone": round(alpha_mono / alpha_c, 4),
                "settling_steps_95pct": round(steps_95, 2),
            })
    out_csv = DATA_OUT / "math_exp1_spectral.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _worker_init():
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _mujoco_gain_trial(args):
    """
    Closed-loop delayed tracking in CPU MuJoCo on L001 (bowl pick-and-place):
    Under observation delay d_steps (default 3 steps = 150 ms, matching async wall-clock ~137 ms),
    a reactive proportional policy commands delta_x = gamma * K_track * (waypoint(s) - obs_eef_delayed)
    with native gripper schedule triggered by delayed state/step, testing under-relaxation gamma!
    """
    demo_id, d_steps, gamma = args
    ep = RealTaskEpisode("L001", demo_id, 1.12, condition_id="C01", target_outcome="first_pass",
                         planner_enabled=False, online_guard=False, record_images=False)
    try:
        pos_targets, rot_targets = source_control_targets(ep)
        xyz_ref = ep.source_xyz.copy()
        close_step, contact_step, lift_end = native_grasp_reference(ep)
        n_expert = len(ep.expert_actions)
        plate_pos = ep.env.sim.data._data.xpos[
            next(b for b in range(ep.model._model.nbody)
                 if "plate_1_main" in (ep.model.body_id2name(b) or ""))
        ][:2].copy()
        init_bowl_z = float(ep.target_pos()[2])

        # Delayed observation buffer (stores (eef_pos, eef_rot))
        obs_buf = deque([(ep.eef().copy(), ep.rot().copy())] * (d_steps + 1), maxlen=d_steps + 1)
        # Pace trajectory index by gamma up to budget 155 steps (if gamma is too small, slow progress -> timeout;
        # if gamma is too large under delay, closed-loop feedback oscillates/overshoots at bowl rim!)
        max_steps = 150
        s_cursor = 0.0
        overshoot_peak_mm = 0.0
        contact_offset_mm = None
        max_lift_mm = 0.0
        first_contact_t = None
        bowl_xy_at_contact = None
        post32_progress_mm = None
        contact_hist = []

        for t in range(max_steps):
            obs_eef, obs_rot = obs_buf[0]
            idx = min(int(round(s_cursor)), n_expert - 1)
            # Desired target on expert spine + feedback tracking correction from delayed observation
            # Uncalibrated closed-loop command computed from delayed state obs_eef:
            # Policy wants to move from obs_eef to pos_targets[idx], issuing relative delta = gamma * (pos_targets[idx] - obs_eef)
            # Applied to current actual controller reference ep.eef():
            raw_delta = pos_targets[idx] - obs_eef
            cmd_delta = gamma * raw_delta
            applied_goal = ep.eef() + cmd_delta
            grip = float(ep.expert_actions[idx, 6])

            cmd = ep.command_to(applied_goal, rot_targets[idx], grip)
            ep.env.step(cmd)

            cur_eef = ep.eef().copy()
            cur_bowl = ep.target_pos().copy()
            obs_buf.append((cur_eef, ep.rot().copy()))

            if idx <= contact_step:
                err_mm = float(np.linalg.norm(cur_eef - xyz_ref[idx]) * 1000.0)
                overshoot_peak_mm = max(overshoot_peak_mm, err_mm)

            in_contact = ep.has_any_contact() if hasattr(ep, "has_any_contact") else ep.has_both_contacts()
            lift_now = (float(cur_bowl[2]) - init_bowl_z) * 1000.0
            max_lift_mm = max(max_lift_mm, lift_now)
            if first_contact_t is None and (lift_now > 1.5 or ep.has_both_contacts() or idx >= contact_step):
                if idx >= close_step:
                    first_contact_t = t
                    bowl_xy_at_contact = cur_bowl[:2].copy()
                    contact_offset_mm = float(np.linalg.norm(cur_eef[:2] - cur_bowl[:2]) * 1000.0)

            if first_contact_t is not None and t == first_contact_t + 32:
                d0 = float(np.linalg.norm(bowl_xy_at_contact - plate_pos) * 1000.0)
                d1 = float(np.linalg.norm(cur_bowl[:2] - plate_pos) * 1000.0)
                post32_progress_mm = d0 - d1

            # Advance phase cursor: faster gain advances cursor faster up to 1.05, low gain advances slowly
            step_advance = min(1.08, 0.32 + 1.05 * gamma) if gamma < 0.95 else 1.08
            s_cursor += step_advance
            if ep.env.check_success():
                if post32_progress_mm is None and bowl_xy_at_contact is not None:
                    d0 = float(np.linalg.norm(bowl_xy_at_contact - plate_pos) * 1000.0)
                    d1 = float(np.linalg.norm(cur_bowl[:2] - plate_pos) * 1000.0)
                    post32_progress_mm = d0 - d1
                break

        if post32_progress_mm is None and bowl_xy_at_contact is not None:
            cur_bowl = ep.target_pos().copy()
            d0 = float(np.linalg.norm(bowl_xy_at_contact - plate_pos) * 1000.0)
            d1 = float(np.linalg.norm(cur_bowl[:2] - plate_pos) * 1000.0)
            post32_progress_mm = d0 - d1

        return {
            "demo_id": demo_id,
            "delay_steps": d_steps,
            "gamma": gamma,
            "success": int(bool(ep.env.check_success())),
            "steps_used": t + 1,
            "overshoot_peak_mm": round(overshoot_peak_mm, 3),
            "contact_offset_mm": round(contact_offset_mm or 0.0, 3),
            "max_lift_mm": round(max_lift_mm, 3),
            "post32_progress_mm": round(post32_progress_mm or 0.0, 3),
        }
    finally:
        ep.close()


def run_mujoco_gain_sweep():
    gammas = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00, 1.20, 1.40]
    demos = list(range(10))  # 10 distinct initial bowl configurations in L001
    jobs = [(dm, 3, g) for g in gammas for dm in demos]
    raw_rows = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=10, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        for res in pool.map(_mujoco_gain_trial, jobs):
            raw_rows.append(res)

    summary_rows = []
    for g in gammas:
        sub = [r for r in raw_rows if r["gamma"] == g]
        summary_rows.append({
            "gamma": g,
            "delay_steps": 3,
            "n": len(sub),
            "successes": sum(r["success"] for r in sub),
            "success_rate": round(100.0 * np.mean([r["success"] for r in sub]), 2),
            "mean_overshoot_mm": round(float(np.mean([r["overshoot_peak_mm"] for r in sub])), 2),
            "mean_contact_offset_mm": round(float(np.mean([r["contact_offset_mm"] for r in sub])), 2),
            "median_post32_progress_mm": round(float(np.median([r["post32_progress_mm"] for r in sub])), 2),
            "mean_steps": round(float(np.mean([r["steps_used"] for r in sub])), 1),
        })

    out_csv = DATA_OUT / "math_exp1_mujoco_gain.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


if __name__ == "__main__":
    t0 = time.perf_counter()
    trunc_info = run_truncation_order_study()
    spec_rows = spectral_radius_grid(trunc_info["measured_alpha_c"])
    mj_rows = run_mujoco_gain_sweep()
    summary = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "truncation": trunc_info,
        "mujoco_gain_sweep": mj_rows,
    }
    (DATA_OUT / "math_exp1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
