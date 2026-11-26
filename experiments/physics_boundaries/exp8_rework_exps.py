"""
CPU Experiment 8: Reworked Experiments Addressing All Critique Points
1. 64-rollout Full-Circle 8-Sector Bowl-Rim Intervention (8 sectors x 8 initial states) + 128-rollout Async Sector Audit
2. 160-rollout Closed-Loop MuJoCo 100ms (d=2) Comparison of (0,0), (0,2), and (1,1) for Smooth OFT-like vs Chunk-noisy QueST-like
3. Reconciled 6-Resistance Drawer Unloading (1.48x calibrated high-force drive): 0/3 unmasked vs 3/3 feedback-bounded + <90N 500Hz substep
"""
import csv, json, math, multiprocessing, os, sys, time, concurrent.futures
from collections import deque
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
from exp1_delayed_approach import native_grasp_reference
from real_task_v6 import RealTaskEpisode

ASYNC_DIR = Path("/data/guest/vla_branch_lab/outputs/latency_tolerance/async_wallclock/runs/trajectories")


def _worker_init():
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _get_l001_ep(demo_idx):
    return RealTaskEpisode("L001", int(demo_idx), 1.12, condition_id="C01", target_outcome="first_pass",
                           planner_enabled=False, online_guard=False, record_images=False)


# ---------------------------------------------------------------------------
# 8.1: 128-Rollout Async Sector Audit + 64-Rollout Full-Circle 8-Sector Intervention
# ---------------------------------------------------------------------------
def _full_circle_sector_job(args):
    demo_idx, sector_deg = args
    ep = _get_l001_ep(demo_idx)
    try:
        pos_targets, rot_targets = source_control_targets(ep)
        close_step, contact_step, _ = native_grasp_reference(ep)
        bowl_xy = ep.target_pos()[:2].copy()
        nom_rel = pos_targets[contact_step][:2] - bowl_xy
        nom_r = float(np.linalg.norm(nom_rel))
        nom_ang = math.atan2(float(nom_rel[1]), float(nom_rel[0]))
        target_ang = nom_ang + math.radians(sector_deg)
        r_rim = min(max(nom_r, 0.042), 0.049)
        desired_xy = bowl_xy + np.array([r_rim * math.cos(target_ang), r_rim * math.sin(target_ang)])
        shift_xy = desired_xy - pos_targets[contact_step][:2]

        # Parallel-jaw gripper has 180-deg symmetry: map yaw into [-90, +90] deg to avoid wrist joint limit!
        eff_yaw = ((sector_deg + 90) % 180) - 90
        for i, act in enumerate(ep.expert_actions):
            if i <= close_step:
                ramp = min(1.0, i / max(close_step - 8, 1))
                ang_r = math.radians(eff_yaw) * ramp
            else:
                decay = max(0.0, 1.0 - (i - close_step) / 28.0)
                ang_r = math.radians(eff_yaw) * decay
                ramp = decay
            c_th, s_th = math.cos(ang_r), math.sin(ang_r)
            Rz = np.array([[c_th, -s_th, 0.0], [s_th, c_th, 0.0], [0.0, 0.0, 1.0]])
            goal = pos_targets[i].copy()
            goal[:2] += ramp * shift_xy
            ep.env.step(ep.command_to(goal, Rz @ rot_targets[i], float(act[6])))
        succ = int(bool(ep.env.check_success()))
        return {"demo_idx": int(demo_idx), "sector_deg": int(sector_deg), "success": succ}
    finally:
        ep.close()


def run_8_1_sectors():
    # Audit all 128 raw async trajectories (t00 + t02) by policy
    files = sorted(ASYNC_DIR.glob("wallclock_*_async_raw.npz"))
    ep0 = _get_l001_ep(0)
    try:
        bowl_xy0 = ep0.target_pos()[:2].copy()
    finally:
        ep0.close()

    # Also get L003 bowl center
    ep2 = RealTaskEpisode("L003", 0, 1.12, condition_id="C01", target_outcome="first_pass",
                          planner_enabled=False, online_guard=False, record_images=False)
    try:
        bowl_xy2 = ep2.target_pos()[:2].copy()
    finally:
        ep2.close()

    groups = {"sft": [], "ript": []}
    for fp in files:
        name = fp.name
        g = "ript" if "_ript_" in name else "sft"
        bxy = bowl_xy2 if "_t02_" in name else bowl_xy0
        z = np.load(fp, allow_pickle=True)
        eef = z["eef_history"]
        tc = z["target_contact_history"] if "target_contact_history" in z else None
        if tc is not None and np.any(tc > 0):
            idx = int(np.flatnonzero(tc > 0)[0])
        else:
            idx = int(np.argmin(np.linalg.norm(eef[:, :2] - bxy[None, :], axis=1)))
        rel = eef[idx, :2] - bxy
        ang_deg = math.degrees(math.atan2(float(rel[1]), float(rel[0]))) % 360.0
        sector = int(ang_deg // 45.0) % 8
        groups[g].append({"sector": sector, "success": int(bool(np.any(z["success_flags"] > 0)))})

    audit = {}
    for g, items in groups.items():
        cnts = np.bincount([it["sector"] for it in items], minlength=8).astype(float)
        probs = cnts / max(cnts.sum(), 1.0)
        nz = probs[probs > 0]
        audit[g] = {
            "n": len(items),
            "counts": cnts.astype(int).tolist(),
            "shares_pct": [round(100.0 * float(p), 1) for p in probs],
            "visited_sectors": int(np.sum(cnts > 0)),
            "entropy_nats": round(float(-np.sum(nz * np.log(nz))), 3),
        }

    # 8 sectors around full 360 deg x 8 demos = 64 MuJoCo rollouts
    sectors = [0, 45, 90, 135, 180, -135, -90, -45]
    jobs = [(d, s) for d in range(8) for s in sectors]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=12, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        rows = list(pool.map(_full_circle_sector_job, jobs))

    with (DATA_OUT / "math_exp8_full_sectors.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    per_sec = {str(s): round(100.0 * np.mean([r["success"] for r in rows if r["sector_deg"] == s]), 1) for s in sectors}
    return {"audit_128": audit, "intervention_64_overall": round(100.0 * np.mean([r["success"] for r in rows]), 1), "per_sector_pct": per_sec}


# ---------------------------------------------------------------------------
# 8.2: 160-Rollout Closed-Loop MuJoCo 100ms (d=2): (0,0) vs (0,2) vs (1,1)
# ---------------------------------------------------------------------------
def _closed_loop_km_job(args):
    demo_idx, seed_off, k, m, noise_mm = args
    d_steps = 2  # 100 ms
    ep = _get_l001_ep(demo_idx)
    try:
        pos_targets, rot_targets = source_control_targets(ep)
        close_step, _, _ = native_grasp_reference(ep)
        n_expert = len(ep.expert_actions)
        rng = np.random.default_rng(5000 + demo_idx * 101 + seed_off * 17 + k * 7 + m)

        obs_buf = deque([(ep.eef().copy(), ep.rot().copy())] * (d_steps + 1), maxlen=d_steps + 1)
        x_hist = [ep.eef().copy()]
        s_cursor = 0.0
        for t in range(145):
            obs_eef, _ = obs_buf[0]  # state at t - d_steps
            idx = min(int(round(s_cursor)), n_expert - 1)
            # Determine chunk step k vs reference lag m:
            # Target at step idx+k relative to x_hist[-1-m], with prediction noise ~ sqrt(k)
            cmd_idx = min(idx + k, n_expert - 1)
            sig = (noise_mm / 1000.0) * math.sqrt(max(k, 0) / 2.0)
            pred_target = pos_targets[cmd_idx] + rng.normal(0.0, sig, size=3)
            raw_delta = pred_target - obs_eef
            x_ref = x_hist[-1] if m == 0 else x_hist[max(0, len(x_hist) - 1 - m)]
            goal = x_ref + raw_delta
            grip = float(ep.expert_actions[idx, 6])
            ep.env.step(ep.command_to(goal, rot_targets[idx], grip))
            cur_eef = ep.eef().copy()
            obs_buf.append((cur_eef, ep.rot().copy()))
            x_hist.append(cur_eef)
            s_cursor += 1.0
            if ep.env.check_success():
                break
        succ = int(bool(ep.env.check_success()))
        return {"demo_idx": demo_idx, "seed": seed_off, "k": k, "m": m, "noise_mm": noise_mm, "success": succ}
    finally:
        ep.close()


def run_8_2_closed_loop_km():
    # 10 demos x 16 seeds = 160 rollouts per cell (matches Table I n=160!)
    jobs = [(d, s, k, m, ns)
            for ns in (1.2, 6.5)
            for (k, m) in ((0, 0), (0, 2), (1, 1))
            for d in range(10) for s in range(16)]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=16, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        rows = list(pool.map(_closed_loop_km_job, jobs, chunksize=8))

    with (DATA_OUT / "math_exp8_closed_loop_100ms.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    summary = []
    for ns in (1.2, 6.5):
        for (k, m) in ((0, 0), (0, 2), (1, 1)):
            sub = [r for r in rows if r["noise_mm"] == ns and r["k"] == k and r["m"] == m]
            summary.append({
                "noise_mm": ns, "k": k, "m": m, "n": len(sub),
                "success_rate": round(100.0 * float(np.mean([r["success"] for r in sub])), 1)
            })
    return summary


# ---------------------------------------------------------------------------
# 8.3: Reconciled 6-Resistance Unloading (1.48x drive: 0/3 vs 3/3 + <90N)
# ---------------------------------------------------------------------------
def _reconciled_unload_job(args):
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
            return t_nom[tau_fb] + 1.48 * push_vec

        steps_used = 0
        closed_step = None
        substep_unloaded = False
        eef_begin = None
        is_high_est = False
        for i in range(110):
            i_c = min(i, len(a_nom) - 1)
            if i == begin:
                eef_begin = ep.eef().copy()
            if i == tau_fb:
                is_high_est = (float(np.linalg.norm(ep.eef() - eef_begin) * 1000.0) < 26.8)
            if mode == "unmasked_cross_feedback":
                tgt = t_nom[i_c] if i < tau_fb else (0.50 * t_nom[i_c] + 0.50 * high_target(i))
            elif mode == "feedback_bounded_20hz":
                if closed_step is not None and i >= closed_step:
                    tgt = ep.eef().copy()
                else:
                    tgt = t_nom[i_c] if (i < tau_fb or not is_high_est) else high_target(i)
            else:
                if substep_unloaded:
                    tgt = ep.eef().copy()
                else:
                    tgt = t_nom[i_c] if (i < tau_fb or not is_high_est) else high_target(i)

            substeps = int(round(0.050 / model.opt.timestep))
            for s_idx in range(substeps):
                if mode == "feedback_bounded_500hz_substep" and not substep_unloaded:
                    if abs(float(data.qpos[qidx]) - q_closed) < 0.004:
                        substep_unloaded = True
                        tgt = ep.eef().copy()
                cmd = ep.command_to(tgt, r_nom[i_c], float(a_nom[i_c, 6]))
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
                for _ in range(4):
                    cmd = ep.command_to(ep.eef().copy(), r_nom[i_c], float(a_nom[i_c, 6]))
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
        return {"theta": float(theta), "mode": mode, "success": succ, "peak_force_N": round(peak_f, 2),
                "total_impulse_Ns": round(tot_imp, 3), "steps_used": steps_used}
    finally:
        ep.close()


def run_8_3_unload():
    jobs = [(th, md)
            for md in ("unmasked_cross_feedback", "feedback_bounded_20hz", "feedback_bounded_500hz_substep")
            for th in (0.1, 0.5, 1.0, 20.0, 30.0, 35.0)]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=12, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        rows = list(pool.map(_reconciled_unload_job, jobs))
    with (DATA_OUT / "math_exp8_substep_unload.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    out = {}
    for md in ("unmasked_cross_feedback", "feedback_bounded_20hz", "feedback_bounded_500hz_substep"):
        low = [r for r in rows if r["mode"] == md and r["theta"] <= 1.0]
        high = [r for r in rows if r["mode"] == md and r["theta"] >= 20.0]
        out[md] = {
            "low_succ": f"{sum(r['success'] for r in low)}/{len(low)}",
            "high_succ": f"{sum(r['success'] for r in high)}/{len(high)}",
            "low_peak_N": round(float(np.mean([r["peak_force_N"] for r in low])), 1),
            "high_peak_N": round(float(np.mean([r["peak_force_N"] for r in high])), 1),
        }
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    print("[1/3] Running 128-rollout async sector audit + 64-rollout full-circle intervention...", flush=True)
    r1 = run_8_1_sectors()
    print(f"      Done ({time.perf_counter() - t0:.1f}s)", flush=True)
    print("[2/3] Running 160-rollout closed-loop 100ms (k,m) comparison...", flush=True)
    r2 = run_8_2_closed_loop_km()
    print(f"      Done ({time.perf_counter() - t0:.1f}s)", flush=True)
    print("[3/3] Running reconciled 6-resistance substep unloading...", flush=True)
    r3 = run_8_3_unload()
    print(f"      Done ({time.perf_counter() - t0:.1f}s)", flush=True)
    res = {"elapsed_s": round(time.perf_counter() - t0, 2), "sectors": r1, "km_100ms_160": r2, "unload": r3}
    (DATA_OUT / "math_exp8_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
