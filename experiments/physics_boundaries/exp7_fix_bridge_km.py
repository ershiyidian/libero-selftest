import csv, json, math, multiprocessing, os, sys, time, concurrent.futures
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

def _worker_init():
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

def _get_l001_ep(demo_idx):
    return RealTaskEpisode("L001", int(demo_idx), 1.12, condition_id="C01", target_outcome="first_pass",
                           planner_enabled=False, online_guard=False, record_images=False)

def _cold_corridor_job(args):
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

        for i, act in enumerate(ep.expert_actions):
            if i <= close_step:
                ramp = min(1.0, i / max(close_step - 8, 1))
                ang_r = math.radians(sector_deg) * ramp
            else:
                decay = max(0.0, 1.0 - (i - close_step) / 28.0)
                ang_r = math.radians(sector_deg) * decay
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

def _km_mujoco_full_job(args):
    demo_idx, d_steps, k, m, noise_std_mm = args
    ep = _get_l001_ep(demo_idx)
    try:
        pos_targets, rot_targets = source_control_targets(ep)
        xyz_ref = ep.source_xyz.copy()
        rng = np.random.default_rng(1000 + demo_idx * 17 + k * 7 + m)
        n_steps = len(ep.expert_actions)
        x_hist = [ep.eef().copy()]
        dg_nom = pos_targets - xyz_ref[:n_steps]
        total_steps = n_steps + d_steps + 6
        for i in range(total_steps):
            if i < d_steps:
                idx = min(i, n_steps - 1)
                goal = pos_targets[idx]
                rot_i = rot_targets[idx]
                grip_i = float(ep.expert_actions[idx, 6])
            else:
                obs_idx = min(i - d_steps, n_steps - 1)
                cmd_idx = min(obs_idx + k, n_steps - 1)
                x_ref = x_hist[max(0, i - m)]
                sig = (noise_std_mm / 1000.0) * math.sqrt(max(k, 0) / 4.0)
                dg_pred = dg_nom[cmd_idx] + rng.normal(0.0, sig, size=3)
                goal = x_ref + dg_pred
                rot_i = rot_targets[cmd_idx]
                grip_i = float(ep.expert_actions[cmd_idx, 6])
            ep.env.step(ep.command_to(goal, rot_i, grip_i))
            x_hist.append(ep.eef().copy())
            if ep.env.check_success():
                break
        succ = int(bool(ep.env.check_success()))
        return {"demo_idx": int(demo_idx), "delay_ms": int(d_steps * 50), "k": int(k), "m": int(m),
                "noise_mm": float(noise_std_mm), "success": succ}
    finally:
        ep.close()

def _substep_unload_100_job(args):
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
            extra = max(0, idx - (len(t_nom) - 1)) * 0.005 * (push_vec / max(float(np.linalg.norm(push_vec)), 1e-6))
            return t_nom[tau_fb] + 1.58 * push_vec + extra

        steps_used = 0
        closed_step = None
        substep_unloaded = False
        eef_begin = None
        is_high_est = False
        max_steps = 115
        for i in range(max_steps):
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

if __name__ == "__main__":
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=12, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        corridor_jobs = [(d, s) for d in (0, 1) for s in (-45, -30, -15, 0, 15, 30, 45)]
        corridor_rows = list(pool.map(_cold_corridor_job, corridor_jobs))

        km_jobs = [(d, ds, k, m, ns)
                   for ds, combos in [(2, [(0, 0), (0, 2), (1, 1), (2, 0)]), (4, [(0, 0), (0, 4), (2, 2), (4, 0)])]
                   for k, m in combos for ns in (1.8, 6.5) for d in range(4)]
        km_rows = list(pool.map(_km_mujoco_full_job, km_jobs))

        sub_jobs = [(th, md)
                    for md in ("unmasked_cross_feedback", "feedback_bounded_20hz", "feedback_bounded_500hz_substep")
                    for th in (0.1, 0.5, 1.0, 20.0, 30.0, 35.0)]
        sub_rows = list(pool.map(_substep_unload_100_job, sub_jobs))

    with (DATA_OUT / "math_exp7_cold_corridor.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corridor_rows[0].keys())); w.writeheader(); w.writerows(corridor_rows)
    with (DATA_OUT / "math_exp7_km_mujoco_full.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(km_rows[0].keys())); w.writeheader(); w.writerows(km_rows)
    with (DATA_OUT / "math_exp7_substep_unload.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sub_rows[0].keys())); w.writeheader(); w.writerows(sub_rows)

    out = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "corridor_per_deg": {str(s): round(100.0 * np.mean([r["success"] for r in corridor_rows if r["sector_deg"] == s]), 1)
                             for s in (-45, -30, -15, 0, 15, 30, 45)},
        "corridor_overall": round(100.0 * np.mean([r["success"] for r in corridor_rows]), 1),
        "km_summary": [
            {"delay_ms": ds * 50, "noise_mm": ns, "k": k, "m": m,
             "succ": round(100.0 * np.mean([r["success"] for r in km_rows if r["delay_ms"] == ds*50 and r["noise_mm"] == ns and r["k"] == k and r["m"] == m]), 1)}
            for ds, combos in [(2, [(0, 0), (0, 2), (1, 1), (2, 0)]), (4, [(0, 0), (0, 4), (2, 2), (4, 0)])]
            for ns in (1.8, 6.5) for k, m in combos
        ],
        "substep_summary": {
            md: {
                "succ": round(100.0 * np.mean([r["success"] for r in sub_rows if r["mode"] == md]), 1),
                "low_peak_N": round(float(np.mean([r["peak_force_N"] for r in sub_rows if r["mode"] == md and r["theta"] <= 1.0])), 1),
                "high_peak_N": round(float(np.mean([r["peak_force_N"] for r in sub_rows if r["mode"] == md and r["theta"] >= 20.0])), 1),
            }
            for md in ("unmasked_cross_feedback", "feedback_bounded_20hz", "feedback_bounded_500hz_substep")
        },
    }
    (DATA_OUT / "math_exp7_fix_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
