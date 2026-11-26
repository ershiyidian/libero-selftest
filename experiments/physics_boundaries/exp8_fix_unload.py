import csv, json, multiprocessing, os, sys, concurrent.futures
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
    os.environ["MKL_NUM_THREADS"] = "1"


def _unload_job(args):
    theta, mode = args
    ep_nom = create("L031", 0, 0.1)
    t_nom, r_nom = source_control_targets(ep_nom)
    a_nom = ep_nom.expert_actions.copy()
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

        # True high-resistance continuation: extend push along closure axis beyond nominal stroke
        push_dir = t_nom[-1] - t_nom[tau_fb]
        push_dir = push_dir / max(float(np.linalg.norm(push_dir)), 1e-6)
        def high_target(idx):
            if idx < tau_fb:
                return t_nom[idx]
            # Smoothly advance 3.2 mm/step along push axis up to closure
            return t_nom[tau_fb] + (idx - tau_fb + 1) * 0.0032 * push_dir

        steps_used = 0
        closed_step = None
        substep_unloaded = False
        is_high_est = False
        q_at_begin = float(ep.joint_pos())

        for i in range(115):
            i_c = min(i, len(a_nom) - 1)
            if i == begin:
                q_at_begin = float(ep.joint_pos())
            if i == tau_fb:
                # Joint displacement over 4-step probe separates normal (~20 mm) vs high (<15 mm)
                j_disp_mm = abs(float(ep.joint_pos()) - q_at_begin) * 1000.0
                is_high_est = (j_disp_mm < 15.0)

            if mode == "unmasked_cross_feedback":
                # 50/50 cross-feedback mode average over the 71-step nominal horizon
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
    jobs = [(th, md)
            for md in ("unmasked_cross_feedback", "feedback_bounded_20hz", "feedback_bounded_500hz_substep")
            for th in (0.1, 0.5, 1.0, 20.0, 30.0, 35.0)]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=12, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        rows = list(pool.map(_unload_job, jobs))
    with (DATA_OUT / "math_exp8_substep_unload.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    out = {}
    for md in ("unmasked_cross_feedback", "feedback_bounded_20hz", "feedback_bounded_500hz_substep"):
        low = [r for r in rows if r["mode"] == md and r["theta"] <= 1.0]
        high = [r for r in rows if r["mode"] == md and r["theta"] >= 20.0]
        out[md] = {"low_succ": f"{sum(r['success'] for r in low)}/{len(low)}",
                   "high_succ": f"{sum(r['success'] for r in high)}/{len(high)}",
                   "low_peak_N": round(float(np.mean([r["peak_force_N"] for r in low])), 1),
                   "high_peak_N": round(float(np.mean([r["peak_force_N"] for r in high])), 1)}
    print(json.dumps(out, indent=2))
