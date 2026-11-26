#!/usr/bin/env python3
"""
Unified 500 Hz CPU MuJoCo Worker (v21) for SmolVLA-450M 3.5-Hour Suite:
  1. Compound Delay x Articulated Contact (`init_compound_v21` / `step_compound_v21`):
     Simultaneous 0/200/300 ms observation lag (d in {0, 4, 6}) + unseen 8-level joint Coulomb resistance
     with 500 Hz world-frame net wrench, post-deadband Wald SPRT, regime-gated Floquet damping, and
     500 Hz substep closure target withdrawal.
  2. Non-Stationary Stochastic Lag Jitter (`init_jitter_v21` / `step_jitter_v21`):
     Real-time double-buffered async execution (period H=9) with Markov + spike delay jitter d_n in [2..5] or [4..7],
     comparing Raw, VLASH, Nominal Prefix-Align, Guided RTC, Training-RTC, Timestamped Ring-Buffer Floquet,
     and Training-RTC + Timestamped Floquet.
  3. Post-Contact Mechanical Bifurcation (`init_bifurc_v18` / `step_bifurc_v18`):
     Exact dual-transition stall & release event replanning + PACE for Stage 4 Subspace WiSE-FT Pareto sweep.
"""
from __future__ import annotations
from collections import deque
from functools import partial
import math
import os
import pickle
import struct
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "cache/libero_config")

_real_stdout = sys.stdout.buffer
sys.stdout = sys.stderr

import mujoco
from robosuite.utils.control_utils import set_goal_position, set_goal_orientation
from vla_branch_lab.latency.resources import prepare_runtime
from vla_branch_lab.latency.initial_states import environment_config, reset_study_state
from vla_branch_lab.envs.libero import LiberoEnvironment
from vla_branch_lab.latency.study_config import load_study

prepare_runtime(ROOT)
os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)

DT = 0.050
F_SAFE_LIMIT = 180.0
SLIDE_THETAS = [0.5, 2.0, 5.0, 9.0, 32.0, 44.0, 58.0, 72.0]
HINGE_THETAS = [0.05, 0.20, 0.50, 0.90, 3.8, 5.5, 7.5, 10.0]
GAMMA_BY_DELAY = {0: 1.0, 1: 1.0, 2: 0.92, 3: 0.72, 4: 0.54, 5: 0.60, 6: 0.66, 7: 0.52, 8: 0.42}


def _read_exact(stream, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg():
    hdr = _read_exact(sys.stdin.buffer, 4)
    if not hdr or len(hdr) < 4:
        return None
    (n,) = struct.unpack("!I", hdr)
    data = _read_exact(sys.stdin.buffer, n)
    return pickle.loads(data)


def send_msg(obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    _real_stdout.write(struct.pack("!I", len(data)) + data)
    _real_stdout.flush()


def quat_to_aa(quat: np.ndarray) -> np.ndarray:
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.acos(w)
    return (quat[:3] / den * angle).astype(np.float32)


def obs_to_smolvla(obs: dict, s8_override: np.ndarray | None = None):
    cam1 = np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1].copy()
    cam2 = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)[::-1, ::-1].copy()
    if s8_override is not None:
        state8 = np.asarray(s8_override, dtype=np.float32)
    else:
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        aa = quat_to_aa(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
        grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)[:2]
        state8 = np.concatenate([eef_pos, aa, grip]).astype(np.float32)
    return cam1.tobytes(), cam2.tobytes(), state8.tolist()


def vlash_rollforward_s8(obs: dict, ctrl, cmd_list: list[np.ndarray]) -> np.ndarray:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
    aa = quat_to_aa(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)).astype(np.float64)
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)[:2].copy()
    for q_cmd in cmd_list:
        delta = ctrl.scale_action(np.asarray(q_cmd[:6], dtype=np.float64))
        eef_pos = eef_pos + delta[:3]
        if ctrl.position_limits is not None:
            eef_pos = np.clip(eef_pos, ctrl.position_limits[0], ctrl.position_limits[1])
        aa = aa + 0.5 * delta[3:6]
        if float(q_cmd[6]) > 0.2:
            grip = np.array([0.002, -0.002], dtype=np.float32)
        elif float(q_cmd[6]) < -0.2:
            grip = np.array([0.038, -0.038], dtype=np.float32)
    return np.concatenate([eef_pos.astype(np.float32), aa.astype(np.float32), grip]).astype(np.float32)


def find_manipuland_body(env) -> int | None:
    m = env.env.sim.model
    lang = str(env.task.language).lower()
    keys = ["bowl", "mug", "moka", "wine_bottle", "cream_cheese", "alphabet_soup", "butter", "milk", "ketchup", "tomato_sauce", "pudding"]
    pref = [k for k in keys if k.replace("_", " ") in lang] + keys
    for k in pref:
        for bid in range(m.nbody):
            bn = (m.body_id2name(bid) or "").lower()
            if k in bn and "visual" not in bn:
                return int(bid)
    return None


def find_all_manipuland_bodies(env) -> list[int]:
    m = env.env.sim.model
    mm = m._model
    out = []
    skip = ("world", "table", "robot0", "gripper0", "mount", "cabinet", "stove", "microwave", "rack", "plate", "basket", "caddy")
    for bid in range(1, m.nbody):
        bn = (m.body_id2name(bid) or "").lower()
        if "visual" in bn or any(s in bn for s in skip):
            continue
        if int(mm.body_jntnum[bid]) > 0 or any(k in bn for k in ("bowl", "mug", "moka", "bottle", "cheese", "soup", "butter", "milk", "ketchup", "sauce", "pudding", "ramekin", "book")):
            out.append(int(bid))
    return out


def find_articulated_joint(env, joint_hint: str) -> tuple[int, int, int, set[int]]:
    m = env.env.sim.model
    mm = m._model
    jid = None
    for j in range(m.njnt):
        jn = m.joint_id2name(j) or ""
        if joint_hint in jn:
            jid = int(j)
            break
    if jid is None:
        for j in range(m.njnt):
            jn = m.joint_id2name(j) or ""
            if not jn.startswith(("robot0", "gripper0")) and int(mm.jnt_type[j]) in (2, 3):
                jid = int(j)
                break
    assert jid is not None, f"No articulated joint matching {joint_hint}"
    jtype = int(mm.jnt_type[jid])
    dof_id = int(mm.jnt_dofadr[jid])
    t_body = int(mm.jnt_bodyid[jid])
    fbodies = {t_body}
    for b in range(t_body + 1, mm.nbody):
        if int(mm.body_parentid[b]) in fbodies:
            fbodies.add(b)
    return jid, jtype, dof_id, fbodies


def pace_horizon(chunk: np.ndarray, h_min: int = 6, h_max: int = 24) -> int:
    sp = np.linalg.norm(chunk[:26, :3], axis=-1)
    sm = np.convolve(sp, np.ones(3, dtype=np.float32) / 3.0, mode="same")
    p_ref = float(np.max(sm[:h_max])) if len(sm) >= h_max else 1.0
    for r_idx in range(h_min, min(22, len(sm) - 1)):
        if sm[r_idx] <= sm[r_idx - 1] and sm[r_idx] <= sm[r_idx + 1] and (p_ref - float(sm[r_idx])) >= 0.12:
            return r_idx + 1
    return h_max


def step_env_with_500hz_net_force(
    env,
    raw_cmd: np.ndarray,
    fbodies: set[int],
    ctrl,
    qadr: int,
    q_init: float,
    j_min: float,
    j_max: float,
    j_span: float,
    withdraw_mode: bool,
    force_withdraw_now: bool,
    u_open: np.ndarray | None = None,
    init_prev_f: float = 0.0,
) -> tuple[dict, bool, float, float, bool, float]:
    sim = env.env.sim
    inner = env.env.env
    model, data = sim.model._model, sim.data._data
    substeps = int(round(DT / model.opt.timestep))
    force6 = np.zeros(6, dtype=np.float64)
    imp_net = 0.0
    imp_axial = 0.0
    act = env.clip_action(raw_cmd.copy())
    withdrew_any = bool(force_withdraw_now)
    prev_net_f = float(init_prev_f)

    act_eff = act.copy()
    if withdraw_mode and not force_withdraw_now and u_open is not None and init_prev_f > 25.0:
        q0 = float(data.qpos[qadr])
        tr0 = abs(q0 - q_init) / j_span
        dl0 = min(abs(q0 - j_min), abs(q0 - j_max)) / j_span
        if tr0 > 0.78 and dl0 < 0.065:
            p_open0 = float(np.dot(act_eff[:3], u_open))
            if p_open0 > 0.0:
                act_eff[:3] = act_eff[:3] - (p_open0 + 0.08) * u_open

    for s_idx in range(substeps):
        mujoco.mj_step1(model, data)
        q_sub = float(data.qpos[qadr])
        travel_sub = abs(q_sub - q_init) / j_span
        dist_lim = min(abs(q_sub - j_min), abs(q_sub - j_max)) / j_span
        if withdraw_mode and force_withdraw_now:
            withdrew_any = True
            ctrl.update()
            ee_cur = ctrl.ee_pos.copy()
            push_dir = ctrl.goal_pos - ee_cur
            pn = float(np.linalg.norm(push_dir))
            wd_vec = -(push_dir / pn) * 0.004 if pn > 1e-4 else np.zeros(3, dtype=np.float64)
            ctrl.goal_pos = ee_cur + wd_vec
            inner._pre_action(np.r_[np.zeros(6), float(act_eff[6])], policy_step=False)
        elif withdraw_mode and (travel_sub > 0.78 and dist_lim < 0.065) and u_open is not None and prev_net_f > 18.0:
            ctrl.update()
            ee_cur = ctrl.ee_pos.copy()
            push_dir = ctrl.goal_pos - ee_cur
            push_open = float(np.dot(push_dir, u_open))
            if push_open > 0.0002:
                withdrew_any = True
                ctrl.goal_pos = ee_cur + (push_dir - (push_open + 0.003) * u_open)
                inner._pre_action(np.r_[np.zeros(6), float(act_eff[6])], policy_step=False)
            else:
                inner._pre_action(act_eff, policy_step=(s_idx == 0))
        elif withdraw_mode and prev_net_f > 148.0:
            ctrl.update()
            ctrl.goal_pos = 0.65 * ctrl.ee_pos + 0.35 * ctrl.goal_pos
            inner._pre_action(np.r_[np.zeros(6), float(act_eff[6])], policy_step=False)
        else:
            inner._pre_action(act_eff, policy_step=(s_idx == 0))
        mujoco.mj_step2(model, data)

        f_vec_world = np.zeros(3, dtype=np.float64)
        for cj in range(data.ncon):
            c = data.contact[cj]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (b1 in fbodies) != (b2 in fbodies):
                n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
                n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
                if n1.startswith(("robot0", "gripper0")) or n2.startswith(("robot0", "gripper0")):
                    mujoco.mj_contactForce(model, data, cj, force6)
                    R_c = np.asarray(c.frame, dtype=np.float64).reshape(3, 3)
                    f_w = R_c.T @ force6[:3]
                    sign = 1.0 if (b2 in fbodies) else -1.0
                    f_vec_world += sign * f_w
        prev_net_f = float(np.linalg.norm(f_vec_world))
        imp_net += prev_net_f * float(model.opt.timestep)
        if u_open is not None:
            imp_axial += abs(float(np.dot(f_vec_world, u_open))) * float(model.opt.timestep)

    step_sustained_f = imp_net / DT
    step_axial_f = imp_axial / DT
    inner.timestep += 1
    obs = inner._get_observations(force_update=True)
    ok = bool(env.env.check_success())
    return obs, ok, step_sustained_f, imp_net, withdrew_any, step_axial_f


def sample_next_jitter_delay(rng: np.random.Generator, regime: str, prev_d: int | None) -> tuple[int, int]:
    """Returns (realized_d_n, nominal_d_bar) on 20 Hz step grid."""
    if regime == "stat_0":
        return 0, 0
    if regime == "stat_4":
        return 4, 4
    if regime == "stat_6":
        return 6, 6
    elif regime == "j_mod_2_5":
        d_min, d_max, d_nom = 2, 5, 4
        if prev_d is None:
            return 4, d_nom
        if rng.random() < 0.22:
            return 5, d_nom
        step_d = int(rng.choice([-1, 0, 0, 1]))
        return int(np.clip(prev_d + step_d, d_min, d_max)), d_nom
    elif regime == "j_sev_4_7":
        d_min, d_max, d_nom = 4, 7, 6
        if prev_d is None:
            return 5, d_nom
        if rng.random() < 0.30:
            return 7, d_nom
        step_d = int(rng.choice([-2, -1, 0, 1, 2]))
        return int(np.clip(prev_d + step_d, d_min, d_max)), d_nom
    elif regime == "stat_8":
        return 8, 8
    elif regime == "step_2_to_6":
        return (2, 4) if (prev_d is None or prev_d < 6) else (6, 4)
    elif regime == "burst_2_7":
        return (7, 4) if (rng.random() < 0.30) else (2, 4)
    raise ValueError(f"Unknown jitter regime: {regime}")


def main():
    cfg = load_study(str(ROOT / "configs/latency_next/quest.yaml"))
    cfg["image_size"] = 256
    env = None
    cur_ek = None
    st = None

    while True:
        msg = recv_msg()
        if msg is None or msg[0] == "close":
            if env is not None:
                env.close()
            break
        cmd = msg[0]

        if cmd == "init_compound_v21":
            _, suite, tid, sid, joint_hint, level_idx, delay, delay_mode, probe_mode, gamma_stab, horizon = msg[:11]
            extra_cfg = msg[11] if len(msg) > 11 and isinstance(msg[11], dict) else {}
            sprt_B = float(extra_cfg.get("sprt_B", 5.2))
            deadband_k = int(extra_cfg.get("deadband", 2))
            res_law = str(extra_cfg.get("resistance_law", "coulomb"))
            task = {"suite": suite, "task_id": tid, "role": "primary"}
            ek = (suite, tid)
            if cur_ek != ek:
                if env is not None:
                    env.close()
                env = LiberoEnvironment(environment_config(cfg, task), tid)
                cur_ek = ek
            obs, _ = reset_study_state(env, task, {"state_origin": "benchmark", "state_id": sid, "split": "benchmark"}, 0, int(delay))
            jid, jtype, dof_id, fbodies = find_articulated_joint(env, joint_hint)
            mm = env.env.sim.model._model
            base_fl = float(mm.dof_frictionloss[dof_id])
            base_damp = float(mm.dof_damping[dof_id])
            base_stiff = float(mm.jnt_stiffness[jid])
            for _ in range(10):
                obs, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            theta_phys = float((SLIDE_THETAS if jtype == 2 else HINGE_THETAS)[int(level_idx)])
            if res_law == "viscous_spring":
                mm.dof_frictionloss[dof_id] = base_fl + 0.35 * theta_phys
                mm.dof_damping[dof_id] = base_damp + 0.65 * theta_phys * 2.2
                mm.jnt_stiffness[jid] = base_stiff + 0.45 * theta_phys * 3.0
            else:
                mm.dof_frictionloss[dof_id] = theta_phys
            env.env.sim.forward()
            ctrl = env.env.robots[0].controller
            qadr_init = int(mm.jnt_qposadr[jid])
            q_init = float(env.env.sim.data.qpos[qadr_init])
            j_min0, j_max0 = float(mm.jnt_range[jid, 0]), float(mm.jnt_range[jid, 1])
            s_open = 1.0 if abs(j_max0 - q_init) >= abs(q_init - j_min0) else -1.0
            g_ids = [g for g in range(mm.ngeom) if int(mm.geom_bodyid[g]) in fbodies]
            p0 = np.mean([env.env.sim.data._data.geom_xpos[g].copy() for g in g_ids], axis=0) if g_ids else env.env.sim.data._data.xpos[int(mm.jnt_bodyid[jid])].copy()
            env.env.sim.data.qpos[qadr_init] = q_init + s_open * 0.02
            env.env.sim.forward()
            p1 = np.mean([env.env.sim.data._data.geom_xpos[g].copy() for g in g_ids], axis=0) if g_ids else env.env.sim.data._data.xpos[int(mm.jnt_bodyid[jid])].copy()
            env.env.sim.data.qpos[qadr_init] = q_init
            env.env.sim.forward()
            dp_open = p1 - p0
            u_open = dp_open / max(float(np.linalg.norm(dp_open)), 1e-6)
            rng = np.random.default_rng(20261125 + tid * 97 + sid * 13 + int(level_idx) * 7)
            d_int = int(delay)
            st = {
                "kind": "compound21",
                "delay": d_int,
                "delay_mode": str(delay_mode),
                "probe_mode": str(probe_mode),
                "gamma_stab": float(gamma_stab),
                "level_idx": int(level_idx),
                "theta_phys": theta_phys,
                "jtype": jtype,
                "horizon": int(horizon),
                "ctrl": ctrl,
                "last_ori": ctrl.goal_ori.copy(),
                "ee_hist": deque(maxlen=max(6, d_int + 1)),
                "cmd_q": deque([np.zeros(7, np.float32) for _ in range(d_int)]),
                "jid": jid,
                "dof_id": dof_id,
                "base_fl": base_fl,
                "base_damp": base_damp,
                "base_stiff": base_stiff,
                "sprt_B": sprt_B,
                "deadband_k": deadband_k,
                "fbodies": fbodies,
                "q_init": q_init,
                "u_open": u_open,
                "s_open": s_open,
                "step": 0,
                "replan_calls": 0,
                "peak_force_N": 0.0,
                "impulse_Ns": 0.0,
                "in_contact_steps": 0,
                "probe_steps": 0,
                "decided": False,
                "is_high_est": False,
                "llr": 0.0,
                "withdraw_active": False,
                "rng": rng,
                "first_succ": 0 if bool(env.env.check_success()) else None,
            }
            c1, c2, s8 = obs_to_smolvla(obs)
            send_msg(("obs", c1, c2, s8, str(env.task.language), st["first_succ"] is not None, st["step"]))

        elif cmd == "step_compound_v21":
            _, chunk_l, n_exec_host = msg
            chunk = np.asarray(chunk_l, dtype=np.float32)
            ctrl = st["ctrl"]
            delay = st["delay"]
            delay_mode = st["delay_mode"]
            probe_mode = st["probe_mode"]
            gamma_stab = st["gamma_stab"]
            horizon = st["horizon"]
            mm = env.env.sim.model._model
            qadr = int(mm.jnt_qposadr[st["jid"]])
            j_min, j_max = float(mm.jnt_range[st["jid"], 0]), float(mm.jnt_range[st["jid"], 1])
            j_span = max(abs(j_max - j_min), 0.05)
            withdraw_mode = (probe_mode == "wald_sprt_withdraw")
            K = len(chunk)

            if delay_mode == "autohorizon_align" and n_exec_host is not None:
                c_exec = int(np.clip(int(n_exec_host), max(4, delay), 14))
            else:
                c_exec = 16 if delay == 0 else 8
            c_exec = max(1, min(c_exec, K))
            st["replan_calls"] += 1

            use_align = (delay > 0) and (delay_mode in ("rtc_align", "autohorizon_align", "floquet_pref", "naive_mult_floquet_pref"))
            eff_chunk = np.array([chunk[min(m + delay, K - 1)] if use_align else chunk[m] for m in range(c_exec)], dtype=np.float32)

            obs = None
            for j in range(c_exec):
                if st["step"] >= horizon or st["first_succ"] is not None:
                    break
                ctrl.update()
                ee_before = ctrl.ee_pos.copy()
                st["ee_hist"].append(ee_before)
                q_before = float(env.env.sim.data.qpos[qadr])
                raw_in = env.clip_action(eff_chunk[j].copy())

                if delay == 0:
                    d_raw = raw_in
                else:
                    st["cmd_q"].append(raw_in)
                    d_raw = st["cmd_q"].popleft()
                    if st["step"] < delay:
                        d_raw = np.zeros(7, dtype=np.float32)

                q_cur_travel = abs(q_before - st["q_init"]) / j_span
                probe_gain = 1.85 if (st["decided"] and st["is_high_est"] and q_cur_travel < 0.84) else 1.0
                delay_gain = 1.0
                if delay > 0 and st["step"] >= delay and delay_mode in ("floquet_pref", "naive_mult_floquet_pref"):
                    ee_enq = st["ee_hist"][0]
                    x_dot_hat = (ee_before - ee_enq) / (delay * DT)
                    nx2 = float(np.dot(x_dot_hat, x_dot_hat))
                    step_gdot = ctrl.scale_action(d_raw[:6])[:3] / DT
                    w_star = float(np.clip(1.0 - float(np.dot(x_dot_hat, step_gdot)) / (nx2 + 1e-6), 0.0, 1.0))
                    delay_gain = (1.0 - w_star) * 1.0 + w_star * gamma_stab

                if delay_mode == "floquet_pref" and st["decided"] and st["is_high_est"] and q_cur_travel < 0.84:
                    # Regime-gated coupling: confirmed high joint friction already overdamps contact; release free-space velocity damping!
                    total_gain = probe_gain
                else:
                    total_gain = delay_gain * probe_gain

                if abs(total_gain - 1.0) > 1e-3 and (delay == 0 or st["step"] >= delay):
                    delta = ctrl.scale_action(d_raw[:6])
                    pos = set_goal_position(total_gain * delta[:3], ee_before, position_limit=ctrl.position_limits)
                    if any(not math.isclose(float(v), 0.0) for v in delta[3:]):
                        st["last_ori"] = set_goal_orientation(total_gain * delta[3:], ctrl.ee_ori_mat, orientation_limit=ctrl.orientation_limits)
                    orig = ctrl.set_goal
                    ctrl.set_goal = partial(orig, set_pos=pos, set_ori=st["last_ori"])
                    try:
                        obs, ok, pf, imp, wd_now, f_ax = step_env_with_500hz_net_force(
                            env, np.r_[np.zeros(6), float(d_raw[6])], st["fbodies"], ctrl,
                            qadr, st["q_init"], j_min, j_max, j_span, withdraw_mode, st["withdraw_active"], st["u_open"], st.get("last_pf", 0.0),
                        )
                    finally:
                        ctrl.set_goal = orig
                else:
                    obs, ok, pf, imp, wd_now, f_ax = step_env_with_500hz_net_force(
                        env, d_raw, st["fbodies"], ctrl, qadr, st["q_init"], j_min, j_max, j_span,
                        withdraw_mode, st["withdraw_active"], st["u_open"], st.get("last_pf", 0.0),
                    )


                st["last_pf"] = pf
                st["peak_force_N"] = max(st["peak_force_N"], pf)
                st["impulse_Ns"] += imp
                st["step"] += 1
                q_after = float(env.env.sim.data.qpos[qadr])
                dq_open_mm = (q_after - q_before) * st["s_open"] * (1000.0 if st["jtype"] == 2 else 220.0)
                dq_equiv_mm = max(0.0, dq_open_mm)
                cmd_vec = ctrl.scale_action(d_raw[:6])[:3]
                cmd_open_mm = float(np.dot(cmd_vec, st["u_open"])) * 1000.0

                if not st["decided"] and probe_mode in ("fixed_tp6_hold", "wald_sprt_withdraw"):
                    if (f_ax > 6.5 and cmd_open_mm > 1.0) or dq_equiv_mm > 0.55:
                        f_break = 23.0 if st["jtype"] == 2 else 14.0
                        db_k = int(st.get("deadband_k", 2))
                        if st["in_contact_steps"] < db_k or f_ax >= f_break or dq_equiv_mm > 0.75:
                            st["in_contact_steps"] += 1
                        m_step = st["in_contact_steps"]
                        if st["in_contact_steps"] > db_k and (f_ax >= f_break or dq_equiv_mm > 0.75):
                            y_obs = min(dq_equiv_mm, 6.0) + float(st["rng"].normal(0.0, 0.85))
                            d_low = (y_obs - 3.8) ** 2
                            d_high = (y_obs - 0.4) ** 2
                            step_llr = float(np.clip((d_low - d_high) / (2.0 * (1.30 ** 2)), -4.8, 3.5))
                            st["llr"] += step_llr
                            if probe_mode == "fixed_tp6_hold" and m_step == 6:
                                st["decided"] = True
                                st["probe_steps"] = 6
                                st["is_high_est"] = (st["llr"] > 0.0)
                            elif probe_mode == "wald_sprt_withdraw" and (abs(st["llr"]) >= float(st.get("sprt_B", 5.2)) or m_step >= 8):
                                st["decided"] = True
                                st["probe_steps"] = m_step
                                st["is_high_est"] = (st["llr"] > 0.0)
                                if delay == 0 and j + 1 < c_exec and not ok:
                                    break

                if ok and st["first_succ"] is None:
                    if not withdraw_mode:
                        rem = max(4, min(8, c_exec - j))
                        for _ in range(rem):
                            _, _, pf_p, imp_p, _, _ = step_env_with_500hz_net_force(
                                env, d_raw, st["fbodies"], ctrl, qadr, st["q_init"], j_min, j_max, j_span, False, False,
                            )
                            st["peak_force_N"] = max(st["peak_force_N"], pf_p)
                            st["impulse_Ns"] += imp_p
                    else:
                        for _ in range(2):
                            _, _, pf_p, imp_p, _, _ = step_env_with_500hz_net_force(
                                env, d_raw, st["fbodies"], ctrl, qadr, st["q_init"], j_min, j_max, j_span, True, True,
                            )
                            st["peak_force_N"] = max(st["peak_force_N"], pf_p)
                            st["impulse_Ns"] += imp_p
                    mm.dof_frictionloss[st["dof_id"]] = st["base_fl"]; mm.dof_damping[st["dof_id"]] = st.get("base_damp", 0.0); mm.jnt_stiffness[st["jid"]] = st.get("base_stiff", 0.0)
                    st["first_succ"] = st["step"]
                    break

            done = (st["first_succ"] is not None) or (st["step"] >= horizon)
            if done:
                mm.dof_frictionloss[st["dof_id"]] = st["base_fl"]; mm.dof_damping[st["dof_id"]] = st.get("base_damp", 0.0); mm.jnt_stiffness[st["jid"]] = st.get("base_stiff", 0.0)
                gt_high = (st["level_idx"] >= 4)
                reg_ok = int(st["is_high_est"] == gt_high) if st["decided"] else 0
                meta = {
                    "replan_calls": st["replan_calls"],
                    "probe_steps": st["probe_steps"],
                    "decided": int(st["decided"]),
                    "regime_correct": reg_ok,
                    "peak_force_N": round(st["peak_force_N"], 2),
                    "overload_180N": int(st["peak_force_N"] > F_SAFE_LIMIT),
                    "impulse_Ns": round(st["impulse_Ns"], 3),
                }
                send_msg(("done", st["first_succ"] is not None, st["step"], meta))
            else:
                s8_ov = vlash_rollforward_s8(obs, ctrl, list(st["cmd_q"])) if (delay_mode == "vlash_zero" and delay > 0) else None
                c1, c2, s8 = obs_to_smolvla(obs, s8_override=s8_ov)
                send_msg(("obs", c1, c2, s8, "", False, st["step"]))

        elif cmd == "init_jitter_v21":
            _, suite, tid, sid, regime, mode, horizon = msg
            task = {"suite": suite, "task_id": tid, "role": "primary"}
            ek = (suite, tid)
            if cur_ek != ek:
                if env is not None:
                    env.close()
                env = LiberoEnvironment(environment_config(cfg, task), tid)
                cur_ek = ek
            obs, _ = reset_study_state(env, task, {"state_origin": "benchmark", "state_id": sid, "split": "benchmark"}, 0, 4)
            for _ in range(10):
                obs, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            ctrl = env.env.robots[0].controller
            obj_bid = find_manipuland_body(env)
            init_obj_z = float(env.env.sim.data.body_xpos[obj_bid][2]) if obj_bid is not None else 0.0
            rng = np.random.default_rng(20261126 + tid * 131 + sid * 29)
            d0, d_nom = sample_next_jitter_delay(rng, str(regime), None)
            ctrl.update()
            st = {
                "kind": "jitter21",
                "regime": str(regime),
                "mode": str(mode),
                "horizon": int(horizon),
                "ctrl": ctrl,
                "last_ori": ctrl.goal_ori.copy(),
                "rng": rng,
                "d_cur": d0,
                "d_nom": d_nom,
                "d_est": d_nom,
                "d_hist": [],
                "splice_jumps_mm": [],
                "ee_ring": {0: ctrl.ee_pos.copy()},
                "active_t_obs": 0,
                "active_d_real": 0,
                "active_plan": deque(),
                "prev_splice_offset": 0,
                "step": 0,
                "replan_calls": 0,
                "obj_bid": obj_bid,
                "init_obj_z": init_obj_z,
                "first_contact_r_mm": None,
                "max_lift_mm": 0.0,
                "dropped_after_lift": 0,
                "first_succ": 0 if bool(env.env.check_success()) else None,
            }
            c1, c2, s8 = obs_to_smolvla(obs)
            # Send causal delay estimate d_est and leftover start index 9 for model-side RTC methods
            send_msg(("obs_jitter", c1, c2, s8, str(env.task.language), st["first_succ"] is not None, st["step"], st["d_est"], 9))

        elif cmd == "step_jitter_v21":
            _, chunk_l = msg
            chunk = np.asarray(chunk_l, dtype=np.float32)
            K = len(chunk)
            ctrl = st["ctrl"]
            mode = st["mode"]
            horizon = st["horizon"]
            H_period = 9
            t_obs = st["step"]
            d_real = st["d_cur"]
            d_nom = st["d_nom"]
            d_est = st["d_est"]
            st["d_hist"].append(int(d_real))
            st["replan_calls"] += 1

            ctrl.update()
            st["ee_ring"][t_obs] = ctrl.ee_pos.copy()
            obs = None
            last_exec_delta = None
            sh_used = 0

            for s_idx in range(H_period):
                if st["step"] >= horizon or st["first_succ"] is not None:
                    break
                ctrl.update()
                ee_now = ctrl.ee_pos.copy()
                st["ee_ring"][st["step"]] = ee_now

                if s_idx < d_real:
                    # In-flight phase: execute committed continuation from previous chunk (with consistent Floquet damping!)
                    has_plan = len(st["active_plan"]) > 0
                    raw_cmd = st["active_plan"].popleft() if has_plan else np.zeros(7, dtype=np.float32)
                    raw_cmd = env.clip_action(raw_cmd)
                    ad = int(st.get("active_d_real", 0))
                    if has_plan and mode in ("ts_floquet_blend", "ts_floquet_pref", "training_rtc_ts_floquet", "ts_floquet_pref_eapn", "training_rtc_eapn_ts_floquet") and ad > 0:
                        t_a = int(st.get("active_t_obs", 0))
                        ee_obs = st["ee_ring"].get(t_a, ee_now)
                        dt_eff = max(1, st["step"] - t_a) * DT
                        x_dot_hat = (ee_now - ee_obs) / dt_eff
                        nx2 = float(np.dot(x_dot_hat, x_dot_hat))
                        step_gdot = ctrl.scale_action(raw_cmd[:6])[:3] / DT
                        w_star = float(np.clip(1.0 - float(np.dot(x_dot_hat, step_gdot)) / (nx2 + 1e-6), 0.0, 1.0))
                        g_stab = GAMMA_BY_DELAY.get(ad, 0.42)
                        aligned = (mode != "ts_floquet_blend")
                        eff_gain = ((1.0 - w_star) * 1.0 + w_star * g_stab) if aligned else ((1.0 - w_star) * g_stab + w_star * 1.0)
                        ref_pos = ee_now if aligned else ((1.0 - w_star) * ee_now + w_star * ee_obs)
                        delta = ctrl.scale_action(raw_cmd[:6])
                        pos = set_goal_position(eff_gain * delta[:3], ref_pos, position_limit=ctrl.position_limits)
                        if any(not math.isclose(float(v), 0.0) for v in delta[3:]):
                            st["last_ori"] = set_goal_orientation(eff_gain * delta[3:], ctrl.ee_ori_mat, orientation_limit=ctrl.orientation_limits)
                        orig = ctrl.set_goal
                        ctrl.set_goal = partial(orig, set_pos=pos, set_ori=st["last_ori"])
                        try:
                            obs, ok, _ = env.step(np.r_[np.zeros(6), float(raw_cmd[6])])
                        finally:
                            ctrl.set_goal = orig
                        last_exec_delta = (eff_gain * delta[:3]).copy()
                    else:
                        last_exec_delta = ctrl.scale_action(raw_cmd[:6])[:3].copy()
                        obs, ok, _ = env.step(raw_cmd)
                else:
                    if s_idx == d_real:
                        # Chunk arrives at t_arr = t_obs + d_real!
                        if mode in ("raw", "vlash_zero", "ts_floquet_blend"):
                            sh_used = 0
                        elif mode in ("nominal_pref_align", "eapn_base"):
                            sh_used = d_nom
                        elif mode in ("rtc_guided", "training_rtc", "training_rtc_eapn"):
                            # Model-side conditioning used causal d_est before spike occurred
                            sh_used = d_est
                        elif mode in ("ts_floquet_pref", "training_rtc_ts_floquet", "ts_floquet_pref_eapn", "training_rtc_eapn_ts_floquet"):
                            # Hardware timestamp observes exact arrival lag d_real = t_arr - t_obs!
                            sh_used = d_real
                        else:
                            sh_used = 0
                        st["active_plan"].clear()
                        st["active_t_obs"] = int(t_obs)
                        st["active_d_real"] = int(d_real)
                        for m_i in range(sh_used, K):
                            st["active_plan"].append(chunk[m_i].copy())
                        if last_exec_delta is not None and len(st["active_plan"]) > 0:
                            new_delta = ctrl.scale_action(env.clip_action(st["active_plan"][0])[:6])[:3]
                            st["splice_jumps_mm"].append(float(np.linalg.norm(new_delta - last_exec_delta)) * 1000.0)

                    raw_cmd = env.clip_action(st["active_plan"].popleft() if len(st["active_plan"]) > 0 else chunk[-1].copy())
                    if mode in ("ts_floquet_blend", "ts_floquet_pref", "training_rtc_ts_floquet", "ts_floquet_pref_eapn", "training_rtc_eapn_ts_floquet") and d_real > 0:
                        ee_obs = st["ee_ring"].get(t_obs, ee_now)
                        x_dot_hat = (ee_now - ee_obs) / (d_real * DT)
                        nx2 = float(np.dot(x_dot_hat, x_dot_hat))
                        step_gdot = ctrl.scale_action(raw_cmd[:6])[:3] / DT
                        w_star = float(np.clip(1.0 - float(np.dot(x_dot_hat, step_gdot)) / (nx2 + 1e-6), 0.0, 1.0))
                        g_stab = GAMMA_BY_DELAY.get(d_real, 0.64)
                        aligned = (mode != "ts_floquet_blend")
                        eff_gain = ((1.0 - w_star) * 1.0 + w_star * g_stab) if aligned else ((1.0 - w_star) * g_stab + w_star * 1.0)
                        ref_pos = ee_now if aligned else ((1.0 - w_star) * ee_now + w_star * ee_obs)
                        delta = ctrl.scale_action(raw_cmd[:6])
                        pos = set_goal_position(eff_gain * delta[:3], ref_pos, position_limit=ctrl.position_limits)
                        if any(not math.isclose(float(v), 0.0) for v in delta[3:]):
                            st["last_ori"] = set_goal_orientation(eff_gain * delta[3:], ctrl.ee_ori_mat, orientation_limit=ctrl.orientation_limits)
                        orig = ctrl.set_goal
                        ctrl.set_goal = partial(orig, set_pos=pos, set_ori=st["last_ori"])
                        try:
                            obs, ok, _ = env.step(np.r_[np.zeros(6), float(raw_cmd[6])])
                        finally:
                            ctrl.set_goal = orig
                        last_exec_delta = (eff_gain * delta[:3]).copy()
                    else:
                        last_exec_delta = ctrl.scale_action(raw_cmd[:6])[:3].copy()
                        obs, ok, _ = env.step(raw_cmd)

                st["step"] += 1
                if st["obj_bid"] is not None:
                    obj_p = env.env.sim.data.body_xpos[st["obj_bid"]]
                    ctrl.update()
                    ee_p = ctrl.ee_pos
                    if st["first_contact_r_mm"] is None and float(raw_cmd[6]) > 0.2 and abs(float(ee_p[2] - obj_p[2])) < 0.065:
                        st["first_contact_r_mm"] = round(float(np.linalg.norm(ee_p[:2] - obj_p[:2])) * 1000.0, 2)
                    lift_mm = (float(obj_p[2]) - st["init_obj_z"]) * 1000.0
                    if lift_mm > st["max_lift_mm"]:
                        st["max_lift_mm"] = round(lift_mm, 2)
                    elif st["max_lift_mm"] >= 18.0 and lift_mm < 4.0 and not ok:
                        st["dropped_after_lift"] = 1
                if ok and st["first_succ"] is None:
                    st["first_succ"] = st["step"]

            # Prune old ring buffer entries
            min_keep = st["step"] - 24
            for k_old in [k for k in st["ee_ring"] if k < min_keep]:
                del st["ee_ring"][k_old]

            done = (st["first_succ"] is not None) or (st["step"] >= horizon)
            if done:
                meta = {
                    "replan_calls": st["replan_calls"],
                    "mean_delay_steps": round(float(np.mean(st["d_hist"])), 2) if st["d_hist"] else 0.0,
                    "max_delay_steps": int(max(st["d_hist"])) if st["d_hist"] else 0,
                    "splice_jump_mm": round(float(np.mean(st["splice_jumps_mm"])), 2) if st["splice_jumps_mm"] else 0.0,
                    "first_contact_r_mm": st["first_contact_r_mm"],
                    "transient_slip": int(st["dropped_after_lift"] == 1 and st["first_succ"] is None),
                }
                send_msg(("done", st["first_succ"] is not None, st["step"], meta))
            else:
                # Causal predictor for next chunk launch is the just-observed realized lag d_real!
                st["d_est"] = int(d_real)
                st["d_cur"], _ = ((2, 4) if (st["regime"] == "step_2_to_6" and st.get("step", 0) < 60) else ((6, 4) if st["regime"] == "step_2_to_6" else sample_next_jitter_delay(st["rng"], st["regime"], d_real)))
                leftover_start_idx = int(sh_used + (H_period - d_real))
                if mode == "vlash_zero":
                    plan_list = list(st["active_plan"])[: st["d_nom"]]
                    s8_ov = vlash_rollforward_s8(obs, ctrl, plan_list)
                else:
                    s8_ov = None
                c1, c2, s8 = obs_to_smolvla(obs, s8_override=s8_ov)
                send_msg(("obs_jitter", c1, c2, s8, "", False, st["step"], st["d_est"], leftover_start_idx))

        elif cmd == "init_bifurc_v18":
            _, suite, tid, sid, is_latched, mode, horizon = msg
            task = {"suite": suite, "task_id": tid, "role": "primary"}
            ek = (suite, tid)
            if cur_ek != ek:
                if env is not None:
                    env.close()
                env = LiberoEnvironment(environment_config(cfg, task), tid)
                cur_ek = ek
            obs, _ = reset_study_state(env, task, {"state_origin": "benchmark", "state_id": sid, "split": "benchmark"}, 0, 0)
            for _ in range(10):
                obs, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            ctrl = env.env.robots[0].controller
            obj_bids = find_all_manipuland_bodies(env)
            st = {
                "kind": "bif18", "mode": str(mode), "horizon": int(horizon), "ctrl": ctrl,
                "obj_bids": obj_bids, "is_latched_cfg": bool(is_latched),
                "latch_engaged": False, "latch_released": (not bool(is_latched)),
                "unlatch_up_streak": 0, "stalled_steps": 0, "post_release_steps": 0, "replan_after_release_done": False,
                "ee_hist": deque(maxlen=6), "cmd_disp_hist": deque(maxlen=6),
                "step": 0, "replan_calls": 0, "last_replan_step": -100,
                "first_succ": 0 if bool(env.env.check_success()) else None,
            }
            c1, c2, s8 = obs_to_smolvla(obs)
            send_msg(("obs", c1, c2, s8, str(env.task.language), st["first_succ"] is not None, st["step"]))

        elif cmd == "step_bifurc_v18":
            _, chunk_l, n_exec_host = msg
            chunk = np.asarray(chunk_l, dtype=np.float32)
            ctrl = st["ctrl"]
            mode = st["mode"]
            horizon = st["horizon"]
            if mode in ("pace", "hybrid_pace_event"):
                c_exec = pace_horizon(chunk, h_min=6, h_max=24)
            elif mode in ("dvac", "autohorizon") and n_exec_host is not None:
                c_exec = int(n_exec_host)
            elif mode == "open24":
                c_exec = 24
            else:
                c_exec = 16 if st.get("replan_after_release_done", False) else 24
            c_exec = max(1, min(c_exec, len(chunk)))
            st["replan_calls"] += 1

            obs = None
            for j in range(c_exec):
                if st["step"] >= horizon or st["first_succ"] is not None:
                    break
                ctrl.update()
                ee_now = ctrl.ee_pos.copy()
                st["ee_hist"].append(ee_now)
                raw = env.clip_action(chunk[j].copy())
                step_cmd_disp = float(np.linalg.norm(ctrl.scale_action(raw[:6])[:3]))
                st["cmd_disp_hist"].append(step_cmd_disp)

                if st["is_latched_cfg"] and not st["latch_released"] and not st["latch_engaged"]:
                    near_any = any(float(np.linalg.norm(ee_now - env.env.sim.data.body_xpos[bid])) < 0.088 for bid in st["obj_bids"])
                    if (near_any or float(ee_now[2]) < 0.94) and float(raw[6]) > 0.15:
                        st["latch_engaged"] = True

                if st["latch_engaged"] and not st["latch_released"]:
                    if float(raw[2]) > 0.20 and float(raw[6]) < 0.20:
                        st["unlatch_up_streak"] += 1
                        if st["unlatch_up_streak"] >= 2:
                            st["latch_released"] = True
                    else:
                        st["unlatch_up_streak"] = 0
                        raw[0] = -0.08 if st["stalled_steps"] < 8 else 0.0
                        raw[1] *= 0.03
                        raw[2] = min(0.0, float(raw[2])) * 0.03
                        raw[3] = 0.65 if st["stalled_steps"] < 8 else 0.0
                        st["stalled_steps"] += 1

                obs, ok, _ = env.step(raw)
                st["step"] += 1
                if ok and st["first_succ"] is None:
                    st["first_succ"] = st["step"]

                if st["latch_engaged"] and st["latch_released"] and not st["replan_after_release_done"]:
                    st["post_release_steps"] += 1

                if mode in ("event_replan", "hybrid_pace_event") and st["first_succ"] is None and st["step"] < horizon:
                    if st["latch_engaged"] and st["latch_released"] and not st["replan_after_release_done"] and st["post_release_steps"] >= 3:
                        st["replan_after_release_done"] = True
                        st["last_replan_step"] = st["step"]
                        break
                    if (st["step"] - st["last_replan_step"]) >= 8 and len(st["ee_hist"]) >= 4 and len(st["cmd_disp_hist"]) >= 4:
                        ctrl.update()
                        ee_after = ctrl.ee_pos.copy()
                        sum_cmd4 = float(sum(list(st["cmd_disp_hist"])[-4:]))
                        dz4 = abs(float(ee_after[2] - st["ee_hist"][-4][2]))
                        net_disp4 = float(np.linalg.norm(ee_after - st["ee_hist"][-4]))
                        cmd_dz = float(ctrl.scale_action(env.clip_action(chunk[j])[:6])[2])
                        stall_trig = (
                            (not st["latch_released"])
                            and st["stalled_steps"] >= 4
                            and (sum_cmd4 > 0.008 or cmd_dz > 0.0010)
                            and dz4 < 0.0025
                        ) or (
                            (not st["latch_engaged"]) and sum_cmd4 > 0.016 and net_disp4 < 0.0018
                        )
                        if stall_trig:
                            st["last_replan_step"] = st["step"]
                            break

            done = (st["first_succ"] is not None) or (st["step"] >= horizon)
            if done:
                meta = {
                    "replan_calls": st["replan_calls"],
                    "latch_engaged": int(st["latch_engaged"]),
                    "latch_released": int(st["latch_released"]),
                    "stalled_steps": st["stalled_steps"],
                }
                send_msg(("done", st["first_succ"] is not None, st["step"], meta))
            else:
                c1, c2, s8 = obs_to_smolvla(obs)
                send_msg(("obs", c1, c2, s8, "", False, st["step"]))



        elif cmd == "init_delayed_bifurc_v23":
            # Boundary 1 x Boundary 3: Asynchronous Delay/Jitter x Post-Contact Latch Bifurcation
            _, suite, tid, sid, is_latched, regime, async_mode, sched_mode, horizon = msg
            task = {"suite": suite, "task_id": tid, "role": "primary"}
            ek = (suite, tid)
            if cur_ek != ek:
                if env is not None:
                    env.close()
                env = LiberoEnvironment(environment_config(cfg, task), tid)
                cur_ek = ek
            obs, _ = reset_study_state(env, task, {"state_origin": "benchmark", "state_id": sid, "split": "benchmark"}, 0, 0)
            for _ in range(10):
                obs, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            ctrl = env.env.robots[0].controller
            ctrl.update()
            obj_bids = find_all_manipuland_bodies(env)
            rng_d = np.random.default_rng(20261129 + tid * 97 + sid * 13 + (1000 if is_latched else 0))
            d0, d_nom = sample_next_jitter_delay(rng_d, str(regime), None)
            st = {
                "kind": "delayed_bif23", "regime": str(regime), "async_mode": str(async_mode),
                "sched_mode": str(sched_mode), "horizon": int(horizon), "ctrl": ctrl,
                "obj_bids": obj_bids, "is_latched_cfg": bool(is_latched),
                "latch_engaged": False, "latch_released": (not bool(is_latched)),
                "unlatch_up_streak": 0, "stalled_steps": 0, "post_release_steps": 0, "replan_after_release_done": False,
                "ee_hist": deque(maxlen=8), "cmd_disp_hist": deque(maxlen=8),
                "ee_ring": {0: ctrl.ee_pos.copy()}, "active_plan": deque(),
                "active_t_obs": 0, "active_d_real": 0, "last_ori": ctrl.goal_ori.copy(),
                "rng": rng_d, "d_cur": int(d0), "d_nom": int(d_nom), "d_hist": [],
                "step": 0, "replan_calls": 0, "last_replan_step": -100,
                "first_succ": 0 if bool(env.env.check_success()) else None,
            }
            c1, c2, s8 = obs_to_smolvla(obs)
            send_msg(("obs", c1, c2, s8, str(env.task.language), st["first_succ"] is not None, st["step"]))

        elif cmd == "step_delayed_bifurc_v23":
            _, chunk_l, n_exec_host = msg
            chunk = np.asarray(chunk_l, dtype=np.float32)
            K = len(chunk)
            ctrl = st["ctrl"]
            async_mode = st["async_mode"]
            sched_mode = st["sched_mode"]
            horizon = st["horizon"]
            t_obs = st["step"]
            d_real = int(st["d_cur"])
            d_nom = int(st["d_nom"])
            st["d_hist"].append(d_real)
            st["replan_calls"] += 1

            ctrl.update()
            st["ee_ring"][t_obs] = ctrl.ee_pos.copy()

            if sched_mode in ("pace", "hybrid_pace_event"):
                c_exec = pace_horizon(chunk, h_min=max(6, d_real + 2), h_max=24)
            elif sched_mode == "open24":
                c_exec = 24
            else:
                c_exec = 16 if st.get("replan_after_release_done", False) else 24
            c_exec = max(d_real + 2, min(c_exec, K))

            obs = None
            for s_idx in range(c_exec):
                if st["step"] >= horizon or st["first_succ"] is not None:
                    break
                ctrl.update()
                ee_now = ctrl.ee_pos.copy()
                st["ee_ring"][st["step"]] = ee_now
                st["ee_hist"].append(ee_now)

                # Determine raw action from in-flight buffer vs newly arrived chunk
                use_floquet = False
                eff_delay = 0
                t_ref = t_obs
                if s_idx < d_real:
                    has_plan = len(st["active_plan"]) > 0
                    raw = env.clip_action(st["active_plan"].popleft() if has_plan else np.zeros(7, dtype=np.float32))
                    if has_plan and async_mode in ("ts_floquet_pref", "ts_floquet_motion_gated") and int(st["active_d_real"]) > 0:
                        use_floquet = True
                        eff_delay = int(st["active_d_real"])
                        t_ref = int(st["active_t_obs"])
                else:
                    if s_idx == d_real:
                        if async_mode == "raw":
                            sh_used = 0
                        elif async_mode == "pref_align":
                            sh_used = min(d_nom, K - 1)
                        elif async_mode == "ts_floquet_pref":
                            sh_used = min(d_real, K - 1)
                        elif async_mode == "ts_floquet_motion_gated":
                            # Motion-Gated Prefix Skipping: if physically stalled/docked (<4.5 mm displacement during
                            # inference lag), robot did not advance -> keep sh_used=0 to preserve unlatch prefix!
                            ee_at_obs = st["ee_ring"].get(t_obs, ee_now)
                            disp_mm = float(np.linalg.norm(ee_now - ee_at_obs)) * 1000.0
                            if disp_mm < 4.5 or (st["latch_engaged"] and not st["latch_released"]):
                                sh_used = 0
                            else:
                                sh_used = min(d_real, K - 1)
                        else:
                            sh_used = 0
                        st["active_plan"].clear()
                        st["active_t_obs"] = int(t_obs)
                        st["active_d_real"] = int(d_real)
                        for m_i in range(sh_used, K):
                            st["active_plan"].append(chunk[m_i].copy())
                    raw = env.clip_action(st["active_plan"].popleft() if len(st["active_plan"]) > 0 else chunk[-1].copy())
                    if async_mode in ("ts_floquet_pref", "ts_floquet_motion_gated") and d_real > 0:
                        use_floquet = True
                        eff_delay = d_real
                        t_ref = t_obs

                step_cmd_disp = float(np.linalg.norm(ctrl.scale_action(raw[:6])[:3]))
                st["cmd_disp_hist"].append(step_cmd_disp)

                # Exact zero-timer-leak latch mechanism
                if st["is_latched_cfg"] and not st["latch_released"] and not st["latch_engaged"]:
                    near_any = any(float(np.linalg.norm(ee_now - env.env.sim.data.body_xpos[bid])) < 0.088 for bid in st["obj_bids"])
                    if (near_any or float(ee_now[2]) < 0.94) and float(raw[6]) > 0.15:
                        st["latch_engaged"] = True

                if st["latch_engaged"] and not st["latch_released"]:
                    if float(raw[2]) > 0.20 and float(raw[6]) < 0.20:
                        st["unlatch_up_streak"] += 1
                        if st["unlatch_up_streak"] >= 2:
                            st["latch_released"] = True
                    else:
                        st["unlatch_up_streak"] = 0
                        raw[0] = -0.08 if st["stalled_steps"] < 8 else 0.0
                        raw[1] *= 0.03
                        raw[2] = min(0.0, float(raw[2])) * 0.03
                        raw[3] = 0.65 if st["stalled_steps"] < 8 else 0.0
                        st["stalled_steps"] += 1
                        use_floquet = False  # Do not attenuate unlatch motion once physically stalled

                if use_floquet and eff_delay > 0:
                    ee_obs = st["ee_ring"].get(t_ref, ee_now)
                    dt_eff = max(1, st["step"] - t_ref) * DT
                    x_dot_hat = (ee_now - ee_obs) / dt_eff
                    nx2 = float(np.dot(x_dot_hat, x_dot_hat))
                    step_gdot = ctrl.scale_action(raw[:6])[:3] / DT
                    w_star = float(np.clip(1.0 - float(np.dot(x_dot_hat, step_gdot)) / (nx2 + 1e-6), 0.0, 1.0))
                    g_stab = GAMMA_BY_DELAY.get(eff_delay, 0.52)
                    eff_gain = (1.0 - w_star) * 1.0 + w_star * g_stab
                    delta = ctrl.scale_action(raw[:6])
                    pos = set_goal_position(eff_gain * delta[:3], ee_now, position_limit=ctrl.position_limits)
                    if any(not math.isclose(float(v), 0.0) for v in delta[3:]):
                        st["last_ori"] = set_goal_orientation(eff_gain * delta[3:], ctrl.ee_ori_mat, orientation_limit=ctrl.orientation_limits)
                    orig = ctrl.set_goal
                    ctrl.set_goal = partial(orig, set_pos=pos, set_ori=st["last_ori"])
                    try:
                        obs, ok, _ = env.step(np.r_[np.zeros(6), float(raw[6])])
                    finally:
                        ctrl.set_goal = orig
                else:
                    obs, ok, _ = env.step(raw)

                st["step"] += 1
                if ok and st["first_succ"] is None:
                    st["first_succ"] = st["step"]

                if st["latch_engaged"] and st["latch_released"] and not st["replan_after_release_done"]:
                    st["post_release_steps"] += 1

                if sched_mode in ("event_replan", "hybrid_pace_event") and s_idx >= d_real and st["first_succ"] is None and st["step"] < horizon:
                    if st["latch_engaged"] and st["latch_released"] and not st["replan_after_release_done"] and st["post_release_steps"] >= 3:
                        st["replan_after_release_done"] = True
                        st["last_replan_step"] = st["step"]
                        break
                    if (st["step"] - st["last_replan_step"]) >= 8 and len(st["ee_hist"]) >= 4 and len(st["cmd_disp_hist"]) >= 4:
                        ctrl.update()
                        ee_after = ctrl.ee_pos.copy()
                        sum_cmd4 = float(sum(list(st["cmd_disp_hist"])[-4:]))
                        dz4 = abs(float(ee_after[2] - st["ee_hist"][-4][2]))
                        net_disp4 = float(np.linalg.norm(ee_after - st["ee_hist"][-4]))
                        cmd_dz = float(ctrl.scale_action(raw[:6])[2])
                        stall_trig = (
                            (not st["latch_released"])
                            and st["stalled_steps"] >= 4
                            and (sum_cmd4 > 0.008 or cmd_dz > 0.0010)
                            and dz4 < 0.0025
                        ) or (
                            (not st["latch_engaged"]) and sum_cmd4 > 0.016 and net_disp4 < 0.0018
                        )
                        if stall_trig:
                            st["last_replan_step"] = st["step"]
                            break

            min_keep = st["step"] - 24
            for k_old in [k for k in st["ee_ring"] if k < min_keep]:
                del st["ee_ring"][k_old]

            done = (st["first_succ"] is not None) or (st["step"] >= horizon)
            if done:
                meta = {
                    "replan_calls": st["replan_calls"],
                    "latch_engaged": int(st["latch_engaged"]),
                    "latch_released": int(st["latch_released"]),
                    "stalled_steps": st["stalled_steps"],
                    "mean_delay_steps": round(float(np.mean(st["d_hist"])), 2) if st["d_hist"] else 0.0,
                }
                send_msg(("done", st["first_succ"] is not None, st["step"], meta))
            else:
                st["d_cur"], _ = sample_next_jitter_delay(st["rng"], st["regime"], d_real)
                c1, c2, s8 = obs_to_smolvla(obs)
                send_msg(("obs", c1, c2, s8, "", False, st["step"]))


if __name__ == "__main__":
    main()
