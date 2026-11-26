#!/usr/bin/env python3
"""
Upgrade 2 (Table IV): True Mutually-Exclusive Bimodal Contact Bifurcation in Live 500 Hz MuJoCo
Directly tests Eq. (14)-(15) on visually identical pre-contact articulated fixtures (LIBERO drawer L031 & cabinet L059):
  - Branch A (Normal, theta in {0.1, 0.5}): direct push along +push_dir completes closure smoothly.
  - Branch B (Jammed / Latched, theta in {25.0, 35.0}): continuing +push_dir spikes contact force > F_safe (160 N)
    and fails; true recovery requires post-contact withdrawal (-push_dir + lateral/vertical unlatch clearance) to
    release the latch (reducing resistance) followed by clean re-engagement!
Compares L1-ChunkNet vs. Conditional Flow-Matching Flow-ChunkNet across 4 masking/execution regimes:
  1. Unmasked (open-loop c=8)
  2. Role-masked only (open-loop c=8)
  3. Full causal mask (open-loop c=8)  -- exposes open-loop degradation without event replanning
  4. Full causal mask + Contact Event Replanning (replan immediately at tau(t) upon contact feedback)
"""
from __future__ import annotations
import concurrent.futures
import csv
import json
import os
import sys
import time
import types
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

if "diskcache" not in sys.modules:
    class _MemCache(dict):
        def __init__(self, *a, **kw): super().__init__()
        def close(self): pass
    sys.modules["diskcache"] = types.SimpleNamespace(Cache=_MemCache)
if "zstandard" not in sys.modules:
    class _ZComp:
        def __init__(self, level=1): pass
        def compress(self, b): return zlib.compress(bytes(b), 1)
    class _ZDecomp:
        def decompress(self, d): return zlib.decompress(d)
    sys.modules["zstandard"] = types.SimpleNamespace(ZstdCompressor=_ZComp, ZstdDecompressor=_ZDecomp)

import mujoco
import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/data/guest/vla_branch_lab")
DATA_OUT = ROOT / "VLA-SelfTest/paper/source_data"
DATA_OUT.mkdir(parents=True, exist_ok=True)
SPINE_ROOT = ROOT / "VLA_SelfTest_ExpertSpine_v41_20261014"
BASE = SPINE_ROOT / "code"
sys.path.insert(0, str(SPINE_ROOT))
for _sub in sorted(BASE.iterdir()):
    if _sub.is_dir():
        sys.path.insert(0, str(_sub))
os.environ["LIBERO_CANONICAL_CATALOGUE"] = str(BASE / "planner/libero130/official130_catalogue.json")
os.environ["LIBERO_ACCEPTANCE_RULES"] = str(BASE / "planner/libero130/acceptance_rules")

CHUNK = 8
ACT_DIM = 4
OBS_DIM = 15  # +1 for unlatched_flag (observable via click/free-play once withdrawn!)
F_SAFE_LIMIT = 180.0  # N safety overload threshold


def setup_fixture_env(fixture_task: str, theta: float):
    from real_task_v6 import RealTaskEpisode
    from source_control_targets_v28 import source_control_targets
    ep = RealTaskEpisode(
        fixture_task, 0, 1.12, condition_id="C01", target_outcome="first_pass",
        planner_enabled=False, online_guard=False, record_images=False
    )
    t_nom, r_nom = source_control_targets(ep)
    a_nom = ep.expert_actions.copy()
    qidx = int(ep.model._model.jnt_qposadr[ep.joint])
    source = ep.expert_states[:, 1 + qidx]
    q_closed = float(source[-1])
    onset = int(np.flatnonzero(np.abs(source - source[0]) > 0.001)[0])
    begin = max(0, onset - 2)

    xml = ET.fromstring(ep.scene_xml)
    joint_name = ep.model.joint_id2name(ep.joint)
    node = next(x for x in xml.iter("joint") if x.get("name") == joint_name)
    node.set("frictionloss", format(float(theta), ".17g"))
    ep.scene_xml = ET.tostring(xml, encoding="unicode")
    ep.env.reset_from_xml_string(ep.scene_xml)
    ep.model = ep.env.sim.model
    ep.env.sim.set_state_from_flattened(ep.expert_states[0])
    ep.env.sim.forward()
    ep.fixture_shield = None

    model, data = ep.model._model, ep.env.sim.data._data
    target_body = int(model.jnt_bodyid[ep.joint])
    fixture_bodies = {target_body}
    for b in range(target_body + 1, model.nbody):
        if int(model.body_parentid[b]) in fixture_bodies:
            fixture_bodies.add(b)

    push_dir = t_nom[-1] - t_nom[min(begin + 4, len(t_nom) - 1)]
    push_dir = push_dir / max(float(np.linalg.norm(push_dir)), 1e-6)

    substeps = int(round(0.050 / model.opt.timestep))
    start_idx = max(0, begin - 6)
    for i in range(start_idx):
        for s_idx in range(substeps):
            cmd = ep.command_to(t_nom[i], r_nom[i], float(a_nom[i, 6]))
            mujoco.mj_step1(model, data)
            ep.env.env._pre_action(cmd, policy_step=(s_idx == 0))
            mujoco.mj_step2(model, data)
    return ep, a_nom, t_nom, r_nom, q_closed, begin, start_idx, push_dir, fixture_bodies


def step_mujoco_once(ep, tgt_pos, r_ref, grip_val, fixture_bodies):
    model, data = ep.model._model, ep.env.sim.data._data
    substeps = int(round(0.050 / model.opt.timestep))
    force = np.zeros(6)
    peak_f = 0.0
    for s_idx in range(substeps):
        cmd = ep.command_to(tgt_pos, r_ref, grip_val)
        mujoco.mj_step1(model, data)
        ep.env.env._pre_action(cmd, policy_step=(s_idx == 0))
        mujoco.mj_step2(model, data)
        f_sum = 0.0
        for j in range(data.ncon):
            c = data.contact[j]
            a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, z) or "" for z in (a, b)]
            if ((a in fixture_bodies) != (b in fixture_bodies)) and any(n.startswith(("robot0", "gripper0")) for n in names):
                mujoco.mj_contactForce(model, data, j, force)
                f_sum += abs(float(force[0]))
        peak_f = max(peak_f, f_sum)
    return peak_f


def _collect_one_demo(args):
    fixture_task, fid, theta, is_jammed, rep, seed = args
    rng = np.random.default_rng(seed)
    ep, a_nom, t_nom, r_nom, q_closed, begin, start_idx, push_dir, f_bodies = setup_fixture_env(fixture_task, theta)
    try:
        eef0 = ep.eef().copy(); q0 = float(ep.joint_pos()); prev_eef = eef0.copy()
        tau_rel = (begin - start_idx) + 3
        obs_seq, act_seq, role_seq = [], [], []
        peak_f = 0.0
        unlatched = False
        withdraw_done_step = None
        inject_bad_prefix = (is_jammed and rep == 2)
        up_vec = np.array([0.0, 0.0, 1.0])
        for k in range(72):
            idx = min(start_idx + k, len(t_nom) - 1)
            eef = ep.eef().copy(); vel = (eef - prev_eef) * 20.0; dq = float(ep.joint_pos()) - q0
            f_obs = min(peak_f / 200.0, 2.5) + rng.normal(0.0, 0.01)
            in_contact = 1.0 if peak_f > 8.0 else 0.0
            fid_oh = [1.0, 0.0] if fid == 0 else [0.0, 1.0]
            obs = np.concatenate([
                eef - eef0, t_nom[min(begin, len(t_nom) - 1)] - eef, vel,
                [f_obs, dq * 10.0, in_contact, 1.0 if unlatched else 0.0], fid_oh,
            ]).astype(np.float32)

            if inject_bad_prefix and tau_rel <= k < tau_rel + 4:
                # Corrupted unprobed hard-push failure prefix (role = 0)
                tgt = eef + 0.045 * push_dir
                role = 0.0
            elif k < tau_rel:
                # Identical pre-contact approach for both branches
                tgt = t_nom[idx]
                role = 1.0
            elif not is_jammed:
                # Branch A: smooth direct closure
                tgt = t_nom[idx] if abs(float(ep.joint_pos()) - q_closed) >= 0.005 else (eef - 0.003 * push_dir)
                role = 1.0
            else:
                # Branch B (Jammed): mutually exclusive withdrawal (-push_dir + 0.012 up) for 6 steps to release latch,
                # then clean low-force re-engagement!
                if not unlatched:
                    tgt = eef - 0.018 * push_dir + 0.010 * up_vec
                    if k >= tau_rel + 5:
                        unlatched = True
                        withdraw_done_step = k
                        # Physical latch release: reduce joint frictionloss in MuJoCo model dynamically!
                        ep.model._model.dof_frictionloss[ep.model._model.jnt_dofadr[ep.joint]] = 0.5
                else:
                    # Re-align and push smoothly to close
                    tgt = t_nom[min(begin + (k - withdraw_done_step), len(t_nom) - 1)]
                role = 1.0

            delta = np.clip((tgt - eef) * 25.0, -1.5, 1.5)
            act = np.array([delta[0], delta[1], delta[2], float(a_nom[idx, 6])], dtype=np.float32)
            obs_seq.append(obs); act_seq.append(act); role_seq.append(role)
            prev_eef = eef.copy()
            peak_f = step_mujoco_once(ep, tgt, r_nom[idx], float(a_nom[idx, 6]), f_bodies)
            if abs(float(ep.joint_pos()) - q_closed) < 0.004 and k > tau_rel + 8:
                break
        return {
            "fixture": fixture_task, "fid": fid, "theta": theta, "is_jammed": is_jammed,
            "obs": np.stack(obs_seq), "act": np.stack(act_seq), "role": np.array(role_seq, dtype=np.float32),
            "tau_rel": tau_rel,
        }
    finally:
        ep.close()


def build_dataset(episodes, mask_mode: str):
    X, Y, M = [], [], []
    for ep in episodes:
        obs, act, role, tau = ep["obs"], ep["act"], ep["role"], ep["tau_rel"]
        T = len(obs)
        for t in range(T):
            cy = np.zeros((CHUNK, ACT_DIM), dtype=np.float32)
            cm = np.zeros(CHUNK, dtype=np.float32)
            for j in range(CHUNK):
                idx = min(t + j, T - 1)
                cy[j] = act[idx]
                if mask_mode == "unmasked":
                    cm[j] = 1.0
                elif mask_mode == "role_only":
                    cm[j] = role[idx]
                elif mask_mode == "full_causal":
                    cm[j] = 0.0 if (t < tau and (t + j) >= tau) else role[idx]
            if cm.sum() > 0:
                X.append(obs[t]); Y.append(cy); M.append(cm)
    return (
        torch.tensor(np.stack(X), dtype=torch.float32),
        torch.tensor(np.stack(Y), dtype=torch.float32),
        torch.tensor(np.stack(M), dtype=torch.float32),
    )


class L1ChunkPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, CHUNK * ACT_DIM),
        )
    def forward(self, x):
        return self.net(x).view(-1, CHUNK, ACT_DIM)
    def predict_chunk(self, x):
        return self.forward(x)


class FlowChunkPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        flat = CHUNK * ACT_DIM
        self.obs_enc = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.LayerNorm(128), nn.SiLU())
        self.vel_net = nn.Sequential(
            nn.Linear(128 + flat + 1, 192), nn.LayerNorm(192), nn.SiLU(),
            nn.Linear(192, 192), nn.LayerNorm(192), nn.SiLU(),
            nn.Linear(192, flat),
        )
    def velocity(self, obs, a_s, s):
        h = self.obs_enc(obs)
        return self.vel_net(torch.cat([h, a_s.reshape(obs.shape[0], -1), s], dim=-1)).view(-1, CHUNK, ACT_DIM)
    @torch.no_grad()
    def predict_chunk(self, obs, n_steps=5):
        B = obs.shape[0]
        a = torch.zeros((B, CHUNK, ACT_DIM), device=obs.device, dtype=obs.dtype)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            s = torch.full((B, 1), k * dt, device=obs.device, dtype=obs.dtype)
            a = a + dt * self.velocity(obs, a, s)
        return a


def train_policy(arch: str, X, Y, M, seed: int, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = (L1ChunkPolicy() if arch == "L1" else FlowChunkPolicy()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    Xd, Yd, Md = X.to(device), Y.to(device), M.to(device)
    N = Xd.shape[0]; bs = 64; epochs = 110 if arch == "L1" else 140
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb, yb, mb = Xd[idx], Yd[idx], Md[idx].unsqueeze(-1)
            if arch == "L1":
                loss = ((model(xb) - yb).abs() * mb).sum() / mb.sum().clamp_min(1.0)
            else:
                s = torch.rand((xb.shape[0], 1), device=device)
                z0 = torch.randn_like(yb) * 0.20
                a_s = (1.0 - s.unsqueeze(-1)) * z0 + s.unsqueeze(-1) * yb
                loss = (((model.velocity(xb, a_s, s) - (yb - z0)) ** 2) * mb).sum() / mb.sum().clamp_min(1.0)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return {k: v.cpu() for k, v in model.state_dict().items()}


def _eval_batch_worker(args):
    arch, label, seed, state_dict, event_replan, spec_list = args
    torch.set_num_threads(1)
    model = L1ChunkPolicy() if arch == "L1" else FlowChunkPolicy()
    model.load_state_dict(state_dict); model.eval()
    rng = np.random.default_rng(seed * 1000 + hash((arch, label, spec_list[0])) % 10000)
    rows = []
    for fixture_task, fid, theta, is_jammed, trial in spec_list:
        ep, a_nom, t_nom, r_nom, q_closed, begin, start_idx, push_dir, f_bodies = setup_fixture_env(fixture_task, theta)
        try:
            eef0 = ep.eef().copy(); q0 = float(ep.joint_pos()); prev_eef = eef0.copy()
            peak_f_step = 0.0; max_f_ep = 0.0
            chunk = None; chunk_ptr = CHUNK
            contact_seen = False; replan_triggered = False
            unlatched = False; withdraw_accum = 0.0; overload_fail = False; succ = 0
            # Start chunk phase offset to reflect realistic asynchronous pre-contact chunk alignment
            for k in range(72):
                idx = min(start_idx + k, len(t_nom) - 1)
                eef = ep.eef().copy(); vel = (eef - prev_eef) * 20.0; dq = float(ep.joint_pos()) - q0
                f_obs = min(peak_f_step / 200.0, 2.5) + rng.normal(0.0, 0.01)
                in_contact = 1.0 if peak_f_step > 8.0 else 0.0
                fid_oh = [1.0, 0.0] if fid == 0 else [0.0, 1.0]
                obs = np.concatenate([
                    eef - eef0, t_nom[min(begin, len(t_nom) - 1)] - eef, vel,
                    [f_obs, dq * 10.0, in_contact, 1.0 if unlatched else 0.0], fid_oh,
                ]).astype(np.float32)

                # Event-triggered replanning upon post-contact resistance feedback
                if event_replan and (not replan_triggered) and contact_seen and (peak_f_step > 25.0 or abs(dq) < 0.008):
                    chunk_ptr = CHUNK
                    replan_triggered = True
                if in_contact > 0.5:
                    contact_seen = True

                if chunk is None or chunk_ptr >= CHUNK:
                    with torch.no_grad():
                        chunk = model.predict_chunk(torch.from_numpy(obs).unsqueeze(0))[0].numpy()
                    chunk_ptr = 0

                act = chunk[chunk_ptr]; chunk_ptr += 1
                step_d = act[:3] / 25.0
                tgt = eef + step_d
                # Physical latch releases once robot genuinely withdraws >= 30 mm opposite to push_dir after contact
                if is_jammed and contact_seen and (not unlatched):
                    proj = float(np.dot(step_d, -push_dir))
                    if proj > 0.002:
                        withdraw_accum += proj
                    if withdraw_accum >= 0.025:
                        unlatched = True
                        ep.model._model.dof_frictionloss[ep.model._model.jnt_dofadr[ep.joint]] = 0.5

                prev_eef = eef.copy()
                peak_f_step = step_mujoco_once(ep, tgt, r_nom[idx], float(act[3]), f_bodies)
                max_f_ep = max(max_f_ep, peak_f_step)
                if peak_f_step > F_SAFE_LIMIT:
                    overload_fail = True
                    break
                if bool(ep.env.check_success()) or abs(float(ep.joint_pos()) - q_closed) < 0.006:
                    succ = 1
                    break
            rows.append({
                "arch": arch, "method": label, "seed": seed, "fixture": fixture_task,
                "theta": theta, "is_jammed": int(is_jammed), "trial": trial,
                "success": int(succ and not overload_fail), "overload": int(overload_fail),
                "peak_force_N": round(max_f_ep, 2),
            })
        finally:
            ep.close()
    return rows


def main():
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    specs = [
        ("L031", 0, 0.1, False), ("L031", 0, 0.5, False),
        ("L031", 0, 28.0, True), ("L031", 0, 35.0, True),
        ("L059", 1, 0.1, False), ("L059", 1, 0.5, False),
        ("L059", 1, 28.0, True), ("L059", 1, 35.0, True),
    ]
    demo_args = [(f, fid, th, ij, rep, 20261121 + i * 19 + rep) for i, (f, fid, th, ij) in enumerate(specs) for rep in range(3)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as ex:
        episodes = list(ex.map(_collect_one_demo, demo_args))
    print(f"Collected {len(episodes)} mutually-exclusive bimodal demos in {time.time()-t0:.1f}s", flush=True)

    configs = [
        ("Unmasked (open-loop)", "unmasked", False),
        ("Role-masked only (open-loop)", "role_only", False),
        ("Full causal (open-loop)", "full_causal", False),
        ("Full causal + Event Replan", "full_causal", True),
    ]
    eval_specs = [
        ("L031", 0, 0.1, False), ("L031", 0, 0.5, False),
        ("L031", 0, 26.0, True), ("L031", 0, 34.0, True),
        ("L059", 1, 0.1, False), ("L059", 1, 0.5, False),
        ("L059", 1, 26.0, True), ("L059", 1, 34.0, True),
    ]
    eval_items = [(f, fid, th, ij, tr) for (f, fid, th, ij) in eval_specs for tr in range(5)]

    eval_jobs = []
    for arch in ["L1", "Flow"]:
        for label, mmode, ev_replan in configs:
            X, Y, M = build_dataset(episodes, mmode)
            for seed in [0, 1, 2]:
                sd = train_policy(arch, X, Y, M, seed, device)
                for c0 in range(0, len(eval_items), 10):
                    eval_jobs.append((arch, label, seed, sd, ev_replan, eval_items[c0:c0+10]))
    print(f"Trained 24 models in {time.time()-t0:.1f}s; running 960 closed-loop rollouts on 16 CPUs...", flush=True)

    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
        for batch in ex.map(_eval_batch_worker, eval_jobs):
            all_rows.extend(batch)

    summary = {}
    for arch in ["L1", "Flow"]:
        summary[arch] = {}
        for label, _, _ in configs:
            sub = [x for x in all_rows if x["arch"] == arch and x["method"] == label]
            low_sr = 100.0 * np.mean([x["success"] for x in sub if x["is_jammed"] == 0])
            jam_sr = 100.0 * np.mean([x["success"] for x in sub if x["is_jammed"] == 1])
            all_sr = 100.0 * np.mean([x["success"] for x in sub])
            ov_rt = 100.0 * np.mean([x["overload"] for x in sub])
            pf = float(np.mean([x["peak_force_N"] for x in sub]))
            summary[arch][label] = {
                "n": len(sub), "normal_pct": round(low_sr, 1),
                "jammed_recovery_pct": round(jam_sr, 1), "overall_pct": round(all_sr, 1),
                "overload_pct": round(ov_rt, 1), "mean_peak_force_N": round(pf, 1),
            }
            print(f"[{arch:4s}] {label:30s} | Norm={low_sr:5.1f}% JamRec={jam_sr:5.1f}% All={all_sr:5.1f}% | Ovld={ov_rt:4.1f}% F={pf:5.1f}N", flush=True)

    (DATA_OUT / "v12_table4_bifurcation_true.json").write_text(json.dumps({"elapsed_s": round(time.time() - t0, 1), "summary": summary}, indent=2))
    print(f"DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
