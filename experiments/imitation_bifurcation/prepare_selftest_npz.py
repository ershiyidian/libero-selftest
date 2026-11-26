#!/usr/bin/env python3
"""Export 40-task SelfTest-1040 episodes (Spatial/Object/Goal/Long-10) from HDF5 to compact NPZ for torch_env SmolVLA fine-tuning."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import h5py
import numpy as np
from robosuite.utils.transform_utils import axisangle2quat, mat2quat, quat2mat

ROOT = Path("/data/guest/vla_branch_lab")
SRC = ROOT / "VLA_SelfTest_Curated130_v46_20261015/outputs/curated130_exact8_final_v56/episodes"
DST = ROOT / "cache/selftest40_smolvla"
DST.mkdir(parents=True, exist_ok=True)

# Verified exact orthogonal body-frame rotation from gripper0_grip_site (SelfTest eef_rot) to robot0_eef_quat (SmolVLA right_hand)
R_OFF = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

def eef_rot_to_smolvla_aa(rot_mat: np.ndarray) -> np.ndarray:
    q = mat2quat(np.asarray(rot_mat, dtype=np.float64) @ R_OFF)  # (x, y, z, w)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    # Match MuJoCo Panda right_hand body xquat sign convention (qx >= 0 near downward vertical) + LiberoProcessorStep._quat2axisangle
    if q[0] < 0.0:
        q = -q
    w = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.acos(w)
    return (q[:3] / den * angle).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    task_dirs = sorted([d for d in SRC.iterdir() if d.is_dir()])
    selected = []
    for td in task_dirs:
        ep_dirs = sorted([e for e in td.iterdir() if e.is_dir()])
        if not ep_dirs:
            continue
        meta = json.loads((ep_dirs[0] / "metadata.json").read_text())
        bddl = str(meta.get("source_bddl_file", ""))
        sfile = str(meta.get("source_file", ""))
        if "libero_90" in bddl or "/libero_90/" in sfile:
            continue
        suite = sfile.split("/native_sources/")[1].split("/")[0] if "/native_sources/" in sfile else "libero_spatial"
        selected.append((td, suite, ep_dirs))
        if args.smoke and len(selected) >= 2:
            break
        if not args.smoke and len(selected) >= 40:
            break

    suite_counts = {}
    manifest = []
    for td, suite, ep_dirs in selected:
        libero_tid = suite_counts.get(suite, 0)
        suite_counts[suite] = libero_tid + 1
        for ed in (ep_dirs[:2] if args.smoke else ep_dirs):
            out_npz = DST / f"{td.name}_{ed.name}.npz"
            meta = json.loads((ed / "metadata.json").read_text())
            role_mask = np.load(ed / "action_loss_mask.npy").astype(np.float32)
            events = meta.get("events", [])
            fb_steps = []
            if isinstance(events, list):
                for ev in events:
                    if isinstance(ev, dict) and "step" in ev:
                        fb_steps.append(int(ev["step"]))
            elif isinstance(events, dict):
                for v in events.values():
                    if isinstance(v, int):
                        fb_steps.append(v)
                    elif isinstance(v, list):
                        fb_steps.extend(int(x) for x in v if isinstance(x, (int, float)))
            roles = json.loads((ed / "training_roles.json").read_text())
            ctx_ranges = roles.get("context_only_ranges", [])
            for r0, r1 in ctx_ranges:
                fb_steps.append(int(r1))
            with h5py.File(ed / "trajectory.hdf5", "r") as f_ph:
                if "phase" in f_ph:
                    phs = [x.decode("utf-8", errors="ignore") for x in f_ph["phase"][:]]
                    for idx_p in range(1, len(phs)):
                        if phs[idx_p - 1] == "probe" and phs[idx_p] != "probe":
                            fb_steps.append(idx_p)
            fb_steps = sorted(set(s for s in fb_steps if 0 < s < len(role_mask)))
            if args.force or not out_npz.exists():
                with h5py.File(ed / "trajectory.hdf5", "r") as f:
                    actions = np.asarray(f["actions"][:], dtype=np.float32)
                    T_act = len(actions)
                    raw_p = np.asarray(f["observations/proprio"][:T_act], dtype=np.float32)
                    eef_pos = raw_p[:, 0:3].astype(np.float32)
                    if "eef_rot" in f["observations"]:
                        eef_rot = np.asarray(f["observations/eef_rot"][:T_act], dtype=np.float64)
                        aa_all = np.stack([eef_rot_to_smolvla_aa(eef_rot[k]) for k in range(T_act)], axis=0)
                    else:
                        aa_all = np.stack([eef_rot_to_smolvla_aa(quat2mat(axisangle2quat(raw_p[k, 3:6]))) for k in range(T_act)], axis=0)
                    proprio = np.concatenate([eef_pos, aa_all, raw_p[:, 6:8]], axis=-1).astype(np.float32)
                    cam1 = np.asarray(f["observations/agent_rgb"][:T_act], dtype=np.uint8)[:, ::-1, ::-1, :]
                    cam2 = np.asarray(f["observations/wrist_rgb"][:T_act], dtype=np.uint8)[:, ::-1, ::-1, :]
                np.savez(
                    out_npz,
                    actions=actions,
                    proprio=proprio,
                    cam1=cam1,
                    cam2=cam2,
                    role_mask=role_mask,
                    fb_steps=np.asarray(fb_steps, dtype=np.int32),
                )
            manifest.append({
                "npz": str(out_npz),
                "hdf5": str(ed / "trajectory.hdf5"),
                "scene_xml": str(ed / "scene.xml"),
                "task_id": td.name,
                "suite": suite,
                "libero_task_id": libero_tid,
                "episode": ed.name,
                "language": str(meta.get("task_language", "")),
                "bddl": str(meta.get("source_bddl_file", "")),
                "target_outcome": str(meta.get("target_outcome", "")),
                "ctx_only_ranges": ctx_ranges,
                "n_steps": int(len(role_mask)),
                "n_fb": len(fb_steps),
            })
    m_name = "manifest_smoke.json" if args.smoke else "manifest_40tasks.json"
    (DST / m_name).write_text(json.dumps(manifest, indent=2))
    print(f"[Exp4 Prepare] Exported {len(manifest)} episodes (exact state8) to {DST / m_name}", flush=True)

if __name__ == "__main__":
    main()
