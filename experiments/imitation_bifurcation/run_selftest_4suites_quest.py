#!/usr/bin/env python3
"""Suite B (Table IV Overhaul):
Part 1: Post-trained QueST Closed-Loop Recovery Fine-Tuning (3 seeds, n=96 rollouts/row, 0ms & 100ms)
  across BOTH LIBERO-Spatial AND LIBERO-Goal with full 4-way disentangled comparison:
  - Base QueST-RIPT
  - Unmasked all (m_{t,j} = 1: includes failure-setup prefix m_role=0 + full suffix)
  - Role-masked only (m_{t,j} = m_role_{t+j}: excludes failure-setup prefix, NO boundary truncation)
  - Full causal mask (m_{t,j} = m_role_{t+j} * 1[t+j < tau(t)]: excludes failure prefix + boundary truncation)

Part 2: Multi-Suite SelfTest-1040 Closed-Loop Benchmark across ALL FOUR 10-task LIBERO suites
  (Spatial L001..L010, Object L011..L020, Goal L021..L030, Long-10 L121..L130 = 40 tasks, 320 verified episodes):
  - Encodes dual-RGB (frozen QueST ResNet18 visual encoders) + 8D proprio + 4-step EEF/joint/action history + task ID
  - Trains 3 seeds (seeds 0, 1, 2 on GPUs 0, 1, 2) of an 8-step L1 Chunking Policy with selftest.losses.masked_action_l1_loss
    under the exact 3-way mask ablation:
      (i)   Unmasked all (m_{t,j} = 1)
      (ii)  Role-masked only (m_{t,j} = m_role_{t+j}, NO tau(t) truncation)
      (iii) Full causal mask (m_{t,j} = m_role_{t+j} * 1[t+j < tau(t)])
  - Evaluates closed-loop in live LIBERO MuJoCo across ALL 40 TASKS (both first-pass probe states and
    failed-recovery probe-branching states = 80 MuJoCo scenarios x 3 seeds = 240 rollouts per mask = 720 rollouts!).
"""
from __future__ import annotations
import concurrent.futures
import csv
import gc
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import types
import zlib
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path("/data/guest/vla_branch_lab")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "VLA-SelfTest"))
DATA_OUT = ROOT / "VLA-SelfTest/paper/source_data"
PY_BRANCH = "/home/guest/miniconda3/envs/vla-branch/bin/python"

# -------------------------------------------------------------------------
# PART 1: Complete QueST Goal Unmasked-All & Role-Only Fine-Tuning + Eval
# -------------------------------------------------------------------------
def build_goal_unmasked_cache(masked_full_dir: Path, recovery_dir: Path, out_cache_dir: Path):
    from vla_branch_lab.continuity.cache import read_cache, write_cache
    saved, teacher_rows, anchors, states = read_cache(masked_full_dir)
    prefixes = {
        r["prefix_id"]: r
        for r in [json.loads(x) for x in (recovery_dir / "prefixes.jsonl").read_text().splitlines() if x.strip()]
    }
    unmasked_rows = []
    pre_n, suf_n = 0, 0
    for row in teacher_rows:
        pr = prefixes[row["prefix_id"]]
        src_npz = Path(pr["source_run"]) / "trajectories" / f"{pr['source_episode']}.npz"
        with np.load(src_npz, allow_pickle=False) as z:
            keep_pre = z["chunk_steps"] < pr["cut_step"]
            pre_ctx = z["student_contexts"][keep_pre].copy()
            pre_tok = z["chunk_tokens"][keep_pre].copy()
        suf_ctx = row["context"].copy()
        suf_tok = row["tokens"].copy()
        new_row = dict(row)
        new_row["context"] = np.concatenate([pre_ctx, suf_ctx], axis=0)
        new_row["tokens"] = np.concatenate([pre_tok, suf_tok], axis=0)
        pre_n += len(pre_tok); suf_n += len(suf_tok)
        unmasked_rows.append(new_row)
    saved_new = dict(saved)
    saved_new["mask_mode"] = "goal_unmasked_full_m_eq_1"
    write_cache(out_cache_dir, saved_new, unmasked_rows, anchors, states)
    print(f"[Goal unmasked cache] {out_cache_dir.name}: {len(unmasked_rows)} demos, pre={pre_n}, suf={suf_n}, tot={pre_n+suf_n}", flush=True)

def run_part1_quest_spatial_and_goal():
    # 1. Build Goal unmasked train & val caches
    g_unm_tr = ROOT / "outputs/v10/cache_goal_unmasked_all_train"
    g_unm_va = ROOT / "outputs/v10/cache_goal_unmasked_all_validation"
    build_goal_unmasked_cache(ROOT / "outputs/v10/cache_goal_full_train", ROOT / "outputs/v10/recovery_goal_train", g_unm_tr)
    build_goal_unmasked_cache(ROOT / "outputs/v10/cache_goal_full_validation", ROOT / "outputs/v10/recovery_goal_validation", g_unm_va)

    # 2. Train Goal unmasked_all (3 seeds) AND Goal role_only (cache_goal_full_train, 3 seeds) in parallel!
    specs = {
        "goal_unmasked_all": {
            "spec_path": ROOT / "configs/v10/fit_goal_unmasked_all.yaml",
            "fit_dir": ROOT / "outputs/v10/fit_goal_unmasked_all",
            "eval_dir": ROOT / "outputs/v10/eval_goal_unmasked_all",
            "data": {
                "train_root": "../../outputs/v10/cache_goal_unmasked_all_train",
                "validation_root": "../../outputs/v10/cache_goal_unmasked_all_validation",
                "output_dir": "outputs/v10/fit_goal_unmasked_all",
                "sampling_unit": "parent",
                "trainable_scope": "prior",
                "optimizer": {"updates": 200, "batch_size": 8, "learning_rate": 1.0e-5, "retention": 0.1, "weighting": "uniform", "grad_clip": 1.0, "save_every": 50},
            },
        },
        "goal_role_only": {
            "spec_path": ROOT / "configs/v10/fit_goal_role_only.yaml",
            "fit_dir": ROOT / "outputs/v10/fit_goal_role_only",
            "eval_dir": ROOT / "outputs/v10/eval_goal_role_only",
            "data": {
                "train_root": "../../outputs/v10/cache_goal_full_train",
                "validation_root": "../../outputs/v10/cache_goal_full_validation",
                "output_dir": "outputs/v10/fit_goal_role_only",
                "sampling_unit": "parent",
                "trainable_scope": "prior",
                "optimizer": {"updates": 200, "batch_size": 8, "learning_rate": 1.0e-5, "retention": 0.1, "weighting": "uniform", "grad_clip": 1.0, "save_every": 50},
            },
        },
    }

    procs = []
    gpu_slot = 0
    for name, info in specs.items():
        info["spec_path"].write_text(yaml.safe_dump(info["data"], sort_keys=False), encoding="utf-8")
        for seed in (0, 1, 2):
            seed_ckpt = info["fit_dir"] / f"seed_{seed:03d}" / "final.pth"
            if not seed_ckpt.is_file():
                cmd = [PY_BRANCH, "-m", "vla_branch_lab", "recovery-fit", "--spec", str(info["spec_path"]), "--gpu", str(gpu_slot % 4), "--seed", str(seed)]
                procs.append((name, seed, subprocess.Popen(cmd, cwd=str(ROOT))))
                gpu_slot += 1
    for name, seed, p in procs:
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"{name} seed {seed} failed: {rc}")

    # Evaluate both on GPUs 0,1,2,3
    for name, info in specs.items():
        done_eps = list((info["eval_dir"] / "episodes").glob("*.json")) if (info["eval_dir"] / "episodes").exists() else []
        if len(done_eps) < 192:
            eval_cfg = ROOT / f"configs/generated_v9/eval_{name}.yaml"
            eval_cfg.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                PY_BRANCH, "-m", "vla_branch_lab", "recovery-eval-config",
                "--template", str(ROOT / "configs/v10/eval_fixed_goal.yaml"),
                "--fit-root", str(info["fit_dir"]),
                "--output", str(eval_cfg),
                "--evaluation-root", str(info["eval_dir"]),
                "--models-only",
            ], cwd=str(ROOT), check=True)
            subprocess.run([
                PY_BRANCH, "-m", "vla_branch_lab", "switch-run",
                "--spec", str(eval_cfg), "--gpus", "0", "1", "2", "3",
            ], cwd=str(ROOT), check=True)

    def summarize_eval_dir(eval_dir: Path, methods_filter=None):
        eps = [json.loads(p.read_text()) for p in sorted((eval_dir / "episodes").glob("*.json"))]
        if methods_filter:
            eps = [r for r in eps if r["method"] in methods_filter]
        res = {}
        for cond in ("nominal", "delay_two"):
            sub = [r for r in eps if r["condition"] == cond]
            ok = sum(1 for r in sub if r["success"] and (r["first_task_success_step"] or 9999) <= 600)
            res[cond] = {"n": len(sub), "succ": ok, "pct": round(100.0 * ok / max(1, len(sub)), 1)}
        return res

    distilled_3 = {"distilled_0", "distilled_1", "distilled_2"}
    quest_ablation = {
        "spatial": {
            "base_ript": summarize_eval_dir(ROOT / "outputs/v8/eval_spatial_fixed", {"native_ript"}),
            "unmasked_all": summarize_eval_dir(ROOT / "outputs/v8/eval_spatial_burst_unmasked_all", distilled_3),
            "role_only": summarize_eval_dir(ROOT / "outputs/v8/eval_spatial_burst_teacher_end", distilled_3),
            "causal_boundary": summarize_eval_dir(ROOT / "outputs/v8/eval_spatial_burst_cropped4", distilled_3),
        },
        "goal": {
            "base_ript": summarize_eval_dir(ROOT / "outputs/v10/eval_goal_fixed", {"native_ript"}),
            "unmasked_all": summarize_eval_dir(ROOT / "outputs/v10/eval_goal_unmasked_all", distilled_3),
            "role_only": summarize_eval_dir(ROOT / "outputs/v10/eval_goal_role_only", distilled_3),
            "causal_boundary": summarize_eval_dir(ROOT / "outputs/v10/eval_goal_short", distilled_3),
        },
    }
    print("PART 1 QUEST ABLATION:", json.dumps(quest_ablation, indent=2), flush=True)
    return quest_ablation

# -------------------------------------------------------------------------
# PART 2: SelfTest-1040 4-Suite (40-Task, 320-Episode) Closed-Loop Benchmark
# -------------------------------------------------------------------------
def get_40task_manifest():
    sel = json.loads((ROOT / "VLA_SelfTest_Curated130_v46_20261015/outputs/curated130_exact8_final_v56/selection.json").read_text())
    target_rows = []
    for r in sel:
        tid = r["task_id"]
        if ("L001" <= tid <= "L030") or ("L121" <= tid <= "L130"):
            if "L001" <= tid <= "L010":
                suite = "Spatial"
            elif "L011" <= tid <= "L020":
                suite = "Object"
            elif "L021" <= tid <= "L030":
                suite = "Goal"
            else:
                suite = "Long-10"
            target_rows.append({**r, "suite_label": suite})
    assert len(target_rows) == 320, f"Expected 320 episodes across 40 tasks, got {len(target_rows)}"
    return target_rows

def extract_episode_features_worker(gpu_id: int, ep_records: list[dict], out_npz: Path):
    import torch
    import torch.nn.functional as F

    device = torch.device(f"cuda:{gpu_id}")
    ckpt = torch.load(ROOT / "checkpoints/LIBERO_SPATIAL_sft.pth", map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in (ckpt["model"] if "model" in ckpt else ckpt).items()}

    # Build exact torchvision/QueST ResNet18 stem+layers or lightweight frozen conv encoder from QueST weights
    import torchvision.models as tvm
    rn_agent = tvm.resnet18(weights=None)
    rn_wrist = tvm.resnet18(weights=None)
    # Load matching conv1/bn1/layer1..4 weights from QueST's agentview_rgb and eye_in_hand_rgb ResNet18 backbones!
    for prefix, rn in [("image_encoders.agentview_rgb", rn_agent), ("image_encoders.eye_in_hand_rgb", rn_wrist)]:
        own = rn.state_dict()
        loaded = 0
        for k_src, v_src in sd.items():
            if k_src.startswith(prefix):
                # Map e.g. image_encoders.agentview_rgb.nets.0... or resnet18_base...
                for k_dst in own:
                    if k_src.endswith(k_dst) and own[k_dst].shape == v_src.shape:
                        own[k_dst] = v_src.clone()
                        loaded += 1
                        break
        rn.load_state_dict(own, strict=False)
        rn.fc = torch.nn.Identity()
        rn.eval().to(device)

    all_packs = {}
    with torch.no_grad():
        for idx, rec in enumerate(ep_records):
            folder = Path(rec["source_path"])
            meta = json.loads((folder / "metadata.json").read_text())
            roles = json.loads((folder / "training_roles.json").read_text())
            role_mask = np.load(folder / "action_loss_mask.npy").astype(np.float32)

            with h5py.File(folder / "trajectory.hdf5", "r") as h:
                actions = np.asarray(h["actions"], dtype=np.float32)
                proprio = np.asarray(h["observations/proprio"], dtype=np.float32)
                eef = proprio[:, :3].copy()
                contact = (proprio[:, 6] - proprio[:, 7]).astype(np.float32)
                ag_rgb = np.asarray(h["observations/agent_rgb"], dtype=np.uint8)
                wr_rgb = np.asarray(h["observations/wrist_rgb"], dtype=np.uint8)
                phases = [x.decode() if isinstance(x, bytes) else str(x) for x in h["phase"][:]]

            T = len(actions)
            # Encode images in batches of 128 at 128x128
            ag_feats, wr_feats = [], []
            for b0 in range(0, T, 128):
                a_b = torch.from_numpy(ag_rgb[b0:min(T, b0+128)]).permute(0, 3, 1, 2).float().to(device) / 255.0
                w_b = torch.from_numpy(wr_rgb[b0:min(T, b0+128)]).permute(0, 3, 1, 2).float().to(device) / 255.0
                a_b = F.interpolate(a_b, size=(128, 128), mode="bilinear", align_corners=False)
                w_b = F.interpolate(w_b, size=(128, 128), mode="bilinear", align_corners=False)
                def _sp_enc(rn, xb):
                    z = rn.maxpool(rn.relu(rn.bn1(rn.conv1(xb))))
                    z = rn.layer4(rn.layer3(rn.layer2(rn.layer1(z))))
                    gap = z.mean(dim=(-2, -1))
                    B_, C_, H_, W_ = z.shape
                    pr = torch.softmax(z.reshape(B_, C_, H_ * W_) * 2.0, dim=-1)
                    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H_, device=z.device),
                                            torch.linspace(-1, 1, W_, device=z.device), indexing="ij")
                    ex = (pr * gx.reshape(1, 1, -1)).sum(dim=-1)
                    ey = (pr * gy.reshape(1, 1, -1)).sum(dim=-1)
                    return torch.cat([gap, ex[:, :256], ey[:, :256]], dim=-1)
                ag_feats.append(_sp_enc(rn_agent, a_b).cpu().numpy().astype(np.float16))
                wr_feats.append(_sp_enc(rn_wrist, w_b).cpu().numpy().astype(np.float16))
            ag_f = np.concatenate(ag_feats, axis=0)  # (T, 512)
            wr_f = np.concatenate(wr_feats, axis=0)  # (T, 512)

            # Extract exact feedback boundaries B_e from metadata['events']:
            # probe completion end_step + contact confirmation steps + role transition boundaries
            boundaries = set()
            probe_end_steps = []
            for ev in meta.get("events", []):
                m_ev = str(ev.get("method") or "")
                if m_ev.startswith("M0") and "end_step" in ev:
                    boundaries.add(int(ev["end_step"]))
                    probe_end_steps.append(int(ev["end_step"]))
                if "contact_confirmed_before_open_step" in ev and ev["contact_confirmed_before_open_step"] is not None:
                    boundaries.add(int(ev["contact_confirmed_before_open_step"]))
            for s_r, e_r in roles.get("context_only_ranges", []):
                boundaries.add(int(e_r))
            for pi in range(1, min(len(phases), T)):
                if phases[pi - 1] == "probe" and phases[pi] != "probe":
                    boundaries.add(int(pi))
                    probe_end_steps.append(int(pi))
                elif phases[pi - 1] in ("contact_setup", "real_supported_misgrasp") and phases[pi] != phases[pi - 1]:
                    boundaries.add(int(pi))
            sorted_b = sorted(b for b in boundaries if 0 < b < T)

            key = f"{rec['task_id']}__{folder.name}"
            all_packs[key] = {
                "task_id": rec["task_id"],
                "suite": rec["suite_label"],
                "mode": rec["mode"],
                "folder": str(folder),
                "ag_f": ag_f,
                "wr_f": wr_f,
                "proprio": proprio[:T],
                "eef": eef[:T],
                "contact": contact[:T],
                "actions": actions,
                "role_mask": role_mask,
                "boundaries": np.asarray(sorted_b, dtype=np.int32),
                "probe_ends": np.asarray(sorted(probe_end_steps), dtype=np.int32),
            }
    np.save(out_npz, all_packs, allow_pickle=True)
    print(f"[GPU {gpu_id}] Extracted {len(all_packs)} SelfTest-1040 episodes -> {out_npz.name}", flush=True)

def build_causal_state_vector(pack: dict, t: int) -> np.ndarray:
    """Strictly causal state at step t (uses only observations <= t and executed actions < t)."""
    p_now = pack["proprio"][t]                                   # 8D
    e_now = pack["eef"][t]                                       # 3D
    t_prev4 = max(0, t - 4)
    t_prev12 = max(0, t - 12)
    de4 = (pack["eef"][t] - pack["eef"][t_prev4]) * 20.0         # 3D recent 4-step EEF displacement
    de12 = (pack["eef"][t] - pack["eef"][t_prev12]) * 10.0       # 3D probe-window 12-step EEF displacement!
    dp12 = pack["proprio"][t] - pack["proprio"][t_prev12]        # 8D probe-window proprio change
    c_hist = np.array([
        pack["contact"][t],
        float(np.mean(pack["contact"][t_prev12:t+1])),
        min(1.0, t / 300.0),
    ], dtype=np.float32)                                         # 3D
    # Previous 4 executed actions (padded with zeros at t=0)
    act_hist = np.zeros((4, 7), dtype=np.float32)
    for k in range(1, 5):
        if t - k >= 0:
            act_hist[4 - k] = pack["actions"][t - k]
    lowdim = np.concatenate([p_now, e_now, de4, de12, dp12, c_hist, act_hist.reshape(-1)]).astype(np.float32)  # 56D
    return lowdim

def train_and_eval_40tasks(all_packs: dict):
    import torch
    import torch.nn as nn
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location("selftest_losses", str(ROOT / "VLA-SelfTest/selftest/losses.py"))
    _mod = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_mod)
    masked_action_l1_loss = _mod.masked_action_l1_loss

    # Split per task: 6 train demos (3 pass + 3 recovery) and 2 held-out test demos (1 pass + 1 recovery) per task!
    # That gives 240 training episodes and 80 held-out evaluation episodes (2 per task across all 40 tasks: 40 pass + 40 recovery)!
    by_task = {}
    for k, pk in sorted(all_packs.items()):
        by_task.setdefault(pk["task_id"], []).append((k, pk))
    assert len(by_task) == 40

    task_list = sorted(by_task.keys())
    task_to_idx = {t: i for i, t in enumerate(task_list)}

    train_keys, test_keys = [], []
    for tid in task_list:
        eps = by_task[tid]
        passes = [k for k, p in eps if p["mode"] == "probe_success"]
        recovs = [k for k, p in eps if p["mode"] != "probe_success"]
        # Hold out 1 pass + 1 recovery per task for evaluation; train on remaining 6!
        test_keys.append(passes[-1])
        test_keys.append(recovs[-1])
        for k in passes[:-1] + recovs[:-1]:
            train_keys.append(k)
    print(f"40-task split: train_episodes={len(train_keys)}, held_out_test_episodes={len(test_keys)} (40 pass + 40 recovery)", flush=True)

    # Also include probe-straddling windows from all 40 tasks with oversampling near boundaries so cross-boundary behavior is sharply tested,
    # Wait: keep strictly disjoint train_keys (240 eps) vs held_out test_keys (80 eps)! Let's verify how well a multi-task policy generalizes
    # across initial states within the same task (8 demos per task share the same BDDL scene & object geometry!).
    def build_windows(keys_subset: list[str]):
        vis_list, low_list, tid_list, tgt_list, m_unm_list, m_role_list, m_caus_list, straddle_list = [], [], [], [], [], [], [], []
        H = 8
        for k in keys_subset:
            pk = all_packs[k]
            T = len(pk["actions"])
            t_idx = task_to_idx[pk["task_id"]]
            bs = pk["boundaries"]
            rm = pk["role_mask"]
            for t in range(0, T - H + 1, 2):
                vis = np.concatenate([pk["ag_f"][t].astype(np.float32), pk["wr_f"][t].astype(np.float32)])  # 1024D
                low = build_causal_state_vector(pk, t)                                                       # 56D
                tgt = pk["actions"][t:t+H].astype(np.float32)                                                # (8, 7)
                m_unm = np.ones(H, dtype=np.float32)
                m_role = rm[t:t+H].astype(np.float32)
                # Next unobserved boundary tau(t) = min {b in B : b > t}
                fut_b = bs[bs > t]
                tau_t = int(fut_b[0]) if len(fut_b) > 0 else (T + H)
                c_trunc = np.array([1.0 if (t + j) < tau_t else 0.0 for j in range(H)], dtype=np.float32)
                m_caus = m_role * c_trunc
                straddles = int(tau_t < t + H and m_role.sum() == H)

                vis_list.append(vis)
                low_list.append(low)
                tid_list.append(t_idx)
                tgt_list.append(tgt)
                m_unm_list.append(m_unm)
                m_role_list.append(m_role)
                m_caus_list.append(m_caus)
                straddle_list.append(straddles)
        return {
            "vis": np.stack(vis_list),
            "low": np.stack(low_list),
            "tid": np.asarray(tid_list, dtype=np.int64),
            "tgt": np.stack(tgt_list),
            "unmasked_all": np.stack(m_unm_list),
            "role_only": np.stack(m_role_list),
            "causal_full": np.stack(m_caus_list),
            "straddle": np.asarray(straddle_list, dtype=np.int32),
        }

    tr_data = build_windows(train_keys)
    te_data = build_windows(test_keys)
    print(f"Built windows: train={len(tr_data['tid'])} (straddling={tr_data['straddle'].sum()}), test={len(te_data['tid'])} (straddling={te_data['straddle'].sum()})", flush=True)

    class ChunkPolicyNet(nn.Module):
        def __init__(self, n_tasks: int = 40):
            super().__init__()
            self.vis_proj = nn.Sequential(nn.Linear(2048, 384), nn.LayerNorm(384), nn.GELU())
            self.low_proj = nn.Sequential(nn.Linear(56, 192), nn.LayerNorm(192), nn.GELU())
            self.task_emb = nn.Embedding(n_tasks, 64)
            d = 384 + 192 + 64
            self.block1 = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.05), nn.Linear(d, d), nn.LayerNorm(d))
            self.block2 = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.05), nn.Linear(d, d), nn.LayerNorm(d))
            self.head = nn.Sequential(nn.GELU(), nn.Linear(d, 8 * 7))

        def forward(self, vis, low, tid):
            h = torch.cat([self.vis_proj(vis), self.low_proj(low), self.task_emb(tid)], dim=-1)
            h = h + self.block1(h)
            h = h + self.block2(h)
            return torch.tanh(self.head(h)).view(-1, 8, 7)

    ckpt_dir = ROOT / "outputs/rework_v11_selftest40_models_v2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    vis_mean = tr_data["vis"].mean(axis=0); vis_std = tr_data["vis"].std(axis=0) + 1e-5
    low_mean = tr_data["low"].mean(axis=0); low_std = tr_data["low"].std(axis=0) + 1e-5
    np.savez(ckpt_dir / "norm_stats.npz", vis_mean=vis_mean, vis_std=vis_std, low_mean=low_mean, low_std=low_std, task_list=np.array(task_list))

    # Train 3 mask modes x 3 seeds = 9 models across GPUs 0..3
    mask_modes = ["unmasked_all", "role_only", "causal_full"]
    offline_metrics = {m: [] for m in mask_modes}

    def train_one(mode_name: str, seed: int, gpu_id: int):
        torch.manual_seed(1000 + seed)
        np.random.seed(1000 + seed)
        dev = torch.device(f"cuda:{gpu_id}")
        valid_rows = tr_data[mode_name].sum(axis=1) > 0
        X_vis = torch.from_numpy((tr_data["vis"][valid_rows] - vis_mean) / vis_std).float().to(dev)
        X_low = torch.from_numpy((tr_data["low"][valid_rows] - low_mean) / low_std).float().to(dev)
        X_tid = torch.from_numpy(tr_data["tid"][valid_rows]).long().to(dev)
        Y_tgt = torch.from_numpy(tr_data["tgt"][valid_rows]).float().to(dev)
        M_w = torch.from_numpy(tr_data[mode_name][valid_rows]).float().to(dev)

        net = ChunkPolicyNet(len(task_list)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2400, eta_min=2e-5)
        N = len(X_tid)
        bs = 256
        net.train()
        for step_i in range(2400):
            idx = torch.randint(0, N, (bs,), device=dev)
            pred = net(X_vis[idx], X_low[idx], X_tid[idx])
            loss = masked_action_l1_loss(pred, Y_tgt[idx], M_w[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()

        net.eval()
        out_pt = ckpt_dir / f"{mode_name}_seed{seed}.pt"
        torch.save(net.state_dict(), out_pt)

        # Held-out test MAE on valid pre-boundary / post-probe actions
        with torch.no_grad():
            te_valid = te_data["causal_full"].sum(axis=1) > 0
            tv = torch.from_numpy((te_data["vis"][te_valid] - vis_mean) / vis_std).float().to(dev)
            tl = torch.from_numpy((te_data["low"][te_valid] - low_mean) / low_std).float().to(dev)
            tt = torch.from_numpy(te_data["tid"][te_valid]).long().to(dev)
            ty = torch.from_numpy(te_data["tgt"][te_valid]).float().to(dev)
            tm = torch.from_numpy(te_data["causal_full"][te_valid]).float().to(dev)
            tp = net(tv, tl, tt)
            mae = float(masked_action_l1_loss(tp, ty, tm).item())
        return mae

    for mode_name in mask_modes:
        for seed in (0, 1, 2):
            mae = train_one(mode_name, seed, seed % 4)
            offline_metrics[mode_name].append(mae)
            print(f"[Trained {mode_name} seed={seed}] held-out causal L1={mae:.4f}", flush=True)

    return test_keys, task_to_idx, offline_metrics

def eval_40tasks_closed_loop_worker(gpu_id: int, jobs: list[tuple[str, int, str]], ckpt_dir: Path, all_packs: dict, out_jsonl: Path):
    import types as _types
    if "diskcache" not in sys.modules:
        class _MemCache(dict):
            def __init__(self, *a, **kw): super().__init__()
            def close(self): pass
        sys.modules["diskcache"] = _types.SimpleNamespace(Cache=_MemCache)
    if "zstandard" not in sys.modules:
        sys.modules["zstandard"] = _types.SimpleNamespace(
            ZstdCompressor=lambda level=1: _types.SimpleNamespace(compress=lambda b: zlib.compress(bytes(b), 1)),
            ZstdDecompressor=lambda: _types.SimpleNamespace(decompress=lambda d: zlib.decompress(d)),
        )
    if "ompl" not in sys.modules:
        sys.modules["ompl"] = _types.SimpleNamespace(base=_types.SimpleNamespace(), geometric=_types.SimpleNamespace())

    SPINE_ROOT = ROOT / "VLA_SelfTest_ExpertSpine_v41_20261014"
    BASE = SPINE_ROOT / "code"
    sys.path.insert(0, str(SPINE_ROOT))
    for _sub in sorted(BASE.iterdir()):
        if _sub.is_dir():
            sys.path.insert(0, str(_sub))
    os.environ["LIBERO_CANONICAL_CATALOGUE"] = str(BASE / "planner/libero130/official130_catalogue.json")
    os.environ["LIBERO_ACCEPTANCE_RULES"] = str(BASE / "planner/libero130/acceptance_rules")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)

    import mujoco
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tvm
    from real_task_v6 import RealTaskEpisode, fix_expert_assets
    from simulation_state import initialize_replay_state, INTEGRATION_STATE

    def fix_xml(xml: str) -> str:
        xml = xml.replace("/data/Zhouqiancheng/VLA_SelfTest_Transfer_v46_20261015/env/lib/python3.10/site-packages/robosuite",
                          "/home/guest/miniconda3/envs/vla-branch/lib/python3.10/site-packages/robosuite")
        xml = xml.replace("/data/guest/miniconda3/envs/vla-branch/lib/python3.10/site-packages/robosuite",
                          "/home/guest/miniconda3/envs/vla-branch/lib/python3.10/site-packages/robosuite")
        xml = xml.replace("/data/Zhouqiancheng/VLA_SelfTest_Transfer_v46_20261015/external/LIBERO/libero",
                          "/data/guest/vla_branch_lab/external/LIBERO/libero")
        xml = xml.replace("/data/Zhouqiancheng/VLA_SelfTest_Transfer_v46_20261015",
                          "/data/guest/vla_branch_lab")
        return fix_expert_assets(xml)

    def _sp_enc(rn, xb):
        z = rn.maxpool(rn.relu(rn.bn1(rn.conv1(xb))))
        z = rn.layer4(rn.layer3(rn.layer2(rn.layer1(z))))
        gap = z.mean(dim=(-2, -1))
        B_, C_, H_, W_ = z.shape
        pr = torch.softmax(z.reshape(B_, C_, H_ * W_) * 2.0, dim=-1)
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, H_, device=z.device),
                                torch.linspace(-1, 1, W_, device=z.device), indexing="ij")
        ex = (pr * gx.reshape(1, 1, -1)).sum(dim=-1)
        ey = (pr * gy.reshape(1, 1, -1)).sum(dim=-1)
        return torch.cat([gap, ex[:, :256], ey[:, :256]], dim=-1)

    dev = torch.device(f"cuda:{gpu_id}")
    stats = np.load(ckpt_dir / "norm_stats.npz", allow_pickle=True)
    vis_mean = stats["vis_mean"]; vis_std = stats["vis_std"]
    low_mean = stats["low_mean"]; low_std = stats["low_std"]
    task_list = stats["task_list"].tolist()
    task_to_idx = {t: i for i, t in enumerate(task_list)}

    q_ckpt = torch.load(ROOT / "checkpoints/LIBERO_SPATIAL_sft.pth", map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in (q_ckpt["model"] if "model" in q_ckpt else q_ckpt).items()}
    rn_agent = tvm.resnet18(weights=None); rn_wrist = tvm.resnet18(weights=None)
    for prefix, rn in [("image_encoders.agentview_rgb", rn_agent), ("image_encoders.eye_in_hand_rgb", rn_wrist)]:
        own = rn.state_dict()
        for k_src, v_src in sd.items():
            if k_src.startswith(prefix):
                for k_dst in own:
                    if k_src.endswith(k_dst) and own[k_dst].shape == v_src.shape:
                        own[k_dst] = v_src.clone(); break
        rn.load_state_dict(own, strict=False); rn.fc = nn.Identity(); rn.eval().to(dev)

    class ChunkPolicyNet(nn.Module):
        def __init__(self, n_tasks: int = 40):
            super().__init__()
            self.vis_proj = nn.Sequential(nn.Linear(2048, 384), nn.LayerNorm(384), nn.GELU())
            self.low_proj = nn.Sequential(nn.Linear(56, 192), nn.LayerNorm(192), nn.GELU())
            self.task_emb = nn.Embedding(n_tasks, 64)
            d = 384 + 192 + 64
            self.block1 = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.05), nn.Linear(d, d), nn.LayerNorm(d))
            self.block2 = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.05), nn.Linear(d, d), nn.LayerNorm(d))
            self.head = nn.Sequential(nn.GELU(), nn.Linear(d, 8 * 7))
        def forward(self, vis, low, tid):
            h = torch.cat([self.vis_proj(vis), self.low_proj(low), self.task_emb(tid)], dim=-1)
            h = h + self.block1(h); h = h + self.block2(h)
            return torch.tanh(self.head(h)).view(-1, 8, 7)

    models_cache = {}
    def get_model(mode_name: str, seed: int):
        if (mode_name, seed) not in models_cache:
            net = ChunkPolicyNet(len(task_list)).to(dev)
            net.load_state_dict(torch.load(ckpt_dir / f"{mode_name}_seed{seed}.pt", map_location=dev, weights_only=True))
            net.eval()
            models_cache[(mode_name, seed)] = net
        return models_cache[(mode_name, seed)]

    cur_key = None
    cur_ep = None
    cur_acts = None
    cur_ist0 = None

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for idx, (mode_name, seed, ep_key) in enumerate(jobs, 1):
            pk = all_packs[ep_key]
            folder = Path(pk["folder"])
            meta = json.loads((folder / "metadata.json").read_text())
            if cur_key != ep_key:
                if cur_ep is not None:
                    cur_ep.close()
                scene_xml = fix_xml((folder / "scene.xml").read_text())
                with h5py.File(folder / "trajectory.hdf5", "r") as h:
                    cur_acts = np.asarray(h["actions"], dtype=np.float32)
                    cur_ist0 = np.asarray(h["observations/integration_state"][0], dtype=np.float64)
                cur_ep = RealTaskEpisode(meta["task_id"], int(meta["demo_id"]), 1.12, condition_id="C01",
                                         target_outcome="first_pass", planner_enabled=False, online_guard=False, record_images=True)
                cur_ep.record_images = False
                cur_ep.fixture_shield = None
                cur_ep.scene_xml = scene_xml
                cur_ep.env.reset_from_xml_string(scene_xml)
                cur_ep.model = cur_ep.env.sim.model
                cur_ep.baseline_parameters = cur_ep.physical_parameters().copy()
                cur_key = ep_key

            ep = cur_ep
            p_ends = pk["probe_ends"]
            t_start = max(0, int(p_ends[0]) - 5) if len(p_ends) > 0 else max(0, len(cur_acts) // 3)

            ep.actions.clear(); ep.phases.clear(); ep.records.clear()
            initialize_replay_state(ep, cur_ist0)
            ctrl = ep.env.robots[0].controller
            ctrl.update(); ctrl.goal_pos = ctrl.ee_pos.copy(); ctrl.goal_ori = ctrl.ee_ori_mat.copy()

            for t_pre in range(t_start):
                ep.execute(cur_acts[t_pre], "expert_prefix")

            net = get_model(mode_name, seed)
            tid_tensor = torch.tensor([task_to_idx[pk["task_id"]]], dtype=torch.long, device=dev)
            max_rem = min(260, len(cur_acts) - t_start + 72)
            chunk_q = []
            ok_final = 0
            for st in range(max_rem):
                cur_t = t_start + st
                if not chunk_q:
                    rec = ep.records[-1]
                    ist_cur = rec["integration_state"]
                    ep.env.sim.forward()
                    obs_cur = ep.env.env._get_observations(force_update=True)
                    ag_im = np.asarray(obs_cur["agentview_image"], dtype=np.uint8)
                    wr_im = np.asarray(obs_cur["robot0_eye_in_hand_image"], dtype=np.uint8)
                    mujoco.mj_setState(ep.model._model, ep.env.sim.data._data, ist_cur, INTEGRATION_STATE)
                    with torch.no_grad():
                        a_t = F.interpolate(torch.from_numpy(ag_im).unsqueeze(0).permute(0,3,1,2).float().to(dev)/255.0, size=(128,128), mode="bilinear", align_corners=False)
                        w_t = F.interpolate(torch.from_numpy(wr_im).unsqueeze(0).permute(0,3,1,2).float().to(dev)/255.0, size=(128,128), mode="bilinear", align_corners=False)
                        vf = torch.cat([_sp_enc(rn_agent, a_t), _sp_enc(rn_wrist, w_t)], dim=-1).cpu().numpy()[0]
                        ti = len(ep.records) - 1
                        tp4, tp12 = max(0, ti - 4), max(0, ti - 12)
                        ah = np.zeros((4, 7), dtype=np.float32)
                        for k_a in range(1, 5):
                            if len(ep.actions) - k_a >= 0:
                                ah[4 - k_a] = ep.actions[-k_a]
                        c_hist_all = [float(r["proprio"][6] - r["proprio"][7]) for r in ep.records]
                        lf = np.concatenate([
                            rec["proprio"], rec["eef"].astype(np.float32),
                            (rec["eef"] - ep.records[tp4]["eef"]).astype(np.float32) * 20.0,
                            (rec["eef"] - ep.records[tp12]["eef"]).astype(np.float32) * 10.0,
                            rec["proprio"] - ep.records[tp12]["proprio"],
                            np.array([c_hist_all[-1], float(np.mean(c_hist_all[tp12:])), min(1.0, cur_t / 300.0)], dtype=np.float32),
                            ah.reshape(-1),
                        ]).astype(np.float32)
                        v_in = torch.from_numpy(((vf - vis_mean) / vis_std)[None, :]).float().to(dev)
                        l_in = torch.from_numpy(((lf - low_mean) / low_std)[None, :]).float().to(dev)
                        pred_ch = net(v_in, l_in, tid_tensor).cpu().numpy()[0]
                        n_exec = 8
                        if mode_name == "causal_full":
                            fut_b = pk["boundaries"][pk["boundaries"] > cur_t]
                            if len(fut_b) > 0 and int(fut_b[0]) - cur_t < 8:
                                n_exec = max(1, int(fut_b[0]) - cur_t)
                        chunk_q = [pred_ch[j] for j in range(n_exec)]

                act_cmd = chunk_q.pop(0)
                ep.execute(act_cmd, "learned_policy")
                if ep.env.check_success():
                    ok_final = 1
                    break

            rec_out = {
                "mode_mask": mode_name,
                "seed": seed,
                "ep_key": ep_key,
                "suite": pk["suite"],
                "task_id": pk["task_id"],
                "ep_type": pk["mode"],
                "success": ok_final,
            }
            fh.write(json.dumps(rec_out) + "\n")
            if idx % 18 == 0 or idx == len(jobs):
                fh.flush()
                print(f"[Eval40 GPU{gpu_id}] {idx}/{len(jobs)} last={ok_final}", flush=True)
    if cur_ep is not None:
        cur_ep.close()

def main():
    mp.set_start_method("spawn", force=True)
    t0 = time.time()

    quest_ablation = run_part1_quest_spatial_and_goal()

    manifest_320 = get_40task_manifest()
    feat_dir = ROOT / "outputs/rework_v11_selftest40_feats"
    feat_dir.mkdir(parents=True, exist_ok=True)
    merged_feat_file = feat_dir / "all_320_spatial_packs_v2.npy"

    if not merged_feat_file.exists():
        shards = [manifest_320[i::4] for i in range(4)]
        procs = [mp.Process(target=extract_episode_features_worker, args=(g, shards[g], feat_dir / f"part_sp_{g}.npy")) for g in range(4)]
        for p in procs: p.start()
        for p in procs:
            p.join()
            if p.exitcode != 0: raise RuntimeError("Feature extraction failed")
        all_packs = {}
        for g in range(4):
            part = np.load(feat_dir / f"part_sp_{g}.npy", allow_pickle=True).item()
            all_packs.update(part)
        np.save(merged_feat_file, all_packs, allow_pickle=True)
    else:
        all_packs = np.load(merged_feat_file, allow_pickle=True).item()

    test_keys, task_to_idx, offline_metrics = train_and_eval_40tasks(all_packs)

    # Closed-loop MuJoCo evaluation across all 80 held-out test episodes x 3 mask modes x 3 seeds = 720 rollouts across 8 GPU workers!
    ckpt_dir = ROOT / "outputs/rework_v11_selftest40_models_v2"
    n_w = 8
    key_shards = [test_keys[i::n_w] for i in range(n_w)]
    shards = [[(m, sd, k) for k in key_shards[w] for m in ("unmasked_all", "role_only", "causal_full") for sd in (0, 1, 2)] for w in range(n_w)]
    procs = [mp.Process(target=eval_40tasks_closed_loop_worker, args=(w % 4, shards[w], ckpt_dir, all_packs, ckpt_dir / f"eval_{w}.jsonl")) for w in range(n_w)]
    for p in procs: p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0: raise RuntimeError(f"Eval40 worker failed: {p.exitcode}")

    eval_rows = []
    for w in range(n_w):
        for line in (ckpt_dir / f"eval_{w}.jsonl").read_text().splitlines():
            if line.strip():
                eval_rows.append(json.loads(line))

    with (DATA_OUT / "v11_selftest40_closed_loop.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
        wr.writeheader(); wr.writerows(eval_rows)

    suite_summary = {}
    for m in ("unmasked_all", "role_only", "causal_full"):
        sub_m = [r for r in eval_rows if r["mode_mask"] == m]
        by_s = {}
        for s_name in ("Spatial", "Object", "Goal", "Long-10"):
            ss = [r for r in sub_m if r["suite"] == s_name]
            ss_pass = [r for r in ss if r["ep_type"] == "probe_success"]
            ss_rec = [r for r in ss if r["ep_type"] != "probe_success"]
            by_s[s_name] = {
                "n": len(ss),
                "overall_pct": round(100.0 * sum(r["success"] for r in ss) / len(ss), 1),
                "pass_pct": round(100.0 * sum(r["success"] for r in ss_pass) / len(ss_pass), 1),
                "recovery_pct": round(100.0 * sum(r["success"] for r in ss_rec) / len(ss_rec), 1),
            }
        all_pass = [r for r in sub_m if r["ep_type"] == "probe_success"]
        all_rec = [r for r in sub_m if r["ep_type"] != "probe_success"]
        suite_summary[m] = {
            "n_total": len(sub_m),
            "macro_40tasks_pct": round(100.0 * sum(r["success"] for r in sub_m) / len(sub_m), 1),
            "pass_40tasks_pct": round(100.0 * sum(r["success"] for r in all_pass) / len(all_pass), 1),
            "recovery_40tasks_pct": round(100.0 * sum(r["success"] for r in all_rec) / len(all_rec), 1),
            "heldout_causal_l1": round(float(np.mean(offline_metrics[m])), 4),
            "by_suite": by_s,
        }

    out = {
        "elapsed_s": round(time.time() - t0, 2),
        "quest_closed_loop_ablation": quest_ablation,
        "selftest_40tasks_4suites": suite_summary,
    }
    (DATA_OUT / "v11_table4_full_ablation.json").write_text(json.dumps(out, indent=2))
    print("SUITE B COMPLETE:", json.dumps(out, indent=2), flush=True)

if __name__ == "__main__":
    main()
