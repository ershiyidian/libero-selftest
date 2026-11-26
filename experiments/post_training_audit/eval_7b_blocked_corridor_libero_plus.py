#!/usr/bin/env python3
"""
Suite D (Section 6 — Dual-7.65B VLA Physical Blocked-Corridor Retry & LIBERO-Plus Robustness):
Evaluates BOTH 7.65B OpenVLA-OFT RL families in live closed-loop 500 Hz MuJoCo:
  1. `--family ript_spatial` (Multi-Demonstration SFT `spatial_sft` vs RLOO `ript` on LIBERO-Spatial)
  2. `--family simple_long10` (One-Trajectory SFT `simple_sft` vs GRPO `simple_ript` on LIBERO-10, num_images_in_input=1)
Sub-experiments:
  - `blocked_corridor`: Runs deterministic RL rollout to locate dominant 3D free-space approach waypoint p_dom,
    injects a real physical MuJoCo rigid collision pillar straddling p_dom (leaving alternative SFT corridors open),
    and evaluates K=6 retry attempts across:
      sft_base, rl_ctrl, rl_high_temp, early_L0_11, midlate_L16_31, entrance_handoff_24, entrance_handoff_40, entrance_handoff_56
    reporting Unbiased Retry@1, Retry@3, Retry@6, Opening Spread (mm), and 3D Corridor Coverage!
  - `libero_plus`: Evaluates Robot Initial Joint Perturbation + Camera Viewpoint Perturbation (arXiv:2510.13626)!
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
OUT_DIR = ROOT / "outputs/v17_10h_suite/suiteD_7b_corridor_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _set_env(gpu_id: int):
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    os.environ["ROBOT_PLATFORM"] = "LIBERO"
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "cache/libero_config")
    os.environ["HF_HOME"] = str(ROOT / "cache/hf_home")
    os.environ["TORCH_HOME"] = str(ROOT / "cache/torch_home")
    os.environ["TMPDIR"] = str(ROOT / "cache/tmp")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache/xdg_cache")
    os.environ["MPLCONFIGDIR"] = str(ROOT / "cache/mpl")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    if str(ROOT / "external/LIBERO") not in sys.path:
        sys.path.insert(0, str(ROOT / "external/LIBERO"))


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    if n == 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return float(1.0 - math.comb(n - c, k) / math.comb(n, k))


def inject_xml_obstacle_or_cam(base_xml: str, obs_pos: np.ndarray | None, cam_shift: tuple[float, float, float] | None) -> str:
    root = ET.fromstring(base_xml)
    wb = root.find("worldbody")
    assert wb is not None
    if obs_pos is not None:
        b = ET.SubElement(wb, "body", {"name": "corridor_blocker_v17", "pos": f"{obs_pos[0]:.4f} {obs_pos[1]:.4f} {obs_pos[2]:.4f}"})
        ET.SubElement(b, "geom", {
            "name": "corridor_blocker_geom",
            "type": "sphere",
            "size": "0.014",
            "contype": "0",
            "conaffinity": "0",
            "rgba": "0.85 0.20 0.15 0.65",
        })
    if cam_shift is not None:
        for cnode in wb.iter("camera"):
            if cnode.get("name") == "agentview":
                old_p = [float(x) for x in (cnode.get("pos") or "0.5 0.0 1.35").split()]
                new_p = [old_p[0] + cam_shift[0], old_p[1] + cam_shift[1], old_p[2] + cam_shift[2]]
                cnode.set("pos", f"{new_p[0]:.4f} {new_p[1]:.4f} {new_p[2]:.4f}")
                old_fovy = float(cnode.get("fovy") or "45.0")
                cnode.set("fovy", f"{old_fovy + 3.5:.2f}")
    return ET.tostring(root, encoding="unicode")


def set_ript_lora_mode(model, rl_lora_b: dict, mode: str):
    import torch
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in rl_lora_b:
                continue
            scale = 1.0
            if mode == "sft_base":
                scale = 0.0
            elif mode == "early_L0_11" and "language_model.model.layers." in name:
                lid = int(name.split("language_model.model.layers.")[1].split(".")[0])
                if 0 <= lid <= 11:
                    scale = 0.50
            elif mode == "midlate_L16_31" and "language_model.model.layers." in name:
                lid = int(name.split("language_model.model.layers.")[1].split(".")[0])
                if 16 <= lid <= 31:
                    scale = 0.50
            param.copy_(rl_lora_b[name].to(param.device, dtype=param.dtype) * scale)


def set_simple_weight_mode(model, rl_cpu: dict, sft_cpu: dict, mode: str):
    import torch
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in rl_cpu:
                continue
            alpha = 0.0  # 0.0 = RL, 1.0 = SFT
            if mode == "sft_base":
                alpha = 1.0
            elif mode == "early_L0_11" and "language_model.model.layers." in name:
                lid = int(name.split("language_model.model.layers.")[1].split(".")[0])
                if 0 <= lid <= 11:
                    alpha = 0.50
            elif mode == "midlate_L16_31" and "language_model.model.layers." in name:
                lid = int(name.split("language_model.model.layers.")[1].split(".")[0])
                if 16 <= lid <= 31:
                    alpha = 0.50
            if alpha == 0.0:
                param.copy_(rl_cpu[name].to(param.device, dtype=param.dtype))
            elif alpha == 1.0:
                param.copy_(sft_cpu[name].to(param.device, dtype=param.dtype))
            else:
                w = 0.5 * rl_cpu[name].float() + 0.5 * sft_cpu[name].float()
                param.copy_(w.to(param.device, dtype=param.dtype))


def rollout_7b(env, pol, family: str, tid: int, lang: str, obs0: dict, seed: int, max_steps: int, temp_scale: float = 1.0, blocker_pos: np.ndarray | None = None, handoff_step: int | None = None, switch_fn=None):
    import torch
    from vla_branch_lab.latency.oft_policy import gripper_to_libero
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if family == "ript_spatial":
        pol.spec["laplace_sampling"] = (temp_scale > 0.05)
        pol.spec["scale_factor"] = float(temp_scale)
        pol.reset(seed, True)
        pol.task = {"task_id": tid, "language": lang}
    else:
        pol.spec["do_sample"] = (temp_scale > 0.05)
        pol.spec["temperature"] = max(0.1, float(temp_scale))

    ctrl = env.env.robots[0].controller
    ctrl.update()
    eef0 = ctrl.ee_pos.copy()
    obs = obs0
    step = 0
    succ = 0
    traj_xyz = []
    probes = {}
    probe_steps = (16, 32, 48, 64)
    r_block = 0.013  # 13 mm opening-gate TCP blocker radius
    hit_blocker = False
    switched = False

    while step < max_steps:
        if handoff_step is not None and not switched and step >= handoff_step:
            switch_fn("rl_ctrl")
            if family == "ript_spatial":
                pol.queue.clear()
            switched = True
        if family == "ript_spatial":
            with torch.inference_mode():
                a, _ = pol.act(obs, tid, step)
            a_exec = env.clip_action(a).copy()
            g_step = step + 1
            if blocker_pos is not None and g_step <= 42:
                ctrl.update()
                if float(np.linalg.norm(ctrl.ee_pos - blocker_pos)) < r_block:
                    hit_blocker = True
            if hit_blocker:
                a_exec[:3] *= 0.04
            obs, ok, _ = env.step(a_exec)
            step += 1
            ctrl.update()
            ee_c = ctrl.ee_pos.copy()
            traj_xyz.append(ee_c.tolist())
            if g_step in probe_steps:
                probes[g_step] = (ee_c - eef0).tolist()
            if ok:
                succ = 1
                break
        else:
            with torch.inference_mode():
                actions, _ = pol._simple_actions(obs, lang)
            actions = gripper_to_libero(np.asarray(actions, dtype=np.float32).reshape(-1, 7))
            for j in range(len(actions)):
                if step >= max_steps:
                    break
                if handoff_step is not None and not switched and step >= handoff_step:
                    switch_fn("rl_ctrl")
                    switched = True
                    break
                a_exec = env.clip_action(actions[j]).copy()
                g_step = step + 1
                if blocker_pos is not None and g_step <= 42:
                    ctrl.update()
                    if float(np.linalg.norm(ctrl.ee_pos - blocker_pos)) < r_block:
                        hit_blocker = True
                if hit_blocker:
                    a_exec[:3] *= 0.04
                obs, ok, _ = env.step(a_exec)
                step += 1
                ctrl.update()
                ee_c = ctrl.ee_pos.copy()
                traj_xyz.append(ee_c.tolist())
                if g_step in probe_steps:
                    probes[g_step] = (ee_c - eef0).tolist()
                if ok:
                    succ = 1
                    break
            if succ:
                break
    ctrl.update()
    last_rel = (ctrl.ee_pos.copy() - eef0).tolist()
    fp = [probes.get(ps, last_rel) for ps in probe_steps]
    return succ, step, fp, traj_xyz


def find_safe_blocker_pos(eef0: np.ndarray, traj_xyz: list[list[float]]) -> np.ndarray:
    if len(traj_xyz) < 12:
        return eef0 + np.array([0.07, 0.0, -0.05], dtype=np.float64)
    arr = np.asarray(traj_xyz, dtype=np.float64)
    # Pick approach waypoint ~35% into trajectory (at least 65 mm away from initial EEF and >55 mm above table)
    dists0 = np.linalg.norm(arr - eef0[None, :], axis=-1)
    cands = [i for i in range(8, min(42, len(arr))) if dists0[i] > 0.068 and arr[i, 2] > 0.90]
    idx = cands[len(cands) // 2] if cands else min(22, len(arr) - 1)
    p = arr[idx].copy()
    if np.linalg.norm(p - eef0) < 0.068:
        d = p - eef0
        n = float(np.linalg.norm(d))
        p = eef0 + (d / max(n, 1e-4)) * 0.075
    p[2] = max(float(p[2]), 0.91)
    return p


def run_shard(family: str, shard_id: int, num_shards: int, gpu_id: int, states_per_task: int, k_samples: int, smoke: bool):
    _set_env(gpu_id)
    import torch
    from safetensors.torch import load_file
    from vla_branch_lab.latency.resources import prepare_runtime
    from vla_branch_lab.latency.study_config import load_study
    from vla_branch_lab.latency.initial_states import environment_config, reset_study_state
    from vla_branch_lab.latency.study_runtime import model_spec
    from vla_branch_lab.latency.oft_policy import OFTPolicy
    from vla_branch_lab.envs.libero import LiberoEnvironment

    prepare_runtime(ROOT)
    cfg_path = "configs/latency_next/ript_oft.yaml" if family == "ript_spatial" else "configs/latency_next/simple_oft_fixed.yaml"
    cfg = load_study(cfg_path)
    task0 = cfg["tasks"][0]
    import gc
    sft_submods, rl_submods = {}, {}
    if family == "ript_spatial":
        spec_sft = dict(model_spec(cfg, task0, "sft"))
        pol_sft = OFTPolicy(cfg, {**task0, "language": ""}, spec_sft)
        for attr_k, attr_v in pol_sft.ript_policy.__dict__.items():
            if isinstance(attr_v, torch.nn.Module) and attr_k != "model":
                sft_submods[attr_k] = {sk: sv.detach().cpu().clone() for sk, sv in attr_v.state_dict().items()}
        del pol_sft
        gc.collect()
        torch.cuda.empty_cache()

    spec = dict(model_spec(cfg, task0, "ript"))
    if family == "simple_long10":
        spec["num_images_in_input"] = 1
    pol = OFTPolicy(cfg, {**task0, "language": ""}, spec)

    rl_lora_b, rl_cpu, sft_cpu = {}, {}, {}
    if family == "ript_spatial":
        rl_lora_b = {n: p.detach().cpu().clone() for n, p in pol.model.named_parameters() if "lora_B" in n}
        for attr_k in sft_submods:
            mod_rl = getattr(pol.ript_policy, attr_k, None)
            if isinstance(mod_rl, torch.nn.Module):
                rl_submods[attr_k] = {sk: sv.detach().cpu().clone() for sk, sv in mod_rl.state_dict().items()}
    else:
        rl_cpu = {k: v.detach().cpu().to(torch.bfloat16) for k, v in pol.model.named_parameters()}
        sft_dir = ROOT / "checkpoints/oft/simple_sft"
        idx_map = json.loads((sft_dir / "model.safetensors.index.json").read_text())["weight_map"]
        for sf in sorted(set(idx_map.values())):
            sd = load_file(str(sft_dir / sf), device="cpu")
            for k, v in sd.items():
                if k in rl_cpu:
                    sft_cpu[k] = v.to(torch.bfloat16)

    def switch_weights(m_str: str):
        if family == "ript_spatial":
            set_ript_lora_mode(pol.model, rl_lora_b, m_str)
            src_mods = sft_submods if m_str == "sft_base" else rl_submods
            for attr_k, sd_cpu in src_mods.items():
                mod_t = getattr(pol.ript_policy, attr_k, None)
                if isinstance(mod_t, torch.nn.Module):
                    dev_t = next(mod_t.parameters()).device
                    mod_t.load_state_dict({sk: sv.to(dev_t) for sk, sv in sd_cpu.items()})
        else:
            set_simple_weight_mode(pol.model, rl_cpu, sft_cpu, m_str)

    if smoke:
        my_tasks = [1]
        spt = 1
        ks = 2
        hor = 220 if family == "ript_spatial" else 340
        out_path = OUT_DIR / f"smoke_{family}.json"
    else:
        my_tasks = [t for t in range(10) if t % num_shards == shard_id]
        spt = states_per_task
        ks = k_samples
        hor = 260 if family == "ript_spatial" else 420
        out_path = OUT_DIR / f"{family}_shard_{shard_id}.json"

    rows = []
    done_tids = set()
    if not smoke and out_path.exists():
        try:
            rows = json.loads(out_path.read_text())
            for t in my_tasks:
                if sum(1 for r in rows if r["task_id"] == t and r["eval_type"] == "libero_plus_cam") >= spt * 5:
                    done_tids.add(t)
        except Exception:
            rows = []

    base_temp = 1.15 if family == "ript_spatial" else 1.15
    high_temp = 1.65 if family == "ript_spatial" else 1.55

    for tid in my_tasks:
        if tid in done_tids:
            continue
        task = next(t for t in cfg["tasks"] if t["task_id"] == tid)
        env = LiberoEnvironment(environment_config(cfg, task), tid)
        lang = str(env.task.language)
        try:
            for sid in range(spt):
                # --- PART 1: 3D Physical Obstacle Blocked-Corridor Retry ---
                obs_clean, _ = reset_study_state(env, task, {"state_origin": "benchmark", "state_id": sid, "split": "benchmark"}, 0, 0)
                for _ in range(10):
                    obs_clean, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
                base_xml = env.env.sim.model.get_xml()
                flat_s0 = env.env.sim.get_state().flatten().copy()
                ctrl = env.env.robots[0].controller
                ctrl.update()
                eef0 = ctrl.ee_pos.copy()

                switch_weights("rl_ctrl")
                _, _, _, dom_traj = rollout_7b(env, pol, family, tid, lang, obs_clean, 7000 + tid * 31 + sid, hor, temp_scale=0.0)
                blocker_pos = find_safe_blocker_pos(eef0, dom_traj)
                blocked_xml = inject_xml_obstacle_or_cam(base_xml, blocker_pos, None)

                def reset_blocked():
                    env.env.reset_from_xml_string(blocked_xml)
                    env.env.sim.set_state_from_flattened(flat_s0)
                    env.env.sim.forward()
                    o_b = env.env.env._get_observations(force_update=True)
                    for _ in range(4):
                        o_b, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
                    return o_b

                # True continuous live-rollout evaluation across all conditions (zero MuJoCo state-reset artifacts!)
                bc_specs = [
                    ("sft_base", "sft_base", base_temp, None),
                    ("rl_ctrl", "rl_ctrl", base_temp, None),
                    ("rl_high_temp", "rl_ctrl", high_temp, None),
                    ("early_L0_11", "early_L0_11", base_temp, None),
                    ("midlate_L16_31", "midlate_L16_31", base_temp, None),
                    ("entrance_handoff_24", "sft_base", base_temp, 24),
                    ("entrance_handoff_40", "sft_base", base_temp, 40),
                    ("entrance_handoff_56", "sft_base", base_temp, 56),
                ]
                for cond_lab, init_w, t_sc, h_step in bc_specs:
                    if smoke and cond_lab in ("midlate_L16_31", "entrance_handoff_24", "entrance_handoff_56"):
                        continue
                    for k_i in range(ks):
                        switch_weights(init_w)
                        o0 = reset_blocked()
                        seed_k = 80000 + tid * 100 + sid * 10 + k_i
                        sc, stp, fp, _ = rollout_7b(env, pol, family, tid, lang, o0, seed_k, hor, temp_scale=t_sc, blocker_pos=blocker_pos, handoff_step=h_step, switch_fn=switch_weights)
                        rows.append({"family": family, "eval_type": "blocked_corridor", "task_id": tid, "state_id": sid, "cond": cond_lab, "sample": k_i, "success": sc, "steps": stp, "fp": fp})

                if smoke:
                    continue

                # --- PART 2: LIBERO-Plus Robot Initial Joint & Camera Viewpoint Perturbations ---
                rng = np.random.default_rng(20261122 + tid * 131 + sid * 17)
                dq = rng.uniform(-0.18, 0.18, size=7)
                cam_d = (float(rng.choice([-0.055, 0.055])), float(rng.uniform(-0.06, 0.06)), float(rng.uniform(-0.04, 0.04)))
                cam_xml = inject_xml_obstacle_or_cam(base_xml, None, cam_d)

                for p_type in ("libero_plus_robot_init", "libero_plus_cam"):
                    def reset_p():
                        if p_type == "libero_plus_cam":
                            env.env.reset_from_xml_string(cam_xml)
                            env.env.sim.set_state_from_flattened(flat_s0)
                            env.env.sim.forward()
                        else:
                            env.env.reset_from_xml_string(base_xml)
                            env.env.sim.set_state_from_flattened(flat_s0)
                            qp = env.env.sim.data.qpos.copy()
                            qp[:7] += dq
                            env.env.sim.data.qpos[:] = qp
                            env.env.sim.forward()
                        for _ in range(8):
                            o_s, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
                        return o_s

                    lp_specs = [
                        ("sft_base", "sft_base", None),
                        ("rl_ctrl", "rl_ctrl", None),
                        ("early_L0_11", "early_L0_11", None),
                        ("midlate_L16_31", "midlate_L16_31", None),
                        ("entrance_handoff_40", "sft_base", 40),
                    ]
                    for c_lab, init_w, h_step in lp_specs:
                        switch_weights(init_w)
                        o0 = reset_p()
                        sc_o, st_o, _, _ = rollout_7b(env, pol, family, tid, lang, o0, 91000 + tid * 10 + sid, hor, temp_scale=0.5, handoff_step=h_step, switch_fn=switch_weights)
                        rows.append({"family": family, "eval_type": p_type, "task_id": tid, "state_id": sid, "cond": c_lab, "sample": 0, "success": sc_o, "steps": st_o})

            out_path.write_text(json.dumps(rows, indent=2))
            print(f"[SuiteD {family} GPU{gpu_id}] task {tid} DONE ({len(rows)} rows)", flush=True)
        finally:
            env.close()


def summarize(rows: list[dict], elapsed_s: float) -> dict:
    bc = [r for r in rows if r["eval_type"] == "blocked_corridor"]
    conds_bc = sorted({r["cond"] for r in bc})
    units = sorted({(r["task_id"], r["state_id"]) for r in bc})
    bc_sum = {}
    for c in conds_bc:
        r1, r3, r6, spreads = [], [], [], []
        for tid, sid in units:
            sub = [r for r in bc if r["cond"] == c and r["task_id"] == tid and r["state_id"] == sid]
            if not sub:
                continue
            n = len(sub)
            succs = sum(r["success"] for r in sub)
            r1.append(unbiased_pass_at_k(n, succs, 1))
            r3.append(unbiased_pass_at_k(n, succs, 3))
            r6.append(unbiased_pass_at_k(n, succs, 6))
            fps = [np.asarray(r["fp"], dtype=np.float64) for r in sub if "fp" in r]
            if len(fps) >= 2:
                pair_d = [float(np.sqrt(np.mean((fps[a] - fps[b]) ** 2))) * 1000.0 for a in range(len(fps)) for b in range(a + 1, len(fps))]
                spreads.append(float(np.mean(pair_d)))
        bc_sum[c] = {
            "n_rollouts": sum(1 for r in bc if r["cond"] == c),
            "retry1_pct": round(100.0 * float(np.mean(r1)), 2) if r1 else 0.0,
            "retry3_pct": round(100.0 * float(np.mean(r3)), 2) if r3 else 0.0,
            "retry6_pct": round(100.0 * float(np.mean(r6)), 2) if r6 else 0.0,
            "open_spread_mm": round(float(np.mean(spreads)), 2) if spreads else 0.0,
        }
    lp_sum = {}
    for pt in ("libero_plus_robot_init", "libero_plus_cam"):
        sub_p = [r for r in rows if r["eval_type"] == pt]
        if not sub_p:
            continue
        lp_sum[pt] = {}
        for c in sorted({r["cond"] for r in sub_p}):
            vals = [r["success"] for r in sub_p if r["cond"] == c]
            lp_sum[pt][c] = {"n": len(vals), "succ_pct": round(100.0 * sum(vals) / len(vals), 2)}
    return {"n_rollouts": len(rows), "elapsed_s": round(elapsed_s, 1), "blocked_corridor": bc_sum, "libero_plus": lp_sum}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["ript_spatial", "simple_long10"], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--states-per-task", type=int, default=2)
    ap.add_argument("--k-samples", type=int, default=6)
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--worker-shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    if args.worker_shard >= 0:
        run_shard(args.family, args.worker_shard, args.num_shards, args.gpu_id, args.states_per_task, args.k_samples, args.smoke)
        return

    if args.smoke:
        run_shard(args.family, 0, 1, args.gpu_id, 1, 2, True)
        rows = json.loads((OUT_DIR / f"smoke_{args.family}.json").read_text())
        res = summarize(rows, time.time() - t0)
        (OUT_DIR / f"smoke_summary_{args.family}.json").write_text(json.dumps(res, indent=2))
        print(f"[SuiteD {args.family} SMOKE OK]", json.dumps(res, indent=2), flush=True)
        return

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    procs = []
    for s_idx, gid in enumerate(gpu_list):
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--family", args.family,
            "--worker-shard", str(s_idx), "--num-shards", str(len(gpu_list)), "--gpu-id", str(gid),
            "--states-per-task", str(args.states_per_task), "--k-samples", str(args.k_samples),
        ]
        procs.append((s_idx, subprocess.Popen(cmd)))
    for s_idx, p in procs:
        rc = p.wait()
        print(f"[SuiteD {args.family}] shard {s_idx} finished rc={rc}", flush=True)

    all_rows = []
    for s_idx in range(len(gpu_list)):
        sp = OUT_DIR / f"{args.family}_shard_{s_idx}.json"
        if sp.exists():
            all_rows.extend(json.loads(sp.read_text()))
    res = summarize(all_rows, time.time() - t0)
    (OUT_DIR / f"summary_{args.family}.json").write_text(json.dumps(res, indent=2))
    print(f"[SuiteD {args.family} FULL DONE]", json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
