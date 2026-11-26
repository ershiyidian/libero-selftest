#!/usr/bin/env python3
"""
Exp 1: Full-Parameter SimpleVLA-RL (7B GRPO vs SFT) on LIBERO-10
Evaluates: sft_base, grpo_ctrl, early_L0_11, midlate_L16_31, entrance_handoff (48-step SFT -> GRPO)
Metrics: Unbiased Pass@1/4/8 + Multi-scale 3D Physical Corridor Coverage (15mm, 25mm, 35mm RMS).
Strictly keeps all caches/temp under /data/guest/vla_branch_lab/cache/.
"""
from __future__ import annotations
import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
OUT_DIR = ROOT / "outputs/v13_supplement/exp1_simplevla_rl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _set_data_env(gpu_id: int):
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
    os.environ["OMP_NUM_THREADS"] = "2"
    sys.path.insert(0, str(ROOT / "src"))

def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        k = n
    if n == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return float(1.0 - math.comb(n - c, k) / math.comb(n, k))

def apply_layer_interp(model, rl_cpu: dict, sft_cpu: dict, layer_range: tuple[int, int] | None, alpha: float):
    import torch
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in rl_cpu:
                continue
            use_alpha = 0.0
            if layer_range is not None:
                lo, hi = layer_range
                if "language_model.model.layers." in name:
                    lid = int(name.split("language_model.model.layers.")[1].split(".")[0])
                    if lo <= lid <= hi:
                        use_alpha = alpha
            elif alpha == 1.0:
                use_alpha = 1.0
            if use_alpha == 0.0:
                param.copy_(rl_cpu[name].to(param.device, dtype=param.dtype))
            elif use_alpha == 1.0:
                param.copy_(sft_cpu[name].to(param.device, dtype=param.dtype))
            else:
                w = (1.0 - use_alpha) * rl_cpu[name].float() + use_alpha * sft_cpu[name].float()
                param.copy_(w.to(param.device, dtype=param.dtype))

def reset_with_pose_jitter(env, task, state_id: int):
    from vla_branch_lab.latency.initial_states import reset_study_state
    obs, _ = reset_study_state(
        env, task, {"state_origin": "benchmark", "state_id": state_id, "split": "benchmark"}, 0, 0
    )
    rng = np.random.default_rng(20261120 + task["task_id"] * 1000 + state_id * 37)
    sim = env.env.sim
    qpos = sim.data.qpos.copy()
    qpos[:7] += rng.uniform(-0.015, 0.015, size=7)
    sim.data.qpos[:] = qpos
    sim.forward()
    for _ in range(10):
        obs, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
    return obs

def rollout_one(env, pol, obs0, lang: str, seed: int, max_steps: int, stop_at: int | None = None):
    import torch
    from vla_branch_lab.latency.oft_policy import gripper_to_libero
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    ctrl = env.env.robots[0].controller
    ctrl.update()
    eef0 = ctrl.ee_pos.copy()
    obs = obs0
    step = 0
    succ = 0
    probes = {}
    probe_steps = (24, 48, 72, 96)
    saved_state = None

    while step < max_steps:
        with torch.inference_mode():
            actions, _ = pol._simple_actions(obs, lang)
        actions = gripper_to_libero(np.asarray(actions, dtype=np.float32).reshape(-1, 7))
        for j in range(len(actions)):
            if step >= max_steps:
                break
            obs, ok, _ = env.step(env.clip_action(actions[j]))
            step += 1
            if step in probe_steps:
                ctrl.update()
                probes[step] = (ctrl.ee_pos.copy() - eef0).tolist()
            if ok:
                succ = 1
                break
            if stop_at is not None and step == stop_at:
                saved_state = env.env.sim.get_state().flatten().copy()
                return succ, step, probes, saved_state, obs
        if succ:
            break
    ctrl.update()
    last_rel = (ctrl.ee_pos.copy() - eef0).tolist()
    fp = [probes.get(ps, last_rel) for ps in probe_steps]
    return succ, step, fp, saved_state, obs

def gpu_worker(gpu_id: int, work_q, k_samples: int, horizon: int, entrance_steps: int, out_file: Path):
    _set_data_env(gpu_id)
    import queue
    import torch
    from safetensors.torch import load_file
    from vla_branch_lab.latency.study_config import load_study
    from vla_branch_lab.latency.initial_states import environment_config
    from vla_branch_lab.latency.study_runtime import model_spec
    from vla_branch_lab.latency.oft_policy import OFTPolicy
    from vla_branch_lab.envs.libero import LiberoEnvironment

    cfg = load_study("configs/latency_next/simple_oft_fixed.yaml")
    task0 = cfg["tasks"][0]
    spec_rl = dict(model_spec(cfg, task0, "ript"))
    spec_rl["num_images_in_input"] = 1
    spec_rl["do_sample"] = True
    spec_rl["temperature"] = 1.2

    t_load = time.time()
    print(f"[Exp1 GPU{gpu_id}] Loading GRPO onto GPU and SFT shards onto CPU...", flush=True)
    pol = OFTPolicy(cfg, {**task0, "language": ""}, spec_rl)
    rl_cpu = {k: v.detach().cpu().to(torch.bfloat16) for k, v in pol.model.named_parameters()}
    sft_dir = ROOT / "checkpoints/oft/simple_sft"
    idx_map = json.loads((sft_dir / "model.safetensors.index.json").read_text())["weight_map"]
    sft_cpu = {}
    for sf in sorted(set(idx_map.values())):
        sd = load_file(str(sft_dir / sf), device="cpu")
        for k, v in sd.items():
            if k in rl_cpu:
                sft_cpu[k] = v.to(torch.bfloat16)
    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[Exp1 GPU{gpu_id}] Loaded {len(sft_cpu)} SFT tensors in {time.time()-t_load:.1f}s (VRAM={vram_gb:.2f} GB)", flush=True)

    results = []
    if out_file.exists():
        try:
            results = json.loads(out_file.read_text())
        except Exception:
            results = []
    cur_tid = None
    env = None
    lang = ""
    task = None
    try:
        while True:
            try:
                item = work_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                break
            tid, state_id = item
            if cur_tid != tid:
                if env is not None:
                    env.close()
                task = next(t for t in cfg["tasks"] if t["task_id"] == tid)
                env = LiberoEnvironment(environment_config(cfg, task), tid)
                lang = str(env.task.language)
                cur_tid = tid
            # 1. SFT base + save opening snapshots
            apply_layer_interp(pol.model, rl_cpu, sft_cpu, None, 1.0)
            sft_snaps = []
            for s_idx in range(k_samples):
                obs0 = reset_with_pose_jitter(env, task, state_id)
                seed = 90000 + tid * 1000 + state_id * 100 + s_idx
                succ1, st1, probes1, snap, obs_mid = rollout_one(env, pol, obs0, lang, seed, horizon, stop_at=entrance_steps)
                sft_snaps.append((succ1, st1, probes1, snap))
                if succ1 or st1 >= horizon:
                    fp = [probes1.get(ps, [0, 0, 0]) for ps in (24, 48, 72, 96)]
                    results.append({"task_id": tid, "state_id": state_id, "cond": "sft_base", "sample": s_idx, "success": succ1, "steps": st1, "fp": fp})
                else:
                    succ2, st2, fp2, _, _ = rollout_one(env, pol, obs_mid, lang, seed + 50000, horizon - st1)
                    fp = [probes1.get(24, fp2[0]), probes1.get(48, fp2[1]), fp2[2], fp2[3]]
                    results.append({"task_id": tid, "state_id": state_id, "cond": "sft_base", "sample": s_idx, "success": succ2, "steps": st1 + st2, "fp": fp})

            # 2. GRPO ctrl + entrance_handoff
            apply_layer_interp(pol.model, rl_cpu, sft_cpu, None, 0.0)
            for s_idx in range(k_samples):
                obs0 = reset_with_pose_jitter(env, task, state_id)
                seed = 10000 + tid * 1000 + state_id * 100 + s_idx
                succ, st, fp, _, _ = rollout_one(env, pol, obs0, lang, seed, horizon)
                results.append({"task_id": tid, "state_id": state_id, "cond": "grpo_ctrl", "sample": s_idx, "success": succ, "steps": st, "fp": fp})

            for s_idx in range(k_samples):
                succ1, st1, probes1, snap = sft_snaps[s_idx]
                if succ1 or snap is None or st1 >= horizon:
                    fp = [probes1.get(ps, [0, 0, 0]) for ps in (24, 48, 72, 96)]
                    results.append({"task_id": tid, "state_id": state_id, "cond": "entrance_handoff", "sample": s_idx, "success": succ1, "steps": st1, "fp": fp})
                else:
                    reset_with_pose_jitter(env, task, state_id)
                    env.env.sim.set_state_from_flattened(snap)
                    env.env.sim.forward()
                    obs_mid = env.env.env._get_observations(force_update=True)
                    seed = 40000 + tid * 1000 + state_id * 100 + s_idx
                    succ2, st2, fp2, _, _ = rollout_one(env, pol, obs_mid, lang, seed, horizon - st1)
                    fp = [probes1.get(24, fp2[0]), probes1.get(48, fp2[1]), fp2[2], fp2[3]]
                    results.append({"task_id": tid, "state_id": state_id, "cond": "entrance_handoff", "sample": s_idx, "success": succ2, "steps": st1 + st2, "fp": fp})

            # 3. Early L0-11 (alpha=0.50) & Mid-late L16-31 (alpha=0.50)
            for cond_name, lrange, s_off in [("early_L0_11", (0, 11), 20000), ("midlate_L16_31", (16, 31), 30000)]:
                apply_layer_interp(pol.model, rl_cpu, sft_cpu, lrange, 0.50)
                for s_idx in range(k_samples):
                    obs0 = reset_with_pose_jitter(env, task, state_id)
                    seed = s_off + tid * 1000 + state_id * 100 + s_idx
                    succ, st, fp, _, _ = rollout_one(env, pol, obs0, lang, seed, horizon)
                    results.append({"task_id": tid, "state_id": state_id, "cond": cond_name, "sample": s_idx, "success": succ, "steps": st, "fp": fp})

            cur_sr = {c: sum(r["success"] for r in results if r["task_id"] == tid and r["state_id"] == state_id and r["cond"] == c) for c in ["sft_base", "grpo_ctrl", "early_L0_11", "midlate_L16_31", "entrance_handoff"]}
            print(f"[Exp1 GPU{gpu_id}] Task {tid} State {state_id} done succ/{k_samples}={cur_sr} (VRAM={torch.cuda.max_memory_allocated()/1e9:.2f} GB)", flush=True)
            out_file.write_text(json.dumps(results, indent=2))
    finally:
        if env is not None:
            env.close()

def summarize_multiscale(all_rows: list[dict]) -> dict:
    conds = ["sft_base", "grpo_ctrl", "early_L0_11", "midlate_L16_31", "entrance_handoff"]
    pairs = sorted({(r["task_id"], r["state_id"]) for r in all_rows})
    summary = {}
    for c in conds:
        p1, p4, p8 = [], [], []
        covs = {15: [], 25: [], 35: []}
        for tid, sid in pairs:
            sub = [r for r in all_rows if r["task_id"] == tid and r["state_id"] == sid and r["cond"] == c]
            n = len(sub)
            k_succ = sum(r["success"] for r in sub)
            p1.append(unbiased_pass_at_k(n, k_succ, 1))
            p4.append(unbiased_pass_at_k(n, k_succ, 4))
            p8.append(unbiased_pass_at_k(n, k_succ, 8))
            succ_bank = [np.asarray(r["fp"], dtype=np.float64) for r in all_rows if r["task_id"] == tid and r["state_id"] == sid and r["success"] == 1]
            for mm in (15, 25, 35):
                thr = mm / 1000.0
                centers = []
                for fp in succ_bank:
                    if not any(float(np.sqrt(np.mean((fp - ctr) ** 2))) < thr for ctr in centers):
                        centers.append(fp)
                if not centers:
                    covs[mm].append(0.0)
                else:
                    hit = set()
                    for r in sub:
                        if r["success"] == 1:
                            fp = np.asarray(r["fp"], dtype=np.float64)
                            dists = [float(np.sqrt(np.mean((fp - ctr) ** 2))) for ctr in centers]
                            hit.add(int(np.argmin(dists)))
                    covs[mm].append(len(hit) / len(centers))
        summary[c] = {
            "pass1": round(float(np.mean(p1)), 4),
            "pass4": round(float(np.mean(p4)), 4),
            "pass8": round(float(np.mean(p8)), 4),
            "cov_15mm": round(float(np.mean(covs[15])), 4),
            "cov_25mm": round(float(np.mean(covs[25])), 4),
            "cov_35mm": round(float(np.mean(covs[35])), 4),
        }
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--states-per-task", type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()
    if args.smoke:
        out_f = OUT_DIR / "smoke_gpu0.json"
        if out_f.exists():
            out_f.unlink()
        ctx = mp.get_context("spawn")
        wq = ctx.Queue()
        wq.put((0, 0))
        gpu_worker(0, wq, k_samples=2, horizon=380, entrance_steps=48, out_file=out_f)
        rows = json.loads(out_f.read_text())
        summ = summarize_multiscale(rows)
        (OUT_DIR / "smoke_summary.json").write_text(json.dumps(summ, indent=2))
        print("[Exp1 SMOKE OK]", json.dumps(summ, indent=2), f"elapsed={time.time()-t0:.1f}s", flush=True)
        return

    gpus = [int(x) for x in args.gpus.split(",")]
    ctx = mp.get_context("spawn")
    wq = ctx.Queue()
    done_pairs = set()
    for idx in range(len(gpus)):
        f = OUT_DIR / f"gpu{idx}_v14.json"
        if f.exists():
            for r in json.loads(f.read_text()):
                done_pairs.add((r["task_id"], r["state_id"]))
    for tid in range(10):
        for sid in range(args.states_per_task):
            if (tid, sid) not in done_pairs:
                wq.put((tid, sid))
    procs = []
    for idx, g in enumerate(gpus):
        p = ctx.Process(target=gpu_worker, args=(g, wq, 6, 520, 48, OUT_DIR / f"gpu{idx}_v14.json"))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Exp1 worker failed with exitcode {p.exitcode}")

    all_rows = []
    for idx in range(len(gpus)):
        f = OUT_DIR / f"gpu{idx}_v14.json"
        if f.exists():
            all_rows.extend(json.loads(f.read_text()))
    summary = summarize_multiscale(all_rows)
    (OUT_DIR / "summary.json").write_text(json.dumps({"summary": summary, "n_rows": len(all_rows), "elapsed_s": round(time.time()-t0, 1)}, indent=2))
    print("[Exp1 FULL DONE]", json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
