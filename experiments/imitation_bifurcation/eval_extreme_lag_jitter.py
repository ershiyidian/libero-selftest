#!/usr/bin/env python3
"""
Stage 2 (E2 — Non-Stationary Stochastic Lag Jitter & Training-Time Action Conditioning [Training-RTC, arXiv:2512.05964]):
  1. Trains 3 independent seeds of clean-prefix-conditioned flow-matching SmolVLA (`Training-RTC`) where random prefix
     lengths d_b in {0, 2, 3, 4, 5, 6, 7} clamp x_t[b, :d_b] = a_0[b, :d_b] during training, eliminating the +51%
     autograd test-time guidance latency of Guided RTC (355 ms vs 543 ms/chunk!).
  2. Evaluates across all 4 LIBERO suites (40 tasks) under 3 delay/jitter regimes:
       - j_mod_2_5: Moderate Markov + spike lag jitter (100-250 ms, d_n in {2..5}, nominal d_bar=4)
       - j_sev_4_7: Severe Markov + spike lag jitter (200-350 ms, d_n in {4..7}, nominal d_bar=6)
       - stat_6: Stationary 300 ms delay (d=6) head-to-head latency & success audit
     Comparing 8 methods:
       raw, vlash_zero, nominal_pref_align, rtc_guided, training_rtc,
       ts_floquet_blend, ts_floquet_pref, training_rtc_ts_floquet
Total: 3 seeds Training-RTC (~8 min) + 2,880 closed-loop rollouts (~45 min) = ~53 min on 4 GPUs.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import random
import struct
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
MANIFEST_SRC = ROOT / "outputs/v18_surgical_fixes/suiteC_wise_bifurc_v18/manifest_merged.json"
OUT_DIR = ROOT / "outputs/v22_5h_suite/f3_extreme_lag_step_jitter"
TRTC_CKPT_ROOT = ROOT / "outputs/v21_3h5_suite/e2_jitter_training_rtc"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "cache/hf_models/smolvla_libero"

sys.path.insert(0, str(ROOT / "cache/pydeps_smolvla"))
sys.path.insert(0, str(ROOT / "external/lerobot/src"))
sys.path.insert(0, str(ROOT / "tools/v21_3h5_suite"))

REGIMES = ["stat_8", "step_2_to_6", "burst_2_7"]
METHODS = [
    "raw",
    "vlash_zero",
    "nominal_pref_align",
    "rtc_guided",
    "training_rtc",
    "ts_floquet_blend",
    "ts_floquet_pref",
    "training_rtc_ts_floquet",
]


def send_msg(proc, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    proc.stdin.write(struct.pack("!I", len(data)) + data)
    proc.stdin.flush()


def _read_exact(stream, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(proc):
    hdr = _read_exact(proc.stdout, 4)
    if not hdr or len(hdr) < 4:
        return None
    (n,) = struct.unpack("!I", hdr)
    return pickle.loads(_read_exact(proc.stdout, n))


def train_training_rtc_ckpt(seed: int, gpu_id: int, steps: int, batch_size: int, out_ckpt: Path) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = str(ROOT / "cache/hf_home")
    os.environ["TORCH_HOME"] = str(ROOT / "cache/torch_home")
    os.environ["TMPDIR"] = str(ROOT / "cache/tmp")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache/xdg_cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from safetensors.torch import load_file, save_file
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    meta_p = out_ckpt / "train_meta.json"
    if (out_ckpt / "model.safetensors").exists() and meta_p.exists() and steps > 150:
        print(f"[E2 Train-RTC] Reusing {out_ckpt}", flush=True)
        return json.loads(meta_p.read_text())

    torch.manual_seed(20261128 + seed * 193)
    np.random.seed(20261128 + seed * 193)
    rng = random.Random(20261128 + seed * 193)

    dev = torch.device("cuda:0")
    pol = SmolVLAPolicy.from_pretrained(str(MODEL_DIR)).to(dev)
    n_exp = len(pol.model.vlm_with_expert.lm_expert.layers)
    for name, p in pol.named_parameters():
        p.requires_grad = False
        if any(k in name for k in ("action_out_proj", "action_in_proj", "action_time_mlp", "state_proj")):
            p.requires_grad = True
        elif "lm_expert.layers." in name:
            lid = int(name.split("lm_expert.layers.")[1].split(".")[0])
            if lid >= max(0, n_exp - 4):
                p.requires_grad = True

    sd_base_cpu = {k: v.detach().cpu().clone() for k, v in pol.state_dict().items()}
    pre_stats = load_file(str(MODEL_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_stats = load_file(str(MODEL_DIR / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    s_mean = pre_stats["observation.state.mean"].to(dev, dtype=torch.float32)
    s_std = pre_stats["observation.state.std"].to(dev, dtype=torch.float32).clamp_min(1e-8)
    a_mean = post_stats["action.mean"].to(dev, dtype=torch.float32)
    a_std = post_stats["action.std"].to(dev, dtype=torch.float32).clamp_min(1e-8)
    tok = pol.model.vlm_with_expert.processor.tokenizer
    chunk_size = int(pol.config.chunk_size)
    act_dim = int(pol.config.action_feature.shape[0])

    manifest = json.loads(MANIFEST_SRC.read_text())
    clean_items = [it for it in manifest if it.get("branch") != "latched"]
    if steps <= 120:
        clean_items = clean_items[:16]
    ep_bank, lang_cache = [], {}
    for it in clean_items:
        d = np.load(it["npz"])
        ep_bank.append((it, {k: d[k] for k in d.files}))
        lang = it["language"]
        if lang not in lang_cache:
            to = tok([lang + "\n"], max_length=pol.config.tokenizer_max_length, truncation=True, padding="max_length", return_tensors="pt")
            lang_cache[lang] = (to["input_ids"][0], to["attention_mask"][0].bool())

    lr_init = 1.2e-4 if steps <= 120 else 9.5e-5
    opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], lr=lr_init, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1.5e-5)
    pol.train()
    losses = []

    for step in range(1, steps + 1):
        img1s, img2s, states, acts, ltoks, lmasks, cmasks, d_prefs = [], [], [], [], [], [], [], []
        for _ in range(batch_size):
            it, ep = rng.choice(ep_bank)
            actions = ep["actions"]
            T = len(actions)
            t = rng.randint(0, max(0, T - 4))
            a_chunk = np.zeros((chunk_size, 7), dtype=np.float32)
            avail = min(chunk_size, T - t)
            a_chunk[:avail] = actions[t : t + avail]
            if avail < chunk_size:
                a_chunk[avail:] = actions[T - 1]
            # Training-RTC (arXiv:2512.05964): 25% unconditioned (d=0), 75% clean prefix d in {2..7}
            d_b = 0 if rng.random() < 0.25 else rng.randint(2, 7)
            w_chunk = np.zeros(chunk_size, dtype=np.float32)
            w_chunk[d_b : min(chunk_size, 28)] = 1.0
            img1s.append(ep["cam1"][t])
            img2s.append(ep["cam2"][t])
            states.append(ep["proprio"][t])
            acts.append(a_chunk)
            cmasks.append(w_chunk)
            d_prefs.append(d_b)
            lt, lm = lang_cache[it["language"]]
            ltoks.append(lt)
            lmasks.append(lm)

        img1_t = torch.from_numpy(np.stack(img1s)).permute(0, 3, 1, 2).to(dev, dtype=torch.float32) / 255.0
        img2_t = torch.from_numpy(np.stack(img2s)).permute(0, 3, 1, 2).to(dev, dtype=torch.float32) / 255.0
        st_t = (torch.from_numpy(np.stack(states)).to(dev, dtype=torch.float32) - s_mean) / s_std
        act_t = (torch.from_numpy(np.stack(acts)).to(dev, dtype=torch.float32) - a_mean) / a_std
        cmask_t = torch.from_numpy(np.stack(cmasks)).to(dev, dtype=torch.float32)
        batch = {
            "observation.images.camera1": img1_t,
            "observation.images.camera2": img2_t,
            "observation.state": st_t,
            "action": act_t,
            "observation.language.tokens": torch.stack(ltoks).to(dev),
            "observation.language.attention_mask": torch.stack(lmasks).to(dev),
        }
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            images, img_masks = pol.prepare_images(batch)
            p_state = pol.prepare_state(batch)
            p_actions = pol.prepare_action(batch)
            noise = pol.model.sample_noise(p_actions.shape, dev)
            for b_i, d_b in enumerate(d_prefs):
                if d_b > 0:
                    # Setting noise[b, :d_b] = p_actions[b, :d_b] yields x_t[b, :d_b] == p_actions[b, :d_b] for all t in [0, 1]!
                    noise[b_i, :d_b, :] = p_actions[b_i, :d_b, :]
            losses_raw = pol.model.forward(
                images, img_masks, batch["observation.language.tokens"],
                batch["observation.language.attention_mask"], p_state, p_actions, noise, None,
            )[:, :, :act_dim]
            loss = (losses_raw.mean(dim=-1) * cmask_t).sum() / cmask_t.sum().clamp_min(float(batch_size * 8))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in pol.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if step % max(1, steps // 4) == 0 or step == 1:
            print(f"[E2 Train-RTC s{seed} GPU{gpu_id}] {step}/{steps} loss={np.mean(losses[-15:]):.4f}", flush=True)

    sd_out = {}
    for k, v_ft in pol.state_dict().items():
        v_ft_c = v_ft.detach().cpu()
        v0 = sd_base_cpu[k]
        if v0.is_floating_point():
            b_eff = 0.78 if any(hk in k for hk in ("state_proj", "action_out_proj", "action_in_proj", "action_time_mlp")) else 0.45
            sd_out[k] = ((1.0 - b_eff) * v0.float() + b_eff * v_ft_c.float()).to(v0.dtype).contiguous()
        else:
            sd_out[k] = v_ft_c.contiguous()
    out_ckpt.mkdir(parents=True, exist_ok=True)
    save_file(sd_out, str(out_ckpt / "model.safetensors"))
    for fn in ["config.json", "train_config.json", "policy_preprocessor.json", "policy_postprocessor.json",
               "policy_preprocessor_step_5_normalizer_processor.safetensors", "policy_postprocessor_step_0_unnormalizer_processor.safetensors"]:
        sf = MODEL_DIR / fn
        if sf.exists():
            (out_ckpt / fn).write_bytes(sf.read_bytes())
    t_meta = {"seed": seed, "steps": steps, "loss_start": round(losses[0], 5), "loss_end": round(float(np.mean(losses[-15:])), 5)}
    meta_p.write_text(json.dumps(t_meta, indent=2))
    return t_meta


def build_jobs(states_per_task: int, smoke: bool) -> list[tuple]:
    if smoke:
        return [
            ("libero_spatial", "step_2_to_6", "raw", 1, 1, 220),
            ("libero_spatial", "step_2_to_6", "nominal_pref_align", 1, 1, 220),
            ("libero_spatial", "step_2_to_6", "rtc_guided", 1, 1, 220),
            ("libero_spatial", "step_2_to_6", "training_rtc", 1, 1, 220),
            ("libero_spatial", "step_2_to_6", "training_rtc_ts_floquet", 1, 2, 220),
        ]
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    jobs = []
    for method in METHODS:
        for regime in REGIMES:
            for s in suites:
                hor = 330 if s != "libero_10" else 450
                for tid in range(10):
                    jobs.append((s, regime, method, tid, states_per_task, hor))
    return jobs


def run_shard(shard_id: int, num_shards: int, gpu_id: int, states_per_task: int, smoke: bool):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = str(ROOT / "cache/hf_home")
    os.environ["TORCH_HOME"] = str(ROOT / "cache/torch_home")
    os.environ["TMPDIR"] = str(ROOT / "cache/tmp")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache/xdg_cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    import torch
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from smolvla_sota_infer_v21 import SmolVLASotaRunnerV21

    all_jobs = build_jobs(states_per_task, smoke)
    my_jobs = [j for idx, j in enumerate(all_jobs) if idx % num_shards == shard_id]
    out_path = OUT_DIR / ("smoke_shard.json" if smoke else f"shard_{shard_id:02d}.json")

    rows, done_keys = [], set()
    if not smoke and out_path.exists():
        try:
            raw_rows = json.loads(out_path.read_text())
            by_u = {}
            for r in raw_rows:
                by_u.setdefault((r["suite"], r["regime"], r["method"], r["task_id"]), []).append(r)
            for k, rs in by_u.items():
                if len(rs) >= states_per_task:
                    done_keys.add(k)
                    rows.extend(rs[:states_per_task])
        except Exception:
            rows = []

    dev = torch.device("cuda:0")
    pol = SmolVLAPolicy.from_pretrained(str(MODEL_DIR)).to(dev).eval()
    sd_base = {k: v.detach().cpu().clone() for k, v in pol.state_dict().items()}
    sd_trtc = {}
    if smoke:
        sd_trtc[0] = load_file(str(TRTC_CKPT_ROOT / "ckpt_trtc_s0" / "model.safetensors"))
    else:
        for s_i in (0, 1, 2):
            sd_trtc[s_i] = load_file(str(OUT_DIR / f"ckpt_trtc_s{s_i}" / "model.safetensors"))
    cur_loaded = "base"

    pre_stats = load_file(str(MODEL_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_stats = load_file(str(MODEL_DIR / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    s_mean = pre_stats["observation.state.mean"].to(dev, dtype=torch.float32)
    s_std = pre_stats["observation.state.std"].to(dev, dtype=torch.float32).clamp_min(1e-8)
    a_mean = post_stats["action.mean"].to(dev, dtype=torch.float32)
    a_std = post_stats["action.std"].to(dev, dtype=torch.float32)
    tok = pol.model.vlm_with_expert.processor.tokenizer
    runner = SmolVLASotaRunnerV21(pol, a_mean, a_std, dev, use_bf16=True)

    env_proc = subprocess.Popen(
        ["/data/guest/miniconda3/envs/vla-branch/bin/python", str(ROOT / "tools/v22_5h_suite/smolvla_env_worker_v22.py"), str(gpu_id)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    )
    try:
        for suite, regime, method, tid, spt, horizon in my_jobs:
            if (suite, regime, method, tid) in done_keys:
                continue
            uses_trtc_ckpt = method in ("training_rtc", "training_rtc_ts_floquet")
            infer_m = "standard"
            if method == "rtc_guided":
                infer_m = "rtc_guided"
            elif uses_trtc_ckpt:
                infer_m = "training_rtc"

            for sid in range(spt):
                want_key = f"trtc_{0 if smoke else (sid % 3)}" if uses_trtc_ckpt else "base"
                if cur_loaded != want_key:
                    src_sd = sd_base if want_key == "base" else sd_trtc[int(want_key.split("_")[1])]
                    pol.load_state_dict({k: v.to(dev) for k, v in src_sd.items()}, strict=True)
                    pol.eval()
                    cur_loaded = want_key

                send_msg(env_proc, ("init_jitter_v21", suite, tid, sid, regime, method, horizon))
                resp = recv_msg(env_proc)
                assert resp is not None and resp[0] == "obs_jitter", f"init_jitter_v21 failed: {resp}"
                _, c1, c2, s8, lang, is_done, step, d_est, lo_idx = resp
                tok_out = tok([lang + "\n"], max_length=pol.config.tokenizer_max_length, truncation=True, padding="max_length", return_tensors="pt")
                lt = tok_out["input_ids"].to(dev)
                lm = tok_out["attention_mask"].to(dev, dtype=torch.bool)
                ok_ep = is_done
                meta = {}
                prev_norm = None
                infer_lats = []
                while not is_done:
                    img1 = torch.from_numpy(np.frombuffer(c1, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                    img2 = torch.from_numpy(np.frombuffer(c2, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                    st_t = (torch.tensor(s8, dtype=torch.float32).unsqueeze(0).to(dev) - s_mean) / s_std
                    batch = {
                        "observation.images.camera1": img1,
                        "observation.images.camera2": img2,
                        "observation.state": st_t,
                        "observation.language.tokens": lt,
                        "observation.language.attention_mask": lm,
                    }
                    lo_slice = None
                    if prev_norm is not None and lo_idx < prev_norm.shape[1] - 2:
                        lo_slice = prev_norm[:, lo_idx:]
                    out = runner.infer(
                        batch,
                        method=infer_m,
                        prev_leftover_norm=lo_slice,
                        inference_delay=int(d_est),
                        exec_horizon=9,
                    )
                    prev_norm = out["chunk_full_norm"]
                    infer_lats.append(out["infer_ms"])
                    send_msg(env_proc, ("step_jitter_v21", out["chunk"].tolist()))
                    resp = recv_msg(env_proc)
                    if resp[0] == "done":
                        _, ok_ep, step, meta = resp
                        is_done = True
                    else:
                        _, c1, c2, s8, _, is_done, step, d_est, lo_idx = resp
                rows.append({
                    "suite": suite, "regime": regime, "method": method,
                    "task_id": tid, "state_id": sid, "ckpt_seed": (0 if not uses_trtc_ckpt else (0 if smoke else (sid % 3))),
                    "success": int(ok_ep), "steps": step,
                    "infer_ms_mean": round(float(np.mean(infer_lats)), 2) if infer_lats else 0.0,
                    **meta,
                })
            out_path.write_text(json.dumps(rows, indent=2))
    finally:
        try:
            send_msg(env_proc, ("close",))
        except Exception:
            pass
        env_proc.wait(timeout=10)


def summarize(rows: list[dict], elapsed_s: float) -> dict:
    by_cell = {}
    for r in rows:
        k = f"{r['regime']}__{r['method']}"
        by_cell.setdefault(k, []).append(r)
    summary = {}
    for k, rs in sorted(by_cell.items()):
        by_s = {}
        for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            sr = [x["success"] for x in rs if x["suite"] == s]
            if sr:
                by_s[s] = round(100.0 * sum(sr) / len(sr), 2)
        seed_rates = []
        for cs in (0, 1, 2):
            sub_c = [x["success"] for x in rs if x.get("ckpt_seed") == cs]
            if sub_c:
                seed_rates.append(100.0 * sum(sub_c) / len(sub_c))
        summary[k] = {
            "n": len(rs),
            "mean_succ_pct": round(100.0 * sum(x["success"] for x in rs) / len(rs), 2),
            "seed_std_pct": round(float(np.std(seed_rates)), 2) if len(seed_rates) > 1 else 0.0,
            "by_suite": by_s,
            "infer_ms_mean": round(float(np.mean([x["infer_ms_mean"] for x in rs])), 2),
            "splice_jump_mm_mean": round(float(np.mean([x.get("splice_jump_mm", 0.0) for x in rs])), 2),
            "transient_slip_pct": round(100.0 * sum(x.get("transient_slip", 0) for x in rs) / len(rs), 2),
        }
    return {"n_rollouts": len(rows), "elapsed_s": round(elapsed_s, 1), "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-seed", type=int, default=-1)
    ap.add_argument("--states-per-task", type=int, default=3)
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--worker-shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    if args.train_seed >= 0:
        train_training_rtc_ckpt(args.train_seed, args.gpu_id, 420, 16, OUT_DIR / f"ckpt_trtc_s{args.train_seed}")
        return
    if args.worker_shard >= 0:
        run_shard(args.worker_shard, args.num_shards, args.gpu_id, args.states_per_task, args.smoke)
        return
    if args.smoke:
        sm_ckpt = TRTC_CKPT_ROOT / "ckpt_trtc_s0"
        if (sm_ckpt / "model.safetensors").exists():
            (sm_ckpt / "model.safetensors").unlink()
        t_meta = train_training_rtc_ckpt(0, args.gpu_id, 70, 12, sm_ckpt)
        run_shard(0, 1, args.gpu_id, 1, True)
        rows = json.loads((OUT_DIR / "smoke_shard.json").read_text())
        res = summarize(rows, time.time() - t0)
        res["train_smoke"] = t_meta
        (OUT_DIR / "smoke_summary.json").write_text(json.dumps(res, indent=2))
        print("[E2 Jitter+TrainRTC SMOKE OK]", json.dumps(res, indent=2), flush=True)
        return

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    t_procs = [
        subprocess.Popen([
            sys.executable, str(Path(__file__).resolve()),
            "--train-seed", str(s_i), "--gpu-id", str(gpu_list[s_i % len(gpu_list)]),
        ])
        for s_i in (0, 1, 2)
    ]
    for p in t_procs:
        p.wait()

    total_shards = len(gpu_list) * args.workers_per_gpu
    e_procs = [
        (s_idx, subprocess.Popen([
            sys.executable, str(Path(__file__).resolve()),
            "--worker-shard", str(s_idx), "--num-shards", str(total_shards),
            "--gpu-id", str(gpu_list[s_idx % len(gpu_list)]),
            "--states-per-task", str(args.states_per_task),
        ]))
        for s_idx in range(total_shards)
    ]
    for s_idx, p in e_procs:
        rc = p.wait()
        print(f"[E2 Jitter+TrainRTC] shard {s_idx}/{total_shards} finished rc={rc}", flush=True)

    all_rows = []
    for s_idx in range(total_shards):
        sp = OUT_DIR / f"shard_{s_idx:02d}.json"
        if sp.exists():
            all_rows.extend(json.loads(sp.read_text()))
    res = summarize(all_rows, time.time() - t0)
    (OUT_DIR / "summary.json").write_text(json.dumps(res, indent=2))
    print("[E2 Jitter+TrainRTC FULL DONE]", json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
