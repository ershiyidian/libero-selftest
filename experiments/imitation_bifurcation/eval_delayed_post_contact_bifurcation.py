#!/usr/bin/env python3
"""
Stage 2 (G2 — Cross-Boundary Coupling: Boundary 1 [Async Lag/Burst] x Boundary 3 [Post-Contact Latch Bifurcation]):
  Tests whether causal-masked inverted-subspace WiSE-FT SmolVLA-450M preserves nominal retention AND
  recovers latched post-contact bifurcations UNDER 200ms, 300ms, and Bimodal Burst (100/350ms) latency,
  comparing Naive Async vs. Prefix-Align vs. Unified Timestamped Floquet + Prefix paired with PACE & Hybrid Event Replan.
  Total: ~6,288 closed-loop rollouts across 16 workers on 4x RTX 3090 (~2.7 hours).
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import struct
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
CKPT_V18_DIR = ROOT / "outputs/v18_surgical_fixes/suiteC_wise_bifurc_v18"
OUT_DIR = ROOT / "outputs/v23_16h_suite/g2_delayed_post_contact_bifurc"
MODEL_DIR = ROOT / "cache/hf_models/smolvla_libero"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "cache/pydeps_smolvla"))
sys.path.insert(0, str(ROOT / "external/lerobot/src"))
sys.path.insert(0, str(ROOT / "tools/v23_16h_suite"))

BIFURC_TASKS = [
    ("libero_spatial", 0), ("libero_spatial", 1), ("libero_spatial", 2), ("libero_spatial", 6),
    ("libero_object", 0), ("libero_object", 1), ("libero_object", 2),
    ("libero_goal", 1), ("libero_goal", 3), ("libero_goal", 8),
    ("libero_10", 1), ("libero_10", 3),
]
NOM_SUBSET = [
    (s, t) for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10") for t in range(0, 10, 2)
]  # 20 representative tasks across all 4 suites
HEAD_KEYS = ("state_proj", "action_out_proj", "action_in_proj", "action_time_mlp")


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


def build_interpolated_sd(sd_base: dict, sd_ckpt_s: dict, b_head: float, b_exp: float, dev) -> dict:
    out = {}
    for k, v0 in sd_base.items():
        if not v0.is_floating_point() or (b_head == 0.0 and b_exp == 0.0):
            out[k] = v0.to(dev)
            continue
        is_h = any(hk in k for hk in HEAD_KEYS)
        b_saved = 0.92 if is_h else 0.55
        b_new = b_head if is_h else b_exp
        v0_f = v0.float()
        vc_f = sd_ckpt_s[k].float()
        v_ft = (vc_f - (1.0 - b_saved) * v0_f) / b_saved
        v_mix = ((1.0 - b_new) * v0_f + b_new * v_ft).to(v0.dtype)
        out[k] = v_mix.to(dev)
    return out


def build_jobs(smoke: bool):
    # Each job: (split, model_tag, seed, b_head, b_exp, regime, async_mode, sched_mode, suite, tid, is_latched, n_states, hor)
    jobs = []
    if smoke:
        for reg, am, sm, bh, be, tag in [
            ("stat_4", "raw", "pace", 0.50, 0.85, "inv_h050_e085"),
            ("stat_4", "ts_floquet_pref", "pace", 0.50, 0.85, "inv_h050_e085"),
            ("burst_2_7", "ts_floquet_pref", "hybrid_pace_event", 0.50, 0.85, "inv_h050_e085"),
            ("stat_4", "ts_floquet_pref", "pace", 0.0, 0.0, "base"),
        ]:
            for lat in (0, 1):
                jobs.append(("bifurc", tag, 0, bh, be, reg, am, sm, "libero_spatial", 0, lat, 1, 280))
        return jobs

    # 1. Base reference across 4 regimes (0ms, 200ms, 300ms, burst)
    for reg in ("stat_0", "stat_4", "stat_6", "burst_2_7"):
        for am in (("raw",) if reg == "stat_0" else ("raw", "ts_floquet_pref")):
            for suite, tid in BIFURC_TASKS:
                hor = 340 if suite != "libero_10" else 440
                for lat in (0, 1):
                    jobs.append(("bifurc", "base", 0, 0.0, 0.0, reg, am, "pace", suite, tid, lat, 2, hor))

    # 2. 3-seed Uniform(0.75) vs Inverted Subspace(0.50, 0.85) under 200ms, 300ms, Burst
    for seed in (0, 1, 2):
        for tag, bh, be in [("uniform_075", 0.75, 0.75), ("inv_h050_e085", 0.50, 0.85)]:
            for reg in ("stat_4", "stat_6", "burst_2_7"):
                for am in ("raw", "pref_align", "ts_floquet_pref"):
                    for sm in ("pace", "hybrid_pace_event"):
                        for suite, tid in BIFURC_TASKS:
                            hor = 340 if suite != "libero_10" else 440
                            for lat in (0, 1):
                                jobs.append(("bifurc", tag, seed, bh, be, reg, am, sm, suite, tid, lat, 2, hor))
            # Nominal-20 retention under delay (raw vs ts_floquet_pref)
            for reg in ("stat_4", "stat_6", "burst_2_7"):
                for am in ("raw", "ts_floquet_pref"):
                    for suite, tid in NOM_SUBSET:
                        hor = 330 if suite != "libero_10" else 440
                        jobs.append(("nom20", tag, seed, bh, be, reg, am, "pace", suite, tid, 0, 1, hor))
    return jobs


def run_shard(shard_id: int, num_shards: int, gpu_id: int, smoke: bool):
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
    from smolvla_sota_infer_v23 import SmolVLASotaRunnerV23

    all_jobs = build_jobs(smoke)
    my_jobs = [j for idx, j in enumerate(all_jobs) if idx % num_shards == shard_id]
    out_path = OUT_DIR / ("smoke_shard.json" if smoke else f"shard_{shard_id:02d}.json")

    rows, done_keys = [], set()
    if not smoke and out_path.exists():
        try:
            raw_rows = json.loads(out_path.read_text())
            by_u = {}
            for r in raw_rows:
                k = (r["split"], r["model_tag"], r["seed"], r["regime"], r["async_mode"], r["sched_mode"], r["suite"], r["task_id"], r["is_latched"])
                by_u.setdefault(k, []).append(r)
            for k, rs in by_u.items():
                done_keys.add(k)
                rows.extend(rs)
        except Exception:
            rows = []

    dev = torch.device("cuda:0")
    pol = SmolVLAPolicy.from_pretrained(str(MODEL_DIR)).to(dev).eval()
    sd_base = {k: v.detach().cpu().clone() for k, v in pol.state_dict().items()}
    sd_ckpts = {s: load_file(str(CKPT_V18_DIR / f"ckpt_causal_s{s}" / "model.safetensors")) for s in (0, 1, 2)}
    cur_loaded = ("base", 0, 0.0, 0.0)

    pre_stats = load_file(str(MODEL_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_stats = load_file(str(MODEL_DIR / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    s_mean = pre_stats["observation.state.mean"].to(dev, dtype=torch.float32)
    s_std = pre_stats["observation.state.std"].to(dev, dtype=torch.float32).clamp_min(1e-8)
    a_mean = post_stats["action.mean"].to(dev, dtype=torch.float32)
    a_std = post_stats["action.std"].to(dev, dtype=torch.float32)
    tok = pol.model.vlm_with_expert.processor.tokenizer
    runner = SmolVLASotaRunnerV23(pol, a_mean, a_std, dev, use_bf16=True)

    env_proc = subprocess.Popen(
        ["/data/guest/miniconda3/envs/vla-branch/bin/python", str(ROOT / "tools/v23_16h_suite/smolvla_env_worker_v23.py"), str(gpu_id)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    )
    try:
        for split, tag, seed, bh, be, reg, am, sm, suite, tid, lat, n_st, hor in my_jobs:
            key = (split, tag, seed, reg, am, sm, suite, tid, lat)
            if key in done_keys:
                continue
            want_m = (tag, seed, bh, be)
            if cur_loaded != want_m:
                sd_new = build_interpolated_sd(sd_base, sd_ckpts[seed], bh, be, dev)
                pol.load_state_dict(sd_new, strict=True)
                pol.eval()
                cur_loaded = want_m

            for sid in range(n_st):
                send_msg(env_proc, ("init_delayed_bifurc_v23", suite, tid, sid, bool(lat), reg, am, sm, hor))
                resp = recv_msg(env_proc)
                assert resp is not None and resp[0] == "obs", f"init_delayed_bifurc_v23 failed: {resp}"
                _, c1, c2, s8, lang, is_done, step = resp
                tok_out = tok([lang + "\n"], max_length=pol.config.tokenizer_max_length, truncation=True, padding="max_length", return_tensors="pt")
                lt = tok_out["input_ids"].to(dev)
                lm = tok_out["attention_mask"].to(dev, dtype=torch.bool)
                ok_ep, meta = is_done, {}
                while not is_done:
                    img1 = torch.from_numpy(np.frombuffer(c1, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                    img2 = torch.from_numpy(np.frombuffer(c2, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                    st_t = (torch.tensor(s8, dtype=torch.float32).unsqueeze(0).to(dev) - s_mean) / s_std
                    batch = {
                        "observation.images.camera1": img1, "observation.images.camera2": img2,
                        "observation.state": st_t, "observation.language.tokens": lt, "observation.language.attention_mask": lm,
                    }
                    out = runner.infer_v23(batch, method="standard", exec_horizon=24)
                    send_msg(env_proc, ("step_delayed_bifurc_v23", out["chunk"].tolist(), None))
                    resp = recv_msg(env_proc)
                    assert resp is not None, "worker EOF"
                    if resp[0] == "done":
                        _, ok_ep, step, meta = resp
                        is_done = True
                    else:
                        _, c1, c2, s8, _, is_done, step = resp
                rows.append({
                    "split": split, "model_tag": tag, "seed": seed, "b_head": bh, "b_exp": be,
                    "regime": reg, "async_mode": am, "sched_mode": sm,
                    "suite": suite, "task_id": tid, "state_id": sid, "is_latched": int(lat),
                    "success": int(ok_ep), "steps": int(step), **meta,
                })
            out_path.write_text(json.dumps(rows, indent=1))
            if len(rows) % 24 == 0 or smoke:
                print(f"[G2 s{shard_id:02d} GPU{gpu_id}] {len(rows)} rows | {tag}/{reg}/{am}/{sm} lat={lat} succ={rows[-1]['success']}", flush=True)
    finally:
        try:
            send_msg(env_proc, ("close",))
        except Exception:
            pass
        env_proc.wait(timeout=15)


def aggregate(smoke: bool):
    files = [OUT_DIR / "smoke_shard.json"] if smoke else sorted(OUT_DIR.glob("shard_*.json"))
    all_rows = []
    for f in files:
        if f.exists():
            all_rows.extend(json.loads(f.read_text()))
    summary = {}
    groups = sorted({(r["split"], r["model_tag"], r["regime"], r["async_mode"], r["sched_mode"]) for r in all_rows})
    for g in groups:
        sp, tag, reg, am, sm = g
        rs = [r for r in all_rows if (r["split"], r["model_tag"], r["regime"], r["async_mode"], r["sched_mode"]) == g]
        key = f"{sp}|{tag}|{reg}|{am}|{sm}"
        if sp == "bifurc":
            rn = [r for r in rs if r["is_latched"] == 0]
            rl = [r for r in rs if r["is_latched"] == 1]
            summary[key] = {
                "n": len(rs),
                "norm_pct": round(100.0 * sum(r["success"] for r in rn) / max(1, len(rn)), 2),
                "latched_pct": round(100.0 * sum(r["success"] for r in rl) / max(1, len(rl)), 2),
                "all_pct": round(100.0 * sum(r["success"] for r in rs) / max(1, len(rs)), 2),
            }
        else:
            summary[key] = {
                "n": len(rs),
                "nom20_pct": round(100.0 * sum(r["success"] for r in rs) / max(1, len(rs)), 2),
            }
    out_p = OUT_DIR / ("smoke_summary.json" if smoke else "summary.json")
    out_p.write_text(json.dumps(summary, indent=2))
    print(f"[G2] Aggregated {len(all_rows)} rows -> {out_p}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker-shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    if args.worker_shard >= 0:
        run_shard(args.worker_shard, args.num_shards, args.gpu_id, args.smoke)
        return
    if args.smoke:
        g0 = int(args.gpus.split(",")[0])
        run_shard(0, 1, g0, True)
        aggregate(True)
        return
    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    num_shards = len(gpu_list) * args.workers_per_gpu
    procs = []
    for sh in range(num_shards):
        g = gpu_list[sh % len(gpu_list)]
        procs.append(subprocess.Popen([sys.executable, __file__, "--worker-shard", str(sh), "--num-shards", str(num_shards), "--gpu-id", str(g)]))
    rcs = [p.wait() for p in procs]
    assert all(rc == 0 for rc in rcs), f"Some G2 shards failed: {rcs}"
    aggregate(False)


if __name__ == "__main__":
    main()
