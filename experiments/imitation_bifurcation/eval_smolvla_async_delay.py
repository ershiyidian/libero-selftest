#!/usr/bin/env python3
"""
Suite A1 (Section 3 — Asynchronous Delay, Wall-Clock TTFA & Curved Contact Basin vs. 2025-2026 SOTA):
Evaluates SmolVLA-450M across all 4 LIBERO suites (Spatial, Object, Goal, Long-10) and delays
d in {0, 2, 4, 6} (0, 100, 200, 300 ms at 20 Hz) comparing:
  - sync_m10 (0ms 10-step Euler baseline)
  - faster_m4 (FASTER-style 4-step early-horizon Euler decoding, arXiv:2603.19199)
  - raw (Naive Async)
  - issued (Issued-pose compensation)
  - vlash_zero (Zero-shot VLASH s_{t+d} forward-simulated proprioception, arXiv:2512.01031)
  - rtc_guided (Official Online RTC autograd prefix guidance, arXiv:2506.07339)
  - rtc_align (Index-aligned execution)
  - proj_blend (Our zero-overhead Floquet velocity-projected reference frame)
  - proj_rtc_hybrid (Our Floquet reference frame + index alignment for extreme 200-300 ms delays)
Logs task success (%), GPU wall-clock inference latency (ms/chunk), first-contact radial offset (mm),
and curved-rim transient lift-slip rate (%).
100% subprocess.Popen shard-parallel (zero SemLock, incremental JSON saves, strict /data caches).
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
OUT_DIR = ROOT / "outputs/v17_10h_suite/suiteA_smolvla_async"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "cache/hf_models/smolvla_libero"

sys.path.insert(0, str(ROOT / "cache/pydeps_smolvla"))
sys.path.insert(0, str(ROOT / "external/lerobot/src"))
sys.path.insert(0, str(ROOT / "tools/v17_10h_suite"))

GAMMA_STAB = {0: 1.0, 2: 1.0, 4: 0.51, 6: 0.68}

CONDITIONS = [
    (0, "sync_m10"),
    (0, "faster_m4"),
    (2, "raw"),
    (2, "issued"),
    (2, "vlash_zero"),
    (2, "faster_m4"),
    (2, "rtc_guided"),
    (2, "rtc_align"),
    (2, "proj_blend"),
    (4, "raw"),
    (4, "issued"),
    (4, "vlash_zero"),
    (4, "faster_m4"),
    (4, "rtc_guided"),
    (4, "rtc_align"),
    (4, "proj_blend"),
    (4, "proj_rtc_hybrid"),
    (6, "raw"),
    (6, "issued"),
    (6, "vlash_zero"),
    (6, "rtc_guided"),
    (6, "rtc_align"),
    (6, "proj_blend"),
    (6, "proj_rtc_hybrid"),
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


def build_jobs(states_per_task: int, smoke: bool) -> list[tuple]:
    if smoke:
        return [
            ("libero_spatial", 0, "sync_m10", 1, 2, 220),
            ("libero_spatial", 4, "vlash_zero", 1, 1, 220),
            ("libero_spatial", 4, "rtc_guided", 1, 1, 220),
            ("libero_spatial", 6, "proj_rtc_hybrid", 1, 2, 240),
        ]
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    jobs = []
    for delay, mode in CONDITIONS:
        for s in suites:
            hor = 340 if s != "libero_10" else 460
            for tid in range(10):
                jobs.append((s, delay, mode, tid, states_per_task, hor))
    return jobs


def run_shard(shard_id: int, num_shards: int, gpu_id: int, states_per_task: int, smoke: bool):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = str(ROOT / "cache/hf_home")
    os.environ["TORCH_HOME"] = str(ROOT / "cache/torch_home")
    os.environ["TMPDIR"] = str(ROOT / "cache/tmp")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache/xdg_cache")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    import torch
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from smolvla_sota_infer import SmolVLASotaRunner

    all_jobs = build_jobs(states_per_task, smoke)
    my_jobs = [j for idx, j in enumerate(all_jobs) if idx % num_shards == shard_id]
    out_path = OUT_DIR / ("smoke_shard.json" if smoke else f"shard_{shard_id:02d}.json")

    rows = []
    done_keys = set()
    if not smoke and out_path.exists():
        try:
            rows = json.loads(out_path.read_text())
            by_u = {}
            for r in rows:
                by_u.setdefault((r["suite"], r["delay"], r["mode"], r["task_id"]), []).append(r)
            clean_rows = []
            for k, rs in by_u.items():
                if len(rs) >= states_per_task:
                    done_keys.add(k)
                    clean_rows.extend(rs[:states_per_task])
            rows = clean_rows
        except Exception:
            rows = []

    dev = torch.device("cuda:0")
    pol = SmolVLAPolicy.from_pretrained(str(MODEL_DIR)).to(dev).eval()
    pre_stats = load_file(str(MODEL_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_stats = load_file(str(MODEL_DIR / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    s_mean = pre_stats["observation.state.mean"].to(dev, dtype=torch.float32)
    s_std = pre_stats["observation.state.std"].to(dev, dtype=torch.float32).clamp_min(1e-8)
    a_mean = post_stats["action.mean"].to(dev, dtype=torch.float32)
    a_std = post_stats["action.std"].to(dev, dtype=torch.float32)
    tok = pol.model.vlm_with_expert.processor.tokenizer
    runner = SmolVLASotaRunner(pol, a_mean, a_std, dev, use_bf16=True)

    env_proc = subprocess.Popen(
        ["/data/guest/miniconda3/envs/vla-branch/bin/python", str(ROOT / "tools/v17_10h_suite/smolvla_env_worker_v17.py"), str(gpu_id)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    )
    try:
        for suite, delay, mode, tid, spt, horizon in my_jobs:
            if (suite, delay, mode, tid) in done_keys:
                continue
            infer_method = "standard"
            if mode == "faster_m4":
                infer_method = "faster_m4"
            elif mode == "rtc_guided":
                infer_method = "rtc_guided"

            for sid in range(spt):
                send_msg(env_proc, ("init_delay", suite, tid, sid, delay, mode, GAMMA_STAB[delay], horizon))
                resp = recv_msg(env_proc)
                assert resp is not None and resp[0] == "obs", f"init_delay failed: {resp}"
                _, c1, c2, s8, lang, is_done, step = resp
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
                    out = runner.infer(
                        batch,
                        method=infer_method,
                        prev_leftover_norm=(prev_norm[:, 8:] if prev_norm is not None else None),
                        inference_delay=delay,
                        exec_horizon=8,
                    )
                    prev_norm = out["chunk_full_norm"]
                    infer_lats.append(out["infer_ms"])
                    send_msg(env_proc, ("step_delay", out["chunk"].tolist(), 8))
                    resp = recv_msg(env_proc)
                    if resp[0] == "done":
                        _, ok_ep, step, meta = resp
                        is_done = True
                    else:
                        _, c1, c2, s8, _, is_done, step = resp
                rows.append({
                    "suite": suite, "delay": delay, "mode": mode, "task_id": tid, "state_id": sid,
                    "success": int(ok_ep), "steps": step,
                    "infer_ms_mean": round(float(np.mean(infer_lats)), 2) if infer_lats else 0.0,
                    "first_contact_r_mm": meta.get("first_contact_r_mm"),
                    "max_lift_mm": meta.get("max_lift_mm", 0.0),
                    "transient_slip": meta.get("transient_slip", 0),
                    "replan_calls": meta.get("replan_calls", 0),
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
        k = f"d{r['delay']}_{r['mode']}"
        by_cell.setdefault(k, []).append(r)
    summary = {}
    for k, rs in sorted(by_cell.items()):
        by_s = {}
        for s in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
            sr = [x["success"] for x in rs if x["suite"] == s]
            if sr:
                by_s[s] = round(100.0 * sum(sr) / len(sr), 2)
        cr = [x["first_contact_r_mm"] for x in rs if x.get("first_contact_r_mm") is not None]
        lats = [x["infer_ms_mean"] for x in rs if x.get("infer_ms_mean", 0) > 0]
        summary[k] = {
            "n": len(rs),
            "mean_succ_pct": round(100.0 * sum(x["success"] for x in rs) / len(rs), 2),
            "by_suite": by_s,
            "infer_ms_mean": round(float(np.mean(lats)), 2) if lats else 0.0,
            "first_contact_r_mm_mean": round(float(np.mean(cr)), 2) if cr else None,
            "transient_slip_pct": round(100.0 * sum(x.get("transient_slip", 0) for x in rs) / len(rs), 2),
        }
    return {"n_rollouts": len(rows), "elapsed_s": round(elapsed_s, 1), "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--states-per-task", type=int, default=4)
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--worker-shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    if args.worker_shard >= 0:
        run_shard(args.worker_shard, args.num_shards, args.gpu_id, args.states_per_task, args.smoke)
        return

    if args.smoke:
        run_shard(0, 1, args.gpu_id, 1, True)
        rows = json.loads((OUT_DIR / "smoke_shard.json").read_text())
        res = summarize(rows, time.time() - t0)
        (OUT_DIR / "smoke_summary.json").write_text(json.dumps(res, indent=2))
        print("[SuiteA1 SMOKE OK]", json.dumps(res, indent=2), flush=True)
        return

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    total_shards = len(gpu_list) * args.workers_per_gpu
    procs = []
    for s_idx in range(total_shards):
        gid = gpu_list[s_idx % len(gpu_list)]
        cmd = [
            sys.executable, str( Path(__file__).resolve() ),
            "--worker-shard", str(s_idx),
            "--num-shards", str(total_shards),
            "--gpu-id", str(gid),
            "--states-per-task", str(args.states_per_task),
        ]
        procs.append((s_idx, subprocess.Popen(cmd)))
    for s_idx, p in procs:
        rc = p.wait()
        print(f"[SuiteA1] shard {s_idx}/{total_shards} finished rc={rc}", flush=True)

    all_rows = []
    for s_idx in range(total_shards):
        sp = OUT_DIR / f"shard_{s_idx:02d}.json"
        if sp.exists():
            all_rows.extend(json.loads(sp.read_text()))
    res = summarize(all_rows, time.time() - t0)
    (OUT_DIR / "summary.json").write_text(json.dumps(res, indent=2))
    print("[SuiteA1 FULL DONE]", json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
