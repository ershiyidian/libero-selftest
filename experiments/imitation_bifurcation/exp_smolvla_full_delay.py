#!/usr/bin/env python3
"""
Exp 2B: Full-Sample SmolVLA-450M (100 episodes/suite across all 4 LIBERO suites)
Compares 0ms raw and 100/200ms (raw, issued, rtc_align, closed_form, proj_blend).
Runs 4 GPUs in parallel (1 suite per GPU). All caches strictly under /data.
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import pickle
import struct
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
OUT_DIR = ROOT / "outputs/v13_supplement/exp2b_smolvla_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "cache/hf_models/smolvla_libero"

sys.path.insert(0, str(ROOT / "cache/pydeps_smolvla"))
sys.path.insert(0, str(ROOT / "external/lerobot/src"))

GAMMA_STAB = {0: 1.0, 2: 1.0, 4: 0.51}

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

def gpu_queue_worker(worker_id: int, gpu_id: int, work_q, res_q, states_per_task: int, horizon_override: int | None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = str(ROOT / "cache/hf_home")
    os.environ["TORCH_HOME"] = str(ROOT / "cache/torch_home")
    os.environ["TMPDIR"] = str(ROOT / "cache/tmp")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache/xdg_cache")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import queue
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dev = torch.device("cuda:0")
    pol = SmolVLAPolicy.from_pretrained(str(MODEL_DIR)).to(dev)
    pol.eval()

    pre_stats = load_file(str(MODEL_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_stats = load_file(str(MODEL_DIR / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    s_mean = pre_stats["observation.state.mean"].to(dev, dtype=torch.float32)
    s_std = pre_stats["observation.state.std"].to(dev, dtype=torch.float32).clamp_min(1e-8)
    a_mean = post_stats["action.mean"].to(dev, dtype=torch.float32)
    a_std = post_stats["action.std"].to(dev, dtype=torch.float32)
    tok = pol.model.vlm_with_expert.processor.tokenizer

    env_proc = subprocess.Popen(
        ["/data/guest/miniconda3/envs/vla-branch/bin/python", str(ROOT / "tools/v13_supplement/smolvla_env_worker_v13.py"), str(gpu_id)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    )
    try:
        while True:
            try:
                item = work_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                break
            suite, delay, mode, tid = item
            horizon = horizon_override if horizon_override is not None else (360 if suite != "libero_10" else 480)
            batch_rows = []
            for sid in range(states_per_task):
                send_msg(env_proc, ("init", suite, tid, sid, delay, mode, GAMMA_STAB[delay], horizon))
                resp = recv_msg(env_proc)
                assert resp is not None and resp[0] == "obs"
                _, c1, c2, s8, lang, is_done, step = resp
                tok_out = tok([lang + "\n"], max_length=pol.config.tokenizer_max_length, truncation=True, padding="max_length", return_tensors="pt")
                lang_tokens = tok_out["input_ids"].to(dev)
                lang_mask = tok_out["attention_mask"].to(dev, dtype=torch.bool)
                ok_ep = is_done
                while not is_done:
                    img1 = torch.from_numpy(np.frombuffer(c1, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                    img2 = torch.from_numpy(np.frombuffer(c2, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                    st_t = (torch.tensor(s8, dtype=torch.float32).unsqueeze(0).to(dev, dtype=torch.float32) - s_mean) / s_std
                    batch = {
                        "observation.images.camera1": img1,
                        "observation.images.camera2": img2,
                        "observation.state": st_t,
                        "observation.language.tokens": lang_tokens,
                        "observation.language.attention_mask": lang_mask,
                    }
                    with torch.inference_mode():
                        act_norm = pol.predict_action_chunk(batch)
                        act_chunk = (act_norm[0, :, :7] * a_std + a_mean).cpu().numpy().astype(np.float32)
                    send_msg(env_proc, ("step_chunk", act_chunk.tolist()))
                    resp = recv_msg(env_proc)
                    if resp[0] == "done":
                        _, ok_ep, step = resp
                        is_done = True
                    else:
                        _, c1, c2, s8, _, is_done, step = resp
                batch_rows.append({"suite": suite, "delay": delay, "mode": mode, "task_id": tid, "state_id": sid, "success": int(ok_ep), "steps": step})
            res_q.put(("batch", suite, delay, mode, tid, batch_rows))
        res_q.put(("worker_done", worker_id))
    finally:
        try:
            send_msg(env_proc, ("close",))
        except Exception:
            pass
        env_proc.wait(timeout=10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--states-per-task", type=int, default=10)
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    ctx = mp.get_context("spawn")
    if args.smoke:
        wq, rq = ctx.Queue(), ctx.Queue()
        wq.put(("libero_spatial", 0, "raw", 0))
        wq.put(("libero_spatial", 4, "proj_blend", 0))
        gpu_queue_worker(0, 0, wq, rq, states_per_task=1, horizon_override=160)
        smoke_rows = []
        while not rq.empty():
            m = rq.get()
            if m[0] == "batch":
                smoke_rows.extend(m[5])
        (OUT_DIR / "smoke.json").write_text(json.dumps(smoke_rows, indent=2))
        print("[Exp2B SMOKE OK]", json.dumps(smoke_rows, indent=2), f"elapsed={time.time()-t0:.1f}s", flush=True)
        return

    suites = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
    conditions = [
        (0, "raw"),
        (2, "raw"), (2, "issued"), (2, "rtc_align"), (2, "proj_blend"),
        (4, "raw"), (4, "issued"), (4, "rtc_align"), (4, "proj_blend"),
    ]
    suite_rows = {s: [] for s in suites}
    done_keys = set()
    for s in suites:
        sf = OUT_DIR / f"{s}.json"
        if sf.exists():
            try:
                existing = json.loads(sf.read_text())
                # Keep complete (delay, mode, tid) groups with states_per_task entries
                by_k = {}
                for r in existing:
                    by_k.setdefault((r["delay"], r["mode"], r["task_id"]), []).append(r)
                for (d, m, tid), rs in by_k.items():
                    if len(rs) >= args.states_per_task:
                        suite_rows[s].extend(rs[:args.states_per_task])
                        done_keys.add((s, d, m, tid))
            except Exception:
                pass

    wq, rq = ctx.Queue(), ctx.Queue()
    total_units = 0
    for delay, mode in conditions:
        for s in suites:
            for tid in range(10):
                if (s, delay, mode, tid) not in done_keys:
                    wq.put((s, delay, mode, tid))
                    total_units += 1
    print(f"[Exp2B v14] Reused {len(done_keys)} task-units ({len(done_keys)*args.states_per_task} eps); queued {total_units} remaining task-units across {4*args.workers_per_gpu} workers!", flush=True)

    procs = []
    if total_units > 0:
        wid = 0
        for g in range(4):
            for _ in range(args.workers_per_gpu):
                p = ctx.Process(target=gpu_queue_worker, args=(wid, g, wq, rq, args.states_per_task, None))
                p.start()
                procs.append(p)
                wid += 1
        finished_workers = 0
        done_units = 0
        cell_counts = {}
        for s in suites:
            for r in suite_rows[s]:
                cell_counts[(s, r["delay"], r["mode"])] = cell_counts.get((s, r["delay"], r["mode"]), 0) + 1
        while finished_workers < len(procs):
            msg = rq.get()
            if msg[0] == "worker_done":
                finished_workers += 1
                continue
            _, s, d, m, tid, brows = msg
            suite_rows[s].extend(brows)
            done_units += 1
            cell_counts[(s, d, m)] = cell_counts.get((s, d, m), 0) + len(brows)
            if done_units % 4 == 0 or cell_counts[(s, d, m)] == 10 * args.states_per_task:
                (OUT_DIR / f"{s}.json").write_text(json.dumps(suite_rows[s], indent=2))
            if cell_counts[(s, d, m)] == 10 * args.states_per_task:
                vals = [r["success"] for r in suite_rows[s] if r["delay"] == d and r["mode"] == m]
                print(f"[Exp2B Cell Done] {s:14s} d={d} {m:11s}: {100.0*np.mean(vals):5.1f}% (n={len(vals)}) [{done_units}/{total_units} units, elapsed={(time.time()-t0)/60:.1f}m]", flush=True)
        for s in suites:
            (OUT_DIR / f"{s}.json").write_text(json.dumps(suite_rows[s], indent=2))
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Exp2B worker failed: {p.exitcode}")
    summary = {}
    for s in suites:
        sub_all = json.loads((OUT_DIR / f"{s}.json").read_text())
        summary[s] = {}
        for delay, dlab in [(0, "0ms"), (2, "100ms"), (4, "200ms")]:
            summary[s][dlab] = {}
            for mode in ["raw", "issued", "rtc_align", "proj_blend"]:
                vals = [r["success"] for r in sub_all if r["delay"] == delay and r["mode"] == mode]
                if vals:
                    summary[s][dlab][mode] = round(100.0 * float(np.mean(vals)), 1)
    (OUT_DIR / "summary.json").write_text(json.dumps({"summary": summary, "elapsed_s": round(time.time()-t0, 1)}, indent=2))
    print("[Exp2B FULL DONE]", json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
