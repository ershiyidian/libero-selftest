#!/usr/bin/env python3
"""Shard-parallel (zero SemLock) evaluation of SmolVLA-450M across Nominal LIBERO, Mid-Fail Recovery, Jam Bifurcation, and PACE (2026)."""
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
OUT_DIR = ROOT / "outputs/v13_supplement/exp4_smolvla_selftest"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "cache/pydeps_smolvla"))
sys.path.insert(0, str(ROOT / "external/lerobot/src"))

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

def get_model_specs(smoke: bool = False) -> dict[str, str]:
    if smoke:
        return {"smoke_causal_replan": str(OUT_DIR / "smoke_ckpt")}
    return {
        "base": str(ROOT / "cache/hf_models/smolvla_libero"),
        "base_pace": str(ROOT / "cache/hf_models/smolvla_libero"),
        "unmasked": str(OUT_DIR / "ckpt_unmasked"),
        "role_only": str(OUT_DIR / "ckpt_role_only"),
        "causal": str(OUT_DIR / "ckpt_causal"),
        "causal_pace": str(OUT_DIR / "ckpt_causal"),
        "causal_replan": str(OUT_DIR / "ckpt_causal"),
    }

def exec_mode_for_label(label: str) -> str:
    if label.endswith("_replan"):
        return "event_replan"
    if label.endswith("_pace"):
        return "pace"
    return "open16"

def build_all_jobs(states_per_task: int, cached_base_nom: set[tuple[str, int]]) -> list[tuple]:
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    manifest = json.loads((ROOT / "cache/selftest40_smolvla/manifest_40tasks.json").read_text())
    rec_eps_by_task = {}
    for it in manifest:
        if it.get("ctx_only_ranges"):
            rec_eps_by_task[it["task_id"]] = it

    jobs = []
    for lab in get_model_specs(False):
        for s in suites:
            hor = 300 if s != "libero_10" else 440
            for tid in range(10):
                if lab == "base" and (s, tid) in cached_base_nom:
                    continue
                jobs.append(("nominal", lab, s, tid, states_per_task, hor))
        for _, it in sorted(rec_eps_by_task.items()):
            r0, r1 = it["ctx_only_ranges"][0]
            mid_fail = max(0, (int(r0) + int(r1)) // 2)
            hor = 280 if it["suite"] != "libero_10" else 420
            jobs.append(("recovery", lab, "from_midfail", it["suite"], int(it["libero_task_id"]), it["scene_xml"], it["hdf5"], mid_fail, it["episode"], hor))
            jobs.append(("recovery", lab, "from_jam_bifurc", it["suite"], int(it["libero_task_id"]), it["scene_xml"], it["hdf5"], mid_fail, it["episode"], hor))
    return jobs

def run_shard_worker(shard_id: int, num_shards: int, gpu_id: int, states_per_task: int, smoke: bool = False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = str(ROOT / "cache/hf_home")
    os.environ["TORCH_HOME"] = str(ROOT / "cache/torch_home")
    os.environ["TMPDIR"] = str(ROOT / "cache/tmp")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache/xdg_cache")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    model_specs = get_model_specs(smoke)
    if smoke:
        my_jobs = [("nominal", "smoke_causal_replan", "libero_spatial", 0, 1, 140)]
        out_path = OUT_DIR / "smoke_eval.json"
    else:
        cached_nom = set()
        v14_p = OUT_DIR / "eval_all_v14.json"
        if v14_p.exists():
            old = json.loads(v14_p.read_text())
            for s in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
                for tid in range(10):
                    cnt = sum(1 for r in old if r.get("model") == "base" and r.get("kind") == "nominal" and r.get("suite") == s and r.get("task_id") == tid)
                    if cnt >= states_per_task:
                        cached_nom.add((s, tid))
        all_jobs = build_all_jobs(states_per_task, cached_nom)
        my_jobs = [j for idx, j in enumerate(all_jobs) if idx % num_shards == shard_id]
        out_path = OUT_DIR / f"shard_{shard_id}_v16.json"

    dev = torch.device("cuda:0")
    base_dir = Path(next(iter(model_specs.values())))
    pol = SmolVLAPolicy.from_pretrained(str(base_dir)).to(dev)
    pol.eval()
    cur_ckpt = str(base_dir)
    pre_stats = load_file(str(base_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_stats = load_file(str(base_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    s_mean = pre_stats["observation.state.mean"].to(dev, dtype=torch.float32)
    s_std = pre_stats["observation.state.std"].to(dev, dtype=torch.float32).clamp_min(1e-8)
    a_mean = post_stats["action.mean"].to(dev, dtype=torch.float32)
    a_std = post_stats["action.std"].to(dev, dtype=torch.float32)
    tok = pol.model.vlm_with_expert.processor.tokenizer

    env_proc = subprocess.Popen(
        ["/data/guest/miniconda3/envs/vla-branch/bin/python", str(ROOT / "tools/v13_supplement/smolvla_env_worker_v13.py"), str(gpu_id)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    )
    rows = []
    try:
        for job in my_jobs:
            kind = job[0]
            label = job[1]
            base_exec = exec_mode_for_label(label)
            target_ckpt = model_specs[label]
            if target_ckpt != cur_ckpt:
                sd = load_file(str(Path(target_ckpt) / "model.safetensors"), device="cpu")
                pol.load_state_dict(sd, strict=True)
                pol.eval()
                cur_ckpt = target_ckpt

            if kind == "nominal":
                _, label, suite, tid, spt, horizon = job
                for sid in range(spt):
                    send_msg(env_proc, ("init", suite, tid, sid, 0, base_exec, 1.0, horizon))
                    resp = recv_msg(env_proc)
                    assert resp is not None and resp[0] == "obs", f"init failed on {job}"
                    _, c1, c2, s8, lang, is_done, step = resp
                    tok_out = tok([lang + "\n"], max_length=pol.config.tokenizer_max_length, truncation=True, padding="max_length", return_tensors="pt")
                    lt, lm = tok_out["input_ids"].to(dev), tok_out["attention_mask"].to(dev, dtype=torch.bool)
                    ok_ep = is_done
                    while not is_done:
                        img1 = torch.from_numpy(np.frombuffer(c1, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                        img2 = torch.from_numpy(np.frombuffer(c2, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                        st_t = (torch.tensor(s8, dtype=torch.float32).unsqueeze(0).to(dev) - s_mean) / s_std
                        with torch.inference_mode():
                            act_norm = pol.predict_action_chunk({"observation.images.camera1": img1, "observation.images.camera2": img2, "observation.state": st_t, "observation.language.tokens": lt, "observation.language.attention_mask": lm})
                            act_chunk = (act_norm[0, :, :7] * a_std + a_mean).cpu().numpy().astype(np.float32)
                        send_msg(env_proc, ("step_chunk", act_chunk.tolist()))
                        resp = recv_msg(env_proc)
                        if resp[0] == "done":
                            _, ok_ep, step = resp; is_done = True
                        else:
                            _, c1, c2, s8, _, is_done, step = resp
                    rows.append({"kind": "nominal", "model": label, "suite": suite, "task_id": tid, "state_id": sid, "success": int(ok_ep), "steps": step})
            elif kind == "recovery":
                _, label, rec_stage, suite, tid, scene_xml, hdf5_path, pref_steps, ep_name, horizon = job
                rec_exec = f"{base_exec}_jam" if rec_stage == "from_jam_bifurc" else base_exec
                send_msg(env_proc, ("init_selftest", suite, tid, scene_xml, hdf5_path, pref_steps, rec_exec, horizon))
                resp = recv_msg(env_proc)
                assert resp is not None and resp[0] == "obs", f"init_selftest failed on {job}"
                _, c1, c2, s8, lang, is_done, step = resp
                tok_out = tok([lang + "\n"], max_length=pol.config.tokenizer_max_length, truncation=True, padding="max_length", return_tensors="pt")
                lt, lm = tok_out["input_ids"].to(dev), tok_out["attention_mask"].to(dev, dtype=torch.bool)
                ok_ep = is_done
                while not is_done:
                    img1 = torch.from_numpy(np.frombuffer(c1, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                    img2 = torch.from_numpy(np.frombuffer(c2, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                    st_t = (torch.tensor(s8, dtype=torch.float32).unsqueeze(0).to(dev) - s_mean) / s_std
                    with torch.inference_mode():
                        act_norm = pol.predict_action_chunk({"observation.images.camera1": img1, "observation.images.camera2": img2, "observation.state": st_t, "observation.language.tokens": lt, "observation.language.attention_mask": lm})
                        act_chunk = (act_norm[0, :, :7] * a_std + a_mean).cpu().numpy().astype(np.float32)
                    send_msg(env_proc, ("step_chunk", act_chunk.tolist()))
                    resp = recv_msg(env_proc)
                    if resp[0] == "done":
                        _, ok_ep, step = resp; is_done = True
                    else:
                        _, c1, c2, s8, _, is_done, step = resp
                rows.append({"kind": "recovery", "rec_stage": rec_stage, "model": label, "suite": suite, "task_id": tid, "episode": ep_name, "success": int(ok_ep), "steps": step})
            out_path.write_text(json.dumps(rows, indent=2))
    finally:
        try: send_msg(env_proc, ("close",))
        except Exception: pass
        env_proc.wait(timeout=10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker-shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--states-per-task", type=int, default=5)
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    args = ap.parse_args()

    if args.worker_shard >= 0:
        run_shard_worker(args.worker_shard, args.num_shards, args.gpu, args.states_per_task, smoke=False)
        return
    if args.smoke:
        run_shard_worker(0, 1, 0, 1, smoke=True)
        print("[Exp4 SHARD SMOKE OK]", (OUT_DIR / "smoke_eval.json").read_text(), flush=True)
        return

    t0 = time.time()
    total_shards = 4 * args.workers_per_gpu
    for s in range(total_shards):
        sp = OUT_DIR / f"shard_{s}_v16.json"
        if sp.exists():
            sp.unlink()

    procs = []
    for s in range(total_shards):
        p = subprocess.Popen([
            sys.executable, __file__,
            "--worker-shard", str(s),
            "--num-shards", str(total_shards),
            "--gpu", str(s % 4),
            "--states-per-task", str(args.states_per_task),
        ])
        procs.append((s, p))

    for s, p in procs:
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Shard {s} failed with exit code {rc}")

    all_rows = []
    v14_p = OUT_DIR / "eval_all_v14.json"
    if v14_p.exists():
        all_rows.extend([r for r in json.loads(v14_p.read_text()) if r.get("model") == "base" and r.get("kind") == "nominal"])
    for s in range(total_shards):
        sp = OUT_DIR / f"shard_{s}_v16.json"
        if sp.exists():
            all_rows.extend(json.loads(sp.read_text()))
    (OUT_DIR / "eval_all_v16.json").write_text(json.dumps(all_rows, indent=2))

    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    summary = {}
    for lab in get_model_specs(False):
        nom = [r for r in all_rows if r["model"] == lab and r["kind"] == "nominal"]
        rf0 = [r for r in all_rows if r["model"] == lab and r["kind"] == "recovery" and r["rec_stage"] == "from_midfail"]
        rf1 = [r for r in all_rows if r["model"] == lab and r["kind"] == "recovery" and r["rec_stage"] == "from_jam_bifurc"]
        summary[lab] = {s: round(100.0 * float(np.mean([r["success"] for r in nom if r["suite"] == s])), 1) for s in suites}
        summary[lab]["nominal_mean"] = round(100.0 * float(np.mean([r["success"] for r in nom])), 1) if nom else 0.0
        summary[lab]["recovery_from_midfail"] = round(100.0 * float(np.mean([r["success"] for r in rf0])), 1) if rf0 else 0.0
        summary[lab]["recovery_from_jam_bifurc"] = round(100.0 * float(np.mean([r["success"] for r in rf1])), 1) if rf1 else 0.0
    out_obj = {"summary": summary, "elapsed_s": round(time.time() - t0, 1), "total_rows": len(all_rows)}
    (OUT_DIR / "eval_summary_v16.json").write_text(json.dumps(out_obj, indent=2))
    (OUT_DIR / "eval_summary.json").write_text(json.dumps(out_obj, indent=2))
    print("[Exp4 v16 FULL DONE]", json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
