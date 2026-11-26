#!/usr/bin/env python3
"""
Upgrade 1 (Table I - Flow-Matching VLA SmolVLA):
Runs lerobot/smolvla_libero (450M Flow-Matching VLA, 2025-2026) inside torch_env (GPU),
communicating with CPU MuJoCo workers running inside vla-branch env via binary pipe.
Evaluates all 4 LIBERO suites (Spatial, Goal, Object, Long-10) at:
  - 0 ms (Nominal)
  - 100 ms (d=2): Raw, Issued, ACT-TE, RTC-Align, Closed-Form (zero-threshold, gamma*=1.0)
  - 200 ms (d=4): Raw, Issued, ACT-TE, RTC-Align, Closed-Form (zero-threshold, gamma*=0.51)
Runs 4 GPU workers in parallel (1 suite per GPU).
"""
from __future__ import annotations
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
DATA_OUT = ROOT / "VLA-SelfTest/paper/source_data"
DATA_OUT.mkdir(parents=True, exist_ok=True)
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
    data = _read_exact(proc.stdout, n)
    return pickle.loads(data)


def gpu_suite_worker(gpu_id: int, suite: str, out_file: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = str(ROOT / "cache/hf_home")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dev = torch.device("cuda:0")
    print(f"[SmolVLA GPU{gpu_id}] Loading {MODEL_DIR} for {suite}...", flush=True)
    pol = SmolVLAPolicy.from_pretrained(str(MODEL_DIR)).to(dev)
    pol.eval()

    # Load exact pre/post normalizer stats from safetensors
    pre_stats = load_file(str(MODEL_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_stats = load_file(str(MODEL_DIR / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    s_mean = pre_stats["observation.state.mean"].to(dev, dtype=torch.float32)
    s_std = pre_stats["observation.state.std"].to(dev, dtype=torch.float32).clamp_min(1e-8)
    a_mean = post_stats["action.mean"].to(dev, dtype=torch.float32)
    a_std = post_stats["action.std"].to(dev, dtype=torch.float32)

    tok = pol.model.vlm_with_expert.processor.tokenizer

    env_proc = subprocess.Popen(
        ["/home/guest/miniconda3/envs/vla-branch/bin/python", str(ROOT / "tools/smolvla_env_worker.py"), str(gpu_id)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    )

    conditions = [
        (0, "raw"),
        (2, "raw"), (2, "issued"), (2, "act_te"), (2, "rtc_align"), (2, "closed_form"),
        (4, "raw"), (4, "issued"), (4, "act_te"), (4, "rtc_align"), (4, "closed_form"),
    ]
    horizon = 360 if suite != "libero_10" else 480
    rows = []
    try:
        for delay, mode in conditions:
            succs = []
            for tid in range(10):
                for sid in range(2):  # 20 official states per (suite, delay, mode) -> 880 total rollouts
                    send_msg(env_proc, ("init", suite, tid, sid, delay, mode, GAMMA_STAB[delay], horizon))
                    resp = recv_msg(env_proc)
                    assert resp is not None and resp[0] == "obs"
                    _, c1, c2, s8, lang, is_done, step = resp
                    # Tokenize language once per episode
                    tok_out = tok(
                        [lang + "\n"],
                        max_length=pol.config.tokenizer_max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    lang_tokens = tok_out["input_ids"].to(dev)
                    lang_mask = tok_out["attention_mask"].to(dev, dtype=torch.bool)

                    ok_ep = is_done
                    while not is_done:
                        # Build SmolVLA batch dict
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
                            act_norm = pol.predict_action_chunk(batch)  # (1, 50, 7)
                            act_chunk = (act_norm[0, :, :7] * a_std + a_mean).cpu().numpy().astype(np.float32)
                        send_msg(env_proc, ("step_chunk", act_chunk.tolist()))
                        resp = recv_msg(env_proc)
                        if resp[0] == "done":
                            _, ok_ep, step = resp
                            is_done = True
                        else:
                            _, c1, c2, s8, _, is_done, step = resp
                    succs.append(int(ok_ep))
                    rows.append({"suite": suite, "delay": delay, "mode": mode, "task_id": tid, "state_id": sid, "success": int(ok_ep), "steps": step})
            print(f"[SmolVLA GPU{gpu_id}] {suite} d={delay} {mode:11s}: {100.0*np.mean(succs):5.1f}% (n={len(succs)})", flush=True)
            out_file.write_text(json.dumps(rows, indent=2))
    finally:
        try:
            send_msg(env_proc, ("close",))
        except Exception:
            pass
        env_proc.wait(timeout=10)


def main():
    t0 = time.time()
    suites = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
    tmp_dir = DATA_OUT / "v12_smolvla_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    procs = []
    for g, s in enumerate(suites):
        p = ctx.Process(target=gpu_suite_worker, args=(g, s, tmp_dir / f"{s}.json"))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"SmolVLA worker failed with exitcode {p.exitcode}")
    all_rows = []
    summary = {}
    for s in suites:
        sub_all = json.loads((tmp_dir / f"{s}.json").read_text())
        all_rows.extend(sub_all)
        summary[s] = {}
        for delay, dlab in [(0, "0ms"), (2, "100ms"), (4, "200ms")]:
            summary[s][dlab] = {}
            for mode in ["raw", "issued", "act_te", "rtc_align", "closed_form"]:
                vals = [r["success"] for r in sub_all if r["delay"] == delay and r["mode"] == mode]
                if vals:
                    summary[s][dlab][mode] = round(100.0 * float(np.mean(vals)), 1)
    out = {"n_rollouts": len(all_rows), "elapsed_s": round(time.time() - t0, 1), "summary": summary}
    (DATA_OUT / "v12_table1_smolvla.json").write_text(json.dumps(out, indent=2))
    print(f"DONE SmolVLA all 4 suites in {time.time()-t0:.1f}s", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
