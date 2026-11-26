#!/usr/bin/env python3
"""
G3 Surgical Repair:
  1. Completes the 2x2 real-VLA matrix (Unmasked vs Causal x L1 vs Flow) by training 3 seeds of `unmasked_l1`
     (4 top expert layers, frozen heads, 220 steps).
  2. Trains 3 seeds of lightweight 4-layer Frozen-Head Causal Flow (`frozen4_causal_flow`, top 4 expert layers only,
     220 steps, lr=8.5e-5, 2:1 clean:latched replay) and sweeps beta_exp in {0.65, 0.85, 1.00} with beta_head=0.00!
  Total: 6 fast 4-GPU parallel trainings (~6 min) + ~2,160 rollouts (~42 min) = ~48 min.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
MANIFEST_SRC = ROOT / "outputs/v18_surgical_fixes/suiteC_wise_bifurc_v18/manifest_merged.json"
OUT_DIR = ROOT / "outputs/v23_16h_suite/g3_repair_2x2_and_4layer"
MODEL_DIR = ROOT / "cache/hf_models/smolvla_libero"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "cache/pydeps_smolvla"))
sys.path.insert(0, str(ROOT / "external/lerobot/src"))
sys.path.insert(0, str(ROOT / "tools/v23_16h_suite"))

from g3_frozen_head_causal_flow_vs_l1_vla import BIFURC_TASKS, HEAD_KEYS, send_msg, recv_msg


def build_chunk_and_weights(ep: dict, fb_arr: np.ndarray, t_raw: int, chunk_size: int, mode: str):
    actions = ep["actions"]
    roles = ep["role_mask"] if "role_mask" in ep else np.ones(len(actions), dtype=np.float32)
    T = len(actions)
    t = int(np.clip(t_raw, 0, max(0, T - 2)))
    a_chunk = np.zeros((chunk_size, 7), dtype=np.float32)
    avail = min(chunk_size, T - t)
    a_chunk[:avail] = actions[t : t + avail]
    if avail < chunk_size:
        a_chunk[avail:] = actions[T - 1]
    w_chunk = np.zeros(chunk_size, dtype=np.float32)
    w_chunk[:avail] = 1.0
    if mode == "frozen4_causal_flow":
        for j in range(avail):
            if float(roles[t + j]) < 0.5:
                w_chunk[j] = 0.0
        nxt = fb_arr[fb_arr > t]
        if len(nxt) > 0:
            cut = int(nxt[0]) - t
            if 0 < cut < chunk_size:
                w_chunk[cut:] = 0.0
    if w_chunk.sum() < 1.0:
        w_chunk[0] = 1.0
    return t, a_chunk, w_chunk


def train_one(mode: str, seed: int, gpu_id: int, steps: int, batch_size: int, out_ckpt: Path):
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
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    meta_p = out_ckpt / "train_meta.json"
    if (out_ckpt / "model.safetensors").exists() and meta_p.exists() and steps > 100:
        return

    torch.manual_seed(20261215 + seed * 151 + (17 if mode == "unmasked_l1" else 0))
    np.random.seed(20261215 + seed * 151)
    rng = random.Random(20261215 + seed * 151 + hash(mode) % 997)

    dev = torch.device("cuda:0")
    pol = SmolVLAPolicy.from_pretrained(str(MODEL_DIR)).to(dev)
    n_exp = len(pol.model.vlm_with_expert.lm_expert.layers)
    trainable = 0
    for name, p in pol.named_parameters():
        p.requires_grad = False
        if any(hk in name for hk in HEAD_KEYS):
            p.requires_grad = False
        elif "lm_expert.layers." in name:
            lid = int(name.split("lm_expert.layers.")[1].split(".")[0])
            if lid >= max(0, n_exp - 4):  # Top 4 expert layers ONLY!
                p.requires_grad = True
                trainable += p.numel()

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
    ep_bank, lang_cache = [], {}
    for it in manifest:
        d = np.load(it["npz"])
        ep_bank.append((it, {k: d[k] for k in d.files}))
        lang = it["language"]
        if lang not in lang_cache:
            to = tok([lang + "\n"], max_length=pol.config.tokenizer_max_length, truncation=True, padding="max_length", return_tensors="pt")
            lang_cache[lang] = (to["input_ids"][0], to["attention_mask"][0].bool())

    latched_eps = [(it, ep) for it, ep in ep_bank if it["branch"] == "latched"]
    clean_eps = [(it, ep) for it, ep in ep_bank if it["branch"] != "latched"]

    lr_init = 1.2e-4 if mode == "unmasked_l1" else 8.5e-5
    opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], lr=lr_init, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1.5e-5)
    pol.train()
    losses = []

    for step in range(1, steps + 1):
        img1s, img2s, states, acts, ltoks, lmasks, cmasks = [], [], [], [], [], [], []
        for b_idx in range(batch_size):
            if b_idx % 3 == 0:
                it, ep = rng.choice(latched_eps)
                fb_arr = np.asarray(it["fb_steps"], dtype=np.int32)
                if len(fb_arr) > 0:
                    c0 = int(fb_arr[0])
                    u_r = rng.random()
                    if u_r < 0.65:
                        t_raw = rng.randint(c0, min(len(ep["actions"]) - 2, c0 + 5))
                    elif u_r < 0.85:
                        t_raw = rng.randint(max(0, c0 - 16), max(0, c0 - 1))
                    else:
                        t_raw = rng.randint(min(len(ep["actions"]) - 2, c0 + 6), max(0, len(ep["actions"]) - 2))
                else:
                    t_raw = rng.randint(0, max(0, len(ep["actions"]) - 2))
            else:
                it, ep = rng.choice(clean_eps)
                fb_arr = np.asarray(it["fb_steps"], dtype=np.int32)
                t_raw = rng.randint(0, max(0, len(ep["actions"]) - 2))
            t, a_chunk, w_chunk = build_chunk_and_weights(ep, fb_arr, t_raw, chunk_size, mode)
            img1s.append(ep["cam1"][t]); img2s.append(ep["cam2"][t]); states.append(ep["proprio"][t])
            acts.append(a_chunk); cmasks.append(w_chunk)
            lt, lm = lang_cache[it["language"]]
            ltoks.append(lt); lmasks.append(lm)

        img1_t = torch.from_numpy(np.stack(img1s)).permute(0, 3, 1, 2).to(dev, dtype=torch.float32) / 255.0
        img2_t = torch.from_numpy(np.stack(img2s)).permute(0, 3, 1, 2).to(dev, dtype=torch.float32) / 255.0
        st_t = (torch.from_numpy(np.stack(states)).to(dev, dtype=torch.float32) - s_mean) / s_std
        act_t = (torch.from_numpy(np.stack(acts)).to(dev, dtype=torch.float32) - a_mean) / a_std
        cmask_t = torch.from_numpy(np.stack(cmasks)).to(dev, dtype=torch.float32)
        batch = {"observation.images.camera1": img1_t, "observation.images.camera2": img2_t, "observation.state": st_t, "action": act_t, "observation.language.tokens": torch.stack(ltoks).to(dev), "observation.language.attention_mask": torch.stack(lmasks).to(dev)}
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            images, img_masks = pol.prepare_images(batch)
            p_state = pol.prepare_state(batch)
            p_actions = pol.prepare_action(batch)
            if mode == "unmasked_l1":
                m = pol.model
                noise_zero = torch.zeros_like(p_actions)
                t_one = torch.ones((batch_size,), dtype=torch.float32, device=dev)
                prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"], state=p_state)
                suffix_embs, suffix_pad_masks, suffix_att_masks = m.embed_suffix(noise_zero, t_one)
                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
                att_2d = make_att_2d_masks(pad_masks, att_masks)
                pos_ids = torch.cumsum(pad_masks, dim=1) - 1
                (_, suffix_out), _ = m.vlm_with_expert.forward(attention_mask=att_2d, position_ids=pos_ids, past_key_values=None, inputs_embeds=[prefix_embs, suffix_embs], use_cache=False)
                pred_a = -m.action_out_proj(suffix_out[:, -chunk_size:].to(dtype=torch.float32))[:, :, :act_dim]
                l1_per = torch.abs(pred_a - p_actions[:, :, :act_dim]).mean(dim=-1)
                loss = (l1_per * cmask_t).sum() / cmask_t.sum().clamp_min(float(batch_size * 8))
            else:
                losses_raw = pol.model.forward(images, img_masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"], p_state, p_actions, None, None)[:, :, :act_dim]
                loss = (losses_raw.mean(dim=-1) * cmask_t).sum() / cmask_t.sum().clamp_min(float(batch_size * 8))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in pol.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if step % max(1, steps // 4) == 0 or step == 1:
            print(f"[G3-Rep Train {mode} s{seed} GPU{gpu_id}] {step}/{steps} loss={np.mean(losses[-20:]):.4f}", flush=True)

    out_ckpt.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in pol.state_dict().items()}, str(out_ckpt / "model.safetensors"))
    meta_p.write_text(json.dumps({"mode": mode, "seed": seed, "steps": steps, "trainable": trainable, "loss_init": losses[0], "loss_final": float(np.mean(losses[-20:]))}, indent=2))


def build_eval_jobs(smoke: bool):
    jobs = []
    if smoke:
        for m in ("frozen4_causal_flow", "unmasked_l1"):
            for lat in (0, 1):
                jobs.append(("bifurc", m, 0, 0.85 if m == "frozen4_causal_flow" else 1.0, "pace", "libero_spatial", 0, lat, 1, 280))
        return jobs
    suites_all = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    for seed in (0, 1, 2):
        # 1. frozen4_causal_flow: beta_exp in {0.65, 0.85, 1.00}
        for be in (0.65, 0.85, 1.00):
            for s in suites_all:
                hor = 330 if s != "libero_10" else 440
                for tid in range(10):
                    jobs.append(("nom40", "frozen4_causal_flow", seed, be, "pace", s, tid, 0, 2, hor))
            for sm in ("pace", "hybrid_pace_event"):
                for s, tid in BIFURC_TASKS:
                    hor = 340 if s != "libero_10" else 440
                    for lat in (0, 1):
                        jobs.append(("bifurc", "frozen4_causal_flow", seed, be, sm, s, tid, lat, 2, hor))
        # 2. unmasked_l1: open24 + pace + nom40
        for s in suites_all:
            hor = 330 if s != "libero_10" else 440
            for tid in range(10):
                jobs.append(("nom40", "unmasked_l1", seed, 1.00, "pace", s, tid, 0, 2, hor))
        for sm in ("open24", "pace"):
            for s, tid in BIFURC_TASKS:
                hor = 340 if s != "libero_10" else 440
                for lat in (0, 1):
                    jobs.append(("bifurc", "unmasked_l1", seed, 1.00, sm, s, tid, lat, 2, hor))
    return jobs


def run_eval_shard(shard_id: int, num_shards: int, gpu_id: int, smoke: bool):
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

    all_jobs = build_eval_jobs(smoke)
    my_jobs = [j for idx, j in enumerate(all_jobs) if idx % num_shards == shard_id]
    out_path = OUT_DIR / ("smoke_shard.json" if smoke else f"shard_{shard_id:02d}.json")
    rows, done_keys = [], set()
    if not smoke and out_path.exists():
        try:
            raw_rows = json.loads(out_path.read_text())
            for r in raw_rows:
                done_keys.add((r["split"], r["mode"], r["seed"], r["beta_exp"], r["sched_mode"], r["suite"], r["task_id"], r["is_latched"]))
            rows = raw_rows
        except Exception:
            rows = []

    dev = torch.device("cuda:0")
    pol = SmolVLAPolicy.from_pretrained(str(MODEL_DIR)).to(dev).eval()
    sd_base = {k: v.detach().cpu().clone() for k, v in pol.state_dict().items()}
    sd_cache, cur_loaded = {}, None

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
        for split, mode, seed, be, sm, suite, tid, lat, n_st, hor in my_jobs:
            key = (split, mode, seed, be, sm, suite, tid, lat)
            if key in done_keys:
                continue
            ckpt_name = (f"smoke_ckpt_{mode}_s{seed}" if smoke else f"ckpt_{mode}_s{seed}")
            if (mode, seed) not in sd_cache:
                sd_cache[(mode, seed)] = load_file(str(OUT_DIR / ckpt_name / "model.safetensors"))
            want_m = (mode, seed, be)
            if cur_loaded != want_m:
                sd_ft = sd_cache[(mode, seed)]
                sd_mix = {k: (sd_ft[k].to(dev) if (not v0.is_floating_point() or be >= 0.999) else (((1.0 - be) * v0.float() + be * sd_ft[k].float()).to(v0.dtype)).to(dev)) for k, v0 in sd_base.items()}
                pol.load_state_dict(sd_mix, strict=True)
                pol.eval()
                cur_loaded = want_m
            infer_method = "l1_direct" if mode == "unmasked_l1" else "standard"
            for sid in range(n_st):
                send_msg(env_proc, ("init_bifurc_v18", suite, tid, sid, bool(lat), sm, hor))
                resp = recv_msg(env_proc)
                assert resp is not None and resp[0] == "obs"
                _, c1, c2, s8, lang, is_done, step = resp
                tok_out = tok([lang + "\n"], max_length=pol.config.tokenizer_max_length, truncation=True, padding="max_length", return_tensors="pt")
                lt, lm = tok_out["input_ids"].to(dev), tok_out["attention_mask"].to(dev, dtype=torch.bool)
                ok_ep, meta = is_done, {}
                while not is_done:
                    img1 = torch.from_numpy(np.frombuffer(c1, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                    img2 = torch.from_numpy(np.frombuffer(c2, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                    st_t = (torch.tensor(s8, dtype=torch.float32).unsqueeze(0).to(dev) - s_mean) / s_std
                    out = runner.infer_v23({"observation.images.camera1": img1, "observation.images.camera2": img2, "observation.state": st_t, "observation.language.tokens": lt, "observation.language.attention_mask": lm}, method=infer_method, exec_horizon=24)
                    send_msg(env_proc, ("step_bifurc_v18", out["chunk"].tolist(), None))
                    resp = recv_msg(env_proc)
                    assert resp is not None
                    if resp[0] == "done":
                        _, ok_ep, step, meta = resp
                        is_done = True
                    else:
                        _, c1, c2, s8, _, is_done, step = resp
                rows.append({"split": split, "mode": mode, "seed": seed, "beta_exp": be, "sched_mode": sm, "suite": suite, "task_id": tid, "state_id": sid, "is_latched": int(lat), "success": int(ok_ep), "steps": int(step), **meta})
            out_path.write_text(json.dumps(rows, indent=1))
            if len(rows) % 24 == 0 or smoke:
                print(f"[G3-Rep s{shard_id:02d} GPU{gpu_id}] {len(rows)} rows | {mode} b={be} {sm} lat={lat} succ={rows[-1]['success']}", flush=True)
    finally:
        try:
            send_msg(env_proc, ("close",))
        except Exception:
            pass
        env_proc.wait(timeout=15)


def aggregate(smoke: bool):
    files = [OUT_DIR / "smoke_shard.json"] if smoke else sorted(OUT_DIR.glob("shard_*.json"))
    all_rows = [r for f in files if f.exists() for r in json.loads(f.read_text())]
    summary = {}
    for g in sorted({(r["split"], r["mode"], r["beta_exp"], r["sched_mode"]) for r in all_rows}):
        sp, mode, be, sm = g
        rs = [r for r in all_rows if (r["split"], r["mode"], r["beta_exp"], r["sched_mode"]) == g]
        k = f"{sp}|{mode}|b{be:.2f}|{sm}"
        if sp == "bifurc":
            rn, rl = [r for r in rs if r["is_latched"] == 0], [r for r in rs if r["is_latched"] == 1]
            by_s = {s_i: 100.0 * sum(r["success"] for r in rl if r["seed"] == s_i) / max(1, sum(1 for r in rl if r["seed"] == s_i)) for s_i in sorted({r["seed"] for r in rl})}
            summary[k] = {"n": len(rs), "norm_pct": round(100.0 * sum(r["success"] for r in rn) / max(1, len(rn)), 2), "latched_mean_pct": round(float(np.mean(list(by_s.values()))), 2) if by_s else 0.0, "latched_sd_pct": round(float(np.std(list(by_s.values()))), 2) if len(by_s) > 1 else 0.0, "all_pct": round(100.0 * sum(r["success"] for r in rs) / max(1, len(rs)), 2)}
        else:
            by_s = {s_i: 100.0 * sum(r["success"] for r in rs if r["seed"] == s_i) / max(1, sum(1 for r in rs if r["seed"] == s_i)) for s_i in sorted({r["seed"] for r in rs})}
            summary[k] = {"n": len(rs), "nom40_mean_pct": round(float(np.mean(list(by_s.values()))), 2), "nom40_sd_pct": round(float(np.std(list(by_s.values()))), 2) if len(by_s) > 1 else 0.0}
    out_p = OUT_DIR / ("smoke_summary.json" if smoke else "summary.json")
    out_p.write_text(json.dumps(summary, indent=2))
    print(f"[G3-Rep] {len(all_rows)} rows -> {out_p}\n{json.dumps(summary, indent=2)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-one", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--worker-shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=16)
    args = ap.parse_args()
    if args.train_one:
        ckpt_name = (f"smoke_ckpt_{args.train_one}_s{args.seed}" if args.smoke else f"ckpt_{args.train_one}_s{args.seed}")
        train_one(args.train_one, args.seed, args.gpu_id, args.steps, 8, OUT_DIR / ckpt_name)
        return
    if args.worker_shard >= 0:
        run_eval_shard(args.worker_shard, args.num_shards, args.gpu_id, args.smoke)
        return
    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if args.smoke:
        p1 = subprocess.Popen([sys.executable, __file__, "--train-one", "frozen4_causal_flow", "--seed", "0", "--steps", "40", "--gpu-id", str(gpu_list[0]), "--smoke"])
        p2 = subprocess.Popen([sys.executable, __file__, "--train-one", "unmasked_l1", "--seed", "0", "--steps", "40", "--gpu-id", str(gpu_list[-1]), "--smoke"])
        assert p1.wait() == 0 and p2.wait() == 0
        run_eval_shard(0, 1, gpu_list[0], True)
        aggregate(True)
        return
    train_jobs = [(m, s) for m in ("frozen4_causal_flow", "unmasked_l1") for s in (0, 1, 2)]
    for i in range(0, len(train_jobs), len(gpu_list)):
        wave = train_jobs[i : i + len(gpu_list)]
        procs = [subprocess.Popen([sys.executable, __file__, "--train-one", m, "--seed", str(s), "--steps", str(args.steps), "--gpu-id", str(gpu_list[idx])]) for idx, (m, s) in enumerate(wave)]
        rcs = [p.wait() for p in procs]
        assert all(rc == 0 for rc in rcs), f"G3-Rep train wave failed: {rcs}"
    num_shards = len(gpu_list) * args.workers_per_gpu
    procs = [subprocess.Popen([sys.executable, __file__, "--worker-shard", str(sh), "--num-shards", str(num_shards), "--gpu-id", str(gpu_list[sh % len(gpu_list)])]) for sh in range(num_shards)]
    rcs = [p.wait() for p in procs]
    assert all(rc == 0 for rc in rcs), f"G3-Rep eval shards failed: {rcs}"
    aggregate(False)


if __name__ == "__main__":
    main()
