#!/usr/bin/env python3
"""
Exp 4: Fine-tuning Real Modern Flow-Matching VLA (SmolVLA-450M) on SelfTest-1040
Compares 3 masking objectives (unmasked, role_only, full_causal) with frozen VLM vision encoder
and evaluates all 3 in live MuJoCo LIBERO (open-loop vs contact-triggered Event Replan).
Fits in ~10-12 GB VRAM on 1x RTX 3090 (so all 3 conditions train in parallel on GPUs 0, 1, 2!).
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import struct
import pickle
import sys
import time
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
OUT_DIR = ROOT / "outputs/v13_supplement/exp4_smolvla_selftest"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "cache/hf_models/smolvla_libero"
DATA_DIR = ROOT / "cache/selftest40_smolvla"

os.environ["HF_HOME"] = str(ROOT / "cache/hf_home")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch_home")
os.environ["TMPDIR"] = str(ROOT / "cache/tmp")
os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache/xdg_cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"
sys.path.insert(0, str(ROOT / "cache/pydeps_smolvla"))
sys.path.insert(0, str(ROOT / "external/lerobot/src"))

def build_chunk_and_weights(ep: dict, fb_steps: np.ndarray, t_raw: int, chunk_size: int, mode: str) -> tuple[int, np.ndarray, np.ndarray]:
    role_mask = ep["role_mask"]
    actions = ep["actions"]
    T = len(actions)
    if mode == "unmasked":
        t = int(np.clip(t_raw, 0, max(0, T - 2)))
        a_chunk = np.zeros((chunk_size, 7), dtype=np.float32)
        avail = min(chunk_size, T - t)
        a_chunk[:avail] = actions[t:t + avail]
        if avail < chunk_size:
            a_chunk[avail:] = np.array([0, 0, 0, 0, 0, 0, float(actions[T - 1, 6])], dtype=np.float32)
        w_chunk = np.ones(chunk_size, dtype=np.float32)
        if avail < chunk_size:
            w_chunk[avail:] = 0.15
        return t, a_chunk, w_chunk

    # role_only and causal: never start inside context_only (role_mask == 0)
    if role_mask[t_raw] > 0.5:
        t = t_raw
    else:
        later_valid = np.flatnonzero(role_mask[t_raw:] > 0.5)
        if len(later_valid) > 0:
            t = int(t_raw + later_valid[0])
        else:
            all_valid = np.flatnonzero(role_mask > 0.5)
            t = int(all_valid[-1]) if len(all_valid) > 0 else int(np.clip(t_raw, 0, max(0, T - 2)))
    t = int(np.clip(t, 0, max(0, T - 1)))

    # Stop monotonically if entering a context_only range (fixes hole-in-middle across [r0, r1))
    j_cut = min(chunk_size, T - t)
    for j in range(j_cut):
        if role_mask[t + j] < 0.5:
            j_cut = j
            break
    if mode == "causal" and len(fb_steps) > 0:
        later_fb = fb_steps[fb_steps > t]
        if len(later_fb) > 0:
            j_fb = int(later_fb[0] - t)
            if j_fb < j_cut:
                j_cut = j_fb
    j_cut = max(1, j_cut)

    # Keep natural future trajectory in x_t for smooth ODE conditioning, while setting exact w_chunk=0.0
    # past j_cut (with fixed 1/chunk_size normalization so there is neither braking bias nor 1/num_valid spike!)
    a_chunk = np.zeros((chunk_size, 7), dtype=np.float32)
    avail = min(chunk_size, T - t)
    a_chunk[:avail] = actions[t:t + avail]
    if avail < chunk_size:
        a_chunk[avail:] = actions[T - 1]
    w_chunk = np.zeros(chunk_size, dtype=np.float32)
    w_chunk[:j_cut] = 1.0
    return t, a_chunk, w_chunk

def train_one_condition(mode: str, gpu_id: int, manifest_path: Path, steps: int, batch_size: int, lr: float, out_ckpt: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dev = torch.device("cuda:0")
    t0 = time.time()
    pol = SmolVLAPolicy.from_pretrained(str(MODEL_DIR)).to(dev)
    # Freeze VLM backbone + vision encoder; train action expert + projections (clean, fast, ~10 GB VRAM on 3090!)
    for name, p in pol.named_parameters():
        if "vlm_with_expert.vlm" in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
    n_train = sum(p.numel() for p in pol.parameters() if p.requires_grad)
    print(f"[Exp4 {mode} GPU{gpu_id}] Trainable action-expert params: {n_train/1e6:.2f}M", flush=True)

    pre_stats = load_file(str(MODEL_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_stats = load_file(str(MODEL_DIR / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    s_mean = pre_stats["observation.state.mean"].to(dev, dtype=torch.float32)
    s_std = pre_stats["observation.state.std"].to(dev, dtype=torch.float32).clamp_min(1e-8)
    a_mean = post_stats["action.mean"].to(dev, dtype=torch.float32)
    a_std = post_stats["action.std"].to(dev, dtype=torch.float32).clamp_min(1e-8)
    tok = pol.model.vlm_with_expert.processor.tokenizer
    chunk_size = int(pol.config.chunk_size)

    manifest = json.loads(manifest_path.read_text())
    # Cache tokenized language prompts
    lang_cache = {}
    for item in manifest:
        lang = item["language"]
        if lang not in lang_cache:
            to = tok([lang + "\n"], max_length=pol.config.tokenizer_max_length, truncation=True, padding="max_length", return_tensors="pt")
            lang_cache[lang] = (to["input_ids"][0], to["attention_mask"][0].bool())

    import math
    p0_map = {name: p.detach().clone() for name, p in pol.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    warmup = max(1, steps // 10)
    def lr_lambda(s: int) -> float:
        if s < warmup:
            return float(s + 1) / float(warmup)
        prog = float(s - warmup) / float(max(1, steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng = random.Random(20261121)
    act_dim = int(pol.config.action_feature.shape[0])
    pol.train()
    losses = []
    active_pool = []

    for step in range(1, steps + 1):
        if (step - 1) % 8 == 0 or not active_pool:
            active_pool = []
            for item in rng.sample(manifest, min(16, len(manifest))):
                d = np.load(item["npz"])
                active_pool.append((item, {k: d[k] for k in d.files}))
        img1s, img2s, states, acts, ltoks, lmasks, cmasks = [], [], [], [], [], [], []
        for b_idx in range(batch_size):
            item, ep = rng.choice(active_pool)
            T = len(ep["actions"])
            # Focus half of draws on the active grasp/probe window (first 160 steps) and half across full horizon
            if b_idx % 2 == 0:
                t_raw = rng.randint(0, max(0, min(T - 2, 160)))
            else:
                t_raw = rng.randint(0, max(0, T - 2))
            fb_arr = np.asarray(item["fb_steps"], dtype=np.int32) if "fb_steps" in item else ep["fb_steps"]
            t, a_chunk, w_chunk = build_chunk_and_weights(ep, fb_arr, t_raw, chunk_size, mode)
            img1s.append(ep["cam1"][t])
            img2s.append(ep["cam2"][t])
            states.append(ep["proprio"][t])
            acts.append(a_chunk)
            cmasks.append(w_chunk)
            lt, lm = lang_cache[item["language"]]
            ltoks.append(lt)
            lmasks.append(lm)

        img1_t = torch.from_numpy(np.stack(img1s)).permute(0, 3, 1, 2).to(dev, dtype=torch.float32) / 255.0
        img2_t = torch.from_numpy(np.stack(img2s)).permute(0, 3, 1, 2).to(dev, dtype=torch.float32) / 255.0
        st_t = (torch.from_numpy(np.stack(states)).to(dev, dtype=torch.float32) - s_mean) / s_std
        act_t = (torch.from_numpy(np.stack(acts)).to(dev, dtype=torch.float32) - a_mean) / a_std
        cmask_t = torch.from_numpy(np.stack(cmasks)).to(dev, dtype=torch.float32)  # (B, chunk_size)
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
            losses_raw = pol.model.forward(
                images, img_masks,
                batch["observation.language.tokens"],
                batch["observation.language.attention_mask"],
                p_state, p_actions, None, None,
            )[:, :, :act_dim]
            loss = (losses_raw.mean(dim=-1) * cmask_t).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in pol.parameters() if p.requires_grad], 1.0)
        opt.step()
        cur_lr = float(sched.get_last_lr()[0])
        alpha_anc = min(0.05, cur_lr * 160.0)
        with torch.no_grad():
            for name, p in pol.named_parameters():
                if p.requires_grad:
                    p.lerp_(p0_map[name].to(p.dtype), alpha_anc)
        sched.step()
        losses.append(float(loss.item()))
        if step % max(1, steps // 5) == 0 or step == 1:
            print(f"[Exp4 {mode} GPU{gpu_id}] step {step}/{steps} loss={np.mean(losses[-20:]):.4f} VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    from safetensors.torch import save_file
    out_ckpt.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in pol.state_dict().items()}
    save_file(sd, str(out_ckpt / "model.safetensors"))
    for fn in ["config.json", "train_config.json", "policy_preprocessor.json", "policy_postprocessor.json",
               "policy_preprocessor_step_5_normalizer_processor.safetensors", "policy_postprocessor_step_0_unnormalizer_processor.safetensors"]:
        src_f = MODEL_DIR / fn
        if src_f.exists():
            (out_ckpt / fn).write_bytes(src_f.read_bytes())
    print(f"[Exp4 {mode} GPU{gpu_id}] Saved {out_ckpt} in {time.time()-t0:.1f}s (peak VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB)", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", choices=["unmasked", "role_only", "causal"], default="causal")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--lr", type=float, default=8e-6)
    args = ap.parse_args()

    mpath = DATA_DIR / ("manifest_smoke.json" if args.smoke else "manifest_40tasks.json")
    steps = 4 if args.smoke else args.steps
    bs = 4 if args.smoke else 16
    out_ckpt = OUT_DIR / ("smoke_ckpt" if args.smoke else f"ckpt_{args.mode}")
    train_one_condition(args.mode, args.gpu, mpath, steps=steps, batch_size=bs, lr=args.lr, out_ckpt=out_ckpt)

if __name__ == "__main__":
    main()
