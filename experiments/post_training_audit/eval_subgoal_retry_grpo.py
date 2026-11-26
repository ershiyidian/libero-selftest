#!/usr/bin/env python3
"""
Exp 3 (v14): LIBERO-Subgoal-150 — Multi-Constraint Embodied Kitchen Subgoal Planning,
Online GRPO Entrance Lock, Layer-Selective Interpolation, and Opening-Subgoal Handoff.
Uses local Qwen3-1.7B (<12 GB VRAM on 1x RTX 3090) with KV-cached generation.
Strictly keeps all caches/temp under /data/guest/vla_branch_lab/cache/.
"""
from __future__ import annotations
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
OUT_DIR = ROOT / "outputs/v13_supplement/exp3_subgoal_retry"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(ROOT / "cache/hf_home")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch_home")
os.environ["TMPDIR"] = str(ROOT / "cache/tmp")
os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache/xdg_cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ITEMS = ["bowl", "mug", "butter", "cream", "ketchup", "cheese", "soup", "milk", "cookies", "sauce", "ramekin", "pudding"]
POTS  = ["moka", "kettle", "skillet", "teapot"]

def find_local_qwen3_1p7b() -> str:
    base = Path("/data/guest/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots")
    snaps = sorted([p for p in base.iterdir() if p.is_dir()])
    assert snaps, f"No local Qwen3-1.7B snapshot in {base}"
    return str(snaps[-1])

def generate_scene(seed: int) -> dict | None:
    rng = random.Random(seed)
    mode = seed % 3
    a, b, obs, d1, d2 = rng.sample(ITEMS, 5)
    pot = rng.choice(POTS)
    if mode == 0:
        obs_t = rng.choice(["a", "pot"])
        blocked_item = a if obs_t == "a" else pot
        init = (0, 0, 0, 0, 0, 0, 0)
        def step_fn(st):
            to, bo, oc, ad, bd, ps, so = st
            res = []
            if not to: res.append(("open(top_drawer)", (1, bo, oc, ad, bd, ps, so)))
            if to and ad and not bd: res.append(("close(top_drawer)", (0, bo, oc, ad, bd, ps, so)))
            if not bo and not to: res.append(("open(bot_drawer)", (to, 1, oc, ad, bd, ps, so)))
            if not oc: res.append((f"clear({obs},tray)", (to, bo, 1, ad, bd, ps, so)))
            if to and not ad and (oc or obs_t != "a"): res.append((f"place({a},top_drawer)", (to, bo, oc, 1, bd, ps, so)))
            if bo and not to and not bd: res.append((f"place({b},bot_drawer)", (to, bo, oc, ad, 1, ps, so)))
            if not ps and (oc or obs_t != "pot"): res.append((f"place({pot},stove)", (to, bo, oc, ad, bd, 1, so)))
            if ps and not so: res.append(("turn_on(stove)", (to, bo, oc, ad, bd, ps, 1)))
            return res
        goal_fn = lambda st: (st[3] == 1 and st[4] == 1 and st[6] == 1)
        clauses = [
            f"Store {a} in top_drawer and {b} in bot_drawer.",
            f"Heat {pot} on stove (place before turn_on).",
            "Both drawers start closed; open top_drawer blocks bot_drawer.",
            f"Obstacle: {obs} blocks {blocked_item} (clear to tray).",
            f"Distractors on table: {d1}, {d2}.",
        ]
    elif mode == 1:
        c1, c2 = rng.sample(["wooden_cabinet", "white_cabinet", "fridge_rack"], 2)
        obs_t = rng.choice(["c1", "b"])
        blocked_item = c1 if obs_t == "c1" else b
        extra_cond = f" or {b} is in {c2}" if obs_t == "c1" else ""
        init = (0, 0, 0, 0, 0, 0)
        def step_fn(st):
            c1o, c2o, oc, ad, bd, pd = st
            res = []
            if not c1o and (oc or obs_t != "c1"): res.append((f"open({c1})", (1, c2o, oc, ad, bd, pd)))
            if not c2o: res.append((f"open({c2})", (c1o, 1, oc, ad, bd, pd)))
            if not oc: res.append((f"clear({obs},tray)", (c1o, c2o, 1, ad, bd, pd)))
            if c1o and not ad: res.append((f"place({a},{c1})", (c1o, c2o, oc, 1, bd, pd)))
            if c2o and not bd and (oc or obs_t != "b"): res.append((f"place({b},{c2})", (c1o, c2o, oc, ad, 1, pd)))
            if not pd and ad: res.append((f"stack({d1},rack)", (c1o, c2o, oc, ad, bd, 1)))
            elif not pd and bd and obs_t == "c1": res.append((f"stack({d1},rack)", (c1o, c2o, oc, ad, bd, 1)))
            return res
        goal_fn = lambda st: (st[3] == 1 and st[4] == 1 and st[5] == 1)
        clauses = [
            f"Put {a} into {c1}, put {b} into {c2}, and stack {d1} onto rack (only after {a} is in {c1}{extra_cond}).",
            f"Closed fixtures: {c1}, {c2}.",
            f"Obstacle: {obs} blocks {blocked_item} (clear to tray first).",
            f"Ignore {d2}, {pot}.",
        ]
    else:
        c1, c2 = rng.sample(["wooden_cabinet", "white_cabinet", "bot_drawer"], 2)
        init = (0, 0, 0, 0, 0, 0)
        def step_fn(st):
            c1o, c2o, wp, ad, bd, pd = st
            res = []
            if not c1o: res.append((f"open({c1})", (1, c2o, wp, ad, bd, pd)))
            if not c2o: res.append((f"open({c2})", (c1o, 1, wp, ad, bd, pd)))
            if not wp: res.append((f"clear({obs},tray)", (c1o, c2o, 1, ad, bd, pd)))
            if c1o and wp and not ad: res.append((f"place({a},{c1})", (c1o, c2o, wp, 1, bd, pd)))
            if c2o and not bd: res.append((f"place({b},{c2})", (c1o, c2o, wp, ad, 1, pd)))
            if not pd: res.append((f"place({pot},stove)", (c1o, c2o, wp, ad, bd, 1)))
            return res
        goal_fn = lambda st: (st[3] == 1 and st[4] == 1 and st[5] == 1)
        clauses = [
            f"Goal: {a} in {c1}, {b} in {c2}, and {pot} on stove.",
            f"Closed containers: {c1}, {c2}.",
            f"Obstacle: {obs} blocks {a} (clear to tray before placing {a}).",
            f"Distractors: {d1}, {d2}.",
        ]
    rng.shuffle(clauses)
    q = deque([(init, [])])
    seen = {init: 0}
    valid = []
    min_l = None
    while q:
        st, path = q.popleft()
        if min_l is not None and len(path) > min_l:
            break
        if goal_fn(st):
            min_l = len(path)
            valid.append(path)
            continue
        for act, nst in step_fn(st):
            if seen.get(nst, 99) >= len(path) + 1:
                seen[nst] = len(path) + 1
                q.append((nst, path + [act]))
    fams = sorted({p[0] for p in valid})
    if len(valid) < 6 or len(fams) < 2:
        return None
    fam_groups = {f: ["; ".join(p) for p in valid if p[0] == f] for f in fams}
    prompt = "Kitchen Scene: " + " ".join(clauses) + " Output one valid minimal plan separated by '; '.\nPlan:"
    return {
        "scene_id": seed,
        "prompt": prompt,
        "valid_plans": ["; ".join(p) for p in valid],
        "fam_groups": fam_groups,
        "families": fams,
        "min_len": min_l,
    }

def build_dataset(n_train: int, n_test: int):
    train, test = [], []
    seen_p = set()
    s = 1000
    while len(train) < n_train:
        sc = generate_scene(s)
        s += 1
        if sc and sc["prompt"] not in seen_p:
            seen_p.add(sc["prompt"])
            train.append(sc)
    s = 500000
    while len(test) < n_test:
        sc = generate_scene(s)
        s += 1
        if sc and sc["prompt"] not in seen_p:
            seen_p.add(sc["prompt"])
            test.append(sc)
    return train, test

def parse_plan(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    parts = [p.strip() for p in line.split(";") if p.strip()]
    return "; ".join(parts)

def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    k = min(k, n)
    if n == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return float(1.0 - math.comb(n - c, k) / math.comb(n, k))

def set_gen_mode(model):
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True

def set_train_mode(model):
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

def train_sft(model, tok, scenes: list[dict], multi_branch: bool, steps: int, lr: float, dev):
    import torch
    set_train_mode(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(42 + int(multi_branch))
    pool_scenes = scenes[:min(len(scenes), 160)]
    losses = []
    for step in range(1, steps + 1):
        batch = rng.sample(pool_scenes, min(6, len(pool_scenes)))
        opt.zero_grad()
        tot = 0.0
        for sc in batch:
            if multi_branch:
                f = rng.choice(sc["families"])
                tgt = rng.choice(sc["fam_groups"][f])
            else:
                f0 = sc["families"][0]
                tgt = sc["fam_groups"][f0][0]
            full = sc["prompt"] + " " + tgt + tok.eos_token
            pids = tok(sc["prompt"], return_tensors="pt")["input_ids"]
            enc = tok(full, return_tensors="pt", truncation=True, max_length=192).to(dev)
            lab = enc["input_ids"].clone()
            lab[:, :pids.shape[1]] = -100
            loss = model(**enc, labels=lab).loss / len(batch)
            loss.backward()
            tot += float(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(tot)
        if step % max(1, steps // 4) == 0:
            print(f"  [SFT multi={multi_branch}] step {step}/{steps} loss={np.mean(losses[-10:]):.4f} VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    return model

def train_grpo(model, tok, scenes: list[dict], steps: int, group_size: int, lr: float, dev):
    import torch
    import torch.nn.functional as F
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(2026)
    for step in range(1, steps + 1):
        sc = rng.choice(scenes)
        vs = set(sc["valid_plans"])
        set_gen_mode(model)
        enc = tok(sc["prompt"], return_tensors="pt").to(dev)
        pl = enc["input_ids"].shape[1]
        with torch.no_grad():
            g = model.generate(
                **enc, max_new_tokens=64, do_sample=True, temperature=0.9, top_p=0.95,
                num_return_sequences=group_size, pad_token_id=tok.eos_token_id,
            )
        rews = np.array([1.0 if parse_plan(tok.decode(g[i, pl:], skip_special_tokens=True)) in vs else 0.0 for i in range(group_size)], dtype=np.float32)
        if rews.std() < 1e-6:
            continue
        adv = (rews - rews.mean()) / (rews.std() + 1e-6)
        set_train_mode(model)
        opt.zero_grad()
        for i in range(group_size):
            seq = g[i:i+1]
            logits = model(seq).logits[:, pl - 1:-1, :]
            tt = seq[:, pl:]
            lp = F.log_softmax(logits, dim=-1).gather(-1, tt.unsqueeze(-1)).squeeze(-1)
            msk = (tt != tok.eos_token_id).float()
            loss_i = -float(adv[i]) * (lp * msk).sum() / msk.sum().clamp_min(1.0) / group_size
            loss_i.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 5) == 0:
            print(f"  [GRPO] step {step}/{steps} mean_r={rews.mean():.3f} VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    return model

def apply_block_interp(model, rl_cpu: dict, sft_cpu: dict, layer_range: tuple[int, int] | None, alpha: float, include_head: bool = False):
    import torch
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in rl_cpu:
                continue
            a = 0.0
            if layer_range is not None:
                lo, hi = layer_range
                if "model.layers." in name:
                    lid = int(name.split("model.layers.")[1].split(".")[0])
                    if lo <= lid <= hi:
                        a = alpha
                elif include_head and ("model.norm" in name or "lm_head" in name):
                    a = alpha
            elif alpha == 1.0:
                a = 1.0
            w = (1.0 - a) * rl_cpu[name].float() + a * sft_cpu[name].float()
            param.copy_(w.to(param.device, dtype=param.dtype))

def sample_plans(model, tok, test_scenes: list[dict], k_samples: int, dev) -> list[list[str]]:
    import torch
    set_gen_mode(model)
    all_preds = []
    bs = min(16, k_samples)
    for sc in test_scenes:
        enc = tok(sc["prompt"], return_tensors="pt").to(dev)
        pl = enc["input_ids"].shape[1]
        preds = []
        with torch.no_grad():
            for _ in range(0, k_samples, bs):
                cur = min(bs, k_samples - len(preds))
                g = model.generate(
                    **enc, max_new_tokens=64, do_sample=True, temperature=0.8, top_p=0.95,
                    num_return_sequences=cur, pad_token_id=tok.eos_token_id,
                )
                for i in range(cur):
                    preds.append(parse_plan(tok.decode(g[i, pl:], skip_special_tokens=True)))
        all_preds.append(preds)
    return all_preds

def sample_opening_handoff(model_rl, tok, test_scenes: list[dict], sft_preds_per_scene: list[list[str]], dev) -> list[list[str]]:
    import torch
    set_gen_mode(model_rl)
    handoff_all = []
    for sc, sp in zip(test_scenes, sft_preds_per_scene):
        pl = tok(sc["prompt"], return_tensors="pt")["input_ids"].shape[1]
        cur_out = []
        # Group identical opening primitives b_1 to batch generation fast!
        by_b1 = {}
        for idx, s_plan in enumerate(sp):
            b1 = s_plan.split(";")[0].strip() if s_plan else ""
            by_b1.setdefault(b1, []).append(idx)
        indexed_res = [None] * len(sp)
        with torch.no_grad():
            for b1, idxs in by_b1.items():
                pref = sc["prompt"] + (" " + b1 + ";" if b1 else "")
                enc = tok(pref, return_tensors="pt").to(dev)
                g = model_rl.generate(
                    **enc, max_new_tokens=54, do_sample=True, temperature=0.5, top_p=0.95,
                    num_return_sequences=len(idxs), pad_token_id=tok.eos_token_id,
                )
                for j, orig_i in enumerate(idxs):
                    indexed_res[orig_i] = parse_plan(tok.decode(g[j, pl:], skip_special_tokens=True))
        handoff_all.append(indexed_res)
    return handoff_all

def score_predictions(test_scenes: list[dict], preds_per_scene: list[list[str]]) -> dict:
    p1s, p8s, pks, fcovs, pcovs, ents, retry3 = [], [], [], [], [], [], []
    for sc, preds in zip(test_scenes, preds_per_scene):
        k = len(preds)
        vs = set(sc["valid_plans"])
        vp = [p for p in preds if p in vs]
        c = len(vp)
        p1s.append(unbiased_pass_at_k(k, c, 1))
        p8s.append(unbiased_pass_at_k(k, c, 8))
        pks.append(unbiased_pass_at_k(k, c, k))
        pcovs.append(len(set(vp)) / min(k, len(vs)))
        fams = [p.split(";")[0].strip() for p in vp]
        fcovs.append(len(set(fams)) / len(sc["families"]))
        if fams:
            _, cnts = np.unique(fams, return_counts=True)
            pr = cnts / cnts.sum()
            ents.append(float(-np.sum(pr * np.log(pr + 1e-12))))
            all_open = [p.split(";")[0].strip() for p in preds if p]
            dom = max(set(all_open), key=all_open.count) if all_open else fams[0]
            n_alt = sum(1 for p in vp if p.split(";")[0].strip() != dom)
            retry3.append(unbiased_pass_at_k(k, n_alt, 3))
        else:
            ents.append(0.0)
            retry3.append(0.0)
    return {
        "pass1": round(float(np.mean(p1s)), 4),
        "pass8": round(float(np.mean(p8s)), 4),
        "pass_k": round(float(np.mean(pks)), 4),
        "family_cov": round(float(np.mean(fcovs)), 4),
        "plan_cov": round(float(np.mean(pcovs)), 4),
        "open_entropy": round(float(np.mean(ents)), 4),
        "blocked_retry3": round(float(np.mean(retry3)), 4),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    n_train = 80 if args.smoke else 500
    n_test = 8 if args.smoke else 150
    sft_steps = 12 if args.smoke else 42
    grpo_steps = 15 if args.smoke else 140
    k_eval = 8 if args.smoke else 32

    train_sc, test_sc = build_dataset(n_train, n_test)
    (OUT_DIR / "benchmark_meta.json").write_text(json.dumps({
        "n_train": len(train_sc), "n_test": len(test_sc),
        "mean_valid_plans_test": round(float(np.mean([len(s["valid_plans"]) for s in test_sc])), 2),
        "mean_families_test": round(float(np.mean([len(s["families"]) for s in test_sc])), 2),
    }, indent=2))

    mpath = find_local_qwen3_1p7b()
    print(f"[Exp3 v14] Loading {mpath} on GPU {args.gpu}...", flush=True)
    tok = AutoTokenizer.from_pretrained(mpath)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.bfloat16).to(dev)
    base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    results = {}
    # 1. Multi-branch Warm-Start SFT -> Online GRPO (Primary Entrance-Lock & Mitigation Study)
    train_sft(model, tok, train_sc, multi_branch=True, steps=sft_steps, lr=1.5e-5, dev=dev)
    sft_multi_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sft_multi_preds = sample_plans(model, tok, test_sc, k_eval, dev)
    results["sft_multi_warm"] = score_predictions(test_sc, sft_multi_preds)
    print("[Exp3] sft_multi_warm:", results["sft_multi_warm"], flush=True)

    train_grpo(model, tok, train_sc, steps=grpo_steps, group_size=8, lr=1.0e-5, dev=dev)
    grpo_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    results["grpo_ctrl"] = score_predictions(test_sc, sample_plans(model, tok, test_sc, k_eval, dev))
    print("[Exp3] grpo_ctrl:", results["grpo_ctrl"], flush=True)

    # Opening-Subgoal Handoff (SFT opening primitive b_1 -> GRPO suffix completion b_2..b_T)
    handoff_preds = sample_opening_handoff(model, tok, test_sc, sft_multi_preds, dev)
    results["opening_handoff"] = score_predictions(test_sc, handoff_preds)
    print("[Exp3] opening_handoff:", results["opening_handoff"], flush=True)

    # Matched Early (L0-9) vs Mid-Late+Unembed (L14-27) Weight Interpolations
    for cname, lrange, alpha, inc_h in [
        ("early_L0_9_a50", (0, 9), 0.50, False),
        ("early_L0_9_a65", (0, 9), 0.65, False),
        ("midlate_L14_27_a50", (14, 27), 0.50, True),
        ("midlate_L14_27_a65", (14, 27), 0.65, True),
    ]:
        apply_block_interp(model, grpo_sd, sft_multi_sd, lrange, alpha, include_head=inc_h)
        results[cname] = score_predictions(test_sc, sample_plans(model, tok, test_sc, k_eval, dev))
        print(f"[Exp3] {cname}:", results[cname], flush=True)

    # 2. Single-Plan Warm-Start SFT (M=1) -> Online GRPO (Demonstrates SFT Support Ceiling)
    model.load_state_dict({k: v.to(dev) for k, v in base_sd.items()})
    train_sft(model, tok, train_sc, multi_branch=False, steps=sft_steps, lr=1.5e-5, dev=dev)
    results["sft_single_m1"] = score_predictions(test_sc, sample_plans(model, tok, test_sc, k_eval, dev))
    print("[Exp3] sft_single_m1:", results["sft_single_m1"], flush=True)

    train_grpo(model, tok, train_sc, steps=grpo_steps, group_size=8, lr=1.0e-5, dev=dev)
    results["sft_single_m1_grpo"] = score_predictions(test_sc, sample_plans(model, tok, test_sc, k_eval, dev))
    print("[Exp3] sft_single_m1_grpo:", results["sft_single_m1_grpo"], flush=True)

    out_name = "smoke_summary.json" if args.smoke else "summary.json"
    (OUT_DIR / out_name).write_text(json.dumps({
        "results": results, "elapsed_s": round(time.time() - t0, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }, indent=2))
    print(f"[Exp3 {'SMOKE' if args.smoke else 'FULL'} OK] elapsed={time.time()-t0:.1f}s VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

if __name__ == "__main__":
    main()
