#!/usr/bin/env python3
"""
Stage 5A (G5A — DRIVE-GRPO [Oct 2026, arXiv:2610.09943] vs. Vanilla GRPO vs. Entropy-GRPO vs. Opening Handoff):
  Trains 4 Qwen3-1.7B RL variants in parallel on 4x RTX 3090 from a shared Multi-Branch Warm SFT base:
    - GPU 0: vanilla_grpo   (standard outcome GRPO)
    - GPU 1: entropy_grpo   (token-entropy regularized GRPO, beta_ent=0.04)
    - GPU 2: drive_grpo     (DRIVE success-conditioned intra-group diversity reward, lambda_div=0.35)
    - GPU 3: drive_kl_grpo  (DRIVE + KL anchor to multi-branch SFT, beta_kl=0.04)
  Evaluates all 4 + sft_multi_warm + opening_handoff_vanilla + opening_handoff_drive on:
    1. All 150 held-out LIBERO-Subgoal-150 scenes (k=32, 33,600 rollouts)
    2. 10 MuJoCo-coupled dual-subgoal kitchen scenes -> exports stage6_planner_plans.json for Stage 6!
  Total wall-clock on 4 GPUs: ~65 minutes.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab")
OUT_DIR = ROOT / "outputs/v23_16h_suite/g5a_drive_grpo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "tools/v13_supplement"))
from exp3_subgoal_retry_grpo import (
    build_dataset,
    find_local_qwen3_1p7b,
    parse_plan,
    sample_opening_handoff,
    sample_plans,
    score_predictions,
    set_gen_mode,
    set_train_mode,
    train_sft,
    unbiased_pass_at_k,
)


def _set_env(gpu_id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = str(ROOT / "cache/hf_home")
    os.environ["TORCH_HOME"] = str(ROOT / "cache/torch_home")
    os.environ["TMPDIR"] = str(ROOT / "cache/tmp")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache/xdg_cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_stage6_mujoco_scenes() -> list[dict]:
    """10 deterministic multi-subgoal scenes paired with real LIBERO MuJoCo tasks for Stage 6."""
    _, test_scenes = build_dataset(120, 150)
    # Pick 10 scenes with >= 2 distinct opening families for clean blocked-subgoal physical retry
    picked = [sc for sc in test_scenes if len(sc["families"]) >= 2][:10]
    return picked


def compute_drive_rewards(plans: list[str], valid_set: set[str], lambda_div: float) -> np.ndarray:
    """DRIVE (arXiv:2610.09943): success-conditioned intra-group behavioral diversity reward."""
    n = len(plans)
    base_r = np.array([1.0 if p in valid_set else 0.0 for p in plans], dtype=np.float32)
    if lambda_div <= 0.0 or base_r.sum() < 1.0:
        return base_r
    succ_idxs = [i for i in range(n) if base_r[i] > 0.5]
    fams = [plans[i].split(";")[0].strip() if plans[i] else "" for i in range(n)]
    tok_sets = [set(plans[i].replace(";", " ").split()) for i in range(n)]
    fam_counts = {}
    for i in succ_idxs:
        fam_counts[fams[i]] = fam_counts.get(fams[i], 0) + 1
    out = base_r.copy()
    for i in succ_idxs:
        inv_sqrt_c = 1.0 / math.sqrt(max(1, fam_counts[fams[i]]))
        if len(succ_idxs) > 1:
            dists = []
            for j in succ_idxs:
                if i == j:
                    continue
                u = len(tok_sets[i] | tok_sets[j])
                inter = len(tok_sets[i] & tok_sets[j])
                dists.append(1.0 - (inter / max(1, u)))
            mean_d = float(np.mean(dists))
        else:
            mean_d = 0.0
        out[i] = base_r[i] + lambda_div * (inv_sqrt_c + 0.5 * mean_d)
    return out


def train_grpo_variant(
    model,
    ref_model,
    tok,
    scenes: list[dict],
    variant: str,
    steps: int,
    group_size: int,
    lr: float,
    dev,
):
    import torch
    import torch.nn.functional as F

    lambda_div = 0.35 if variant in ("drive_grpo", "drive_kl_grpo") else 0.0
    beta_ent = 0.04 if variant == "entropy_grpo" else 0.0
    beta_kl = 0.04 if variant == "drive_kl_grpo" else 0.0

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(202612 + hash(variant) % 1000)
    mean_rs, mean_divs = [], []
    for step in range(1, steps + 1):
        sc = rng.choice(scenes)
        vs = set(sc["valid_plans"])
        set_gen_mode(model)
        enc = tok(sc["prompt"], return_tensors="pt").to(dev)
        pl = enc["input_ids"].shape[1]
        with torch.no_grad():
            g = model.generate(
                **enc, max_new_tokens=64, do_sample=True, temperature=0.95, top_p=0.95,
                num_return_sequences=group_size, pad_token_id=tok.eos_token_id,
            )
        plans = [parse_plan(tok.decode(g[i, pl:], skip_special_tokens=True)) for i in range(group_size)]
        rews = compute_drive_rewards(plans, vs, lambda_div)
        mean_rs.append(float(np.mean([1.0 if p in vs else 0.0 for p in plans])))
        succ_fams = {p.split(";")[0].strip() for p in plans if p in vs}
        mean_divs.append(float(len(succ_fams)))
        if rews.std() < 1e-6:
            continue
        adv = (rews - rews.mean()) / (rews.std() + 1e-6)

        set_train_mode(model)
        opt.zero_grad()
        for i in range(group_size):
            seq = g[i : i + 1]
            logits = model(seq).logits[:, pl - 1 : -1, :]
            tt = seq[:, pl:]
            log_probs_all = F.log_softmax(logits, dim=-1)
            lp = log_probs_all.gather(-1, tt.unsqueeze(-1)).squeeze(-1)
            msk = (tt != tok.eos_token_id).float()
            denom = msk.sum().clamp_min(1.0)
            loss_i = -float(adv[i]) * (lp * msk).sum() / denom
            if beta_ent > 0.0:
                probs_all = log_probs_all.exp()
                ent_tok = -(probs_all * log_probs_all).sum(dim=-1)
                loss_i = loss_i - beta_ent * (ent_tok * msk).sum() / denom
            if beta_kl > 0.0 and ref_model is not None:
                with torch.no_grad():
                    ref_logits = ref_model(seq).logits[:, pl - 1 : -1, :]
                    ref_lp = F.log_softmax(ref_logits, dim=-1).gather(-1, tt.unsqueeze(-1)).squeeze(-1)
                kl_approx = ((lp - ref_lp) ** 2 * msk).sum() / denom
                loss_i = loss_i + beta_kl * kl_approx
            (loss_i / group_size).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 5) == 0 or step == 1:
            print(f"  [G5A {variant}] step {step}/{steps} task_r={np.mean(mean_rs[-15:]):.3f} uniq_fams={np.mean(mean_divs[-15:]):.2f} VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    return model


def run_sft_prep(gpu_id: int, smoke: bool):
    _set_env(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sft_path = OUT_DIR / ("smoke_sft_warm.pt" if smoke else "sft_multi_warm.pt")
    preds_path = OUT_DIR / ("smoke_sft_preds.json" if smoke else "sft_preds.json")
    s6_path = OUT_DIR / ("smoke_s6_sft_plans.json" if smoke else "s6_sft_plans.json")

    train_scenes, test_scenes = build_dataset(120 if smoke else 1200, 12 if smoke else 150)
    s6_scenes = build_stage6_mujoco_scenes() if not smoke else test_scenes[:3]

    mpath = find_local_qwen3_1p7b()
    dev = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(mpath, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.bfloat16, local_files_only=True).to(dev)
    sft_steps = 12 if smoke else 70
    train_sft(model, tok, train_scenes, multi_branch=True, steps=sft_steps, lr=2.5e-5, dev=dev)

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, str(sft_path))
    k_eval = 8 if smoke else 32
    sft_preds = sample_plans(model, tok, test_scenes, k_eval, dev)
    sft_metrics = score_predictions(test_scenes, sft_preds)
    preds_path.write_text(json.dumps({"metrics": sft_metrics, "preds": sft_preds}))
    s6_preds = sample_plans(model, tok, s6_scenes, 16 if smoke else 32, dev)
    s6_path.write_text(json.dumps({"scenes": s6_scenes, "preds": s6_preds}))
    print(f"[G5A SFT] Saved {sft_path} | metrics={sft_metrics}", flush=True)


def run_variant_worker(variant: str, gpu_id: int, smoke: bool):
    _set_env(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sft_path = OUT_DIR / ("smoke_sft_warm.pt" if smoke else "sft_multi_warm.pt")
    preds_path = OUT_DIR / ("smoke_sft_preds.json" if smoke else "sft_preds.json")
    s6_sft_path = OUT_DIR / ("smoke_s6_sft_plans.json" if smoke else "s6_sft_plans.json")

    train_scenes, test_scenes = build_dataset(120 if smoke else 1200, 12 if smoke else 150)
    sft_data = json.loads(preds_path.read_text())
    sft_preds = sft_data["preds"]
    s6_data = json.loads(s6_sft_path.read_text())
    s6_scenes, s6_sft_preds = s6_data["scenes"], s6_data["preds"]

    mpath = find_local_qwen3_1p7b()
    dev = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(mpath, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    sd_sft = torch.load(str(sft_path), map_location="cpu")
    model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.bfloat16, local_files_only=True).to(dev)
    model.load_state_dict(sd_sft, strict=True)

    ref_model = None
    if variant == "drive_kl_grpo":
        ref_model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.bfloat16, local_files_only=True).to(dev)
        ref_model.load_state_dict(sd_sft, strict=True)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    rl_steps = 8 if smoke else 180
    train_grpo_variant(model, ref_model, tok, train_scenes, variant, steps=rl_steps, group_size=8, lr=8e-6, dev=dev)

    k_eval = 8 if smoke else 32
    rl_preds = sample_plans(model, tok, test_scenes, k_eval, dev)
    rl_metrics = score_predictions(test_scenes, rl_preds)
    ho_preds = sample_opening_handoff(model, tok, test_scenes, sft_preds, dev)
    ho_metrics = score_predictions(test_scenes, ho_preds)

    # Also sample Stage-6 MuJoCo plans directly from this trained RL model & its Opening Handoff!
    s6_rl_preds = sample_plans(model, tok, s6_scenes, 16 if smoke else 32, dev)
    s6_ho_preds = sample_opening_handoff(model, tok, s6_scenes, s6_sft_preds, dev)

    res = {
        "variant": variant,
        "rl_metrics": rl_metrics,
        "handoff_metrics": ho_metrics,
        "s6_rl_preds": s6_rl_preds,
        "s6_ho_preds": s6_ho_preds,
    }
    out_p = OUT_DIR / (f"smoke_res_{variant}.json" if smoke else f"res_{variant}.json")
    out_p.write_text(json.dumps(res, indent=2))
    print(f"[G5A {variant}] RL={rl_metrics} | Handoff={ho_metrics}", flush=True)


def aggregate(smoke: bool):
    sft_data = json.loads((OUT_DIR / ("smoke_sft_preds.json" if smoke else "sft_preds.json")).read_text())
    s6_sft = json.loads((OUT_DIR / ("smoke_s6_sft_plans.json" if smoke else "s6_sft_plans.json")).read_text())
    variants = ["vanilla_grpo", "drive_grpo"] if smoke else ["vanilla_grpo", "entropy_grpo", "drive_grpo", "drive_kl_grpo"]

    summary = {"sft_multi_warm": sft_data["metrics"]}
    s6_export = {
        "scenes": s6_sft["scenes"],
        "planners": {"sft_multi_warm": s6_sft["preds"]},
    }
    for v in variants:
        p = OUT_DIR / (f"smoke_res_{v}.json" if smoke else f"res_{v}.json")
        if p.exists():
            d = json.loads(p.read_text())
            summary[v] = d["rl_metrics"]
            summary[f"opening_handoff_{v}"] = d["handoff_metrics"]
            s6_export["planners"][v] = d["s6_rl_preds"]
            s6_export["planners"][f"opening_handoff_{v}"] = d["s6_ho_preds"]

    out_sum = OUT_DIR / ("smoke_summary.json" if smoke else "summary.json")
    out_s6 = OUT_DIR / ("smoke_stage6_planner_plans.json" if smoke else "stage6_planner_plans.json")
    out_sum.write_text(json.dumps(summary, indent=2))
    out_s6.write_text(json.dumps(s6_export, indent=2))
    # Remove temporary 3.4GB sft pt file after full run to keep /data clean!
    if not smoke:
        sft_pt = OUT_DIR / "sft_multi_warm.pt"
        if sft_pt.exists():
            sft_pt.unlink()
    print(f"[G5A] Summary -> {out_sum}\n{json.dumps(summary, indent=2)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sft-prep", action="store_true")
    ap.add_argument("--variant", type=str, default="")
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    if args.sft_prep:
        run_sft_prep(args.gpu_id, args.smoke)
        return
    if args.variant:
        run_variant_worker(args.variant, args.gpu_id, args.smoke)
        return

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    # Step 1: Shared Multi-branch Warm SFT
    rc = subprocess.Popen([sys.executable, __file__, "--sft-prep", "--gpu-id", str(gpu_list[0])] + (["--smoke"] if args.smoke else [])).wait()
    assert rc == 0, f"G5A SFT prep failed rc={rc}"

    # Step 2: Parallel RL variants in waves of len(gpu_list) so 1 variant = 1 dedicated GPU!
    variants = ["vanilla_grpo", "drive_grpo"] if args.smoke else ["vanilla_grpo", "entropy_grpo", "drive_grpo", "drive_kl_grpo"]
    for i in range(0, len(variants), len(gpu_list)):
        wave = variants[i : i + len(gpu_list)]
        procs = []
        for idx, v in enumerate(wave):
            g = gpu_list[idx]
            procs.append(subprocess.Popen([sys.executable, __file__, "--variant", v, "--gpu-id", str(g)] + (["--smoke"] if args.smoke else [])))
        rcs = [p.wait() for p in procs]
        assert all(r == 0 for r in rcs), f"G5A variant wave failed: {rcs}"
    aggregate(args.smoke)


if __name__ == "__main__":
    main()
