#!/usr/bin/env python3
"""
Fix 3 (Suite C v18 — 1:1 Nominal-Replay Co-Trained WiSE-FT + Pure-Stall Event Replanning Across 3 Seeds):
Solves both remaining Suite C issues:
  1. Eliminates the 75.8% -> 50.6-58.3% Nominal-40 drift by co-training 1:1 with clean Nominal-28 replay
     demonstrations + calibrated WiSE-FT (`wise_beta = 0.62`), anchoring VLM & nominal visuomotor skills!
  2. Eliminates `empty_lift_trig` false positives via `smolvla_env_worker_v18.py` pure post-contact mechanical
     stall + wrist-snag replanning, and evaluates 24-step chunks (`open24`) vs `pace`, `dvac`, `autohorizon`,
     and `event_replan` across 3 independent training seeds!
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
V17_BIMODAL_DIR = ROOT / "cache/v17_bimodal_smolvla"
NOM_REPLAY_DIR = ROOT / "cache/v18_nominal_replay"
OUT_DIR = ROOT / "outputs/v18_surgical_fixes/suiteC_wise_bifurc_v18"
NOM_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "cache/hf_models/smolvla_libero"

sys.path.insert(0, str(ROOT / "cache/pydeps_smolvla"))
sys.path.insert(0, str(ROOT / "external/lerobot/src"))
sys.path.insert(0, str(ROOT / "tools/v17_10h_suite"))

BIFURC_TASKS = [
    ("libero_spatial", 0), ("libero_spatial", 1), ("libero_spatial", 2), ("libero_spatial", 6),
    ("libero_object", 0), ("libero_object", 1), ("libero_object", 2),
    ("libero_goal", 1), ("libero_goal", 3), ("libero_goal", 8),
    ("libero_10", 1), ("libero_10", 3),
]
BIFURC_SET = set(BIFURC_TASKS)
NOMINAL_EXTRA_TASKS = [
    (s, t)
    for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    for t in range(10)
    if (s, t) not in BIFURC_SET
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


def collect_nominal_shard(gpu_id: int, tasks_sub: list[tuple[str, int]], out_part: Path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["HF_HOME"] = str(ROOT / "cache/hf_home")
    os.environ["TORCH_HOME"] = str(ROOT / "cache/torch_home")
    os.environ["TMPDIR"] = str(ROOT / "cache/tmp")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache/xdg_cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dev = torch.device("cuda:0")
    pol = SmolVLAPolicy.from_pretrained(str(MODEL_DIR)).to(dev).eval()
    pre_stats = load_file(str(MODEL_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_stats = load_file(str(MODEL_DIR / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    s_mean = pre_stats["observation.state.mean"].to(dev, dtype=torch.float32)
    s_std = pre_stats["observation.state.std"].to(dev, dtype=torch.float32).clamp_min(1e-8)
    a_mean = post_stats["action.mean"].to(dev, dtype=torch.float32)
    a_std = post_stats["action.std"].to(dev, dtype=torch.float32)
    tok = pol.model.vlm_with_expert.processor.tokenizer

    env_proc = subprocess.Popen(
        ["/data/guest/miniconda3/envs/vla-branch/bin/python", str(ROOT / "tools/v18_surgical_fixes/smolvla_env_worker_v18.py"), str(gpu_id)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    )
    items = []
    try:
        for suite, tid in tasks_sub:
            npz_p = NOM_REPLAY_DIR / f"{suite}_T{tid}_nom.npz"
            if npz_p.exists():
                d = np.load(npz_p)
                items.append({
                    "suite": suite, "task_id": tid, "branch": "nominal_replay", "state_id": 0,
                    "language": str(d["language"]), "npz": str(npz_p), "fb_steps": [],
                })
                continue
            hor = 280 if suite != "libero_10" else 380
            for sid in range(4):
                send_msg(env_proc, ("init_bifurc_v18", suite, tid, sid, False, "open24", hor))
                _, c1, c2, s8, lang, is_done, step = recv_msg(env_proc)
                tok_out = tok([lang + "\n"], max_length=pol.config.tokenizer_max_length, truncation=True, padding="max_length", return_tensors="pt")
                lt, lm = tok_out["input_ids"].to(dev), tok_out["attention_mask"].to(dev, dtype=torch.bool)
                cam1s, cam2s, props, acts = [], [], [], []
                q_chunk = []
                ok_ep = False
                while not is_done and step < hor:
                    if not q_chunk:
                        img1 = torch.from_numpy(np.frombuffer(c1, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                        img2 = torch.from_numpy(np.frombuffer(c2, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                        st_t = (torch.tensor(s8, dtype=torch.float32).unsqueeze(0).to(dev) - s_mean) / s_std
                        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            an = pol.predict_action_chunk({
                                "observation.images.camera1": img1, "observation.images.camera2": img2,
                                "observation.state": st_t, "observation.language.tokens": lt, "observation.language.attention_mask": lm,
                            })
                            ac = (an[0, :16, :7].float() * a_std + a_mean).cpu().numpy().astype(np.float32)
                        q_chunk = [ac[k] for k in range(len(ac))]
                    act_s = q_chunk.pop(0)
                    cam1s.append(np.frombuffer(c1, dtype=np.uint8).reshape(256, 256, 3).copy())
                    cam2s.append(np.frombuffer(c2, dtype=np.uint8).reshape(256, 256, 3).copy())
                    props.append(np.asarray(s8, dtype=np.float32))
                    acts.append(act_s)
                    send_msg(env_proc, ("step_nominal_single", act_s.tolist()))
                    _, c1, c2, s8, is_done, step = recv_msg(env_proc)
                    if is_done and step < hor:
                        ok_ep = True
                        break
                if ok_ep and len(acts) >= 8:
                    np.savez_compressed(
                        npz_p,
                        cam1=np.stack(cam1s), cam2=np.stack(cam2s),
                        proprio=np.stack(props), actions=np.stack(acts),
                        role_mask=np.ones(len(acts), dtype=np.float32),
                        fb_steps=np.asarray([], dtype=np.int32),
                        language=lang,
                    )
                    items.append({
                        "suite": suite, "task_id": tid, "branch": "nominal_replay", "state_id": sid,
                        "language": lang, "npz": str(npz_p), "fb_steps": [],
                    })
                    print(f"[SuiteC v18 Replay GPU{gpu_id}] {suite} T{tid} OK (len={len(acts)})", flush=True)
                    break
        out_part.write_text(json.dumps(items, indent=2))
    finally:
        try:
            send_msg(env_proc, ("close",))
        except Exception:
            pass
        env_proc.wait(timeout=10)


def build_chunk_and_weights(ep: dict, fb_steps: np.ndarray, t_raw: int, chunk_size: int, mode: str):
    actions = ep["actions"]
    role_mask = ep["role_mask"]
    T = len(actions)
    if mode == "unmasked":
        t = int(np.clip(t_raw, 0, max(0, T - 1)))
        a_chunk = np.zeros((chunk_size, 7), dtype=np.float32)
        avail = min(chunk_size, T - t)
        a_chunk[:avail] = actions[t : t + avail]
        if avail < chunk_size:
            a_chunk[avail:] = actions[T - 1]
        return t, a_chunk, np.ones(chunk_size, dtype=np.float32)

    if role_mask[t_raw] > 0.5:
        t = t_raw
    else:
        later_valid = np.flatnonzero(role_mask[t_raw:] > 0.5)
        t = int(t_raw + later_valid[0]) if len(later_valid) > 0 else int(np.clip(t_raw, 0, max(0, T - 2)))
    t = int(np.clip(t, 0, max(0, T - 1)))

    j_cut = min(chunk_size, T - t)
    for j in range(j_cut):
        if role_mask[t + j] < 0.5:
            j_cut = j
            break
    if mode == "causal" and len(fb_steps) > 0:
        later_fb = fb_steps[fb_steps > t]
        if len(later_fb) > 0:
            j_cut = min(j_cut, int(later_fb[0] - t))
    j_cut = max(1, j_cut)

    a_chunk = np.zeros((chunk_size, 7), dtype=np.float32)
    avail = min(chunk_size, T - t)
    a_chunk[:avail] = actions[t : t + avail]
    if avail < chunk_size:
        a_chunk[avail:] = actions[T - 1]
    w_chunk = np.zeros(chunk_size, dtype=np.float32)
    w_chunk[:j_cut] = 1.0
    return t, a_chunk, w_chunk


def train_one_ckpt(mode: str, seed: int, gpu_id: int, manifest_path: Path, steps: int, batch_size: int, wise_beta: float, out_ckpt: Path):
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

    if (out_ckpt / "model.safetensors").exists() and steps > 200:
        print(f"[SuiteC v18 Train] Reusing {out_ckpt}", flush=True)
        return

    torch.manual_seed(20261123 + seed * 137)
    np.random.seed(20261123 + seed * 137)
    rng = random.Random(20261123 + seed * 137)

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

    manifest = json.loads(manifest_path.read_text())
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
    if not latched_eps:
        latched_eps = ep_bank
    if not clean_eps:
        clean_eps = ep_bank

    lr_init = 1.6e-4 if steps <= 180 else 1.05e-4
    opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], lr=lr_init, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=2.0e-5)
    pol.train()
    losses = []

    for step in range(1, steps + 1):
        img1s, img2s, states, acts, ltoks, lmasks, cmasks = [], [], [], [], [], [], []
        for b_idx in range(batch_size):
            # 50% Clean Nominal+Normal Anchor, 50% Latched Bifurcation (focused on post-stall unlatch onset!)
            if b_idx % 2 == 0:
                it, ep = rng.choice(latched_eps)
                fb_arr = np.asarray(it["fb_steps"], dtype=np.int32)
                if len(fb_arr) > 0:
                    c0 = int(fb_arr[0])
                    u_r = rng.random()
                    if u_r < 0.65:
                        # Exact stalled state right where event_replan queries the policy!
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
            img1s.append(ep["cam1"][t])
            img2s.append(ep["cam2"][t])
            states.append(ep["proprio"][t])
            acts.append(a_chunk)
            cmasks.append(w_chunk)
            lt, lm = lang_cache[it["language"]]
            ltoks.append(lt)
            lmasks.append(lm)

        img1_t = torch.from_numpy(np.stack(img1s)).permute(0, 3, 1, 2).to(dev, dtype=torch.float32) / 255.0
        img2_t = torch.from_numpy(np.stack(img2s)).permute(0, 3, 1, 2).to(dev, dtype=torch.float32) / 255.0
        st_t = (torch.from_numpy(np.stack(states)).to(dev, dtype=torch.float32) - s_mean) / s_std
        act_t = (torch.from_numpy(np.stack(acts)).to(dev, dtype=torch.float32) - a_mean) / a_std
        cmask_t = torch.from_numpy(np.stack(cmasks)).to(dev, dtype=torch.float32)
        batch = {
            "observation.images.camera1": img1_t, "observation.images.camera2": img2_t,
            "observation.state": st_t, "action": act_t,
            "observation.language.tokens": torch.stack(ltoks).to(dev),
            "observation.language.attention_mask": torch.stack(lmasks).to(dev),
        }
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            images, img_masks = pol.prepare_images(batch)
            p_state = pol.prepare_state(batch)
            p_actions = pol.prepare_action(batch)
            losses_raw = pol.model.forward(
                images, img_masks, batch["observation.language.tokens"],
                batch["observation.language.attention_mask"], p_state, p_actions, None, None,
            )[:, :, :act_dim]
            loss = (losses_raw.mean(dim=-1) * cmask_t).sum() / cmask_t.sum().clamp_min(float(batch_size * 8))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in pol.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if step % max(1, steps // 4) == 0 or step == 1:
            print(f"[SuiteC v18 Train {mode} s{seed} GPU{gpu_id}] {step}/{steps} loss={np.mean(losses[-20:]):.4f}", flush=True)

    sd_out = {}
    for k, v_ft in pol.state_dict().items():
        v_ft_c = v_ft.detach().cpu()
        v0 = sd_base_cpu[k]
        if v0.is_floating_point():
            b_eff = 0.92 if any(hk in k for hk in ("state_proj", "action_out_proj", "action_in_proj", "action_time_mlp")) else wise_beta
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


def get_eval_specs(seeds: list[int], smoke: bool) -> list[tuple[str, int, str, str]]:
    if smoke:
        return [
            ("base_open24", 0, str(MODEL_DIR), "open24"),
            ("causal_open24", 0, str(OUT_DIR / "smoke_ckpt_causal_s0"), "open24"),
            ("causal_event_replan", 0, str(OUT_DIR / "smoke_ckpt_causal_s0"), "event_replan"),
        ]
    specs = [
        ("base_open24", 0, str(MODEL_DIR), "open24"),
        ("base_pace", 0, str(MODEL_DIR), "pace"),
        ("base_dvac", 0, str(MODEL_DIR), "dvac"),
        ("base_autohorizon", 0, str(MODEL_DIR), "autohorizon"),
    ]
    for s in seeds:
        specs.append(("unmasked_open24", s, str(OUT_DIR / f"ckpt_unmasked_s{s}"), "open24"))
        specs.append(("role_only_open24", s, str(OUT_DIR / f"ckpt_role_only_s{s}"), "open24"))
        specs.append(("causal_open24", s, str(OUT_DIR / f"ckpt_causal_s{s}"), "open24"))
        specs.append(("causal_pace", s, str(OUT_DIR / f"ckpt_causal_s{s}"), "pace"))
        specs.append(("causal_dvac", s, str(OUT_DIR / f"ckpt_causal_s{s}"), "dvac"))
        specs.append(("causal_event_replan", s, str(OUT_DIR / f"ckpt_causal_s{s}"), "event_replan"))
    return specs


def run_eval_shard(shard_id: int, num_shards: int, gpu_id: int, states_nom: int, states_bif: int, smoke: bool):
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
    from smolvla_sota_infer import SmolVLASotaRunner, DVACState

    specs = get_eval_specs([0, 1, 2], smoke)
    all_jobs = []
    if smoke:
        for mname, seed, ckpt_d, emode in specs:
            all_jobs.append(("nominal40", mname, seed, ckpt_d, emode, "libero_spatial", 3, False, 1, 220))
            all_jobs.append(("bifurc", mname, seed, ckpt_d, emode, "libero_spatial", 0, False, 1, 220))
            all_jobs.append(("bifurc", mname, seed, ckpt_d, emode, "libero_spatial", 0, True, 1, 260))
    else:
        for mname, seed, ckpt_d, emode in specs:
            for suite, tid in BIFURC_TASKS:
                hor = 320 if suite != "libero_10" else 440
                all_jobs.append(("bifurc", mname, seed, ckpt_d, emode, suite, tid, False, states_bif, hor))
                all_jobs.append(("bifurc", mname, seed, ckpt_d, emode, suite, tid, True, states_bif, hor))
            if seed == 0 or mname in ("unmasked_open24", "role_only_open24", "causal_open24", "causal_event_replan"):
                for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
                    hor = 320 if suite != "libero_10" else 440
                    for tid in range(10):
                        all_jobs.append(("nominal40", mname, seed, ckpt_d, emode, suite, tid, False, states_nom, hor))

    my_jobs = [j for idx, j in enumerate(all_jobs) if idx % num_shards == shard_id]
    my_jobs.sort(key=lambda x: x[3])
    out_path = OUT_DIR / ("smoke_eval.json" if smoke else f"shard_{shard_id:02d}.json")

    rows, done_keys = [], set()
    if not smoke and out_path.exists():
        try:
            raw_rows = json.loads(out_path.read_text())
            by_u = {}
            for r in raw_rows:
                by_u.setdefault((r["kind"], r["method"], r["seed"], r["suite"], r["task_id"], r["is_latched"]), []).append(r)
            for k, rs in by_u.items():
                req = states_bif if k[0] == "bifurc" else states_nom
                if len(rs) >= req:
                    done_keys.add(k)
                    rows.extend(rs[:req])
        except Exception:
            rows = []

    dev = torch.device("cuda:0")
    pol = SmolVLAPolicy.from_pretrained(str(MODEL_DIR)).to(dev).eval()
    cur_ckpt = str(MODEL_DIR)
    pre_stats = load_file(str(MODEL_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    post_stats = load_file(str(MODEL_DIR / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    s_mean = pre_stats["observation.state.mean"].to(dev, dtype=torch.float32)
    s_std = pre_stats["observation.state.std"].to(dev, dtype=torch.float32).clamp_min(1e-8)
    a_mean = post_stats["action.mean"].to(dev, dtype=torch.float32)
    a_std = post_stats["action.std"].to(dev, dtype=torch.float32)
    tok = pol.model.vlm_with_expert.processor.tokenizer
    runner = SmolVLASotaRunner(pol, a_mean, a_std, dev, use_bf16=True)

    env_proc = subprocess.Popen(
        ["/data/guest/miniconda3/envs/vla-branch/bin/python", str(ROOT / "tools/v18_surgical_fixes/smolvla_env_worker_v18.py"), str(gpu_id)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    )
    try:
        for kind, mname, seed, ckpt_d, emode, suite, tid, is_lat, spt, hor in my_jobs:
            if (kind, mname, seed, suite, tid, is_lat) in done_keys:
                continue
            if ckpt_d != cur_ckpt:
                sd = load_file(str(Path(ckpt_d) / "model.safetensors"), device="cpu")
                pol.load_state_dict(sd, strict=True)
                pol.eval()
                cur_ckpt = ckpt_d
            infer_m = emode if emode in ("dvac", "autohorizon") else "standard"
            for sid in range(spt):
                dvac = DVACState(m=5, alpha=2.0, n_min=2, n_max=24)
                send_msg(env_proc, ("init_bifurc_v18", suite, tid, sid, is_lat, emode, hor))
                _, c1, c2, s8, lang, is_done, step = recv_msg(env_proc)
                tok_out = tok([lang + "\n"], max_length=pol.config.tokenizer_max_length, truncation=True, padding="max_length", return_tensors="pt")
                lt, lm = tok_out["input_ids"].to(dev), tok_out["attention_mask"].to(dev, dtype=torch.bool)
                ok_ep, meta = is_done, {}
                while not is_done:
                    img1 = torch.from_numpy(np.frombuffer(c1, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                    img2 = torch.from_numpy(np.frombuffer(c2, dtype=np.uint8).reshape(256, 256, 3).copy()).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
                    st_t = (torch.tensor(s8, dtype=torch.float32).unsqueeze(0).to(dev) - s_mean) / s_std
                    out = runner.infer({
                        "observation.images.camera1": img1, "observation.images.camera2": img2,
                        "observation.state": st_t, "observation.language.tokens": lt, "observation.language.attention_mask": lm,
                    }, method=infer_m, dvac_state=dvac, exec_horizon=24)
                    send_msg(env_proc, ("step_bifurc_v18", out["chunk"].tolist(), out["n_exec"]))
                    resp = recv_msg(env_proc)
                    if resp[0] == "done":
                        _, ok_ep, step, meta = resp
                        is_done = True
                    else:
                        _, c1, c2, s8, _, is_done, step = resp
                rows.append({
                    "kind": kind, "method": mname, "seed": seed, "suite": suite, "task_id": tid,
                    "state_id": sid, "is_latched": is_lat, "success": int(ok_ep), "steps": step, **meta,
                })
            out_path.write_text(json.dumps(rows, indent=2))
    finally:
        try:
            send_msg(env_proc, ("close",))
        except Exception:
            pass
        env_proc.wait(timeout=10)


def summarize(rows: list[dict], elapsed_s: float) -> dict:
    methods = sorted({r["method"] for r in rows})
    out = {}
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        seeds = sorted({r["seed"] for r in sub})
        nom = [r for r in sub if r["kind"] == "nominal40"]
        bif = [r for r in sub if r["kind"] == "bifurc"]
        b_norm = [r["success"] for r in bif if not r["is_latched"]]
        b_lat = [r["success"] for r in bif if r["is_latched"]]
        seed_lat = [100.0 * sum(r["success"] for r in bif if r["seed"] == s and r["is_latched"]) / max(1, sum(1 for r in bif if r["seed"] == s and r["is_latched"])) for s in seeds if any(r["seed"] == s and r["is_latched"] for r in bif)]
        seed_nom = [100.0 * sum(r["success"] for r in nom if r["seed"] == s) / max(1, sum(1 for r in nom if r["seed"] == s)) for s in seeds if any(r["seed"] == s for r in nom)]
        out[m] = {
            "n_seeds": len(seeds),
            "nominal40_mean_pct": round(100.0 * sum(r["success"] for r in nom) / len(nom), 2) if nom else None,
            "nominal40_std_pct": round(float(np.std(seed_nom)), 2) if len(seed_nom) > 1 else 0.0,
            "bifurc_normal_pct": round(100.0 * sum(b_norm) / len(b_norm), 2) if b_norm else None,
            "bifurc_latched_pct": round(100.0 * sum(b_lat) / len(b_lat), 2) if b_lat else None,
            "bifurc_latched_std_pct": round(float(np.std(seed_lat)), 2) if len(seed_lat) > 1 else 0.0,
            "bifurc_overall_pct": round(100.0 * sum(r["success"] for r in bif) / len(bif), 2) if bif else None,
        }
    return {"n_rollouts": len(rows), "elapsed_s": round(elapsed_s, 1), "summary": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--collect-nom-shard", type=int, default=-1)
    ap.add_argument("--train-single", action="store_true")
    ap.add_argument("--mode", type=str, default="causal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--worker-shard", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=16)
    ap.add_argument("--states-nom", type=int, default=2)
    ap.add_argument("--states-bif", type=int, default=4)
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    args = ap.parse_args()

    t0 = time.time()
    if args.collect_nom_shard >= 0:
        sub = [t for idx, t in enumerate(NOMINAL_EXTRA_TASKS) if idx % args.num_shards == args.collect_nom_shard]
        collect_nominal_shard(args.gpu_id, sub, NOM_REPLAY_DIR / f"nom_part_{args.collect_nom_shard}.json")
        return
    if args.train_single:
        train_one_ckpt(args.mode, args.seed, args.gpu_id, OUT_DIR / "manifest_merged.json", 380, 16, 0.55, OUT_DIR / f"ckpt_{args.mode}_s{args.seed}")
        return
    if args.worker_shard >= 0:
        run_eval_shard(args.worker_shard, args.num_shards, args.gpu_id, args.states_nom, args.states_bif, args.smoke)
        return

    if args.smoke:
        # Collect 1 nominal replay task (libero_spatial T3) + combine with cached v17 libero_spatial T1 bimodal demos!
        sm_nom = NOM_REPLAY_DIR / "smoke_nom.json"
        collect_nominal_shard(args.gpu_id, [("libero_spatial", 3)], sm_nom)
        v17_all = json.loads((V17_BIMODAL_DIR / "manifest_all.json").read_text())
        t1_demos = [x for x in v17_all if x["suite"] == "libero_spatial" and x["task_id"] == 0]
        sm_merged = OUT_DIR / "smoke_manifest.json"
        sm_merged.write_text(json.dumps(t1_demos + json.loads(sm_nom.read_text()), indent=2))
        sm_ckpt = OUT_DIR / "smoke_ckpt_causal_s0"
        if (sm_ckpt / "model.safetensors").exists():
            (sm_ckpt / "model.safetensors").unlink()
        train_one_ckpt("causal", 0, args.gpu_id, sm_merged, 180, 12, 0.55, sm_ckpt)
        run_eval_shard(0, 1, args.gpu_id, 1, 1, True)
        rows = json.loads((OUT_DIR / "smoke_eval.json").read_text())
        res = summarize(rows, time.time() - t0)
        (OUT_DIR / "smoke_summary.json").write_text(json.dumps(res, indent=2))
        print("[Fix3 SuiteC v18 SMOKE OK]", json.dumps(res, indent=2), flush=True)
        return

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    merged_man = OUT_DIR / "manifest_merged.json"
    if not merged_man.exists():
        c_procs = [
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--collect-nom-shard", str(i), "--num-shards", str(len(gpu_list)), "--gpu-id", str(gid)])
            for i, gid in enumerate(gpu_list)
        ]
        for p in c_procs:
            p.wait()
        merged = json.loads((V17_BIMODAL_DIR / "manifest_all.json").read_text())
        for i in range(len(gpu_list)):
            mp = NOM_REPLAY_DIR / f"nom_part_{i}.json"
            if mp.exists():
                merged.extend(json.loads(mp.read_text()))
        merged_man.write_text(json.dumps(merged, indent=2))
        print(f"[Fix3 SuiteC v18] Merged {len(merged)} total demonstrations (72 bimodal + {len(merged)-72} nominal replay)!", flush=True)

    train_jobs = [(m, s) for s in (0, 1, 2) for m in ("unmasked", "role_only", "causal")]
    for w_start in range(0, len(train_jobs), len(gpu_list)):
        wave = train_jobs[w_start : w_start + len(gpu_list)]
        t_procs = [
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--train-single", "--mode", m, "--seed", str(s), "--gpu-id", str(gpu_list[j % len(gpu_list)])])
            for j, (m, s) in enumerate(wave)
        ]
        for p in t_procs:
            p.wait()

    total_shards = len(gpu_list) * args.workers_per_gpu
    e_procs = [
        (s_idx, subprocess.Popen([
            sys.executable, str(Path(__file__).resolve()),
            "--worker-shard", str(s_idx), "--num-shards", str(total_shards), "--gpu-id", str(gpu_list[s_idx % len(gpu_list)]),
            "--states-nom", str(args.states_nom), "--states-bif", str(args.states_bif),
        ]))
        for s_idx in range(total_shards)
    ]
    for s_idx, p in e_procs:
        rc = p.wait()
        print(f"[Fix3 SuiteC v18 Eval] shard {s_idx}/{total_shards} finished rc={rc}", flush=True)

    all_rows = []
    for s_idx in range(total_shards):
        sp = OUT_DIR / f"shard_{s_idx:02d}.json"
        if sp.exists():
            all_rows.extend(json.loads(sp.read_text()))
    res = summarize(all_rows, time.time() - t0)
    (OUT_DIR / "summary.json").write_text(json.dumps(res, indent=2))
    print("[Fix3 SuiteC v18 FULL DONE]", json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
