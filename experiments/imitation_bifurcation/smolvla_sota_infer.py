#!/usr/bin/env python3
"""
v23 SOTA Inference Wrapper for SmolVLA-450M:
  - Extends SmolVLASotaRunnerV21 with:
    1. EAPN (Execution-Aligned Progressive Noise, arXiv:2610.06090, Oct 2026):
       - Intra-chunk AR(1) temporal correlation (rho_intra=0.65)
       - Inter-chunk execution-aligned noise propagation shifted by realized displacement D (rho_inter=0.70)
       - Compatible both standalone (`eapn_base`) and atop Training-RTC (`eapn_training_rtc`)!
    2. Deterministic 1-step L1 regression readout (`l1_direct`) for SmolVLA-L1 vs SmolVLA-Flow ablation.
"""
from __future__ import annotations
import contextlib
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/data/guest/vla_branch_lab")
if str(ROOT / "tools/v21_3h5_suite") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools/v21_3h5_suite"))

from smolvla_sota_infer_v21 import (
    DVACState,
    SmolVLASotaRunnerV21,
    compute_autohorizon_from_attn,
)


class EAPNNoiseState:
    """Execution-Aligned Progressive Noise (arXiv:2610.06090) latent noise state."""

    def __init__(self, rho_inter: float = 0.70, rho_intra: float = 0.65):
        self.rho_inter = float(rho_inter)
        self.rho_intra = float(rho_intra)
        self.prev_noise: torch.Tensor | None = None

    def reset(self):
        self.prev_noise = None

    def sample(self, shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype, disp: int = 0) -> torch.Tensor:
        bsize, horizon, dim = shape
        # Step 1: Intra-chunk temporally correlated Gaussian noise (unit variance at every step)
        eps = torch.randn((bsize, horizon, dim), device=device, dtype=torch.float32)
        if self.rho_intra > 0.0:
            z = torch.empty_like(eps)
            z[:, 0, :] = eps[:, 0, :]
            scale = math.sqrt(max(1e-6, 1.0 - self.rho_intra * self.rho_intra))
            for h in range(1, horizon):
                z[:, h, :] = self.rho_intra * z[:, h - 1, :] + scale * eps[:, h, :]
        else:
            z = eps

        # Step 2: Inter-chunk execution-aligned propagation shifted by actual displacement D
        if self.prev_noise is not None and self.rho_inter > 0.0 and 0 <= disp < horizon:
            overlap = horizon - disp
            s_inter = math.sqrt(max(1e-6, 1.0 - self.rho_inter * self.rho_inter))
            prev_aligned = self.prev_noise[:, disp : disp + overlap, :].to(device=device, dtype=torch.float32)
            z[:, :overlap, :] = self.rho_inter * prev_aligned + s_inter * z[:, :overlap, :]

        self.prev_noise = z.detach().clone()
        return z.to(dtype=dtype)


class SmolVLASotaRunnerV23(SmolVLASotaRunnerV21):
    def infer_v23(
        self,
        batch: dict[str, torch.Tensor],
        method: str = "standard",
        num_steps: int = 10,
        dvac_state: DVACState | None = None,
        eapn_state: EAPNNoiseState | None = None,
        prev_leftover_norm: torch.Tensor | None = None,
        inference_delay: int = 0,
        exec_disp: int = 9,
        exec_horizon: int = 8,
        autohorizon_p: int = 16,
    ) -> dict:
        if method in ("standard", "faster_m4", "rtc_guided", "training_rtc", "dvac", "autohorizon"):
            return self.infer(
                batch=batch,
                method=method,
                num_steps=num_steps,
                dvac_state=dvac_state,
                prev_leftover_norm=prev_leftover_norm,
                inference_delay=inference_delay,
                exec_horizon=exec_horizon,
                autohorizon_p=autohorizon_p,
            )

        m = self.pol.model
        torch.cuda.synchronize(self.dev)
        t0 = time.perf_counter()
        amp_ctx = (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)) if self.use_bf16 else contextlib.nullcontext

        if method == "l1_direct":
            # Deterministic 1-step direct regression readout from x_1 = 0 at t = 1.0
            with torch.inference_mode(), amp_ctx():
                bsize, prefix_pad_masks, past_kv = self._prefill(batch)
                x_zero = torch.zeros((bsize, m.config.chunk_size, m.config.max_action_dim), device=self.dev, dtype=torch.float32)
                t_one = torch.ones((bsize,), dtype=torch.float32, device=self.dev)
                v_t = m.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_kv,
                    x_t=x_zero,
                    timestep=t_one,
                )
                # Under linear interpolant x_t = t*x_1 + (1-t)*a_0, v = x_1 - a_0 => a_0 = x_1 - v = -v when x_1 = 0
                x_final = -v_t
            n_exec = exec_horizon
            extra = {}
        elif method in ("eapn_base", "eapn_training_rtc"):
            assert eapn_state is not None, "eapn_state required for EAPN methods"
            use_prefix_clamp = (method == "eapn_training_rtc")
            with torch.inference_mode(), amp_ctx():
                bsize, prefix_pad_masks, past_kv = self._prefill(batch)
                x_t = eapn_state.sample(
                    (bsize, m.config.chunk_size, m.config.max_action_dim),
                    device=self.dev,
                    dtype=torch.float32,
                    disp=int(exec_disp),
                )
                d_eff = 0
                if use_prefix_clamp and prev_leftover_norm is not None and inference_delay > 0:
                    d_eff = min(int(inference_delay), int(prev_leftover_norm.shape[1]), m.config.chunk_size - 2)
                dt = -1.0 / num_steps
                for step in range(num_steps):
                    if d_eff > 0:
                        x_t[:, :d_eff, :] = prev_leftover_norm[:, :d_eff, :].to(x_t.dtype)
                    t_scalar = 1.0 + step * dt
                    t_tensor = torch.full((bsize,), t_scalar, dtype=torch.float32, device=self.dev)
                    v_t = m.denoise_step(
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_kv,
                        x_t=x_t,
                        timestep=t_tensor,
                    )
                    x_t = x_t + dt * v_t
                if d_eff > 0:
                    x_t[:, :d_eff, :] = prev_leftover_norm[:, :d_eff, :].to(x_t.dtype)
                x_final = x_t
            n_exec = exec_horizon
            extra = {"d_cond": d_eff, "exec_disp": int(exec_disp)}
        else:
            raise ValueError(f"Unknown v23 inference method: {method}")

        torch.cuda.synchronize(self.dev)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        act_norm_7 = x_final[0, :, :7].detach()
        act_chunk = (act_norm_7 * self.a_std + self.a_mean).cpu().numpy().astype(np.float32)
        return {
            "chunk": act_chunk,
            "chunk_full_norm": x_final.detach(),
            "n_exec": int(n_exec),
            "infer_ms": float(infer_ms),
            **extra,
        }
