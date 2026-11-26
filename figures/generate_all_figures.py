#!/usr/bin/env python3
"""
Generate all 7 publication-grade vector PDF + PNG figures for the refactored manuscript:
  Main paper (5 figures):
    - fig01_boundary_teaser: 3-panel conceptual + quantitative overview (Sec. I)
    - fig02_approach_contact: 1x3 (tau,gamma) phase map, bowl self-locking basin, multi-curvature MAE (Sec. III)
    - fig03_probing_deadband: 1x3 post-deadband separation, Wald SPRT Pareto, 800-rollout peak force (Sec. IV)
    - fig04_bifurcation_imitation: 1x3 bimodal collapse vs flow, single-physics negative control, SmolVLA 300ms gated stall (Sec. V)
    - fig05_rl_narrowing_retry: 2x2 two-stage flow, QueST takeover, 7B OpenVLA-OFT coverage, 500 Hz MuJoCo jammed retry (Sec. VI)
  Appendix (2 figures):
    - fig06_app_sawtooth: single-column sawtooth stride & ring-buffer calibration (Appendix C)
    - fig07_app_countdown: single-column enumerable Countdown Pass@k scaling (Appendix C)
"""
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle
import numpy as np
from PIL import Image

FIG_DIR = Path(__file__).resolve().parent
DATA = FIG_DIR / "source_data"
OUT = FIG_DIR / "generated"
OUT.mkdir(parents=True, exist_ok=True)

WIDTH = 7.16
COL_WIDTH = 3.48
BLUE = "#286B9B"
ORANGE = "#C77A37"
GREEN = "#2E7D5B"
RED = "#B83A2D"
GRAY = "#747C84"
LIGHT_GRAY = "#D9DDE2"
INK = "#222222"
GRID = "#EAECEF"

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 8.6,
    "mathtext.fontset": "stix",
    "axes.labelsize": 8.4,
    "axes.labelpad": 5.5,
    "axes.titlesize": 8.9,
    "axes.titleweight": "normal",
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 6.9,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#6E7378")
    ax.spines["left"].set_color("#6E7378")
    ax.tick_params(length=2.8, width=0.65, pad=33.0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def set_panel(ax, letter, title, pad=33.0):
    ax.set_title(f"({letter}) {title}", loc="left", pad=pad, fontsize=8.9)


def check_all_artist_overlaps(fig, stem, pad_px=1.8):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = []
    for t_idx, txt in enumerate(fig.texts):
        if txt.get_visible() and txt.get_text().strip():
            items.append((f"fig.text[{t_idx}]='{txt.get_text().strip()[:18]}'", txt.get_window_extent(renderer)))
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        for tattr, tname in (("title", "title"), ("_left_title", "ltitle"), ("_right_title", "rtitle")):
            t_obj = getattr(ax, tattr, None)
            if t_obj is not None and t_obj.get_visible() and t_obj.get_text().strip():
                items.append((f"ax[{i}].{tname}='{t_obj.get_text().strip()[:24]}'", t_obj.get_window_extent(renderer)))
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            items.append((f"ax[{i}].legend", leg.get_window_extent(renderer)))
        if ax.axison:
            if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text().strip():
                items.append((f"ax[{i}].xlabel='{ax.xaxis.label.get_text().strip()[:18]}'", ax.xaxis.label.get_window_extent(renderer)))
            if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text().strip():
                items.append((f"ax[{i}].ylabel='{ax.yaxis.label.get_text().strip()[:18]}'", ax.yaxis.label.get_window_extent(renderer)))
            for k, tl in enumerate(ax.get_xticklabels()):
                if tl.get_visible() and tl.get_text().strip():
                    items.append((f"ax[{i}].xtick[{k}]='{tl.get_text().strip()[:12]}'", tl.get_window_extent(renderer)))
            for k, tl in enumerate(ax.get_yticklabels()):
                if tl.get_visible() and tl.get_text().strip():
                    items.append((f"ax[{i}].ytick[{k}]='{tl.get_text().strip()[:12]}'", tl.get_window_extent(renderer)))
        for k, txt in enumerate(ax.texts):
            if txt.get_visible() and txt.get_text().strip():
                items.append((f"ax[{i}].text[{k}]='{txt.get_text().strip()[:18]}'", txt.get_window_extent(renderer)))

    for a_idx in range(len(items)):
        na, ba = items[a_idx]
        if (ba.x1 - ba.x0) < 0.5 or (ba.y1 - ba.y0) < 0.5:
            continue
        for b_idx in range(a_idx + 1, len(items)):
            nb, bb = items[b_idx]
            if (bb.x1 - bb.x0) < 0.5 or (bb.y1 - bb.y0) < 0.5:
                continue
            x_ov = min(ba.x1 + pad_px, bb.x1 + pad_px) - max(ba.x0 - pad_px, bb.x0 - pad_px)
            y_ov = min(ba.y1 + pad_px, bb.y1 + pad_px) - max(ba.y0 - pad_px, bb.y0 - pad_px)
            if x_ov > 0.0 and y_ov > 0.0:
                raw_x = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
                raw_y = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
                raise AssertionError(f"{stem}: OVERLAP between {na} and {nb} (raw x={raw_x:.1f}px, y={raw_y:.1f}px)!")
    print(f"  [bbox-ok] {stem:30s}: {len(items):2d} artists checked, ZERO overlap")


def save_fig(fig, stem):
    fig.set_dpi(240)
    check_all_artist_overlaps(fig, stem, pad_px=1.5)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_fig01_teaser():
    """3-panel Teaser: Smooth Inside, Fragile at Boundaries across Execution, Imitation, and RL."""
    fig = plt.figure(figsize=(WIDTH, 2.42), facecolor="white")
    gs = fig.add_gridspec(1, 3, left=0.015, right=0.985, bottom=0.05, top=0.84, wspace=0.12)

    # Panel (a): Physical contact & probing boundary
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.set_axis_off()
    ax.set_title("(a) Contact boundary (§III–IV)", loc="center", pad=8.0, fontsize=8.3)
    ax.add_patch(Rectangle((0.1, 0.2), 9.8, 5.7, facecolor="#F7F9FB", edgecolor="#D0D7DE", linewidth=0.7))
    ax.add_patch(Rectangle((0.2, 0.3), 5.0, 5.5, facecolor="#EBF3F9", edgecolor="none", alpha=0.65))
    ax.add_patch(Rectangle((5.2, 0.95), 1.35, 4.85, facecolor="#FDEBD0", edgecolor="none", alpha=0.55))
    ax.plot([5.2, 5.2], [1.05, 5.65], color=ORANGE, linestyle="--", linewidth=1.25, zorder=3)
    t1 = np.linspace(0.6, 5.2, 40)
    y_ok = 2.85 + 0.55 * np.sin((t1 - 0.6) * 0.55)
    ax.plot(t1, y_ok, color=BLUE, linewidth=1.9)
    t2 = np.linspace(5.2, 9.3, 40)
    y_slip = y_ok[-1] + 1.50 * (1 - np.exp(-(t2 - 5.2) * 0.9))
    y_safe = y_ok[-1] - 0.55 * (1 - np.exp(-(t2 - 5.2) * 1.1))
    ax.plot(t2, y_slip, color=RED, linestyle="--", linewidth=1.5)
    ax.plot(t2, y_safe, color=BLUE, linewidth=1.9)
    ax.scatter([5.2], [y_ok[-1]], color=ORANGE, edgecolor=INK, s=34, zorder=5)
    ax.text(2.55, 5.52, "Smooth transit", ha="center", va="top", fontsize=6.9, color=BLUE, weight="bold")
    ax.text(5.38, 5.58, "0.10 s\ndeadband", ha="left", va="top", fontsize=6.2, color=ORANGE)
    ax.text(8.05, 3.62, "Lag slip/slam", ha="center", va="bottom", fontsize=6.4, color=RED)
    ax.text(7.75, 1.88, "Bounded+SPRT", ha="center", va="top", fontsize=6.6, color=BLUE, weight="bold")
    ax.text(5.0, 0.46, "41.4%→90.6%  |  452.9→183.6 N", ha="center", va="bottom", fontsize=6.8, color=INK)

    # Panel (b): Post-contact bifurcation boundary (Imitation centerpiece)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.set_axis_off()
    ax.set_title("(b) Bifurcation boundary (§V)", loc="center", pad=8.0, fontsize=8.3)
    ax.add_patch(Rectangle((0.1, 0.2), 9.8, 5.7, facecolor="#F7F9FB", edgecolor=BLUE, linewidth=1.1))
    ax.plot([4.6, 4.6], [1.05, 4.92], color=ORANGE, linestyle="--", linewidth=1.25, zorder=3)
    t_pre = np.linspace(0.6, 4.6, 30)
    ax.plot(t_pre, np.full_like(t_pre, 3.1), color=INK, linewidth=2.0)
    t_post = np.linspace(4.6, 9.2, 40)
    y_norm = 3.1 + 1.42 * np.tanh((t_post - 4.6) * 0.65)
    y_jam = 3.1 - 1.38 * np.tanh((t_post - 4.6) * 0.65)
    y_l1 = 3.1 + 1.12 * np.tanh((t_post - 4.6) * 0.65)
    ax.plot(t_post, y_norm, color=BLUE, linewidth=1.8)
    ax.plot(t_post, y_jam, color=GREEN, linewidth=2.0)
    ax.plot(t_post, y_l1, color=RED, linestyle=":", linewidth=1.6)
    ax.scatter([4.6], [3.1], color=ORANGE, edgecolor=INK, s=38, zorder=5)
    ax.text(2.5, 3.50, "Identical\npre-contact", ha="center", va="bottom", fontsize=7.0, color=INK)
    ax.text(4.6, 5.62, "$t_{\\mathrm{fb}}$", ha="center", va="top", fontsize=8.0, color=ORANGE, weight="bold")
    ax.text(7.55, 4.78, "Normal push", ha="center", va="bottom", fontsize=6.9, color=BLUE)
    ax.text(7.65, 3.66, "$L_1$ collapse", ha="center", va="bottom", fontsize=6.8, color=RED)
    ax.text(7.45, 1.48, "Jammed unlatch", ha="center", va="top", fontsize=6.9, color=GREEN, weight="bold")
    ax.text(5.0, 0.46, "Jam recovery: 35.0%→88.3%", ha="center", va="bottom", fontsize=6.8, color=INK)

    # Panel (c): RL opening-branch boundary
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.set_axis_off()
    ax.set_title("(c) Opening boundary (§VI)", loc="center", pad=8.0, fontsize=8.3)
    ax.add_patch(Rectangle((0.1, 0.2), 9.8, 5.7, facecolor="#F7F9FB", edgecolor="#D0D7DE", linewidth=0.7))
    ax.add_patch(Rectangle((0.2, 0.95), 2.8, 4.85, facecolor="#FDEBD0", edgecolor="none", alpha=0.55))
    ax.plot([3.0, 3.0], [1.05, 5.65], color=ORANGE, linestyle="--", linewidth=1.25, zorder=3)
    t_all = np.linspace(0.7, 9.2, 50)
    y_b1 = 3.1 + 1.50 * np.sin((t_all - 0.7) / 8.5 * np.pi)
    y_b2 = 3.1 + 0.15 * np.sin((t_all - 0.7) / 8.5 * np.pi)
    y_b3 = 3.1 - 1.42 * np.sin((t_all - 0.7) / 8.5 * np.pi)
    ax.plot(t_all, y_b1, color=RED, linewidth=2.0)
    ax.plot(t_all[t_all <= 3.0], y_b2[t_all <= 3.0], color=GREEN, linestyle="--", linewidth=1.6)
    ax.plot(t_all[t_all >= 3.0], y_b2[t_all >= 3.0], color=BLUE, linewidth=1.7)
    ax.plot(t_all[t_all <= 3.0], y_b3[t_all <= 3.0], color=GREEN, linestyle="--", linewidth=1.6)
    ax.plot(t_all[t_all >= 3.0], y_b3[t_all >= 3.0], color=BLUE, linewidth=1.7)
    ax.scatter([0.7], [3.1], color=INK, s=28, zorder=5)
    ax.scatter([5.3], [4.58], marker="X", color=RED, s=44, zorder=6)
    ax.text(1.6, 5.52, "Opening\nhandoff", ha="center", va="top", fontsize=6.8, color=ORANGE, weight="bold")
    ax.text(6.8, 4.84, "RL-locked $B_1$", ha="center", va="bottom", fontsize=6.7, color=RED)
    ax.text(6.55, 1.54, "Intact RL continuation", ha="center", va="top", fontsize=6.8, color=BLUE, weight="bold")
    ax.text(5.0, 0.46, "Cov 0.227→0.342 | Retry 20→78%", ha="center", va="bottom", fontsize=6.7, color=INK)

    save_fig(fig, "fig01_boundary_teaser")


def make_fig02_approach_contact():
    """Streamlined 1x3 Section III figure: (a) (tau,gamma) phase map, (b) bowl self-locking basin, (c) multi-curvature MAE."""
    tg_grid = read_csv("math_exp5_tau_gamma_grid.csv")
    basin = read_csv("v11_bowl_centered_basin_25x25.csv")
    sm_sum = json.loads((DATA / "v11_smooth_curved_selflock_summary.json").read_text())["by_R"]

    fig = plt.figure(figsize=(WIDTH, 2.68), facecolor="white")
    gs = fig.add_gridspec(1, 3, left=0.068, right=0.982, bottom=0.19, top=0.71, wspace=0.72)

    # (a) MuJoCo (tau, gamma) phase map
    ax_a = fig.add_subplot(gs[0, 0])
    taus = sorted({int(r["delay_ms"]) for r in tg_grid})
    gams = sorted({float(r["gamma"]) for r in tg_grid})
    Z = np.zeros((len(gams), len(taus)))
    for r in tg_grid:
        Z[gams.index(float(r["gamma"])), taus.index(int(r["delay_ms"]))] = float(r["success_rate"])
    cmap_b = LinearSegmentedColormap.from_list("bl", ["#F4F6F8", "#9EC1DB", "#286B9B"])
    im_a = ax_a.imshow(Z, origin="lower", aspect="auto", cmap=cmap_b, vmin=0, vmax=100,
                       extent=[min(taus) - 25, max(taus) + 25, min(gams) - 0.05, max(gams) + 0.05])
    ds = np.linspace(1.0, 5.2, 60)
    eta_c = 0.2549
    g_crit = 2.0 * np.sin(np.pi / (2.0 * (2.0 * ds + 1.0))) / eta_c
    g_mono = (ds ** ds) / ((ds + 1.0) ** (ds + 1.0) * eta_c)
    ax_a.plot(ds * 50.0, g_crit, color=ORANGE, linestyle="--", linewidth=1.4, label="$\\gamma_{\\mathrm{crit}}$")
    ax_a.plot(ds * 50.0, g_mono, color=INK, linestyle=":", linewidth=1.3, label="$\\gamma_{\\mathrm{mono}}$")
    ax_a.set(xlim=(25, 275), ylim=(0.35, 1.05), xlabel="Delay $\\tau$ (ms)", ylabel="Chunk gain $\\gamma$")
    ax_a.set_xticks([50, 150, 250])
    ax_a.set_yticks([0.4, 0.7, 1.0])
    style_ax(ax_a, None)
    set_panel(ax_a, "a", "MuJoCo $(\\tau,\\gamma)$ stability", pad=33.0)
    ax_a.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.03, 1.01),
                handlelength=1.3, columnspacing=0.5, handletextpad=0.22, fontsize=6.9)
    pos_a = ax_a.get_position()
    cax = fig.add_axes([pos_a.x1 + 0.006, pos_a.y0, 0.010, pos_a.height])
    cb = fig.colorbar(im_a, cax=cax, ticks=[0, 50, 100])
    cb.ax.set_yticklabels(["0%", "50%", "100%"])
    cb.ax.tick_params(labelsize=6.4, length=2.0, pad=33.0)
    cb.outline.set_linewidth(0.55)
    cb.outline.set_edgecolor("#6E7378")

    # (b) Bowl-centered basin
    ax_b = fig.add_subplot(gs[0, 1])
    xs = sorted({float(r["rel_bowl_x_mm"]) for r in basin})
    ys = sorted({float(r["rel_bowl_y_mm"]) for r in basin})
    code_map = {"miss": 0, "rim_graze": 1, "transient_lift_slip": 1, "stable_grasp": 2}
    Zb = np.zeros((len(ys), len(xs)), dtype=int)
    for r in basin:
        Zb[ys.index(float(r["rel_bowl_y_mm"])), xs.index(float(r["rel_bowl_x_mm"]))] = code_map[r["regime"]]
    cmap = ListedColormap(["#ECEFF3", "#DF9255", "#286B9B"])
    ax_b.set_facecolor("#ECEFF3")
    ax_b.imshow(Zb, origin="lower", extent=[-75.0, 75.0, -75.0, 75.0], cmap=cmap, vmin=0, vmax=2)
    ax_b.add_patch(Circle((0.0, 0.0), 52.3, fill=False, edgecolor=INK, linestyle="--", linewidth=1.35))
    ax_b.set_aspect("equal", adjustable="datalim")
    ax_b.set(ylim=(-74, 74), xlabel="$x - x_c$ (mm)", ylabel="$y - y_c$ (mm)")
    ax_b.set_xticks([-60, 0, 60])
    ax_b.set_yticks([-60, 0, 60])
    style_ax(ax_b, None)
    for _sp in ("top", "right"):
        ax_b.spines[_sp].set_visible(True)
        ax_b.spines[_sp].set_color("#6E7378")
        ax_b.spines[_sp].set_linewidth(0.65)
    set_panel(ax_b, "b", "Curved-rim basin", pad=33.0)
    handles_b = [
        Patch(facecolor="#286B9B", label="Grasp"),
        Patch(facecolor="#DF9255", label="Slip"),
        Line2D([0], [0], color=INK, linestyle="--", linewidth=1.25, label="$r_{\\mathrm{crit}}$"),
    ]
    ax_b.legend(handles_b, ["Grasp", "Slip", "$r_{\\mathrm{crit}}$"], frameon=False, ncol=3,
                loc="lower left", bbox_to_anchor=(-0.06, 1.01), handlelength=0.85,
                columnspacing=0.38, handletextpad=0.20, fontsize=6.7)

    # (c) Multi-curvature x multi-friction
    ax_c = fig.add_subplot(gs[0, 2])
    mus = [0.3, 0.5, 0.7, 0.9]
    mu_dense = np.linspace(0.25, 0.95, 80)
    for R_str, col, mk, lbl in [("40", GRAY, "o", "$R=40$"), ("55", GREEN, "s", "$R=55$"), ("75", BLUE, "^", "$R=75$"), ("95", ORANGE, "D", "$R=95$")]:
        R_val = float(R_str) + 2.0
        fc, half_mg = 24.0, 0.5 * 0.18 * (9.81 + 0.35)
        theo = []
        for mv in mu_dense:
            als = np.linspace(0.0, 1.35, 1000)
            va = als[(fc * np.sin(als) + half_mg * np.cos(als)) <= mv * np.maximum(0.0, fc * np.cos(als) - half_mg * np.sin(als))]
            theo.append(R_val * math.sin(float(va[-1])) if len(va) else 0.0)
        ax_c.plot(mu_dense, theo, color=col, linestyle="--", linewidth=1.1)
        emp = [sm_sum[R_str]["by_mu"][str(m)]["h_emp_mm"] for m in mus]
        ax_c.plot(mus, emp, color=col, linestyle="none", marker=mk, markersize=3.8, label=lbl)
    ax_c.set(xlim=(0.24, 0.96), ylim=(5, 68), xlabel="Friction $\\mu$", ylabel="$h_{\\mathrm{crit}}$ (mm)")
    ax_c.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax_c.set_yticks([15, 35, 55])
    style_ax(ax_c, "y")
    set_panel(ax_c, "c", "Self-locking ($0.46\\,\\mathrm{mm}$ MAE)", pad=33.0)
    ax_c.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.04, 1.01),
                handlelength=0.85, columnspacing=0.45, handletextpad=0.22, fontsize=6.6)

    save_fig(fig, "fig02_approach_contact")


def make_fig03_probing():
    """Unified 1x3 Section IV figure: (a) post-deadband separation, (b) Wald SPRT Pareto, (c) 800-rollout peak force."""
    curves = read_csv("math_exp3_theta_curves.csv")
    sprt = read_csv("math_exp6_sprt_probing.csv")
    cl_sprt = json.loads((DATA / "v11_contact_handle_sprt_summary.json").read_text())["closed_loop_sprt"]

    fig = plt.figure(figsize=(WIDTH, 2.75), facecolor="white")
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.978, bottom=0.19, top=0.69, wspace=0.42)

    # (a) Post-deadband separation
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axvspan(0.0, 0.10, color="#FDEBD0", alpha=0.65)
    for th, col, ls, mk, lbl in [(5.0, GRAY, "--", "o", "$\\theta=5$"), (12.0, GREEN, "-.", "s", "$\\theta=12$"), (30.0, BLUE, "-", "^", "$\\theta=30$")]:
        sub = [r for r in curves if abs(float(r["theta"]) - th) < 1e-3 and int(r["probe_steps"]) <= 8]
        ax_a.plot([float(r["probe_time_s"]) for r in sub], [float(r["eef_gap_to_nom_mm"]) for r in sub],
                  color=col, linestyle=ls, marker=mk, markersize=3.0, linewidth=1.4, label=lbl)
    ax_a.axvline(0.10, color=ORANGE, linestyle="--", linewidth=1.0)
    ax_a.text(0.05, 19.5, "Dead-\nband", ha="center", va="top", fontsize=6.5, color=ORANGE)
    ax_a.set(xlim=(-0.01, 0.41), ylim=(-0.5, 23.0), xlabel="Probe duration (s)", ylabel="Response gap (mm)")
    ax_a.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax_a.set_yticks([0, 10, 20])
    style_ax(ax_a, "y")
    set_panel(ax_a, "a", "Post-deadband separation", pad=33.0)
    ax_a.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(-0.06, 1.01),
                handlelength=1.0, columnspacing=0.35, handletextpad=0.18, fontsize=6.5)

    # (b) Wald SPRT vs fixed Tp
    ax_b = fig.add_subplot(gs[0, 1])
    for task, col, mk, lbl in [("L031", BLUE, "o", "Drawer"), ("L059", GREEN, "s", "Cabinet")]:
        fixed = sorted([r for r in sprt if r["task"] == task and r["policy"].startswith("fixed_Tp_") and int(r["policy"].split("_")[-1]) >= 2],
                       key=lambda z: int(z["policy"].split("_")[-1]))
        ax_b.plot([float(r["mean_probe_steps"]) for r in fixed], [float(r["overall_error_pct"]) for r in fixed],
                  color=col, linestyle="--", marker=mk, markersize=3.2, linewidth=1.2, label=f"{lbl} $T_p$")
        sp_all = sorted([r for r in sprt if r["task"] == task and r["policy"].startswith("wald_sprt_")],
                        key=lambda z: float(z["mean_probe_steps"]))
        ax_b.plot([float(r["mean_probe_steps"]) for r in sp_all], [float(r["overall_error_pct"]) for r in sp_all],
                  color=ORANGE, linestyle="-" if task == "L031" else "-.", linewidth=1.35,
                  label="Wald SPRT" if task == "L031" else None)
        sp_row = next(r for r in sp_all if "B6.5" in r["policy"])
        ax_b.scatter([float(sp_row["mean_probe_steps"])], [float(sp_row["overall_error_pct"])],
                     color=ORANGE, edgecolor=INK, linewidth=0.5, marker="*", s=72, zorder=6,
                     label="$B=6.5$" if task == "L031" else None)
    ax_b.axvline(2.0, color="#888888", linestyle=":", linewidth=0.9)
    ax_b.set(xlim=(1.65, 8.35), ylim=(-1.5, 65.0), xlabel="Expected probe steps", ylabel="Regime error (%)")
    ax_b.set_xticks([2, 4, 6, 8])
    ax_b.set_yticks([0, 20, 40, 60])
    style_ax(ax_b, "y")
    set_panel(ax_b, "b", "Wald SPRT vs. fixed $T_p$", pad=33.0)
    ax_b.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.06, 1.01),
                handlelength=0.95, columnspacing=0.38, handletextpad=0.20, fontsize=6.5)

    # (c) 800-rollout peak force
    ax_c = fig.add_subplot(gs[0, 2])
    modes = [
        ("unmasked_cross_feedback", GRAY, "Unmask."),
        ("fixed_tp6_hold", ORANGE, "$T_p\\!=\\!6$ hold"),
        ("fixed_tp2_withdraw", GREEN, "$T_p\\!=\\!2$ w/d"),
        ("wald_sprt_withdraw", BLUE, "SPRT w/d"),
    ]
    x = np.arange(2)
    for idx, (md, col, lbl) in enumerate(modes):
        vals = [cl_sprt[md]["low_peak_force_N"], cl_sprt[md]["high_peak_force_N"]]
        ax_c.bar(x + (idx - 1.5) * 0.19, vals, width=0.175, color=col, label=lbl)
    ax_c.set_xticks(x, ["Low ($\\theta < 8$)", "High ($\\theta \\geq 8$)"])
    ax_c.set(ylabel="Peak force (N)", ylim=(0, 650))
    ax_c.set_yticks([0, 300, 600])
    style_ax(ax_c, "y")
    set_panel(ax_c, "c", "Closed-loop force ($n=800$)", pad=33.0)
    ax_c.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.06, 1.01),
                handlelength=0.85, columnspacing=0.38, handletextpad=0.20, fontsize=6.6)

    save_fig(fig, "fig03_probing_deadband")


def make_fig04_bifurcation():
    """NEW Section V Centerpiece 1x3 figure: (a) Bimodal collapse vs flow, (b) Negative control, (c) SmolVLA 300ms gated stall."""
    fig = plt.figure(figsize=(WIDTH, 2.75), facecolor="white")
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.978, bottom=0.21, top=0.70, wspace=0.40)

    # (a) Bimodal post-contact bifurcation (Table III-A)
    ax_a = fig.add_subplot(gs[0, 0])
    cats = ["$L_1$\nUnm.", "$L_1$\nC+R", "Flow\nUnm.", "Flow\nRole", "Flow\nCaus.", "Flow\nC+R"]
    norm_v = [71.7, 90.0, 80.0, 86.7, 80.0, 78.3]
    jam_v = [0.0, 3.3, 35.0, 70.0, 81.7, 88.3]
    x = np.arange(len(cats))
    ax_a.bar(x - 0.18, norm_v, width=0.34, color=LIGHT_GRAY, edgecolor=INK, linewidth=0.55, label="Normal")
    ax_a.bar(x + 0.18, jam_v, width=0.34, color=BLUE, label="Jammed rec.")
    ax_a.set_xticks(x, cats)
    ax_a.tick_params(axis="x", labelsize=6.6)
    ax_a.set(ylim=(0, 108), ylabel="Completion (%)")
    ax_a.set_yticks([0, 50, 100])
    style_ax(ax_a, "y")
    set_panel(ax_a, "a", "Bimodal bifurcation", pad=33.0)
    ax_a.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.04, 1.01),
                handlelength=0.9, columnspacing=0.45, handletextpad=0.22, fontsize=6.8)

    # (b) Single-physics negative control (Table III-B, 40 tasks)
    ax_b = fig.add_subplot(gs[0, 1])
    cats_b = ["Unmasked", "Role only", "Full causal\n(open)"]
    all_v = [32.5, 39.2, 32.1]
    rec_v = [23.3, 26.7, 28.3]
    xb = np.arange(3)
    ax_b.bar(xb - 0.17, all_v, width=0.32, color=ORANGE, label="40-task All")
    ax_b.bar(xb + 0.17, rec_v, width=0.32, color=GREEN, label="Recovery sub.")
    ax_b.set_xticks(xb, cats_b)
    ax_b.tick_params(axis="x", labelsize=7.0)
    ax_b.set(ylim=(0, 52), ylabel="Completion (%)")
    ax_b.set_yticks([0, 25, 50])
    style_ax(ax_b, "y")
    set_panel(ax_b, "b", "Single-physics control", pad=33.0)
    ax_b.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.05, 1.01),
                handlelength=0.9, columnspacing=0.40, handletextpad=0.22, fontsize=6.7)

    # (c) Pretrained SmolVLA-450M under 300ms stalled lag (Table IV-B)
    ax_c = fig.add_subplot(gs[0, 2])
    cats_c = ["300ms\nRaw", "300ms\nUncond.", "300ms\nGated", "Burst\nGated"]
    nom_c = [11.7, 43.3, 53.3, 61.7]
    lat_c = [4.2, 0.0, 20.8, 33.3]
    xc = np.arange(4)
    ax_c.bar(xc - 0.17, nom_c, width=0.32, color=LIGHT_GRAY, edgecolor=INK, linewidth=0.55, label="Nom.-20")
    ax_c.bar(xc + 0.17, lat_c, width=0.32, color=BLUE, label="Latched")
    ax_c.set_xticks(xc, cats_c)
    ax_c.tick_params(axis="x", labelsize=6.8)
    ax_c.set(ylim=(0, 76), ylabel="Completion (%)")
    ax_c.set_yticks([0, 35, 70])
    style_ax(ax_c, "y")
    set_panel(ax_c, "c", "SmolVLA-450M stall gating", pad=33.0)
    ax_c.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.04, 1.01),
                handlelength=0.9, columnspacing=0.45, handletextpad=0.22, fontsize=6.8)

    save_fig(fig, "fig04_bifurcation_imitation")


def make_fig05_rl_narrowing_retry():
    """Upgraded 2x2 Section VI figure: (a) Two-stage flow, (b) QueST takeover, (c) 7B OpenVLA-OFT coverage, (d) 500Hz MuJoCo blocked retry."""
    ode_true = read_csv("math_exp7_true_ode.csv")
    bdata = json.loads((DATA / "rework_quest_branch_collapse.json").read_text())
    sv = json.loads((DATA / "v12_table5_oft7b.json").read_text())["summary"]

    fig = plt.figure(figsize=(WIDTH, 4.90), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.095, right=0.972, bottom=0.095, top=0.878, wspace=0.38, hspace=0.68)

    # (a) Two-stage gradient flow
    ax_a = fig.add_subplot(gs[0, 0])
    cold = [r for r in ode_true if r["regime"] == "cold_start"]
    warm = [r for r in ode_true if r["regime"] == "warm_start"]
    ts = [float(r["t"]) for r in cold]
    ax_a.plot(ts, [float(r["q1"]) for r in cold], color=ORANGE, linestyle="-", linewidth=1.5, label="Cold $q_1$ ($\\beta=0.002$)")
    ax_a.plot(ts, [float(r["q1"]) for r in warm], color=BLUE, linestyle="-", linewidth=1.5, label="Warm $q_1$ ($\\beta=0.03$)")
    ax_a.plot(ts, [float(r["E2"]) for r in cold], color=GREEN, linestyle="-.", linewidth=1.4, label="Cold $E_2(\\phi)$")
    ax_a.plot(ts, [float(r["pass1"]) for r in cold], color=GRAY, linestyle="--", linewidth=1.3, label="Cold pass@1")
    ax_a.set(xlim=(0, 40), ylim=(0, 1.06), xlabel="Gradient-flow time $s$", ylabel="Probability")
    ax_a.set_xticks([0, 20, 40])
    ax_a.set_yticks([0.0, 0.5, 1.0])
    style_ax(ax_a, "y")
    set_panel(ax_a, "a", "Two-stage gradient flow ($B=4$)", pad=21.0)
    ax_a.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.03, 1.01),
                handlelength=1.1, columnspacing=0.44, handletextpad=0.22, fontsize=6.7)

    # (b) QueST opening-skill share & takeover
    sub_gs = gs[0, 1].subgridspec(2, 1, hspace=0.45, height_ratios=[1.15, 1.0])
    ax_top = fig.add_subplot(sub_gs[0])
    ax_bot = fig.add_subplot(sub_gs[1])
    branches = bdata["branch_eval"]["Layout_B"]["branches"]
    xp = np.arange(len(branches))
    sft_share = [float(b["sft_emp_64"]) / 64.0 for b in branches]
    ript_share = [float(b["ript_emp_64"]) / 64.0 for b in branches]
    takeover_rate = [float(b["ript_takeover_succ"]) / float(b["ript_takeover_n"]) for b in branches]

    ax_top.bar(xp - 0.17, sft_share, width=0.32, color=LIGHT_GRAY, edgecolor=INK, linewidth=0.6, label="SFT")
    ax_top.bar(xp + 0.17, ript_share, width=0.32, color=ORANGE, edgecolor=INK, linewidth=0.6, label="RIPT")
    ax_top.set_xticks(xp, ["", "", "", ""])
    ax_top.set(ylim=(0, 1.08), ylabel="Share")
    ax_top.set_yticks([0.0, 0.5, 1.0])
    style_ax(ax_top, "y")
    set_panel(ax_top, "b", "QueST skill share & takeover", pad=21.0)
    ax_top.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.03, 1.03),
                  handlelength=0.90, columnspacing=0.48, handletextpad=0.22, fontsize=6.7)

    ax_bot.bar(xp, takeover_rate, width=0.46, color=BLUE)
    ax_bot.set_xticks(xp, ["B1", "B2", "B3", "B4"])
    ax_bot.set(ylim=(0, 1.12), xlabel="Opening skill", ylabel="Takeover")
    ax_bot.set_yticks([0.0, 0.5, 1.0])
    style_ax(ax_bot, "y")

    # (c) 7B OpenVLA-OFT 3D corridor coverage
    ax_c = fig.add_subplot(gs[1, 0])
    keys = ["sft_base", "rl_ctrl", "early_L0_11", "midlate_L16_31", "entrance_handoff"]
    lbls = ["SFT", "RL", "Early", "Late", "Entry"]
    cols = [LIGHT_GRAY, GRAY, ORANGE, BLUE, GREEN]
    cov_vals = [sv[k]["phys_cov8"] for k in keys]
    p1_vals = [sv[k]["pass1"] for k in keys]
    bx = np.arange(len(keys))
    ax_c.bar(bx, cov_vals, width=0.52, color=cols, edgecolor=INK, linewidth=0.65, label="Phys-Cov@8")
    ax_c.plot(bx, p1_vals, color=RED, linestyle="none", marker="D", markeredgecolor="white",
              markeredgewidth=0.5, markersize=5.0, zorder=5, label="Pass@1")
    ax_c.set_xticks(bx, lbls)
    ax_c.set(xlim=(-0.55, 4.55), ylim=(0.0, 0.85), xlabel="7B OpenVLA-OFT", ylabel="Coverage / Pass@1")
    ax_c.set_yticks([0.0, 0.4, 0.8])
    style_ax(ax_c, "y")
    set_panel(ax_c, "c", "7B OpenVLA-OFT 3D coverage", pad=21.0)
    ax_c.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.03, 1.03),
                handlelength=0.95, columnspacing=0.48, handletextpad=0.24, fontsize=6.8)

    # (d) 500 Hz MuJoCo blocked-subgoal 3-attempt closed-loop retry
    ax_d = fig.add_subplot(gs[1, 1])
    cats_d = ["GRPO\n0ms", "GRPO\n200ms", "HO+Raw\n200ms", "HO+Fl.\n200ms", "HO\n0ms", "DRIVE\n0ms"]
    vals_d = [28.0, 20.0, 52.0, 78.0, 84.0, 88.0]
    cols_d = [GRAY, RED, ORANGE, BLUE, GREEN, LIGHT_GRAY]
    xd = np.arange(len(cats_d))
    ax_d.bar(xd, vals_d, width=0.54, color=cols_d, edgecolor=INK, linewidth=0.6)
    ax_d.set_xticks(xd, cats_d)
    ax_d.tick_params(axis="x", labelsize=6.3)
    ax_d.set(xlim=(-0.55, 5.55), ylim=(0, 106), xlabel="Hierarchical planner + SmolVLA", ylabel="MuJoCo@3 retry (%)")
    ax_d.set_yticks([0, 50, 100])
    style_ax(ax_d, "y")
    set_panel(ax_d, "d", "Closed-loop jammed retry ($500\\,\\mathrm{Hz}$)", pad=21.0)

    fig.canvas.draw()
    assert abs(ax_a.get_position().y1 - ax_top.get_position().y1) < 1e-4, "Top edge of (a) and (b) must match!"
    assert abs(ax_a.get_position().y0 - ax_bot.get_position().y0) < 1e-4, "Bottom edge of (a) and (b) must match!"
    save_fig(fig, "fig05_rl_narrowing_retry")


def make_fig06_app_sawtooth():
    saw = read_csv("math_exp6_sawtooth_mujoco.csv")
    fig = plt.figure(figsize=(COL_WIDTH, 2.08), facecolor="white")
    ax = fig.add_axes([0.16, 0.22, 0.80, 0.62])
    prots = [
        ("sawtooth_c8_raw_g1.0", "$c=8$\nRaw"),
        ("sawtooth_c8_damped_g0.55", "$c=8$\nDamp"),
        ("sawtooth_c4_raw_g1.0", "$c=4$\nRaw"),
        ("sawtooth_c2_raw_g1.0", "$c=2$\nRaw"),
        ("sawtooth_c8_age_calibrated", "$c=8$\nRing"),
    ]
    xb = np.arange(len(prots))
    v_bowl = [float(next(r["success_rate"] for r in saw if r["task"] == "L001" and r["protocol"] == pk)) for pk, _ in prots]
    v_plate = [float(next(r["success_rate"] for r in saw if r["task"] == "L003" and r["protocol"] == pk)) for pk, _ in prots]
    ax.bar(xb - 0.17, v_bowl, width=0.32, color=ORANGE, label="Bowl grasp")
    ax.bar(xb + 0.17, v_plate, width=0.32, color=BLUE, label="Plate-stove")
    ax.set_xticks(xb, [pl for _, pl in prots])
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set(ylim=(0, 108), ylabel="Completion (%)")
    ax.set_yticks([0, 50, 100])
    style_ax(ax, "y")
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.03, 1.02),
              handlelength=0.92, columnspacing=0.48, handletextpad=0.22, fontsize=7.0)
    save_fig(fig, "fig06_app_sawtooth")


def make_fig07_app_countdown():
    fig = plt.figure(figsize=(COL_WIDTH, 2.18), facecolor="white")
    ax = fig.add_axes([0.18, 0.23, 0.78, 0.58])
    ks = np.array([1, 4, 8, 16, 64], dtype=float)
    ax.plot(ks, [0.430, 0.478, 0.499, 0.523, 0.585], color=GRAY, linestyle="--", marker="o",
            markerfacecolor="white", markeredgecolor=GRAY, markersize=3.4, linewidth=1.3, label="GRPO ctrl")
    ax.plot(ks, [0.366, 0.482, 0.518, 0.550, 0.607], color=GREEN, linestyle="-", marker="o",
            markersize=3.4, linewidth=1.4, label="GRPO entry")
    ax.plot(ks, [0.283, 0.334, 0.356, 0.373, 0.407], color=GRAY, linestyle="--", marker="s",
            markerfacecolor="white", markeredgecolor=GRAY, markersize=3.2, linewidth=1.3, label="PPO ctrl")
    ax.plot(ks, [0.292, 0.368, 0.399, 0.425, 0.467], color=BLUE, linestyle="-", marker="s",
            markersize=3.2, linewidth=1.4, label="PPO L16-32")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels(["1", "4", "8", "16", "64"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set(xlim=(0.82, 78.0), ylim=(0.25, 0.65), xlabel="Sampling budget $k$", ylabel="Pass@$k$")
    ax.set_yticks([0.30, 0.45, 0.60])
    style_ax(ax, "y")
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.04, 1.02),
              handlelength=1.0, columnspacing=0.44, handletextpad=0.22, fontsize=6.8)
    save_fig(fig, "fig07_app_countdown")


if __name__ == "__main__":
    make_fig01_teaser()
    make_fig02_approach_contact()
    make_fig03_probing()
    make_fig04_bifurcation()
    make_fig05_rl_narrowing_retry()
    make_fig06_app_sawtooth()
    make_fig07_app_countdown()
    fig_root = OUT.parent
    stems = (
        "fig01_boundary_teaser",
        "fig02_approach_contact",
        "fig03_probing_deadband",
        "fig04_bifurcation_imitation",
        "fig05_rl_narrowing_retry",
        "fig06_app_sawtooth",
        "fig07_app_countdown",
    )
    bad = []
    for stem in stems:
        for ext in (".pdf", ".png"):
            shutil.copy2(OUT / f"{stem}{ext}", fig_root / f"{stem}{ext}")
        im = np.asarray(Image.open(OUT / f"{stem}.png").convert("RGB"))
        t, b = int(im[:4, :, :].min()), int(im[-4:, :, :].min())
        l, r = int(im[:, :4, :].min()), int(im[:, -4:, :].min())
        print(f"  {stem:30s} shape={im.shape} margins=({t},{b},{l},{r})")
        if min(t, b, l, r) < 255:
            bad.append((stem, t, b, l, r))
    assert not bad, f"Clipped margins: {bad}"
    print("ALL 7 V13 FIGURES GENERATED AND VERIFIED CLEANLY!")
