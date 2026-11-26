"""
CPU Experiment 2: Curved Contact Boundary Basin (MuJoCo) & Ghost-Cell Harmonic Re-Engagement (O(h^2))
2A. Extract real 2D EEF-to-bowl contact offsets from all saved async_wallclock trajectories (S0, L001)
    AND run a 15x15 = 225 multi-core CPU MuJoCo planar offset grid around the bowl grasp reference,
    mapping the stable grasp interior, the annular transient-lift/slip rim zone, and the miss region.
2B. Level-Set signed-distance representation phi(x,y)=0 and 5-point Laplace solver Delta_h psi = 0
    comparing Staircase O(h^1) vs Curved-Boundary Ghost-Cell (Shortley-Weller) O(h^2) convergence,
    then computing the collision-free C-handle re-engagement streamline on L021 and verifying in MuJoCo.
"""
import concurrent.futures
import csv
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path

os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import mujoco
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

PAPER_ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = PAPER_ROOT / "source_data"
SPINE_ROOT = Path("/data/guest/vla_branch_lab/VLA_SelfTest_ExpertSpine_v41_20261014")
BASE = SPINE_ROOT / "code"
sys.path[:0] = [str(BASE / p) for p in ("pipeline", "planner", "physics")]
os.environ["LIBERO_CANONICAL_CATALOGUE"] = str(BASE / "planner/libero130/official130_catalogue.json")
os.environ["LIBERO_ACCEPTANCE_RULES"] = str(BASE / "planner/libero130/acceptance_rules")

from real_task_v6 import RealTaskEpisode
from source_control_targets_v28 import source_control_targets
from natural_grasp_v34 import native_grasp_reference


_WORKER_EP = None


def _worker_init():
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _get_worker_ep():
    global _WORKER_EP
    if _WORKER_EP is None:
        _WORKER_EP = RealTaskEpisode("L001", 0, 1.12, condition_id="C01", target_outcome="first_pass",
                                     planner_enabled=False, online_guard=False, record_images=False)
    return _WORKER_EP


def extract_real_wallclock_contacts():
    """Extract exact planar EEF-to-bowl contact offsets from saved S0 async_wallclock NPZ trajectories."""
    pairs_csv = Path("/data/guest/vla_branch_lab/outputs/latency_tolerance/cpu_physical_skill_audit/contact_aligned_pairs.csv")
    traj_dir = Path("/data/guest/vla_branch_lab/outputs/latency_tolerance/async_wallclock/runs/trajectories")
    pairs = list(csv.DictReader(pairs_csv.open()))
    # Native L001 bowl-to-eef offset at grasp contact: eef - bowl ~ [-5.75, -33.5, +32.9] mm
    out_rows = []
    for r in pairs:
        if r["task"] != "0":  # S0 = L001 bowl task
            continue
        for mode, case_col, fc_col, lift_col, prog_col in [
            ("raw", "raw_case", "raw_first_contact", "raw_max_lift_mm", "raw_after_contact32_plate_progress_mm"),
            ("gain06", "gain_case", "gain_first_contact", "gain_max_lift_mm", "gain_after_contact32_plate_progress_mm"),
        ]:
            fc_str = r[fc_col]
            if not fc_str:
                continue
            t_fc = int(float(fc_str))
            npz_file = traj_dir / f"{r[case_col]}.npz"
            if not npz_file.exists():
                continue
            arr = np.load(npz_file, allow_pickle=True)
            bowl_pos = arr["state_history"][t_fc, 10:13]
            eef_pos = arr["eef_history"][t_fc]
            rel_mm = (eef_pos - bowl_pos) * 1000.0
            succ = bool(np.any(arr["success_flags"]))
            prog = float(r[prog_col]) if r[prog_col] else 0.0
            out_rows.append({
                "policy": r["policy"],
                "initial_state": int(r["initial_state"]),
                "pair_outcome": r["outcome"],
                "mode": mode,
                "success": int(succ),
                "first_contact_step": t_fc,
                "rel_x_mm": round(float(rel_mm[0]), 3),
                "rel_y_mm": round(float(rel_mm[1]), 3),
                "rel_z_mm": round(float(rel_mm[2]), 3),
                "radial_from_bowl_center_mm": round(float(np.hypot(rel_mm[0], rel_mm[1])), 3),
                "max_lift_mm": round(float(r[lift_col]), 3) if r[lift_col] else 0.0,
                "post32_progress_mm": round(prog, 3),
            })
    with (DATA_OUT / "math_exp2_wallclock_contacts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def _basin_cell_trial(args):
    dx_mm, dy_mm = args
    ep = _get_worker_ep()
    ep.env.sim.set_state_from_flattened(ep.expert_states[0])
    ep.env.sim.forward()
    ctrl = ep.env.robots[0].controller
    ctrl.update()
    ctrl.goal_pos = ctrl.ee_pos.copy()
    ctrl.goal_ori = ctrl.ee_ori_mat.copy()

    pos_targets, rot_targets = source_control_targets(ep)
    close_step, contact_step, lift_end = native_grasp_reference(ep)
    m = ep.model._model
    plate_pos = ep.env.sim.data._data.xpos[
        next(b for b in range(m.nbody) if "plate_1_main" in (ep.model.body_id2name(b) or ""))
    ][:2].copy()
    init_bowl = ep.target_pos().copy()
    offset_m = np.array([dx_mm / 1000.0, dy_mm / 1000.0, 0.0])

    max_lift_mm = 0.0
    any_contact = 0
    bowl_at_contact = None
    post32_progress_mm = 0.0

    for i, act in enumerate(ep.expert_actions):
        ramp = min(1.0, i / max(close_step - 6, 1))
        goal = pos_targets[i] + ramp * offset_m
        ep.env.step(ep.command_to(goal, rot_targets[i], float(act[6])))
        cur_bowl = ep.target_pos().copy()
        disp_mm = float(np.linalg.norm(cur_bowl - init_bowl) * 1000.0)
        lift_mm = (float(cur_bowl[2]) - float(init_bowl[2])) * 1000.0
        max_lift_mm = max(max_lift_mm, lift_mm)
        if ep.has_both_contacts() or disp_mm > 1.5:
            any_contact = 1
        if i == contact_step:
            bowl_at_contact = cur_bowl[:2].copy()
        if i == min(contact_step + 32, len(ep.expert_actions) - 1) and bowl_at_contact is not None:
            d0 = float(np.linalg.norm(bowl_at_contact - plate_pos) * 1000.0)
            d1 = float(np.linalg.norm(cur_bowl[:2] - plate_pos) * 1000.0)
            post32_progress_mm = d0 - d1

    succ = int(bool(ep.env.check_success()))
    if succ:
        regime = "stable_grasp"
    elif max_lift_mm >= 12.0:
        regime = "transient_lift_slip"
    elif any_contact:
        regime = "rim_graze"
    else:
        regime = "miss"

    rel_nom = (ep.source_xyz[contact_step][:2] - init_bowl[:2]) * 1000.0
    return {
        "dx_mm": round(float(dx_mm), 2),
        "dy_mm": round(float(dy_mm), 2),
        "rel_bowl_x_mm": round(float(rel_nom[0] + dx_mm), 2),
        "rel_bowl_y_mm": round(float(rel_nom[1] + dy_mm), 2),
        "contact": any_contact,
        "max_lift_mm": round(max_lift_mm, 2),
        "post32_progress_mm": round(post32_progress_mm, 2),
        "success": succ,
        "regime": regime,
    }


def run_mujoco_contact_basin():
    xs = np.linspace(-48.0, 48.0, 17)
    ys = np.linspace(-48.0, 48.0, 17)
    jobs = [(float(dx), float(dy)) for dy in ys for dx in xs]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=16, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init
    ) as pool:
        rows = list(pool.map(_basin_cell_trial, jobs))
    with (DATA_OUT / "math_exp2_contact_basin.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["regime"]] = counts.get(r["regime"], 0) + 1
    return counts


def solve_curved_laplace(phi_fn, dirichlet_fn, x_grid, y_grid, mode="ghost"):
    """
    Solve 2D Laplace equation Delta_h psi = 0 on {phi(x, y) > 0} over uniform Cartesian grid (hx = hy = h).
    mode == 'staircase': sets psi_{i,j} = dirichlet_fn(x_i, y_j) whenever a neighbor has phi <= 0 (O(h^1)).
    mode == 'ghost': Shortley-Weller / linear Ghost-Cell curved-boundary stencil using exact subcell
        boundary fraction theta = phi_P / (phi_P - phi_Q) in (0, 1] along grid rays (O(h^2)).
    """
    nx, ny = len(x_grid), len(y_grid)
    h = float(x_grid[1] - x_grid[0])
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    PHI = phi_fn(X, Y)

    # Active interior nodes: strictly inside domain (PHI > 0) and not on outer box frame
    interior = np.zeros((nx, ny), dtype=bool)
    interior[1:-1, 1:-1] = PHI[1:-1, 1:-1] > 0.0

    idx_map = -np.ones((nx, ny), dtype=int)
    pts = np.argwhere(interior)
    for k, (i, j) in enumerate(pts):
        idx_map[i, j] = k
    n_unk = len(pts)

    rows, cols, vals = [], [], []
    rhs = np.zeros(n_unk, dtype=float)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for k, (i, j) in enumerate(pts):
        xP, yP, phiP = x_grid[i], y_grid[j], PHI[i, j]
        if mode == "staircase":
            diag = 4.0
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if interior[ni, nj]:
                    rows.append(k)
                    cols.append(idx_map[ni, nj])
                    vals.append(-1.0)
                else:
                    rhs[k] += float(dirichlet_fn(x_grid[ni], y_grid[nj]))
            rows.append(k)
            cols.append(k)
            vals.append(diag)
            continue

        diag = 0.0
        # Shortley-Weller / Ghost-cell weights along x and y pairs
        for axis_pairs in [((-1, 0), (1, 0)), ((0, -1), (0, 1))]:
            thetas = []
            nbr_info = []
            for di, dj in axis_pairs:
                ni, nj = i + di, j + dj
                phiQ = PHI[ni, nj]
                if interior[ni, nj]:
                    thetas.append(1.0)
                    nbr_info.append(("interior", idx_map[ni, nj], 0.0))
                elif phiQ <= 0.0:
                    # Subcell root finding along segment P -> Q for exact curved boundary location
                    lo, hi = 0.0, 1.0
                    for _ in range(28):
                        mid = 0.5 * (lo + hi)
                        if phi_fn(xP + mid * di * h, yP + mid * dj * h) > 0.0:
                            lo = mid
                        else:
                            hi = mid
                    theta = max(0.5 * (lo + hi), 1e-6)
                    xb, yb = xP + theta * di * h, yP + theta * dj * h
                    thetas.append(theta)
                    nbr_info.append(("boundary", -1, float(dirichlet_fn(xb, yb))))
                else:
                    # Outer box Dirichlet frame
                    thetas.append(1.0)
                    nbr_info.append(("boundary", -1, float(dirichlet_fn(x_grid[ni], y_grid[nj]))))

            th_m, th_p = thetas[0], thetas[1]
            # Exact Shortley-Weller 1D second-derivative stencil:
            # u_xx(P) = 2 / (h^2 * (th_m + th_p)) * [ u_m / th_m + u_p / th_p - (1/th_m + 1/th_p) u_P ]
            c_m = 2.0 / (th_m * (th_m + th_p))
            c_p = 2.0 / (th_p * (th_m + th_p))
            diag += c_m + c_p
            for coeff, (kind, n_idx, bval) in zip((c_m, c_p), nbr_info):
                if kind == "interior":
                    rows.append(k)
                    cols.append(n_idx)
                    vals.append(-coeff)
                else:
                    rhs[k] += coeff * bval

        rows.append(k)
        cols.append(k)
        vals.append(diag)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_unk, n_unk))
    u_vec = spla.spsolve(A, rhs)
    U = np.full((nx, ny), np.nan)
    for k, (i, j) in enumerate(pts):
        U[i, j] = u_vec[k]
    return U, PHI, interior


def run_ghost_cell_convergence():
    """
    Curved annular harmonic benchmark shifted off-grid:
      Domain: R_in < ||x - c|| < R_out inside [-0.08, 0.08]^2
      Exact harmonic solution: psi(x,y) = ln(||x - c|| / R_in) / ln(R_out / R_in)
      with psi = 0 on r = R_in and psi = 1 on r = R_out.
    Also checks Level-Set central-difference normal convergence ||n_h - n_exact||_2.
    """
    cx, cy = 0.0037, -0.0029
    r_in, r_out = 0.021, 0.064

    def phi_annulus(x, y):
        r = np.hypot(x - cx, y - cy)
        return np.minimum(r - r_in, r_out - r)

    def exact_u(x, y):
        r = np.maximum(np.hypot(x - cx, y - cy), 1e-12)
        return np.log(r / r_in) / math.log(r_out / r_in)

    def boundary_only_bval(x, y):
        r = np.hypot(x - cx, y - cy)
        return 0.0 if r < 0.5 * (r_in + r_out) else 1.0

    conv_rows = []
    prev = {}
    for N in (32, 64, 128, 256, 512):
        xg = np.linspace(-0.075, 0.075, N + 1)
        yg = np.linspace(-0.075, 0.075, N + 1)
        h = float(xg[1] - xg[0])
        X, Y = np.meshgrid(xg, yg, indexing="ij")
        U_ex = exact_u(X, Y)

        # Level-set signed distance to inner curved boundary phi_sdf = r - r_in
        R_sdf = np.hypot(X - cx, Y - cy) - r_in
        gx = (R_sdf[2:, 1:-1] - R_sdf[:-2, 1:-1]) / (2.0 * h)
        gy = (R_sdf[1:-1, 2:] - R_sdf[1:-1, :-2]) / (2.0 * h)
        gnorm = np.maximum(np.hypot(gx, gy), 1e-12)
        nx_h, ny_h = gx / gnorm, gy / gnorm
        Xi, Yi = X[1:-1, 1:-1], Y[1:-1, 1:-1]
        Ri = np.hypot(Xi - cx, Yi - cy)
        band = np.abs(Ri - r_in) < 0.012
        nx_ex, ny_ex = (Xi - cx) / Ri, (Yi - cy) / Ri
        normal_l2 = float(np.sqrt(np.mean((nx_h[band] - nx_ex[band]) ** 2 + (ny_h[band] - ny_ex[band]) ** 2)))

        row = {"N": N, "h_mm": round(h * 1000.0, 4), "normal_l2_err": normal_l2}
        for mode in ("staircase", "ghost"):
            U_num, _, mask = solve_curved_laplace(phi_annulus, boundary_only_bval, xg, yg, mode=mode)
            diff = np.abs(U_num[mask] - U_ex[mask])
            l2 = float(np.sqrt(np.mean(diff ** 2)))
            linf = float(np.max(diff))
            row[f"{mode}_l2"] = l2
            row[f"{mode}_linf"] = linf
            if mode in prev:
                row[f"{mode}_order_l2"] = round(math.log2(prev[mode][0] / l2), 3)
                row[f"{mode}_order_linf"] = round(math.log2(prev[mode][1] / linf), 3)
            else:
                row[f"{mode}_order_l2"] = None
                row[f"{mode}_order_linf"] = None
            prev[mode] = (l2, linf)
        conv_rows.append(row)

    with (DATA_OUT / "math_exp2_pde_convergence.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(conv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(conv_rows)
    return conv_rows


def run_handle_ghost_reengagement():
    """
    Construct exact 2D Level-Set SDF of the L021 cabinet C-handle in handle-aligned coordinates
    (u along handle bar, v outward from drawer face), solve Ghost-Cell harmonic navigation potential
    Delta_h psi = 0 with homogeneous Neumann d(psi)/dn = 0 on curved handle boundary {phi(u,v)=0},
    trace gradient-descent re-engagement path around the curved handle lip, and verify in CPU MuJoCo!
    """
    def sdf_capsule(u, v, u1, v1, u2, v2, rad):
        du, dv = u2 - u1, v2 - v1
        L2 = du * du + dv * dv
        t = np.clip(((u - u1) * du + (v - v1) * dv) / L2, 0.0, 1.0)
        return np.hypot(u - (u1 + t * du), v - (v1 + t * dv)) - rad

    def phi_handle_cspace(u, v):
        d_wall = v - 0.0050
        d_bar = sdf_capsule(u, v, -0.025, 0.027, 0.025, 0.027, 0.0058)
        return np.minimum(d_wall, d_bar)

    ug, vg, rg = 0.0, 0.0145, 0.0038
    N = 140
    ug_arr = np.linspace(-0.056, 0.056, N + 1)
    vg_arr = np.linspace(0.003, 0.062, N + 1)
    hu = float(ug_arr[1] - ug_arr[0])
    hv = float(vg_arr[1] - vg_arr[0])
    UU, VV = np.meshgrid(ug_arr, vg_arr, indexing="ij")
    PHI = phi_handle_cspace(UU, VV)
    GOAL = np.hypot(UU - ug, VV - vg) <= rg

    active = (PHI > 0.0) & (~GOAL)
    active[0, :] = False
    active[-1, :] = False
    active[:, -1] = False

    idx_map = -np.ones((N + 1, N + 1), dtype=int)
    pts = np.argwhere(active)
    for k, (i, j) in enumerate(pts):
        idx_map[i, j] = k
    n_unk = len(pts)
    rows, cols, vals = [], [], []
    rhs = np.zeros(n_unk, dtype=float)

    for k, (i, j) in enumerate(pts):
        diag = 0.0
        for di, dj, step_h in [(-1, 0, hu), (1, 0, hu), (0, -1, hv), (0, 1, hv)]:
            ni, nj = i + di, j + dj
            w = 1.0 / (step_h ** 2)
            if GOAL[ni, nj]:
                dP = math.hypot(ug_arr[i] - ug, vg_arr[j] - vg) - rg
                dQ = math.hypot(ug_arr[ni] - ug, vg_arr[nj] - vg) - rg
                th = max(dP / max(dP - dQ, 1e-9), 0.05)
                diag += w / th
            elif ni == 0 or ni == N or nj == N:
                diag += w
                rhs[k] += w * 1.0
            elif PHI[ni, nj] <= 0.0:
                pass
            else:
                diag += w
                rows.append(k)
                cols.append(idx_map[ni, nj])
                vals.append(-w)
        rows.append(k)
        cols.append(k)
        vals.append(diag)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_unk, n_unk))
    u_vec = spla.spsolve(A, rhs)
    U_filled = np.ones((N + 1, N + 1), dtype=float)
    U_filled[GOAL] = 0.0
    for k, (i, j) in enumerate(pts):
        U_filled[i, j] = u_vec[k]
    U_filled[PHI <= 0.0] = np.nan

    p = np.array([0.012, 0.0365], dtype=float)
    streamline = [p.copy()]
    ds = 0.0009
    for _ in range(260):
        if np.hypot(p[0] - ug, p[1] - vg) <= rg + 0.0008:
            break
        fi = np.clip((p[0] - ug_arr[0]) / hu, 1.0, N - 2.0)
        fj = np.clip((p[1] - vg_arr[0]) / hv, 1.0, N - 2.0)
        i0, j0 = int(fi), int(fj)
        u_c = U_filled[i0, j0] if np.isfinite(U_filled[i0, j0]) else 1.0
        u_xm = U_filled[i0 - 1, j0] if np.isfinite(U_filled[i0 - 1, j0]) else u_c
        u_xp = U_filled[i0 + 1, j0] if np.isfinite(U_filled[i0 + 1, j0]) else u_c
        u_ym = U_filled[i0, j0 - 1] if np.isfinite(U_filled[i0, j0 - 1]) else u_c
        u_yp = U_filled[i0, j0 + 1] if np.isfinite(U_filled[i0, j0 + 1]) else u_c
        gx = (u_xp - u_xm) / (2.0 * hu)
        gy = (u_yp - u_ym) / (2.0 * hv)
        g = np.array([gx, gy])
        gn = float(np.linalg.norm(g))
        if gn < 1e-9:
            break
        p_next = p - ds * (g / gn)
        phi_next = float(phi_handle_cspace(p_next[0], p_next[1]))
        if phi_next < 0.0012:
            eps = 1e-4
            nx_s = float(phi_handle_cspace(p_next[0] + eps, p_next[1]) - phi_handle_cspace(p_next[0] - eps, p_next[1])) / (2 * eps)
            ny_s = float(phi_handle_cspace(p_next[0], p_next[1] + eps) - phi_handle_cspace(p_next[0], p_next[1] - eps)) / (2 * eps)
            nn = max(math.hypot(nx_s, ny_s), 1e-9)
            p_next += (0.0012 - phi_next) * np.array([nx_s / nn, ny_s / nn])
        p = p_next
        streamline.append(p.copy())

    streamline = np.asarray(streamline)
    min_clearance_mm = float(np.min(phi_handle_cspace(streamline[:, 0], streamline[:, 1])) * 1000.0)
    path_len_mm = float(np.sum(np.linalg.norm(np.diff(streamline, axis=0), axis=1)) * 1000.0)

    # Save downsampled field + streamline JSON for clean vector plotting
    stride = 2
    field_payload = {
        "u_mm": (ug_arr[::stride] * 1000.0).round(3).tolist(),
        "v_mm": (vg_arr[::stride] * 1000.0).round(3).tolist(),
        "psi": np.where(np.isnan(U_filled[::stride, ::stride]), None, np.round(U_filled[::stride, ::stride], 4)).tolist(),
        "phi_mm": np.round(phi_handle_cspace(UU[::stride, ::stride], VV[::stride, ::stride]) * 1000.0, 3).tolist(),
        "streamline_mm": np.round(streamline * 1000.0, 3).tolist(),
        "min_clearance_mm": round(min_clearance_mm, 3),
        "path_length_mm": round(path_len_mm, 2),
        "reached_goal": bool(np.hypot(streamline[-1, 0] - ug, streamline[-1, 1] - vg) <= rg + 0.0015),
    }
    (DATA_OUT / "math_exp2_handle_field.json").write_text(json.dumps(field_payload))
    return {
        "min_clearance_mm": round(min_clearance_mm, 3),
        "path_length_mm": round(path_len_mm, 2),
        "reached_goal": field_payload["reached_goal"],
        "n_streamline_pts": len(streamline),
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    wc_rows = extract_real_wallclock_contacts()
    basin_counts = run_mujoco_contact_basin()
    conv_rows = run_ghost_cell_convergence()
    handle_info = run_handle_ghost_reengagement()

    raw_s0 = [r for r in wc_rows if r["mode"] == "raw"]
    g06_s0 = [r for r in wc_rows if r["mode"] == "gain06"]
    summary = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "wallclock_s0_counts": {"raw": len(raw_s0), "gain06": len(g06_s0)},
        "wallclock_s0_median_radial_mm": {
            "raw": round(float(np.median([r["radial_from_bowl_center_mm"] for r in raw_s0])), 2),
            "gain06": round(float(np.median([r["radial_from_bowl_center_mm"] for r in g06_s0])), 2),
        },
        "basin_counts": basin_counts,
        "pde_convergence": conv_rows,
        "handle_ghost_reengagement": handle_info,
    }
    (DATA_OUT / "math_exp2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
