"""
spectral_analysis.py

Spectral analysis of the Newton Jacobian at different warm-starts.

For a given sample and a list of checkpoints (models trained with different
solver budgets M), this script:
  1. Loads each model, runs forward pass → warm-start prediction.
  2. Assembles the Newton Jacobian at the warm-start using NGSolve.
  3. Also assembles the Jacobian at the converged solution (cold-start reference).
  4. Computes the k smallest eigenvalues for each Jacobian.
  5. Plots the eigenvalue spectra side-by-side.

Usage: python spectral_analysis.py   (prompts for all inputs interactively)
"""

import os
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])

import torch
import yaml
import h5py

from ngsolve import *
from netgen.occ import *

from solver import sampling_points_2d_grid, solve_pde
from utils_train_with_solver import (
    PDEDataset, create_model, get_precision_config,
    convert_model_precision, apply_bc_mask_np, resolve_run_paths
)

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8,
})


# ─────────────────────────────────────────────────────────────────────────────
# Jacobian assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_jacobian(u_field, k, n, A, mu1, mu2,
                      dirichlet_bcs, dirichlet_value,
                      x_pts, y_pts, maxh=0.1):
    """
    Assemble the Newton Jacobian J = dR/du at the field u_field without running
    the full Newton solve.

    Parameters
    ----------
    u_field : np.ndarray, shape (n_pts, n_pts) or (1, n_pts, n_pts)
        Field at which to evaluate the Jacobian. If None, uses zero field.

    Returns
    -------
    J_free : scipy.sparse.csr_matrix
        Jacobian restricted to free (interior) DOFs.
    free_dof_indices : list[int]
        Indices of free DOFs (for reference).
    ndof_free : int
    """
    SetNumThreads(2)

    shape = Rectangle(1, 1).Face()
    shape.edges.Min(X).name = "left"
    shape.edges.Max(X).name = "right"
    shape.edges.Min(Y).name = "bottom"
    shape.edges.Max(Y).name = "top"
    geom = OCCGeometry(shape, dim=2)
    mesh = Mesh(geom.GenerateMesh(maxh=maxh))

    bcs_clean = [b for b in dirichlet_bcs if b != 0]

    V = H1(mesh, order=3, dirichlet=bcs_clean)
    u, v = V.TnT()
    gfu = GridFunction(V)
    gfu.Set(CF(float(dirichlet_value)), BND)

    # Interpolate u_field onto mesh
    if u_field is not None:
        field_2d = u_field[0] if u_field.ndim == 3 else u_field
        init_cf = VoxelCoefficient((0, 0), (1, 1), np.float64(field_2d), linear=True)
        gfu_init = GridFunction(V)
        gfu_init.Set(init_cf)
        free_dofs = V.FreeDofs()
        for dof in range(V.ndof):
            if free_dofs[dof]:
                gfu.vec.data[dof] = gfu_init.vec[dof]

    # Set up weak form
    f_cf = A * (mu1**2 + mu2**2 + x**2 + y**2) * sin(mu1 * pi * x) * sin(mu2 * pi * y)
    a = BilinearForm(V)
    a += (grad(u) * grad(v) + (k**2) * u**n * v - f_cf * v) * dx

    # Assemble linearization at gfu
    a.AssembleLinearization(gfu.vec)

    # Extract free DOF indices
    fd = V.FreeDofs()
    free_idx = [i for i in range(V.ndof) if fd[i]]
    ndof_free = len(free_idx)

    # Convert full Jacobian to scipy sparse via COO
    try:
        rows_all, cols_all, vals_all = a.mat.COO()
    except AttributeError:
        # fallback: dense (only feasible for small problems)
        J_dense = np.array(a.mat.ToDense())
        free_idx_arr = np.array(free_idx)
        J_free_dense = J_dense[np.ix_(free_idx_arr, free_idx_arr)]
        return sp.csr_matrix(J_free_dense), free_idx, ndof_free

    rows_all = np.array(rows_all, dtype=np.int32)
    cols_all = np.array(cols_all, dtype=np.int32)
    vals_all = np.array(vals_all, dtype=np.float64)

    ndof_total = V.ndof
    J_full = sp.csr_matrix((vals_all, (rows_all, cols_all)),
                            shape=(ndof_total, ndof_total))

    # Restrict to free DOFs
    free_idx_arr = np.array(free_idx, dtype=np.int32)
    J_free = J_full[np.ix_(free_idx_arr, free_idx_arr)].tocsr()

    return J_free, free_idx, ndof_free


# ─────────────────────────────────────────────────────────────────────────────
# Newton PDE residual norm
# ─────────────────────────────────────────────────────────────────────────────

def compute_newton_residual(u_field, k, n, A, mu1, mu2,
                            dirichlet_bcs, dirichlet_value,
                            x_pts, y_pts, maxh=0.1):
    """
    Evaluate the nonlinear PDE residual F(u_field) and return
    its free-DOF L2 norm:  ||F(u)||_2  (unscaled and relative).

    Uses BilinearForm.Apply which computes y = A(x) for a nonlinear form,
    i.e. the assembled residual vector at u_field.
    """
    SetNumThreads(2)

    shape = Rectangle(1, 1).Face()
    shape.edges.Min(X).name = "left"
    shape.edges.Max(X).name = "right"
    shape.edges.Min(Y).name = "bottom"
    shape.edges.Max(Y).name = "top"
    geom = OCCGeometry(shape, dim=2)
    mesh = Mesh(geom.GenerateMesh(maxh=maxh))

    bcs_clean = [b for b in dirichlet_bcs if b != 0]
    V = H1(mesh, order=3, dirichlet=bcs_clean)
    u, v = V.TnT()
    gfu = GridFunction(V)
    gfu.Set(CF(float(dirichlet_value)), BND)

    if u_field is not None:
        field_2d = u_field[0] if u_field.ndim == 3 else u_field
        init_cf = VoxelCoefficient((0, 0), (1, 1), np.float64(field_2d), linear=True)
        gfu_init = GridFunction(V)
        gfu_init.Set(init_cf)
        free_dofs = V.FreeDofs()
        for dof in range(V.ndof):
            if free_dofs[dof]:
                gfu.vec.data[dof] = gfu_init.vec[dof]

    f_cf = A * (mu1**2 + mu2**2 + x**2 + y**2) * sin(mu1 * pi * x) * sin(mu2 * pi * y)
    a = BilinearForm(V)
    a += (grad(u) * grad(v) + (k**2) * u**n * v - f_cf * v) * dx

    res = gfu.vec.CreateVector()
    a.Apply(gfu.vec, res)

    fd = V.FreeDofs()
    free_idx = np.array([i for i in range(V.ndof) if fd[i]], dtype=np.int32)
    res_np = res.FV().NumPy()[free_idx]

    return float(np.linalg.norm(res_np))


# ─────────────────────────────────────────────────────────────────────────────
# Eigenvalue computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_smallest_eigenvalues(J_sparse, k=20):
    """
    Compute k smallest eigenvalues of J_sparse (real part, sorted ascending).

    Uses shift-invert ARPACK (sigma=0) for reliable small-eigenvalue extraction.
    Falls back to dense eigvalsh for small matrices.
    """
    n = J_sparse.shape[0]
    k = min(k, n - 2)  # ARPACK requires k < n

    if n <= 2000:
        # Dense path — exact
        J_dense = J_sparse.toarray()
        evals = np.linalg.eigvalsh(J_dense)
        return np.sort(evals)[:k]
    else:
        try:
            evals, _ = spla.eigsh(J_sparse, k=k, sigma=0.0, which='LM',
                                  tol=1e-10, maxiter=10000)
            return np.sort(np.real(evals))
        except Exception as e:
            print(f"  [eigsh shift-invert failed: {e}] — falling back to dense")
            J_dense = J_sparse.toarray()
            evals = np.linalg.eigvalsh(J_dense)
            return np.sort(evals)[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Condition number (sparse, efficient)
# ─────────────────────────────────────────────────────────────────────────────

def _cond_from_sparse(J_sparse):
    """
    Estimate condition number κ = λ_max / |λ_min| of a symmetric sparse matrix.
    Uses eigsh for large matrices; falls back to dense for small ones.
    """
    n = J_sparse.shape[0]
    if n <= 2000:
        evals = np.linalg.eigvalsh(J_sparse.toarray())
        return float(np.max(np.abs(evals)) / max(np.min(np.abs(evals)), 1e-300))
    try:
        lam_max = float(np.abs(spla.eigsh(J_sparse, k=1, which='LM',
                                           tol=1e-8, maxiter=5000, return_eigenvectors=False)[0]))
        lam_min = float(np.abs(spla.eigsh(J_sparse, k=1, sigma=0.0, which='LM',
                                           tol=1e-8, maxiter=5000, return_eigenvectors=False)[0]))
        return lam_max / max(lam_min, 1e-300)
    except Exception:
        evals = np.linalg.eigvalsh(J_sparse.toarray())
        return float(np.max(np.abs(evals)) / max(np.min(np.abs(evals)), 1e-300))


# ─────────────────────────────────────────────────────────────────────────────
# Newton solve with per-step Jacobian tracking
# ─────────────────────────────────────────────────────────────────────────────

def solve_with_jacobian_tracking(u_field, k, n, A, mu1, mu2,
                                 dirichlet_bcs, dirichlet_value,
                                 x_pts, y_pts, maxh=0.1,
                                 maxiters=100, linear_solver='direct',
                                 cg_maxiter=1000, cg_tol=1e-12,
                                 dampfactor=1.0):
    """
    Newton solve from u_field (or zero if None) that assembles J at every step
    and records cond(J^(k)) = κ(J) at each Newton iterate.

    Returns
    -------
    cond_per_step : list[float]   κ(J^(k)) at k = 0, 1, 2, …
    n_iters       : int
    residuals     : list[float]   energy-norm residual per step
    """
    SetNumThreads(2)

    shape = Rectangle(1, 1).Face()
    shape.edges.Min(X).name = "left"
    shape.edges.Max(X).name = "right"
    shape.edges.Min(Y).name = "bottom"
    shape.edges.Max(Y).name = "top"
    geom = OCCGeometry(shape, dim=2)
    mesh = Mesh(geom.GenerateMesh(maxh=maxh))

    bcs_clean = [b for b in dirichlet_bcs if b != 0]
    V = H1(mesh, order=3, dirichlet=bcs_clean)
    u, v = V.TnT()
    gfu = GridFunction(V)
    gfu.Set(CF(float(dirichlet_value)), BND)

    if u_field is not None:
        field_2d = u_field[0] if u_field.ndim == 3 else u_field
        init_cf = VoxelCoefficient((0, 0), (1, 1), np.float64(field_2d), linear=True)
        gfu_init = GridFunction(V)
        gfu_init.Set(init_cf)
        free_dofs = V.FreeDofs()
        for dof in range(V.ndof):
            if free_dofs[dof]:
                gfu.vec.data[dof] = gfu_init.vec[dof]

    f_cf = A * (mu1**2 + mu2**2 + x**2 + y**2) * sin(mu1 * pi * x) * sin(mu2 * pi * y)
    a = BilinearForm(V)
    a += (grad(u) * grad(v) + (k**2) * u**n * v - f_cf * v) * dx

    res = gfu.vec.CreateVector()
    du  = gfu.vec.CreateVector()
    fd  = V.FreeDofs()
    free_idx = np.array([i for i in range(V.ndof) if fd[i]], dtype=np.int32)

    a.Apply(gfu.vec, res)
    res0_L2 = float(sqrt(abs(InnerProduct(res, res))))

    cond_per_step = []
    residuals     = []
    n_iters       = 0

    res_atol, res_rtol = 1e-6, 1e-8
    du_atol,  du_rtol  = 1e-18, 1e-14

    with TaskManager():
        for it in range(maxiters):
            a.Apply(gfu.vec, res)
            a.AssembleLinearization(gfu.vec)

            # --- extract free-DOF Jacobian and record κ(J^(k)) ---
            try:
                rows_all, cols_all, vals_all = a.mat.COO()
                rows_all = np.array(rows_all, dtype=np.int32)
                cols_all = np.array(cols_all, dtype=np.int32)
                vals_all = np.array(vals_all, dtype=np.float64)
                J_full = sp.csr_matrix((vals_all, (rows_all, cols_all)),
                                       shape=(V.ndof, V.ndof))
                J_free = J_full[np.ix_(free_idx, free_idx)].tocsr()
            except AttributeError:
                J_full_dense = np.array(a.mat.ToDense())
                J_free = sp.csr_matrix(J_full_dense[np.ix_(free_idx, free_idx)])
            cond_per_step.append(_cond_from_sparse(J_free))

            if linear_solver == 'cg':
                pre = a.mat.CreateSmoother(fd)
                from ngsolve.krylovspace import CGSolver
                inv = CGSolver(mat=a.mat, pre=pre, maxiter=cg_maxiter,
                               tol=cg_tol, printrates=False)
                du.data = inv * res
            else:
                du.data = a.mat.Inverse(fd) * res

            alpha_damp = min(1.0, dampfactor * (it + 1))
            unorm_old  = float(sqrt(abs(InnerProduct(gfu.vec, gfu.vec))))
            gfu.vec.data -= alpha_damp * du

            stopcritval  = float(sqrt(abs(InnerProduct(du, du))))
            residual_L2  = float(sqrt(abs(InnerProduct(res, res))))
            energy_L2    = float(sqrt(abs(InnerProduct(du, res))))
            residuals.append(energy_L2)
            n_iters = it + 1

            if residual_L2 < res_atol + res_rtol * res0_L2:
                break
            if stopcritval < du_atol + du_rtol * unorm_old:
                break

    return cond_per_step, n_iters, residuals


# ─────────────────────────────────────────────────────────────────────────────
# Model inference → warm-start
# ─────────────────────────────────────────────────────────────────────────────

def get_warm_start(ckpt_path, config, sample_data, device):
    """
    Load checkpoint, run inference, apply BC mask → warm-start array (n_pts, n_pts).
    """
    precision_config = get_precision_config(config)
    model = create_model(config)
    model = convert_model_precision(model, precision_config['model_dtype'])
    model = model.to(device)

    ckpt = torch.load(ckpt_path, weights_only=False)
    sd = ckpt['model_state_dict']
    sd.pop('_metadata', None)
    model.load_state_dict(sd)
    model.eval()

    prec = get_precision_config(config)
    inp = torch.from_numpy(sample_data['input_tensor']).unsqueeze(0) \
              .to(device=device, dtype=prec['input_dtype'])

    with torch.no_grad():
        pred = model(inp)

    pred_np = pred.squeeze().cpu().to(dtype=torch.float64).numpy()

    bcs = sample_data['bcs']
    bc_val = float(sample_data['bc_values'])
    n_pts = sample_data['input_tensor'].shape[-1]
    warm = apply_bc_mask_np(pred_np, bcs.tolist() if hasattr(bcs, 'tolist') else list(bcs),
                            bc_val, n_pts)
    del model
    return warm, pred_np


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_eigenvalue_spectra(averaged_results, plot_dir, sample_tag, n_eigs_show=20):
    """
    averaged_results : list of dicts with keys:
        label            : str
        evals            : np.ndarray  mean eigenvalues (sorted ascending)
        evals_std        : np.ndarray  std over samples (optional)
        is_ref           : bool
        cond             : float  mean condition number (optional)
        cond_std         : float  std condition number (optional)
        l2_warm_start_error     : float  mean warm-start L2 error (optional)
        l2_warm_start_error_std : float  std (optional)
    sample_tag : str  used in titles / filenames (e.g. "samples_20-22" or "sample_20")
    """
    os.makedirs(plot_dir, exist_ok=True)

    color_map = {r['label']: plt.cm.tab10(i / max(len(averaged_results), 1))
                 for i, r in enumerate(averaged_results)}
    mode_indices = np.arange(n_eigs_show)
    x_ticks = np.arange(0, n_eigs_show, 5)

    # ── 1. Eigenvalue vs mode index (mean ± std) ──────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for res in averaged_results:
        c = color_map[res['label']]
        evals = res['evals'][:n_eigs_show]
        idx = mode_indices[:len(evals)]
        ls = '-' if res.get('is_ref') else '--'
        lw = 2.0 if res.get('is_ref') else 1.2
        ax.plot(idx, evals, label=res['label'], color=c,
                linestyle=ls, linewidth=lw, marker='o', markersize=3)
        if 'evals_std' in res:
            std = res['evals_std'][:n_eigs_show]
            ax.fill_between(idx, evals - std, evals + std, color=c, alpha=0.15)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'Newton Jacobian eigenvalues  ({sample_tag})')
    ax.set_xticks(x_ticks)
    ax.legend(loc='upper left', frameon=True, framealpha=0.7)
    fig.tight_layout()
    out = os.path.join(plot_dir, f'eigenvalues_{sample_tag}.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f'  Saved: {out}')
    plt.close(fig)

    # ── 2. Condition number bar chart ──────────────────────────────────────
    cond_res = [r for r in averaged_results if 'cond' in r]
    if cond_res:
        labels    = [r['label'] for r in cond_res]
        cond_vals = [r['cond'] for r in cond_res]
        cond_errs = [r.get('cond_std', 0.0) for r in cond_res]
        fig2, ax2 = plt.subplots(figsize=(max(4, len(labels) * 0.9), 3.5))
        bar_colors = [color_map[lbl] for lbl in labels]
        ax2.bar(labels, cond_vals, yerr=cond_errs, color=bar_colors,
                error_kw=dict(elinewidth=1.2, capsize=4))
        ax2.set_ylabel('Condition number $\\kappa(J)$')
        ax2.set_title(f'Jacobian condition number  ({sample_tag})')
        ax2.set_yscale('log')
        plt.xticks(rotation=20, ha='right')
        fig2.tight_layout()
        out2 = os.path.join(plot_dir, f'condition_{sample_tag}.pdf')
        fig2.savefig(out2, bbox_inches='tight')
        print(f'  Saved: {out2}')
        plt.close(fig2)

    # ── 3. Warm-start residual bar chart ───────────────────────────────────
    ws_res = [r for r in averaged_results if 'l2_warm_start_error' in r]
    if ws_res:
        ws_labels = [r['label'] for r in ws_res]
        ws_vals   = [r['l2_warm_start_error'] for r in ws_res]
        ws_errs   = [r.get('l2_warm_start_error_std', 0.0) for r in ws_res]
        fig3, ax3 = plt.subplots(figsize=(max(4, len(ws_labels) * 0.9), 3.5))
        ws_colors = [color_map[lbl] for lbl in ws_labels]
        ax3.bar(ws_labels, ws_vals, yerr=ws_errs, color=ws_colors,
                error_kw=dict(elinewidth=1.2, capsize=4))
        ax3.set_xlabel('Solver budget')
        ax3.set_ylabel('Newton PDE residual $\\|F(u_{\\mathrm{warm}})\\|_2$')
        ax3.set_title(f'Newton PDE residual at warm-start  ({sample_tag})')
        ax3.set_yscale('log')
        plt.xticks(rotation=20, ha='right')
        fig3.tight_layout()
        out3 = os.path.join(plot_dir, f'warmstart_residual_{sample_tag}.pdf')
        fig3.savefig(out3, bbox_inches='tight')
        print(f'  Saved: {out3}')
        plt.close(fig3)


def plot_cond_per_newton_step(averaged_results, plot_dir, sample_tag):
    """
    Plot κ(J^(k)) vs Newton iteration k for each label.
    averaged_results entries must have 'cond_per_step' (mean array)
    and optionally 'cond_per_step_std'.
    """
    os.makedirs(plot_dir, exist_ok=True)

    has_data = [r for r in averaged_results if 'cond_per_step' in r]
    if not has_data:
        return

    color_map = {r['label']: plt.cm.tab10(i / max(len(averaged_results), 1))
                 for i, r in enumerate(averaged_results)}

    fig, ax = plt.subplots(figsize=(7, 4))
    for res in has_data:
        c   = color_map[res['label']]
        cps = np.array(res['cond_per_step'])
        xs  = np.arange(1, len(cps) + 1)
        ls  = '-' if res.get('is_ref') else '--'
        lw  = 2.0 if res.get('is_ref') else 1.2
        ax.semilogy(xs, cps, label=res['label'], color=c,
                    linestyle=ls, linewidth=lw, marker='o', markersize=3)
        if 'cond_per_step_std' in res:
            std = np.array(res['cond_per_step_std'])
            ax.fill_between(xs, np.maximum(cps - std, 1e-10), cps + std,
                            color=c, alpha=0.15)
    ax.set_xlabel('Newton iteration $k$')
    ax.set_ylabel(r'$\kappa(J^{(k)})$')
    ax.set_title(f'Jacobian condition number per Newton step  ({sample_tag})')
    ax.legend(loc='upper right', frameon=True, framealpha=0.7)
    fig.tight_layout()
    out = os.path.join(plot_dir, f'cond_per_step_{sample_tag}.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f'  Saved: {out}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sample(sample_idx, dataset, config, ckpt_entries, device, n_eigs):
    """
    Run cold-start reference + all warm-start analyses for one sample.
    Returns list of result dicts (one per label, including 'Converged (reference)').
    Each dict contains: label, evals, is_ref, iters, residuals,
    cond (float), and optionally l2_warm_start_error, t_warm_s, time_speedup_pct.
    """
    input_tensor, alpha, f_param, mu1, mu2, n, bcs, bc_values, orig_idx = dataset[sample_idx]
    sample_data = {
        'input_tensor': input_tensor,
        'alpha': float(alpha),
        'f_param': float(f_param),
        'mu1': float(mu1),
        'mu2': float(mu2),
        'n': int(n),
        'bcs': bcs,
        'bc_values': float(bc_values),
    }
    n_pts = config['data_config']['n_pts']
    x_pts, y_pts, *_ = sampling_points_2d_grid(n_pts=n_pts)
    solver_config = config.get('solver_config', {})

    print(f"\nSample {sample_idx} (orig idx {orig_idx}): "
          f"alpha={alpha:.3f}, n={n}, A={f_param:.3f}, "
          f"mu1={mu1:.3f}, mu2={mu2:.3f}, bcs={bcs}, bc_val={bc_values:.3f}")

    pde_kwargs = dict(
        k=sample_data['alpha'], n=sample_data['n'],
        A=sample_data['f_param'], mu1=sample_data['mu1'], mu2=sample_data['mu2'],
        dirichlet_bcs=bcs.tolist() if hasattr(bcs, 'tolist') else list(bcs),
        dirichlet_value=sample_data['bc_values'],
        x_pts=x_pts, y_pts=y_pts,
        maxiters=100, verbose=False,
        linear_solver=solver_config.get('linear_solver', 'direct'),
        cg_maxiter=int(solver_config.get('cg_maxiter', 1000)),
        cg_tol=float(solver_config.get('cg_tol', 1e-12)),
        dampfactor=1.0
    )

    jac_kwargs = dict(
        k=sample_data['alpha'], n=sample_data['n'],
        A=sample_data['f_param'], mu1=sample_data['mu1'], mu2=sample_data['mu2'],
        dirichlet_bcs=pde_kwargs['dirichlet_bcs'],
        dirichlet_value=sample_data['bc_values'],
        x_pts=x_pts, y_pts=y_pts,
        maxiters=100,
        linear_solver=solver_config.get('linear_solver', 'direct'),
        cg_maxiter=int(solver_config.get('cg_maxiter', 1000)),
        cg_tol=float(solver_config.get('cg_tol', 1e-12)),
        dampfactor=1.0,
    )

    results = []

    # Reference: cold-start with per-step Jacobian tracking
    print("  [Reference] Running cold-start solver with Jacobian tracking …")
    t0 = time.perf_counter()
    cond_ref_steps, iters_ref, resid_ref = solve_with_jacobian_tracking(
        u_field=None, **jac_kwargs
    )
    t_ref = time.perf_counter() - t0
    print(f"    Converged in {iters_ref} Newton iters  ({t_ref:.2f}s)")
    print(f"    κ(J) per step: {[f'{c:.2e}' for c in cond_ref_steps]}")

    # Jacobian at converged solution (for eigenvalue analysis)
    u_converged, _, _, _, _, _ = solve_pde(u_init=None, **pde_kwargs)
    J_ref, _, ndof_free = assemble_jacobian(
        u_converged,
        k=sample_data['alpha'], n=sample_data['n'],
        A=sample_data['f_param'], mu1=sample_data['mu1'], mu2=sample_data['mu2'],
        dirichlet_bcs=pde_kwargs['dirichlet_bcs'],
        dirichlet_value=sample_data['bc_values'],
        x_pts=x_pts, y_pts=y_pts,
    )
    evals_ref = compute_smallest_eigenvalues(J_ref, k=n_eigs)
    cond_ref = cond_ref_steps[-1] if cond_ref_steps else _cond_from_sparse(J_ref)
    print(f"    Free DOFs: {ndof_free}   cond(J_converged): {cond_ref:.4e}")

    results.append({
        'label': 'Converged (reference)',
        'evals': evals_ref,
        'is_ref': True,
        'iters': iters_ref,
        'residuals': resid_ref,
        'cond': cond_ref,
        'cond_per_step': cond_ref_steps,
        't_ref_s': t_ref,
    })

    # Warm-start per checkpoint
    for entry in ckpt_entries:
        ckpt_path = entry['path']
        label = entry['label']
        if not os.path.exists(ckpt_path):
            print(f"  [{label}] Checkpoint not found, skipping.")
            continue

        warm_start, _ = get_warm_start(ckpt_path, config, sample_data, device)
        pde_res = compute_newton_residual(
            warm_start[np.newaxis, :, :],
            k=sample_data['alpha'], n=sample_data['n'],
            A=sample_data['f_param'], mu1=sample_data['mu1'], mu2=sample_data['mu2'],
            dirichlet_bcs=pde_kwargs['dirichlet_bcs'],
            dirichlet_value=sample_data['bc_values'],
            x_pts=x_pts, y_pts=y_pts,
        )

        t1 = time.perf_counter()
        cond_ws_steps, iters_ws, resid_ws = solve_with_jacobian_tracking(
            u_field=warm_start[np.newaxis, :, :], **jac_kwargs
        )
        t_ws = time.perf_counter() - t1
        speedup_pct = (1.0 - t_ws / t_ref) * 100.0
        print(f"  [{label}] {iters_ws} iters  "
              f"pde_res={pde_res:.4e}  speedup={speedup_pct:+.1f}%")
        print(f"    κ(J) per step: {[f'{c:.2e}' for c in cond_ws_steps]}")

        J_ws, _, _ = assemble_jacobian(
            warm_start[np.newaxis, :, :],
            k=sample_data['alpha'], n=sample_data['n'],
            A=sample_data['f_param'], mu1=sample_data['mu1'], mu2=sample_data['mu2'],
            dirichlet_bcs=pde_kwargs['dirichlet_bcs'],
            dirichlet_value=sample_data['bc_values'],
            x_pts=x_pts, y_pts=y_pts,
        )
        evals_ws = compute_smallest_eigenvalues(J_ws, k=n_eigs)
        cond_ws = cond_ws_steps[0] if cond_ws_steps else _cond_from_sparse(J_ws)

        results.append({
            'label': label,
            'evals': evals_ws,
            'is_ref': False,
            'iters': iters_ws,
            'residuals': resid_ws,
            'cond': cond_ws,
            'cond_per_step': cond_ws_steps,
            'l2_warm_start_error': pde_res,
            'pde_res_abs': pde_res,
            't_warm_s': t_ws,
            'time_speedup_pct': speedup_pct,
        })

    return results


def aggregate_results(all_sample_results):
    """
    all_sample_results : list[list[dict]]  — outer: samples, inner: labels
    Returns averaged list[dict] with mean/std fields.
    """
    # Collect all labels in order from first sample
    labels_order = [r['label'] for r in all_sample_results[0]]

    averaged = []
    for label in labels_order:
        per_sample = [r for sample in all_sample_results for r in sample if r['label'] == label]
        if not per_sample:
            continue
        evals_stack = np.stack([r['evals'] for r in per_sample], axis=0)
        agg = {
            'label': label,
            'is_ref': per_sample[0]['is_ref'],
            'evals': evals_stack.mean(axis=0),
            'evals_std': evals_stack.std(axis=0),
            'cond': np.mean([r['cond'] for r in per_sample]),
            'cond_std': np.std([r['cond'] for r in per_sample]),
        }
        if 'l2_warm_start_error' in per_sample[0]:
            vals = [r['l2_warm_start_error'] for r in per_sample]
            agg['l2_warm_start_error'] = float(np.mean(vals))
            agg['l2_warm_start_error_std'] = float(np.std(vals))
        # aggregate per-step condition numbers (pad to max length with nan)
        if 'cond_per_step' in per_sample[0]:
            cps_list = [np.array(r['cond_per_step'], dtype=np.float64) for r in per_sample]
            max_len = max(len(c) for c in cps_list)
            padded = np.full((len(cps_list), max_len), np.nan)
            for i, c in enumerate(cps_list):
                padded[i, :len(c)] = c
            agg['cond_per_step']     = list(np.nanmean(padded, axis=0))
            agg['cond_per_step_std'] = list(np.nanstd(padded, axis=0))
        averaged.append(agg)
    return averaged


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Spectral Analysis of Newton Jacobian ===\n")

    # ── User inputs ────────────────────────────────────────────────────────
    config_path = './configs/unet_configs/config_256_solver_M5.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = resolve_run_paths(config)

    ckpt_entries = [
        {'path': './runs/256_UNet2D_dd/256_UNet2D_dd_20260824_1050/checkpoints/best_model.pt','label': 'DD'},
        {'path': './runs/256_UNet2D_solver_M1_fixed/256_UNet2D_solver_M1_fixed_20260824_1051/checkpoints/best_model.pt', 'label': 'M1'},
        {'path': './runs/256_UNet2D_solver_M3_fixed/256_UNet2D_solver_M3_fixed_20260824_1051/checkpoints/best_model.pt', 'label': 'M3'},
        {'path': './runs/256_UNet2D_solver_M5_fixed/256_UNet2D_solver_M5_fixed_20260824_1052/checkpoints/best_model.pt', 'label': 'M5'},
    ]

    if not ckpt_entries:
        print("No checkpoints provided. Exiting.")
        return

    dataset_choice = 'test'
    sample_indices = np.arange(0, 32, 1).tolist()   # list of one or more sample indices
    n_eigs = 20

    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    plot_dir = os.path.join(
        config.get('run_root', './runs'),
        config.get('run_name', 'run'),
        'spectral_analysis'
    )
    os.makedirs(plot_dir, exist_ok=True)

    # ── Load dataset ───────────────────────────────────────────────────────
    data_cfg = config['data_config']
    data_path = data_cfg['train_path'] if dataset_choice == 'train' else data_cfg['test_path']
    dataset = PDEDataset(data_path, config, sample_type='train')

    # ── Per-sample loop ────────────────────────────────────────────────────
    all_sample_results = []
    for sidx in sample_indices:
        print(f"\n{'─'*60}")
        print(f"Analyzing sample {sidx}  ({sample_indices.index(sidx)+1}/{len(sample_indices)})")
        print(f"{'─'*60}")
        sample_results = analyze_sample(sidx, dataset, config, ckpt_entries, device, n_eigs)
        all_sample_results.append(sample_results)

    # ── Aggregate ──────────────────────────────────────────────────────────
    averaged = aggregate_results(all_sample_results)

    # ── Print summary table ────────────────────────────────────────────────
    n_samples = len(sample_indices)
    tag = f'sample{sample_indices[0]}' if n_samples == 1 else \
          f'samples_{sample_indices[0]}-{sample_indices[-1]}_n{n_samples}'

    print(f"\n{'='*80}")
    print(f"Averaged over {n_samples} sample(s): {sample_indices}")
    print(f"{'Label':<30} {'λ_min (mean)':>15} {'λ_min (std)':>13} "
          f"{'cond (mean)':>14} {'||F(u_ws)||_rel':>16}")
    print(f"{'-'*80}")
    for r in averaged:
        l2_str = f"{r['l2_warm_start_error']:.4e}" if 'l2_warm_start_error' in r else '        (ref)'
        print(f"{r['label']:<30} {r['evals'][0]:>15.4e} {r['evals_std'][0]:>13.4e} "
              f"{r['cond']:>14.4e} {l2_str:>14}")
    print(f"{'='*80}")

    # ── Save averaged eigenvalues + per-step cond ─────────────────────────
    npz_path = os.path.join(plot_dir, f'eigenvalues_{tag}.npz')
    save_dict = {}
    for r in averaged:
        key = r['label'].replace(' ', '_').replace('/', '_')
        save_dict[f'evals_{key}']     = r['evals']
        save_dict[f'evals_std_{key}'] = r['evals_std']
        if 'cond_per_step' in r:
            save_dict[f'cond_per_step_{key}']     = np.array(r['cond_per_step'])
            save_dict[f'cond_per_step_std_{key}'] = np.array(r['cond_per_step_std'])
    np.savez(npz_path, **save_dict)
    print(f"\nEigenvalues + per-step cond saved: {npz_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_eigenvalue_spectra(averaged, plot_dir, tag, n_eigs_show=n_eigs)
    plot_cond_per_newton_step(averaged, plot_dir, tag)

    print(f"\nAll outputs saved to: {plot_dir}")


if __name__ == '__main__':
    main()
