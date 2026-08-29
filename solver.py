import numpy as np
from ngsolve import *
from ngsolve.webgui import Draw

from ngsolve import *
# from ngsolve.webgui import *
from netgen.occ import *
from ngsolve.solvers import * 
import matplotlib.pyplot as plt
import time

from numpy import linalg as LA
from utils_plots import plot_2d_scatter_tensor


# generate sampling points
def sampling_points_2d_grid(n_pts = 32, domain_size = 1):
    
    # set sampling points
    # define sampling points over the volume
    # no. of points along y and z based on ratio of side lengths
    x_pts = np.linspace(0, domain_size, n_pts)
    y_pts = np.linspace(0, domain_size, n_pts)

    #print(x_pts.shape, y_pts.shape, z_pts.shape)
    nx, ny = x_pts.size, y_pts.size

    # create a meshgrid
    xv, yv= np.meshgrid(x_pts, y_pts, indexing = 'ij')
    #print(xv.shape, yv.shape, zv.shape)
    #print(f'Total points:{x_pts.size*y_pts.size*z_pts.size}')

    return x_pts, y_pts, xv, yv, nx, ny

# sample from gfu solution
def sample_on_2d_grid_tensor(gfu, mesh, x_pts, y_pts, void_mask=None):
    """Sample a scalar GridFunction on a structured 2D grid.
    
    Parameters
    ----------
    gfu : GridFunction
        The NGSolve solution field.
    mesh : ngsolve.Mesh
        The NGSolve mesh.
    x_pts, y_pts : np.ndarray
        1D arrays of sampling coordinates.
    void_mask : np.ndarray of bool, shape (nx, ny), optional
        Pre-computed mask where True = void (skip evaluation).
        If None, evaluates all points with try/except fallback.
        
        For best performance, pre-compute with:
          - build_void_mask_2d_analytical() for known shapes (instant)
          - build_void_mask_2d_from_mesh() for arbitrary geometry (probes once)
          - np.zeros((nx,ny), dtype=bool) for simple rectangular domains (no voids)
    
    Returns
    -------
    u_tensor : np.ndarray, shape (1, nx, ny)
    """
    nx, ny = len(x_pts), len(y_pts)
    u_tensor = np.full((1, nx, ny), np.nan, dtype=np.float64)

    t0 = time.time()

    if void_mask is not None:
        # ---- FAST PATH: skip known voids, no try/except on valid points ----
        n_void = np.count_nonzero(void_mask)
        n_valid = nx * ny - n_void

        for i, xx in enumerate(x_pts):
            # Get valid j-indices for this row
            valid_j = np.where(~void_mask[i, :])[0]
            for j in valid_j:
                try:
                    mp = mesh(xx, y_pts[j])
                    u_tensor[0, i, j] = gfu(mp)
                except Exception:
                    u_tensor[0, i, j] = np.nan

        elapsed = time.time() - t0
        # print(f"Sampling: {elapsed:.3f} s ({n_valid} valid pts, "
        #       f"{n_valid / max(elapsed, 1e-9):.0f} pts/s, {n_void} void skipped)")

    else:
        # ---- FALLBACK: evaluate all points with try/except ----
        void_count = 0
        for i, xx in enumerate(x_pts):
            for j, yy in enumerate(y_pts):
                try:
                    mp = mesh(xx, yy)
                    u_tensor[0, i, j] = gfu(mp)
                except Exception:
                    u_tensor[0, i, j] = np.nan
                    void_count += 1

        elapsed = time.time() - t0
        n_total = nx * ny
        # print(f"Sampling: {elapsed:.3f} s ({n_total} pts, {void_count} void, "
        #       f"{(n_total - void_count) / max(elapsed, 1e-9):.0f} valid pts/s)")

    return u_tensor


def SimpleNewtonSolve(gfu, a, mesh, x_pts, y_pts, 
                      res_atol=1.0e-6, res_rtol=1.0e-8, du_atol=1.0e-18, du_rtol=1.0e-14, maxits=None,
                      callback=lambda gfu: None, verbose=False, plot_name=None,
                      linear_solver='direct', cg_maxiter=1000, cg_tol=1e-12,
                      dampfactor=1.0):
    """Newton solver with adaptive tolerances and progressive damping.

    Parameters
    ----------
    linear_solver : str
        'direct' or 'cg' for the linear solve.
    cg_maxiter, cg_tol : int, float
        CG parameters (only used when linear_solver='cg').
    dampfactor : float
        Initial damping factor in (0, 1]. Step size = min(1, dampfactor * (it+1)) * du.
        dampfactor=1.0 means no damping (full Newton step from iteration 0).
        dampfactor=0.1 ramps from 10% to 100% over the first 10 iterations.
    """
    res = gfu.vec.CreateVector()
    du = gfu.vec.CreateVector()
    fes = gfu.space
    callback(gfu)

    a.Apply(gfu.vec, res)
    res0_L2 = sqrt(abs(InnerProduct(res, res)))

    cg_iterations_per_newton_step = []
    newton_residuals = []

    with TaskManager():
        for it in range(maxits):
            a.Apply(gfu.vec, res)
            a.AssembleLinearization(gfu.vec)

            if linear_solver == 'cg':
                pre = a.mat.CreateSmoother(fes.FreeDofs())
                from ngsolve.krylovspace import CGSolver
                inv = CGSolver(mat=a.mat, pre=pre, maxiter=cg_maxiter,
                               tol=cg_tol, printrates=verbose)
                du.data = inv * res
                cg_iterations_per_newton_step.append(inv.iterations)
                if verbose:
                    print(f"Newton iterat {it}: Linear solve with {inv.iterations} CG iters")
            else:
                du.data = a.mat.Inverse(fes.FreeDofs()) * res

            # Progressive damping: ramp step size from dampfactor → 1 over iterations
            alpha = min(1.0, dampfactor * (it + 1))

            unorm_old = sqrt(abs(InnerProduct(gfu.vec, gfu.vec)))
            gfu.vec.data -= alpha * du
            callback(gfu)

            stopcritval = sqrt(abs(InnerProduct(du, du)))
            residual_L2 = sqrt(abs(InnerProduct(res, res)))
            energy_L2 = sqrt(abs(InnerProduct(du, res)))
            newton_residuals.append(float(energy_L2))

            du_threshold = du_atol + du_rtol * unorm_old
            res_threshold = res_atol + res_rtol * res0_L2

            if verbose:
                print(f"  Newton {it} (alpha={alpha:.3f}): ||du|| = {stopcritval:.2e}, ||res|| = {residual_L2:.2e}")

            if residual_L2 < res_threshold:
                if verbose:
                    print(f"  Converged (residual criterion at iter {it})")
                break

            elif stopcritval < du_threshold:
                if verbose:
                    print(f"  Converged (update criterion at iter {it})")
                break

    no_voids = np.zeros((len(x_pts), len(y_pts)), dtype=bool)
    u_sol = sample_on_2d_grid_tensor(gfu, mesh, x_pts, y_pts, void_mask=no_voids)

    if linear_solver == 'cg':
        return u_sol, it + 1, gfu, cg_iterations_per_newton_step, newton_residuals
    else:
        return u_sol, it + 1, gfu, 0, newton_residuals


# Original ngsolve built-in Newton method for comparison
def ngsolve_newton(gfu, a, mesh, x_pts, y_pts, maxit=100, tol=1e-12):
    """
    This function is a wrapper around the underlying Newton solver shipped with NGSolve
    
    Help on function Newton in module ngsolve.nonlinearsolvers:

    Newton(a, u, freedofs=None, maxit=100, maxerr=1e-11, inverse='', dirichletvalues=None, dampfactor=1, printing=True, callback=None)
    Newton's method for solving non-linear problems of the form A(u)=0.

    Parameters
    ----------
    a : BilinearForm
      The BilinearForm of the non-linear variational problem. It does not have to be assembled.

    u : GridFunction
      The GridFunction where the solution is saved. The values are used as initial guess for Newton's method.

    freedofs : BitArray
      The FreeDofs on which the assembled matrix is inverted. If argument is 'None' then the FreeDofs of the underlying FESpace is used.

    maxit : int
      Number of maximal iteration for Newton. If the maximal number is reached before the maximal error Newton might no converge and a warning is displayed.

    maxerr : float
      The maximal error which Newton should reach before it stops. The error is computed by the square root of the inner product of the residuum and the correction.

    inverse : string
      A string of the sparse direct solver which should be solved for inverting the assembled Newton matrix.

    dampfactor : float
      Set the damping factor for Newton's method. If dampfactor is 1 then no damping is done. If value is < 1 then the damping is done by the formula 'min(1,dampfactor*numit)' for the correction, where 'numit' denotes the Newton iteration
    """
    # init 
    gfu.vec[:] = 0

    status, iters = Newton(a, gfu, freedofs = gfu.space.FreeDofs(), maxit=maxit, maxerr=tol, inverse="sparsecholesky", dampfactor=0.1, printing=True)
    no_voids = np.zeros((len(x_pts), len(y_pts)), dtype=bool)
    u_sol = sample_on_2d_grid_tensor(gfu, mesh, x_pts, y_pts, void_mask=no_voids)
    if status == 0:
        return u_sol, iters, gfu
    else:
        print('Newton did not converge!!!')
        return u_sol, iters, gfu

def solve_pde(u_init, k, n, A, mu1, mu2, dirichlet_bcs, dirichlet_value=0.0, 
              x_pts=None, y_pts=None, maxh=0.1, maxiters=5000, verbose=False, 
              plot_name=None, linear_solver='direct', cg_maxiter=1000, cg_tol=1e-12,
              dampfactor=1.0):
    
    # num threads for multiprocessing
    SetNumThreads(2)

    shape = Rectangle(1,1).Face()
    shape.edges.Min(X).name="left"
    shape.edges.Max(X).name="right"
    shape.edges.Min(Y).name="bottom"
    shape.edges.Max(Y).name="top"
    geom = OCCGeometry(shape, dim=2)
    mesh = Mesh(geom.GenerateMesh(maxh=maxh))
    #print(mesh.GetBoundaries())
    
    dirichlet_bcs = [boundary for boundary in dirichlet_bcs if boundary != 0]
    #print(f'solver BC input_ {dirichlet_bcs}')

    V = H1(mesh, order=3, dirichlet=dirichlet_bcs)
    u,v = V.TnT()
    gfu = GridFunction(V)
    gfu.Set(CF(dirichlet_value), BND)
    gfu_int = GridFunction(V)
    
    # Get initial guess from the input for interior points
    if u_init is not None:
        init_guess = VoxelCoefficient((0, 0), (1, 1), np.float64(u_init), linear=True)
        gfu_int.Set(init_guess)
        
        # Copy interior values from initial guess while preserving boundary values
        free_dofs = V.FreeDofs()
        for dof in range(V.ndof):
            if free_dofs[dof]:
                gfu.vec.data[dof] = gfu_int.vec[dof]
    
    # Set up PDE weak form
    a = BilinearForm(V)
    f = A * (mu1**2 + mu2**2 + x**2 + y**2)*sin(mu1*pi*x)*sin(mu2*pi*y)
    a += (grad(u) * grad(v) + (k**2) * u**n * v - f * v)*dx

    # core Newton-Raphson Newton Solve
    u_sol, iters, gfu, cg_iters, newton_residuals = SimpleNewtonSolve(gfu, a, mesh, x_pts, y_pts, maxits=maxiters,
                                                    verbose=verbose, plot_name=plot_name,
                                                    linear_solver=linear_solver, cg_maxiter=cg_maxiter, cg_tol=cg_tol,
                                                    dampfactor=dampfactor)

    # ngsolve builtin variant for comparison
        #u_sol, iters, gfu = ngsolve_newton(gfu, a, mesh, x_pts, y_pts, tol = 1e-12)

    # compute relative L2 between u_init and u_sol to identify improvement in adaptive M strategy
    if u_init is not None:
        l2_init_sol = np.linalg.norm((u_init - u_sol)) / (np.linalg.norm(u_sol))
    else:
        u_init = np.zeros_like(u_sol) # by default initialize field as zero
        l2_init_sol = np.linalg.norm((u_init - u_sol)) / (np.linalg.norm(u_sol))

    return u_sol, iters, gfu, cg_iters, l2_init_sol, newton_residuals

def main():
    # domain discretization for model input
    n_pts = 64
    x_pts, y_pts, xv, yv, nx, ny = sampling_points_2d_grid(n_pts = n_pts)

    # PDE params
    A = 30
    k=3 
    n=13
    mu1=2.2345
    mu2=1.687

    print(f'PDE params: A: {A}, k: {k}, n: {n}, mu1: {mu1}, mu2: {mu2}')

    # compute truth
    u_init = np.ones((64, 64))*0.0
    u_sol, iters, gfu, cg_iters, improvement, residuals = solve_pde(u_init = u_init, k = k, n = n, A = A, mu1 = mu1, mu2 = mu2, dirichlet_bcs= [1, 2, 3, 4], dirichlet_value=0.0, x_pts=x_pts, y_pts=y_pts, maxiters = 100, linear_solver='cg', verbose = False)

    print(f'Sample solver iters: {iters}')
    print(f'List of Newton-iter-wise CG iters: {cg_iters}')
    print(f'Relative improvement in u: {improvement}')
    print(f'Residuals: {residuals[:3]}')

    plot_name = f'truth_A{A}_k{k}_n{n}_mu1{mu1}_mu2{mu2}_1_2_3_4_bc0_CustomNewton'
    plot_2d_scatter_tensor(xv, yv, 1001, f'./plots/truth_range_limits/', u_sol, savefile=plot_name)

if __name__=='__main__':
    main()


