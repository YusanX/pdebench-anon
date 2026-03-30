"""
Template-guided prompt builder for P2 (API Decoupling) experiment.

PURPOSE
-------
Address ICML reviewer concern (eVPk W2, K4KR Q1):
  "The benchmark conflates DOLFINx API proficiency with numerical reasoning ability."

DESIGN
------
This prompt variant provides a complete DOLFINx skeleton with all API boilerplate
pre-filled (mesh creation, function space setup, point-sampling output, solver call).
The model only needs to fill in:
  1. Variational form  — a(u,v) and L(v)  [numerical reasoning / PDE math]
  2. Boundary conditions — BC values from case_spec  [mathematical understanding]
  3. Numerical parameters — mesh resolution, element degree, ksp/pc type, tolerance  [numerical judgment]

INTERPRETATION
--------------
If template_guided pass-rate ≈ standard pass-rate
  → API knowledge is NOT the bottleneck; math/numerics are
  → Benchmark reliably measures numerical reasoning

If template_guided pass-rate > standard pass-rate
  → Standard prompt under-estimates models by imposing API overhead
  → Benchmark is conservative (stronger result for authors)
"""

from typing import Dict, Any, Optional
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Utility code injected at the top of every skeleton
# This provides the "hard API" parts so the model can focus on mathematics
# ─────────────────────────────────────────────────────────────────────────────
_UTILITY_CODE = '''
# ════════════════════════════════════════════════════════════════════════
#  PROVIDED UTILITIES  —  do NOT modify these functions
# ════════════════════════════════════════════════════════════════════════
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as _dmesh, fem, geometry as _geo
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
import ufl

def _sample_scalar(u_h, nx: int, ny: int) -> np.ndarray:
    """Sample a scalar DOLFINx Function onto an (ny, nx) uniform grid [0,1]²."""
    msh = u_h.function_space.mesh
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])
    tree = _geo.bb_tree(msh, msh.topology.dim)
    cands = _geo.compute_collisions_points(tree, pts)
    colliding = _geo.compute_colliding_cells(msh, cands, pts)
    found_pts, cells, idx = [], [], []
    for i in range(len(pts)):
        links = colliding.links(i)
        if len(links):
            found_pts.append(pts[i]); cells.append(links[0]); idx.append(i)
    vals = np.full(nx * ny, np.nan)
    if found_pts:
        ev = u_h.eval(np.array(found_pts), np.array(cells, dtype=np.int32))
        vals[idx] = ev[:, 0]
    return vals.reshape(ny, nx)


def _sample_vector_magnitude(u_h, nx: int, ny: int) -> np.ndarray:
    """Sample ‖u‖₂ of a vector DOLFINx Function onto an (ny, nx) uniform grid."""
    msh = u_h.function_space.mesh
    gdim = msh.geometry.dim
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])
    tree = _geo.bb_tree(msh, msh.topology.dim)
    cands = _geo.compute_collisions_points(tree, pts)
    colliding = _geo.compute_colliding_cells(msh, cands, pts)
    found_pts, cells, idx = [], [], []
    for i in range(len(pts)):
        links = colliding.links(i)
        if len(links):
            found_pts.append(pts[i]); cells.append(links[0]); idx.append(i)
    vals = np.full((nx * ny, gdim), np.nan)
    if found_pts:
        ev = u_h.eval(np.array(found_pts), np.array(cells, dtype=np.int32))
        vals[idx] = ev
    return np.linalg.norm(vals, axis=1).reshape(ny, nx)


def _all_boundary_dofs(msh, V):
    """Return DOF indices on the entire boundary ∂Ω."""
    fdim = msh.topology.dim - 1
    facets = _dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    return fem.locate_dofs_topological(V, fdim, facets)


def _bc_from_str(msh, V, expr_str: str) -> fem.DirichletBC:
    """Build a Dirichlet BC u=g on ∂Ω from a string expression g(x,y).
    Handles constants (e.g. '0.0') and symbolic expressions (e.g. 'sin(pi*x)*sin(pi*y)').
    """
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)
    try:
        val = float(expr_str)
        g = fem.Function(V)
        g.x.array[:] = val
    except ValueError:
        sym = sp.sympify(expr_str, locals={"x": sx, "y": sy, "pi": sp.pi})
        g_np = sp.lambdify((sx, sy), sym, modules="numpy")
        g = fem.Function(V)
        g.interpolate(lambda pts: g_np(pts[0], pts[1]).astype(np.float64))
    dofs = _all_boundary_dofs(msh, V)
    return fem.dirichletbc(g, dofs)


def _bc_vec_zero(msh, V) -> fem.DirichletBC:
    """Homogeneous Dirichlet BC for a vector function space."""
    fdim = msh.topology.dim - 1
    facets = _dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    g = fem.Function(V)
    g.x.array[:] = 0.0
    return fem.dirichletbc(g, dofs)


def _kappa_from_spec(msh, kappa_spec: dict):
    """Parse a kappa coefficient spec into a UFL scalar expression.
    Spec examples:
      {'type': 'constant', 'value': 1.0}
      {'type': 'expr', 'expr': '1 + 0.5*sin(2*pi*x)*sin(2*pi*y)'}
    """
    import sympy as sp
    ktype = kappa_spec.get("type", "constant")
    x = ufl.SpatialCoordinate(msh)
    if ktype == "constant":
        return ufl.as_ufl(float(kappa_spec["value"]))
    elif ktype == "piecewise_x":
        left  = float(kappa_spec["left"])
        right = float(kappa_spec["right"])
        split = float(kappa_spec.get("x_split", 0.5))
        return ufl.conditional(x[0] < split, ufl.as_ufl(left), ufl.as_ufl(right))
    elif ktype == "expr":
        sx, sy = sp.symbols("x y", real=True)
        sym = sp.sympify(kappa_spec["expr"],
                         locals={"x": sx, "y": sy, "pi": sp.pi})
        kappa_fn = sp.lambdify((sx, sy), sym, modules="numpy")
        kappa_h = fem.Function(fem.functionspace(msh, ("DG", 0)))
        kappa_h.interpolate(lambda pts: kappa_fn(pts[0], pts[1]).astype(np.float64))
        return kappa_h
    else:
        return ufl.as_ufl(1.0)


def _f_from_str(msh, expr_str: str):
    """Parse a source-term string to a UFL expression or fem.Function."""
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)
    try:
        return ufl.as_ufl(float(expr_str))
    except (ValueError, TypeError):
        pass
    sym = sp.sympify(expr_str, locals={"x": sx, "y": sy, "pi": sp.pi})
    x = ufl.SpatialCoordinate(msh)
    # Build a DG0 interpolation for general expressions
    f_np = sp.lambdify((sx, sy), sym, modules="numpy")
    V_dg = fem.functionspace(msh, ("DG", 1))
    f_h = fem.Function(V_dg)
    f_h.interpolate(lambda pts: f_np(pts[0], pts[1]).astype(np.float64))
    return f_h


def _manufactured_f_and_bc(msh, V, pde_cfg: dict, kappa_spec: dict):
    """From a manufactured_solution spec, derive f = -div(κ ∇u_exact) symbolically
    and return (f_expr_ufl, bc) for use in the variational form.
    """
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)
    u_str = pde_cfg["manufactured_solution"]["u"]
    u_sym = sp.sympify(u_str, locals={"x": sx, "y": sy, "pi": sp.pi})

    ktype = kappa_spec.get("type", "constant")
    if ktype == "constant":
        kap_sym = sp.sympify(kappa_spec.get("value", 1.0))
    elif ktype == "expr":
        kap_sym = sp.sympify(kappa_spec["expr"],
                             locals={"x": sx, "y": sy, "pi": sp.pi})
    else:
        kap_sym = sp.sympify(1.0)

    f_sym = -(sp.diff(kap_sym * sp.diff(u_sym, sx), sx)
              + sp.diff(kap_sym * sp.diff(u_sym, sy), sy))
    f_sym = sp.simplify(f_sym)

    f_np = sp.lambdify((sx, sy), f_sym, modules="numpy")
    V_dg = fem.functionspace(msh, ("DG", 1))
    f_h = fem.Function(V_dg)
    f_h.interpolate(lambda pts: f_np(pts[0], pts[1]).astype(np.float64))

    g_np = sp.lambdify((sx, sy), u_sym, modules="numpy")
    g_h = fem.Function(V)
    g_h.interpolate(lambda pts: g_np(pts[0], pts[1]).astype(np.float64))
    dofs = _all_boundary_dofs(msh, V)
    bc = fem.dirichletbc(g_h, dofs)
    return f_h, bc


def _manufactured_helmholtz_f_and_bc(msh, V, u_str: str, k: float):
    """Compute f = -Δu - k²u and BC = u|∂Ω from a manufactured solution string.
    Returns (f_h, bc) ready for use in the Helmholtz variational form.
    """
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)
    u_sym = sp.sympify(u_str, locals={"x": sx, "y": sy, "pi": sp.pi})
    k_sym = sp.sympify(float(k))
    f_sym = -(sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)) - k_sym**2 * u_sym
    f_sym = sp.simplify(f_sym)

    f_np = sp.lambdify((sx, sy), f_sym, modules="numpy")
    V_dg = fem.functionspace(msh, ("DG", 1))
    f_h = fem.Function(V_dg)
    f_h.interpolate(lambda pts: f_np(pts[0], pts[1]).astype(np.float64))

    g_np = sp.lambdify((sx, sy), u_sym, modules="numpy")
    g_h = fem.Function(V)
    g_h.interpolate(lambda pts: g_np(pts[0], pts[1]).astype(np.float64))
    dofs = _all_boundary_dofs(msh, V)
    bc = fem.dirichletbc(g_h, dofs)
    return f_h, bc


def _manufactured_biharmonic_data(msh, V, u_str: str):
    """Compute f = Δ²u, bc_u = u|∂Ω, bc_w = −Δu|∂Ω from a manufactured solution.
    Used for the two-step Poisson formulation of the biharmonic equation:
        Step 1: −Δw = f  (find auxiliary w)
        Step 2: −Δu = w  (find solution u)
    Returns (f_h, bc_u, bc_w).
    """
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)
    u_sym = sp.sympify(u_str, locals={"x": sx, "y": sy, "pi": sp.pi})
    w_sym = -(sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2))   # w = -Δu
    f_sym = -(sp.diff(w_sym, sx, 2) + sp.diff(w_sym, sy, 2))   # f = -Δw = Δ²u
    f_sym = sp.simplify(f_sym)

    dofs = _all_boundary_dofs(msh, V)
    V_dg = fem.functionspace(msh, ("DG", 1))

    f_np = sp.lambdify((sx, sy), f_sym, modules="numpy")
    f_h = fem.Function(V_dg)
    f_h.interpolate(lambda pts: f_np(pts[0], pts[1]).astype(np.float64))

    u_np = sp.lambdify((sx, sy), u_sym, modules="numpy")
    u_bc_fn = fem.Function(V)
    u_bc_fn.interpolate(lambda pts: u_np(pts[0], pts[1]).astype(np.float64))
    bc_u = fem.dirichletbc(u_bc_fn, dofs)

    w_np = sp.lambdify((sx, sy), w_sym, modules="numpy")
    w_bc_fn = fem.Function(V)
    w_bc_fn.interpolate(lambda pts: w_np(pts[0], pts[1]).astype(np.float64))
    bc_w = fem.dirichletbc(w_bc_fn, dofs)

    return f_h, bc_u, bc_w


def _manufactured_convdiff_f_and_bc(msh, V, u_str: str, epsilon: float, beta):
    """Compute f = -ε Δu + β·∇u and BC = u|∂Ω for a steady convection-diffusion
    manufactured solution.  Returns (f_h, bc).
    """
    import sympy as sp
    sx, sy = sp.symbols("x y", real=True)
    u_sym = sp.sympify(u_str, locals={"x": sx, "y": sy, "pi": sp.pi})
    bx, by = float(beta[0]), float(beta[1])
    f_sym = (
        -epsilon * (sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2))
        + bx * sp.diff(u_sym, sx)
        + by * sp.diff(u_sym, sy)
    )
    f_sym = sp.simplify(f_sym)

    f_np = sp.lambdify((sx, sy), f_sym, modules="numpy")
    V_dg = fem.functionspace(msh, ("DG", 1))
    f_h = fem.Function(V_dg)
    f_h.interpolate(lambda pts: f_np(pts[0], pts[1]).astype(np.float64))

    g_np = sp.lambdify((sx, sy), u_sym, modules="numpy")
    g_h = fem.Function(V)
    g_h.interpolate(lambda pts: g_np(pts[0], pts[1]).astype(np.float64))
    dofs = _all_boundary_dofs(msh, V)
    bc = fem.dirichletbc(g_h, dofs)
    return f_h, bc


def _f_time_dep(msh, f_str: str, t: float):
    """Evaluate a time-dependent source f(x, y, t) at a given time t.
    f_str may contain symbols x, y, t (and pi).
    Returns a fem.Function on a DG1 space suitable for use in a linear form.
    """
    import sympy as sp
    sx, sy, st = sp.symbols("x y t", real=True)
    f_sym = sp.sympify(f_str, locals={"x": sx, "y": sy, "t": st, "pi": sp.pi})
    f_np = sp.lambdify((sx, sy, st), f_sym, modules="numpy")
    V_dg = fem.functionspace(msh, ("DG", 1))
    fh = fem.Function(V_dg)
    _t = float(t)
    fh.interpolate(lambda pts: f_np(pts[0], pts[1], _t).astype(np.float64))
    return fh


def _update_bc_at_t(g_fn, u_str: str, t: float):
    """Update a mutable Dirichlet BC function g_fn in-place to u(x,y,t).
    Usage:
        g_fn = fem.Function(V)
        bc   = fem.dirichletbc(g_fn, dofs)
        ...
        _update_bc_at_t(g_fn, u_str, t)   # updates BC for next solve
    u_str may contain symbols x, y, t (and pi).
    """
    import sympy as sp
    sx, sy, st = sp.symbols("x y t", real=True)
    u_sym = sp.sympify(u_str, locals={"x": sx, "y": sy, "t": st, "pi": sp.pi})
    u_np = sp.lambdify((sx, sy, st), u_sym, modules="numpy")
    _t = float(t)
    g_fn.interpolate(lambda pts: u_np(pts[0], pts[1], _t).astype(np.float64))

def _manufactured_stokes_f(msh, f1_str: str, f2_str: str):
    """Create a 2-component body force vector from pre-computed strings f1(x,y), f2(x,y).
    Evaluates each component into a DG-1 scalar function and combines with ufl.as_vector.
    """
    import sympy as _sp
    _sx, _sy = _sp.symbols("x y", real=True)
    _loc = {"x": _sx, "y": _sy, "pi": _sp.pi}
    _f_syms = [_sp.sympify(s, locals=_loc) for s in [f1_str, f2_str]]
    _f_nps  = [_sp.lambdify((_sx, _sy), _f, modules="numpy") for _f in _f_syms]
    _V1     = fem.functionspace(msh, ("DG", 1))
    _fh     = []
    for _fn in _f_nps:
        _fi = fem.Function(_V1)
        _fi.interpolate(lambda pts, _fn=_fn: np.asarray(_fn(pts[0], pts[1]), dtype=np.float64)
                        if np.ndim(_fn(pts[0], pts[1])) > 0
                        else np.full(pts.shape[1], float(_fn(pts[0][0], pts[1][0]))))
        _fh.append(_fi)
    return ufl.as_vector(_fh)

def _stokes_exact_bc(msh, W, V_col, u1_str: str, u2_str: str):
    """Dirichlet BC: velocity = [u1(x,y), u2(x,y)] (manufactured solution) on all ∂Ω."""
    import sympy as _sp
    _sx, _sy = _sp.symbols("x y", real=True)
    _loc    = {"x": _sx, "y": _sy, "pi": _sp.pi}
    _u_nps  = [_sp.lambdify((_sx, _sy), _sp.sympify(s, locals=_loc), modules="numpy")
               for s in [u1_str, u2_str]]
    _u_ex   = fem.Function(V_col)
    _u_ex.interpolate(lambda pts: np.array([
        np.asarray(_u_nps[0](pts[0], pts[1]), dtype=np.float64)
        if np.ndim(_u_nps[0](pts[0], pts[1])) > 0
        else np.full(pts.shape[1], float(_u_nps[0](0.0, 0.0))),
        np.asarray(_u_nps[1](pts[0], pts[1]), dtype=np.float64)
        if np.ndim(_u_nps[1](pts[0], pts[1])) > 0
        else np.full(pts.shape[1], float(_u_nps[1](0.0, 0.0))),
    ], dtype=np.float64))
    _fdim   = msh.topology.dim - 1
    _facets = _dmesh.locate_entities_boundary(msh, _fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    _dofs   = fem.locate_dofs_topological((W.sub(0), V_col), _fdim, _facets)
    return fem.dirichletbc(_u_ex, _dofs, W.sub(0))

def _stokes_bc_from_config(msh, W, V_col, bc_data: list):
    """Build velocity Dirichlet BCs from a list of {on, value} config dicts.
    'on': 'all'|'x0'|'x1'|'y0'|'y1'.  'value': [ux_str, uy_str].
    Enables BC config to be inlined at prompt-gen time without exposing case_spec.
    """
    import sympy as _sp
    _sx, _sy = _sp.symbols("x y", real=True)
    _loc = {"x": _sx, "y": _sy, "pi": _sp.pi}
    _sel = {
        "all": lambda x: np.ones(x.shape[1], dtype=bool),
        "x0":  lambda x: np.isclose(x[0], 0.0),
        "x1":  lambda x: np.isclose(x[0], 1.0),
        "y0":  lambda x: np.isclose(x[1], 0.0),
        "y1":  lambda x: np.isclose(x[1], 1.0),
    }
    _fdim = msh.topology.dim - 1
    _bcs  = []
    for _cfg in bc_data:
        _on  = _cfg.get("on", "all").lower()
        _val = _cfg.get("value", ["0.0", "0.0"])
        if _on in ("all", "*"):
            _fcts = _dmesh.locate_entities_boundary(msh, _fdim,
                        lambda x: np.ones(x.shape[1], dtype=bool))
            _dofs = fem.locate_dofs_topological((W.sub(0), V_col), _fdim, _fcts)
        else:
            _dofs = fem.locate_dofs_geometrical((W.sub(0), V_col), _sel.get(_on, _sel["all"]))
        _u_lambdas = []
        for _s in _val:
            try:
                _c = float(_s)
                _u_lambdas.append(lambda pts, c=_c: np.full(pts.shape[1], c))
            except (ValueError, TypeError):
                _sym = _sp.sympify(_s, locals=_loc)
                _fn  = _sp.lambdify((_sx, _sy), _sym, modules="numpy")
                _u_lambdas.append(lambda pts, fn=_fn: np.asarray(fn(pts[0], pts[1]), dtype=np.float64))
        _bc_fn = fem.Function(V_col)
        _bc_fn.interpolate(lambda pts: np.array([f(pts) for f in _u_lambdas], dtype=np.float64))
        _bcs.append(fem.dirichletbc(_bc_fn, _dofs, W.sub(0)))
    return _bcs

def _pressure_pin(msh, W):
    """Pin pressure at (0, 0) to remove the hydrostatic null space.
    Returns a DirichletBC on W.sub(1), or None if no DOF found at origin.
    """
    _Q, _ = W.sub(1).collapse()
    _dofs  = fem.locate_dofs_geometrical(
        (W.sub(1), _Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(_dofs) == 0:
        return None
    _p0 = fem.Function(_Q)
    _p0.x.array[:] = 0.0
    return fem.dirichletbc(_p0, _dofs, W.sub(1))

def _le_exact_bc(msh, V, u1_str: str, u2_str: str):
    """Dirichlet BC: displacement = [u1(x,y), u2(x,y)] on all ∂Ω (manufactured solution).
    V must be a 2D vector Lagrange function space (not mixed).
    """
    import sympy as _sp
    _sx, _sy = _sp.symbols("x y", real=True)
    _loc    = {"x": _sx, "y": _sy, "pi": _sp.pi}
    _u_nps  = [_sp.lambdify((_sx, _sy), _sp.sympify(s, locals=_loc), modules="numpy")
               for s in [u1_str, u2_str]]
    _u_ex   = fem.Function(V)
    _u_ex.interpolate(lambda pts: np.array([
        np.asarray(_u_nps[i](pts[0], pts[1]), dtype=np.float64)
        if np.ndim(_u_nps[i](pts[0], pts[1])) > 0
        else np.full(pts.shape[1], float(_u_nps[i](0.0, 0.0)))
        for i in range(2)
    ], dtype=np.float64))
    _fdim   = msh.topology.dim - 1
    _facets = _dmesh.locate_entities_boundary(msh, _fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    _dofs   = fem.locate_dofs_topological(V, _fdim, _facets)
    return fem.dirichletbc(_u_ex, _dofs)

def _le_bc_from_config(msh, V, bc_data: list):
    """Build displacement Dirichlet BCs for linear elasticity from a config list.
    'on': 'all'|'x0'|'x1'|'y0'|'y1'.  'value': [ux_str, uy_str].
    Works with a plain (non-mixed) vector function space V.
    """
    import sympy as _sp
    _sx, _sy = _sp.symbols("x y", real=True)
    _loc = {"x": _sx, "y": _sy, "pi": _sp.pi}
    _sel = {
        "all": lambda x: np.ones(x.shape[1], dtype=bool),
        "x0":  lambda x: np.isclose(x[0], 0.0),
        "x1":  lambda x: np.isclose(x[0], 1.0),
        "y0":  lambda x: np.isclose(x[1], 0.0),
        "y1":  lambda x: np.isclose(x[1], 1.0),
    }
    _fdim = msh.topology.dim - 1
    _bcs  = []
    for _cfg in bc_data:
        _on  = _cfg.get("on", "all").lower()
        _val = _cfg.get("value", ["0.0", "0.0"])
        if _on in ("all", "*"):
            _fcts = _dmesh.locate_entities_boundary(msh, _fdim,
                        lambda x: np.ones(x.shape[1], dtype=bool))
            _dofs = fem.locate_dofs_topological(V, _fdim, _fcts)
        else:
            _dofs = fem.locate_dofs_geometrical(V, _sel.get(_on, _sel["all"]))
        _u_lambdas = []
        for _s in _val:
            try:
                _c = float(_s)
                _u_lambdas.append(lambda pts, c=_c: np.full(pts.shape[1], c))
            except (ValueError, TypeError):
                _sym = _sp.sympify(_s, locals=_loc)
                _fn  = _sp.lambdify((_sx, _sy), _sym, modules="numpy")
                _u_lambdas.append(lambda pts, fn=_fn: np.asarray(fn(pts[0], pts[1]), dtype=np.float64))
        _bc_fn = fem.Function(V)
        _bc_fn.interpolate(lambda pts: np.array([f(pts) for f in _u_lambdas], dtype=np.float64))
        _bcs.append(fem.dirichletbc(_bc_fn, _dofs))
    return _bcs
# ════════════════════════════════════════════════════════════════════════
#  END OF PROVIDED UTILITIES
# ════════════════════════════════════════════════════════════════════════
'''

# ─────────────────────────────────────────────────────────────────────────────
# Poisson: case-specific dynamic skeleton (PDE data inlined at prompt-gen time)
# ─────────────────────────────────────────────────────────────────────────────

def _build_poisson_skeleton(case: Dict) -> str:
    """
    Generate a Poisson skeleton with ALL PDE math data pre-filled.
    The model only writes: variational form a, L + numerical parameter choices.
    No case_spec access needed at runtime.
    """
    pde_cfg    = case["oracle_config"]["pde"]
    kappa_spec = pde_cfg.get("coefficients", {}).get("kappa",
                                                     {"type": "constant", "value": 1.0})
    manufactured = pde_cfg.get("manufactured_solution", {})
    out_cfg    = case["oracle_config"]["output"]["grid"]
    nx, ny     = out_cfg["nx"], out_cfg["ny"]

    # Cell type for mesh creation
    cell_type_str = case["oracle_config"]["mesh"].get("cell_type", "triangle")
    if cell_type_str == "quadrilateral":
        mesh_call = "    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N,\n" \
                    "               cell_type=_dmesh.CellType.quadrilateral)"
    else:
        mesh_call = "    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)"

    # PDE data section — fully pre-filled, no case_spec needed
    kappa_repr = repr(kappa_spec)
    if manufactured.get("u"):
        u_str = manufactured["u"]
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    kappa   = _kappa_from_spec(msh, {kappa_repr})\n"
            f"    f_h, bc = _manufactured_f_and_bc(\n"
            f"        msh, V,\n"
            f"        {{'manufactured_solution': {{'u': {repr(u_str)}}}}},\n"
            f"        {kappa_repr}\n"
            f"    )"
        )
    else:
        f_str  = pde_cfg.get("source_term", "0.0")
        bc_val = pde_cfg.get("bc", {}).get("dirichlet", {}).get("value", "0.0")
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    kappa = _kappa_from_spec(msh, {kappa_repr})\n"
            f"    f_h   = _f_from_str(msh, {repr(f_str)})\n"
            f"    bc    = _bc_from_str(msh, V, {repr(bc_val)})"
        )

    return f'''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (← YOUR CHOICES, no defaults given) ────────
    N        = None  # ← choose mesh resolution (integer)
    degree   = None  # ← choose FE polynomial degree: 1, 2, or 3
    ksp_type = None  # ← choose linear solver: "cg" or "gmres"
    pc_type  = None  # ← choose preconditioner: "hypre", "ilu", or "jacobi"
    rtol     = None  # ← choose relative tolerance

    # ── SETUP  (PROVIDED — do not modify) ────────────────────────────────
{mesh_call}
    V   = fem.functionspace(msh, ("Lagrange", degree))

{pde_data}

    # ── VARIATIONAL FORM  (← YOUR CODE) ──────────────────────────────────
    # PDE:  -∇·(κ ∇u) = f  in Ω,   u = g on ∂Ω
    # Weak form:  find u ∈ V s.t.  ∫ κ ∇u·∇v dx = ∫ f v dx   ∀ v ∈ V
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # ← YOUR CODE: bilinear form a(u, v)
    L = ...   # ← YOUR CODE: linear form L(v)

    # ── SOLVE  (PROVIDED — do not modify) ────────────────────────────────
    u_h = LinearProblem(a, L, bcs=[bc],
                        petsc_options={{"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}},
                        petsc_options_prefix="p2_poisson_").solve()

    # ── OUTPUT  (PROVIDED — do not modify) ───────────────────────────────
    return {{
        "u": _sample_scalar(u_h, {nx}, {ny}),
        "solver_info": {{"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol}},
    }}
'''


# ─────────────────────────────────────────────────────────────────────────────
# Heat: case-specific dynamic skeleton (PDE data inlined at prompt-gen time)
# ─────────────────────────────────────────────────────────────────────────────

def _build_heat_skeleton(case: Dict) -> str:
    """
    Generate a Heat skeleton with ALL PDE math data pre-filled.
    The model writes: variational form a, L + time-loop + numerical choices.
    No case_spec access needed at runtime.
    """
    pde_cfg    = case["oracle_config"]["pde"]
    kappa_spec = pde_cfg.get("coefficients", {}).get("kappa",
                                                     {"type": "constant", "value": 1.0})
    manufactured = pde_cfg.get("manufactured_solution", {})
    time_cfg   = pde_cfg.get("time", {})
    t_end      = float(time_cfg.get("t_end", 1.0))
    ic_str     = pde_cfg.get("initial_condition", "0.0")
    out_cfg    = case["oracle_config"]["output"]["grid"]
    nx, ny     = out_cfg["nx"], out_cfg["ny"]

    kappa_repr = repr(kappa_spec)
    if manufactured.get("u"):
        u_str = manufactured["u"]
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    kappa   = _kappa_from_spec(msh, {kappa_repr})\n"
            f"    f_h, bc = _manufactured_f_and_bc(\n"
            f"        msh, V,\n"
            f"        {{'manufactured_solution': {{'u': {repr(u_str)}}}}},\n"
            f"        {kappa_repr}\n"
            f"    )"
        )
    else:
        f_str  = pde_cfg.get("source_term", "0.0")
        bc_val = pde_cfg.get("bc", {}).get("dirichlet", {}).get("value", "0.0")
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    kappa = _kappa_from_spec(msh, {kappa_repr})\n"
            f"    f_h   = _f_from_str(msh, {repr(f_str)})\n"
            f"    bc    = _bc_from_str(msh, V, {repr(bc_val)})"
        )

    # Initial condition
    ic_code = (
        f"    u_n = fem.Function(V)\n"
        f"    u_n.interpolate(lambda pts: __import__('sympy').lambdify(\n"
        f"        (__import__('sympy').symbols('x y')),\n"
        f"        __import__('sympy').sympify({repr(ic_str)},\n"
        f"            locals={{'x': __import__('sympy').Symbol('x'),\n"
        f"                     'y': __import__('sympy').Symbol('y'),\n"
        f"                     'pi': __import__('sympy').pi}}),\n"
        f"        modules='numpy')(pts[0], pts[1]).astype(float))"
    )
    # Simplified: just use _bc_from_str logic for IC
    ic_code = (
        f"    # Initial condition (pre-filled — do not modify):\n"
        f"    u_n = fem.Function(V)\n"
        f"    _ic = _bc_from_str(msh, V, {repr(ic_str)})  # reuse BC helper for interpolation\n"
        f"    u_n.x.array[:] = _ic.g.x.array[:]  # copy interpolated values"
        if ic_str != "0.0" else
        f"    # Initial condition (pre-filled — do not modify):\n"
        f"    u_n = fem.Function(V)  # zero initial condition"
    )

    return f'''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (← YOUR CHOICES, no defaults given) ────────
    N           = None  # ← choose mesh resolution (integer)
    degree      = None  # ← choose FE polynomial degree: 1, 2, or 3
    dt          = None  # ← choose time step (float); t_end = {t_end}
    time_scheme = None  # ← choose: "backward_euler" or "crank_nicolson"
    ksp_type    = None  # ← choose linear solver: "cg" or "gmres"
    pc_type     = None  # ← choose preconditioner: "hypre", "ilu", or "jacobi"
    rtol        = None  # ← choose relative tolerance

    # ── SETUP  (PROVIDED — do not modify) ────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))

{pde_data}

{ic_code}

    # ── VARIATIONAL FORM  (← YOUR CODE) ──────────────────────────────────
    # PDE:  ∂u/∂t - ∇·(κ ∇u) = f   in Ω × (0, {t_end}]
    # Discretise time with your chosen scheme.
    # For Backward Euler weak form:
    #   ∫ u v dx + dt ∫ κ ∇u·∇v dx = ∫ u_n v dx + dt ∫ f v dx   ∀ v
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # ← YOUR CODE: bilinear form (must include mass term for time-stepping)
    L = ...   # ← YOUR CODE: linear form   (must include u_n term)

    # ── TIME LOOP  (PROVIDED structure — fill in the solve call) ─────────
    n_steps = max(1, round({t_end} / dt))
    u_h = fem.Function(V)

    for _ in range(n_steps):
        u_h = LinearProblem(a, L, bcs=[bc],
                            petsc_options={{"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}},
                            petsc_options_prefix="p2_heat_").solve()
        u_n.x.array[:] = u_h.x.array  # update previous step

    # ── OUTPUT  (PROVIDED — do not modify) ───────────────────────────────
    return {{
        "u": _sample_scalar(u_h, {nx}, {ny}),
        "solver_info": {{"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol,
                        "dt": dt, "n_steps": n_steps, "time_scheme": time_scheme}},
    }}
'''

# ─────────────────────────────────────────────────────────────────────────────
# Helmholtz: dynamic skeleton (PDE data inlined at prompt-gen time)
# ─────────────────────────────────────────────────────────────────────────────

def _build_helmholtz_skeleton(case: Dict) -> str:
    """
    Generate a Helmholtz skeleton with ALL PDE math data pre-filled.
    The model writes: variational form a, L + numerical parameter choices.
    PDE:  -Δu - k² u = f   in Ω,   u = g on ∂Ω
    """
    pde_cfg      = case["oracle_config"]["pde"]
    params       = pde_cfg.get("pde_params", {})
    k            = float(params.get("k", params.get("wave_number", 10.0)))
    manufactured = pde_cfg.get("manufactured_solution", {})
    out_cfg      = case["oracle_config"]["output"]["grid"]
    nx, ny       = out_cfg["nx"], out_cfg["ny"]

    # Helmholtz is indefinite → GMRES+ILU or direct (lu) required; CG will diverge
    # The rule-of-thumb N ≳ 10·k/π ensures at least 10 nodes per wavelength
    n_min = max(32, int(10 * k / 3.1416) + 1)

    if manufactured.get("u"):
        u_str = manufactured["u"]
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    k     = {k!r}  # wavenumber\n"
            f"    f_h, bc = _manufactured_helmholtz_f_and_bc(\n"
            f"        msh, V, {u_str!r}, k\n"
            f"    )"
        )
    else:
        f_str  = pde_cfg.get("source_term", "0.0")
        bc_val = pde_cfg.get("bc", {}).get("dirichlet", {}).get("value", "0.0")
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    k   = {k!r}  # wavenumber\n"
            f"    f_h = _f_from_str(msh, {f_str!r})\n"
            f"    bc  = _bc_from_str(msh, V, {bc_val!r})"
        )

    return f'''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (← YOUR CHOICES, no defaults given) ────────
    # Helmholtz is indefinite — do NOT use CG; use "gmres" or "preonly".
    # Mesh rule of thumb: N ≳ 10·k/π ≈ {n_min} for k={k} (at least 10 pts/wavelength).
    N        = None  # ← choose mesh resolution (integer); recommend ≥ {n_min}
    degree   = None  # ← choose FE polynomial degree: 1, 2, or 3 (higher → fewer pts needed)
    ksp_type = None  # ← choose: "gmres" or "preonly"
    pc_type  = None  # ← choose: "ilu", "lu", or "hypre"
    rtol     = None  # ← choose relative tolerance

    # ── SETUP  (PROVIDED — do not modify) ────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))

{pde_data}

    # ── VARIATIONAL FORM  (← YOUR CODE) ──────────────────────────────────
    # PDE:  -Δu - k² u = f   in Ω,   u = g on ∂Ω
    # Weak form:  ∫ ∇u·∇v dx - k² ∫ u v dx = ∫ f v dx   ∀ v ∈ V
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # ← YOUR CODE: bilinear form a(u, v)
    L = ...   # ← YOUR CODE: linear form L(v)

    # ── SOLVE  (PROVIDED — do not modify) ────────────────────────────────
    u_h = LinearProblem(a, L, bcs=[bc],
                        petsc_options={{"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}},
                        petsc_options_prefix="p2_helmholtz_").solve()

    # ── OUTPUT  (PROVIDED — do not modify) ───────────────────────────────
    return {{
        "u": _sample_scalar(u_h, {nx}, {ny}),
        "solver_info": {{"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol, "k": k}},
    }}
'''


# ─────────────────────────────────────────────────────────────────────────────
# Biharmonic: dynamic skeleton (PDE data inlined, two-step Poisson approach)
# ─────────────────────────────────────────────────────────────────────────────

def _build_biharmonic_skeleton(case: Dict) -> str:
    """
    Generate a Biharmonic skeleton with ALL PDE math data pre-filled.
    Uses the two-step Poisson formulation matching the oracle:
        Step 1: -Δw = f      (find auxiliary variable w)
        Step 2: -Δu = w      (find primary solution u)
    The model writes: variational forms a_w, L_w, a_u, L_u + numerical choices.
    """
    pde_cfg      = case["oracle_config"]["pde"]
    manufactured = pde_cfg.get("manufactured_solution", {})
    out_cfg      = case["oracle_config"]["output"]["grid"]
    nx, ny       = out_cfg["nx"], out_cfg["ny"]

    if manufactured.get("u"):
        u_str = manufactured["u"]
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    # Computes f = Δ²u_exact, bc_u = u_exact|∂Ω, bc_w = -Δu_exact|∂Ω\n"
            f"    f_h, bc_u, bc_w = _manufactured_biharmonic_data(msh, V, {u_str!r})"
        )
    else:
        f_str  = pde_cfg.get("source_term", "0.0")
        bc_val = (pde_cfg.get("bc", {}).get("dirichlet", {}).get("value", "0.0"))
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    f_h   = _f_from_str(msh, {f_str!r})\n"
            f"    bc_u  = _bc_from_str(msh, V, {bc_val!r})  # u = g on ∂Ω\n"
            f"    bc_w  = _bc_from_str(msh, V, '0.0')       # w = 0 on ∂Ω"
        )

    return f'''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (← YOUR CHOICES, no defaults given) ────────
    # P2 elements recommended for 4th-order accuracy.
    N        = None  # ← choose mesh resolution (integer)
    degree   = None  # ← choose FE polynomial degree: 2 or 3 (P1 may be insufficient)
    ksp_type = None  # ← choose linear solver: "cg" or "gmres"
    pc_type  = None  # ← choose preconditioner: "hypre", "ilu", or "lu"
    rtol     = None  # ← choose relative tolerance

    # ── SETUP  (PROVIDED — do not modify) ────────────────────────────────
    # Two-step formulation: Δ²u = f  ⟺  (-Δw = f) then (-Δu = w)
    # Both sub-problems are standard Poisson → share one scalar function space.
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))
    petsc_opts = {{"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}}

{pde_data}

    # ── STEP 1: solve for w  (← YOUR CODE) ───────────────────────────────
    # Sub-problem:  -Δw = f   in Ω,   w = bc_w on ∂Ω
    # Weak form:    ∫ ∇w·∇v dx = ∫ f v dx   ∀ v
    w_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a_w = ...   # ← YOUR CODE: bilinear form for the w sub-problem
    L_w = ...   # ← YOUR CODE: linear form  for the w sub-problem

    w_h = LinearProblem(a_w, L_w, bcs=[bc_w],
                        petsc_options=petsc_opts,
                        petsc_options_prefix="p2_biharm_w_").solve()

    # ── STEP 2: solve for u  (← YOUR CODE) ───────────────────────────────
    # Sub-problem:  -Δu = w_h  in Ω,   u = bc_u on ∂Ω
    # Weak form:    ∫ ∇u·∇q dx = ∫ w_h q dx   ∀ q
    u_t = ufl.TrialFunction(V)
    q   = ufl.TestFunction(V)

    a_u = ...   # ← YOUR CODE: bilinear form for the u sub-problem
    L_u = ...   # ← YOUR CODE: linear form  for the u sub-problem (involves w_h)

    u_h = LinearProblem(a_u, L_u, bcs=[bc_u],
                        petsc_options=petsc_opts,
                        petsc_options_prefix="p2_biharm_u_").solve()

    # ── OUTPUT  (PROVIDED — do not modify) ───────────────────────────────
    return {{
        "u": _sample_scalar(u_h, {nx}, {ny}),
        "solver_info": {{"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol}},
    }}
'''


# ─────────────────────────────────────────────────────────────────────────────
# ConvectionDiffusion: dynamic skeleton (PDE data inlined at prompt-gen time)
# ─────────────────────────────────────────────────────────────────────────────

def _build_convdiff_skeleton(case: Dict) -> str:
    """
    Generate a convection-diffusion skeleton with ALL PDE math data pre-filled.
    PDE:  -ε ∇²u + β·∇u = f   in Ω  (steady)
      or  ∂u/∂t - ε ∇²u + β·∇u = f  in Ω × (0, T]  (transient)
    Handles 4 sub-cases: {steady, transient} × {manufactured, no-exact}.
    """
    import sympy as sp

    pde_cfg      = case["oracle_config"]["pde"]
    params       = pde_cfg.get("pde_params", {})
    epsilon      = float(params.get("epsilon", 0.01))
    beta         = params.get("beta", [1.0, 1.0])
    bx, by       = float(beta[0]), float(beta[1])
    manufactured = pde_cfg.get("manufactured_solution", {})
    time_cfg     = pde_cfg.get("time")
    out_cfg      = case["oracle_config"]["output"]["grid"]
    nx, ny       = out_cfg["nx"], out_cfg["ny"]

    beta_norm = (bx**2 + by**2) ** 0.5
    peclet    = beta_norm / epsilon if epsilon > 0 else float("inf")
    supg_note = f"  # Pe ≈ {peclet:.1f} — SUPG stabilization strongly recommended" if peclet > 10 else f"  # Pe ≈ {peclet:.1f} — SUPG optional"

    # ── Steady ────────────────────────────────────────────────────────────
    if time_cfg is None:
        if manufactured.get("u"):
            u_str = manufactured["u"]
            pde_data = (
                f"    # PDE data (pre-filled — do not modify):\n"
                f"    epsilon = {epsilon!r}\n"
                f"    beta_v  = ufl.as_vector([{bx!r}, {by!r}])\n"
                f"    f_h, bc = _manufactured_convdiff_f_and_bc(\n"
                f"        msh, V, {u_str!r}, {epsilon!r}, [{bx!r}, {by!r}]\n"
                f"    )"
            )
        else:
            f_str  = pde_cfg.get("source_term", "0.0")
            bc_val = pde_cfg.get("bc", {}).get("dirichlet", {}).get("value", "0.0")
            pde_data = (
                f"    # PDE data (pre-filled — do not modify):\n"
                f"    epsilon = {epsilon!r}\n"
                f"    beta_v  = ufl.as_vector([{bx!r}, {by!r}])\n"
                f"    f_h = _f_from_str(msh, {f_str!r})\n"
                f"    bc  = _bc_from_str(msh, V, {bc_val!r})"
            )

        return f'''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (← YOUR CHOICES, no defaults given) ────────
    {supg_note}
    N        = None  # ← choose mesh resolution (integer)
    degree   = None  # ← choose FE polynomial degree: 1 or 2
    ksp_type = None  # ← choose: "gmres" (required — system is non-symmetric)
    pc_type  = None  # ← choose: "ilu", "hypre", or "lu"
    rtol     = None  # ← choose relative tolerance
    use_supg = None  # ← choose True or False (SUPG stabilisation for high Péclet)

    # ── SETUP  (PROVIDED — do not modify) ────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))

{pde_data}

    # ── VARIATIONAL FORM  (← YOUR CODE) ──────────────────────────────────
    # PDE:  -ε ∇²u + β·∇u = f   in Ω,   u = g on ∂Ω
    # Standard weak form:  ε ∫ ∇u·∇v dx + ∫ (β·∇u) v dx = ∫ f v dx   ∀ v
    # SUPG (if use_supg):  add τ ∫ (β·∇v)(β·∇u - ε Δu) dx  to a
    #                      add τ ∫ (β·∇v) f dx              to L
    #   where τ = h / (2 ‖β‖)  and  h = ufl.CellDiameter(msh)
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # ← YOUR CODE: bilinear form (with optional SUPG)
    L = ...   # ← YOUR CODE: linear form   (with optional SUPG)

    # ── SOLVE  (PROVIDED — do not modify) ────────────────────────────────
    u_h = LinearProblem(a, L, bcs=[bc],
                        petsc_options={{"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}},
                        petsc_options_prefix="p2_convdiff_").solve()

    # ── OUTPUT  (PROVIDED — do not modify) ───────────────────────────────
    return {{
        "u": _sample_scalar(u_h, {nx}, {ny}),
        "solver_info": {{"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol,
                        "epsilon": epsilon, "use_supg": use_supg}},
    }}
'''

    # ── Transient ─────────────────────────────────────────────────────────
    t0    = float(time_cfg.get("t0",    0.0))
    t_end = float(time_cfg.get("t_end", 1.0))

    if manufactured.get("u"):
        u_str = manufactured["u"]
        sx, sy, st = sp.symbols("x y t", real=True)
        u_sym = sp.sympify(u_str, locals={"x": sx, "y": sy, "t": st, "pi": sp.pi})
        u_t_sym = sp.diff(u_sym, st)
        f_sym = (
            u_t_sym
            - epsilon * (sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2))
            + bx * sp.diff(u_sym, sx)
            + by * sp.diff(u_sym, sy)
        )
        f_sym = sp.simplify(f_sym)
        f_str_computed = str(f_sym)

        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    epsilon = {epsilon!r}\n"
            f"    beta_v  = ufl.as_vector([{bx!r}, {by!r}])\n"
            f"    t_end   = {t_end!r}\n"
            f"    # Manufactured source f(x,y,t) = ∂u/∂t - ε∆u + β·∇u (pre-computed):\n"
            f"    _f_str = {f_str_computed!r}\n"
            f"    _u_str = {u_str!r}   # manufactured solution for BC update\n"
            f"    # Initial condition = manufactured u at t={t0}:\n"
            f"    _g_fn = fem.Function(V)\n"
            f"    _update_bc_at_t(_g_fn, _u_str, {t0!r})\n"
            f"    u_n = fem.Function(V); u_n.x.array[:] = _g_fn.x.array[:]\n"
            f"    bc  = fem.dirichletbc(_g_fn, _all_boundary_dofs(msh, V))"
        )
    else:
        f_str  = pde_cfg.get("source_term", "0.0")
        bc_val = pde_cfg.get("bc", {}).get("dirichlet", {}).get("value", "0.0")
        ic_str = pde_cfg.get("initial_condition", "0.0")
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    epsilon = {epsilon!r}\n"
            f"    beta_v  = ufl.as_vector([{bx!r}, {by!r}])\n"
            f"    t_end   = {t_end!r}\n"
            f"    _f_str = {f_str!r}   # source term (may be time-independent)\n"
            f"    _u_str = {bc_val!r}  # BC value\n"
            f"    # Initial condition:\n"
            f"    _g_fn = fem.Function(V)\n"
            f"    _update_bc_at_t(_g_fn, _u_str, {t0!r})\n"
            f"    u_n = _f_from_str(msh, {ic_str!r})  # or fem.Function(V)\n"
            f"    u_n = fem.Function(V)\n"
            f"    u_n.interpolate(lambda pts: __import__('numpy').full(pts.shape[1], 0.0))\n"
            f"    bc  = fem.dirichletbc(_g_fn, _all_boundary_dofs(msh, V))"
        )

    return f'''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (← YOUR CHOICES, no defaults given) ────────
    {supg_note}
    N        = None  # ← choose mesh resolution (integer)
    degree   = None  # ← choose FE polynomial degree: 1 or 2
    dt       = None  # ← choose time step (float); t_end = {t_end!r}
    ksp_type = None  # ← choose: "gmres"
    pc_type  = None  # ← choose: "ilu", "hypre", or "lu"
    rtol     = None  # ← choose relative tolerance
    use_supg = None  # ← choose True or False

    # ── SETUP  (PROVIDED — do not modify) ────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))

{pde_data}

    # ── VARIATIONAL FORM  (← YOUR CODE) ──────────────────────────────────
    # PDE: ∂u/∂t - ε ∇²u + β·∇u = f   in Ω × (0, {t_end!r}]
    # Backward Euler bilinear form (assembled once):
    #   (1/dt) ∫ u v dx + ε ∫ ∇u·∇v dx + ∫ (β·∇u) v dx  [+ SUPG terms]
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # ← YOUR CODE: bilinear form (time-independent; include mass + diffusion + advection)

    # ── TIME LOOP  (PROVIDED structure — fill in L below) ─────────────────
    n_steps = max(1, round({t_end!r} / dt))
    t = {t0!r}
    u_h = fem.Function(V)

    for _ in range(n_steps):
        t += dt
        f_h = _f_time_dep(msh, _f_str, t)       # PROVIDED: f at current time
        _update_bc_at_t(_g_fn, _u_str, t)        # PROVIDED: update BC in-place

        L = ...   # ← YOUR CODE: linear form at this step (involves u_n and f_h)

        u_h = LinearProblem(a, L, bcs=[bc],
                            petsc_options={{"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}},
                            petsc_options_prefix="p2_convdiff_").solve()
        u_n.x.array[:] = u_h.x.array

    # ── OUTPUT  (PROVIDED — do not modify) ───────────────────────────────
    return {{
        "u": _sample_scalar(u_h, {nx}, {ny}),
        "solver_info": {{"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol,
                        "epsilon": epsilon, "dt": dt, "use_supg": use_supg}},
    }}
'''


# ─────────────────────────────────────────────────────────────────────────────
# ReactionDiffusion: dynamic skeleton (PDE data inlined at prompt-gen time)
# All benchmark cases are transient; supports linear and nonlinear reactions.
# ─────────────────────────────────────────────────────────────────────────────

def _build_rxndiff_skeleton(case: Dict) -> str:
    """
    Generate a reaction-diffusion skeleton with ALL PDE math data pre-filled.
    PDE:  ∂u/∂t - ε Δu + R(u) = f   in Ω × (0, T]   (all cases are transient)
    Reaction types: linear (α u), cubic (α u + β u³), allen_cahn (λ(u³-u)),
                    logistic (ρ u(1-u)).
    Linear reactions → LinearProblem each step.
    Nonlinear reactions → Newton (NewtonSolver) each step.
    """
    import sympy as sp

    pde_cfg      = case["oracle_config"]["pde"]
    params       = pde_cfg.get("pde_params", {})
    epsilon      = float(params.get("epsilon", params.get("diffusion", 0.1)))
    reaction     = params.get("reaction", {"type": "linear", "alpha": 0.0})
    rtype        = str(reaction.get("type", "linear")).lower()
    manufactured = pde_cfg.get("manufactured_solution", {})
    time_cfg     = pde_cfg.get("time", {})
    t0    = float(time_cfg.get("t0",    0.0))
    t_end = float(time_cfg.get("t_end", 1.0))
    out_cfg = case["oracle_config"]["output"]["grid"]
    nx, ny  = out_cfg["nx"], out_cfg["ny"]

    is_linear_rxn = rtype == "linear"

    # ── Reaction term string for UFL (to be filled by model) ─────────────
    if rtype == "linear":
        alpha = float(reaction.get("alpha", 0.0))
        rxn_ufl_hint = f"alpha * u_t * v  where alpha = {alpha!r}"
        rxn_params   = f"    alpha = {alpha!r}   # linear reaction coefficient"
    elif rtype in ("cubic", "poly3"):
        alpha = float(reaction.get("alpha", 0.0))
        beta  = float(reaction.get("beta",  1.0))
        rxn_ufl_hint = f"(alpha * u + beta * u**3) * v  where alpha={alpha!r}, beta={beta!r}"
        rxn_params   = f"    alpha = {alpha!r}; beta = {beta!r}   # cubic: R(u) = α u + β u³"
    elif rtype in ("allen_cahn", "allen-cahn"):
        lam = float(reaction.get("lambda", reaction.get("lam", 1.0)))
        rxn_ufl_hint = f"lam * (u**3 - u) * v  where lam = {lam!r}"
        rxn_params   = f"    lam = {lam!r}   # Allen-Cahn: R(u) = λ(u³ - u)"
    elif rtype in ("logistic", "fisher_kpp", "fisher-kpp"):
        rho = float(reaction.get("rho", 1.0))
        rxn_ufl_hint = f"rho * u * (1 - u) * v  where rho = {rho!r}"
        rxn_params   = f"    rho = {rho!r}   # Logistic/Fisher-KPP: R(u) = ρ u(1-u)"
    else:
        rxn_ufl_hint = "# unknown reaction type — inspect reaction_spec"
        rxn_params   = f"    # reaction_spec = {reaction!r}"

    # ── Compute manufactured source f(x,y,t) symbolically ─────────────────
    if manufactured.get("u"):
        u_str = manufactured["u"]
        sx, sy, st = sp.symbols("x y t", real=True)
        u_sym = sp.sympify(u_str, locals={"x": sx, "y": sy, "t": st, "pi": sp.pi})
        u_t_sym = sp.diff(u_sym, st)
        lap_u   = sp.diff(u_sym, sx, 2) + sp.diff(u_sym, sy, 2)

        # R(u_sym) symbolically
        if rtype == "linear":
            R_sym = float(reaction.get("alpha", 0.0)) * u_sym
        elif rtype in ("cubic", "poly3"):
            R_sym = float(reaction.get("alpha", 0.0)) * u_sym + float(reaction.get("beta", 1.0)) * u_sym**3
        elif rtype in ("allen_cahn", "allen-cahn"):
            lam_v = float(reaction.get("lambda", reaction.get("lam", 1.0)))
            R_sym = lam_v * (u_sym**3 - u_sym)
        elif rtype in ("logistic", "fisher_kpp", "fisher-kpp"):
            rho_v = float(reaction.get("rho", 1.0))
            R_sym = rho_v * u_sym * (1 - u_sym)
        else:
            R_sym = sp.sympify(0)

        f_sym = sp.simplify(u_t_sym - epsilon * lap_u + R_sym)
        f_str_computed = str(f_sym)

        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    epsilon = {epsilon!r}\n"
            f"    t_end   = {t_end!r}\n"
            f"{rxn_params}\n"
            f"    # Manufactured source f(x,y,t) = ∂u/∂t - ε∆u + R(u_exact):\n"
            f"    _f_str = {f_str_computed!r}\n"
            f"    _u_str = {u_str!r}   # manufactured solution for BC/IC\n"
            f"    # Initial condition and BC:\n"
            f"    _g_fn = fem.Function(V)\n"
            f"    _update_bc_at_t(_g_fn, _u_str, {t0!r})\n"
            f"    u_n = fem.Function(V); u_n.x.array[:] = _g_fn.x.array[:]\n"
            f"    bc  = fem.dirichletbc(_g_fn, _all_boundary_dofs(msh, V))"
        )
    else:
        f_str  = pde_cfg.get("source_term", "0.0")
        bc_val = pde_cfg.get("bc", {}).get("dirichlet", {}).get("value", "0.0")
        ic_str = pde_cfg.get("initial_condition", "0.0")
        pde_data = (
            f"    # PDE data (pre-filled — do not modify):\n"
            f"    epsilon = {epsilon!r}\n"
            f"    t_end   = {t_end!r}\n"
            f"{rxn_params}\n"
            f"    _f_str = {f_str!r}   # source term (time-independent)\n"
            f"    _u_str = {bc_val!r}  # BC value\n"
            f"    _ic_str = {ic_str!r} # initial condition\n"
            f"    # IC and BC:\n"
            f"    _g_fn = fem.Function(V)\n"
            f"    _update_bc_at_t(_g_fn, _u_str, {t0!r})\n"
            f"    u_n = fem.Function(V)\n"
            f"    u_n.interpolate(lambda pts: _f_time_dep(msh, _ic_str, {t0!r}).x.array.reshape(-1))\n"
            f"    bc  = fem.dirichletbc(_g_fn, _all_boundary_dofs(msh, V))"
        )

    if is_linear_rxn:
        variational_section = f'''
    # ── VARIATIONAL FORM  (← YOUR CODE) ──────────────────────────────────
    # PDE: ∂u/∂t - ε Δu + α u = f   in Ω × (0, {t_end!r}]
    # Backward Euler weak form (each step):
    #   (1/dt) ∫ u v dx + ε ∫ ∇u·∇v dx + α ∫ u v dx = (1/dt) ∫ u_n v dx + ∫ f v dx
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # ← YOUR CODE: bilinear form (mass + diffusion + linear reaction)
              # Hint:  {rxn_ufl_hint}

    # ── TIME LOOP  (PROVIDED structure — fill in L) ────────────────────────
    n_steps = max(1, round({t_end!r} / dt))
    t = {t0!r}
    u_h = fem.Function(V)

    for _ in range(n_steps):
        t += dt
        f_h = _f_time_dep(msh, _f_str, t)     # PROVIDED: source at time t
        _update_bc_at_t(_g_fn, _u_str, t)     # PROVIDED: update BC in-place

        L = ...   # ← YOUR CODE: linear form (involves u_n and f_h)

        u_h = LinearProblem(a, L, bcs=[bc],
                            petsc_options={{"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol}},
                            petsc_options_prefix="p2_rxndiff_").solve()
        u_n.x.array[:] = u_h.x.array'''
    else:
        variational_section = f'''
    # ── VARIATIONAL FORM  (← YOUR CODE) ──────────────────────────────────
    # PDE: ∂u/∂t - ε Δu + R(u) = f   in Ω × (0, {t_end!r}]
    # R(u) is NONLINEAR → use Newton linearisation each time step.
    # Backward Euler nonlinear residual:
    #   F(u) = (u - u_n)/dt * v dx + ε ∫ ∇u·∇v dx + ∫ R(u) v dx - ∫ f v dx = 0
    # Newton: J = dF/du  (computed automatically via ufl.derivative)

    v   = ufl.TestFunction(V)
    u_h = fem.Function(V)     # iterate (Newton solution variable)

    # ── TIME LOOP  (PROVIDED structure — fill in F below) ─────────────────
    from dolfinx.nls.petsc import NewtonSolver as _NewtonSolver
    from dolfinx.fem.petsc import NonlinearProblem as _NLP

    n_steps = max(1, round({t_end!r} / dt))
    t = {t0!r}
    u_h.x.array[:] = u_n.x.array  # initialise iterate

    for _ in range(n_steps):
        t += dt
        f_h = _f_time_dep(msh, _f_str, t)     # PROVIDED: source at time t
        _update_bc_at_t(_g_fn, _u_str, t)     # PROVIDED: update BC in-place

        # ← YOUR CODE: nonlinear residual F(u_h) using u_h (fem.Function)
        # Reaction term hint:  {rxn_ufl_hint}
        F = ...   # e.g. ((u_h - u_n)/dt * v + ε ∫∇u_h·∇v dx + R(u_h)*v - f_h*v) * dx

        J       = ufl.derivative(F, u_h)       # PROVIDED: automatic Jacobian
        problem = _NLP(F, u_h, bcs=[bc], J=J)
        solver  = _NewtonSolver(MPI.COMM_WORLD, problem)
        solver.rtol   = newton_rtol
        solver.max_it = newton_max_it
        solver.solve(u_h)
        u_n.x.array[:] = u_h.x.array'''

    solver_params = (
        "    ksp_type    = None  # ← choose: \"gmres\"\n"
        "    pc_type     = None  # ← choose: \"ilu\" or \"lu\"\n"
        "    rtol        = None  # ← choose relative tolerance"
        if is_linear_rxn else
        "    ksp_type    = None  # ← choose: \"gmres\"\n"
        "    pc_type     = None  # ← choose: \"ilu\" or \"lu\"\n"
        "    newton_rtol = None  # ← choose Newton relative tolerance\n"
        "    newton_max_it = None  # ← choose max Newton iterations (e.g. 20–50)"
    )

    return f'''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (← YOUR CHOICES, no defaults given) ────────
    N        = None  # ← choose mesh resolution (integer)
    degree   = None  # ← choose FE polynomial degree: 1 or 2
    dt       = None  # ← choose time step (float); t_end = {t_end!r}
{solver_params}

    # ── SETUP  (PROVIDED — do not modify) ────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))

{pde_data}
{variational_section}

    # ── OUTPUT  (PROVIDED — do not modify) ───────────────────────────────
    return {{
        "u": _sample_scalar(u_h, {nx}, {ny}),
        "solver_info": {{"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type,
                        "rtol": {("rtol" if is_linear_rxn else "newton_rtol")}, "epsilon": epsilon, "dt": dt}},
    }}
'''


# ─────────────────────────────────────────────────────────────────────────────
# Stokes: dynamic skeleton (PDE data inlined at prompt-gen time)
# ─────────────────────────────────────────────────────────────────────────────

def _build_stokes_skeleton(case: Dict) -> str:
    """Generate a Stokes skeleton with body force and BCs pre-filled.
    Manufactured cases: f = -ν∆u + ∇p inlined; BC = u_exact on all ∂Ω.
    No-exact cases: source_term and BC config inlined.
    """
    import sympy as sp

    oracle_cfg = case["oracle_config"]
    pde_cfg    = oracle_cfg["pde"]
    params     = pde_cfg.get("pde_params", {})
    nu         = float(params.get("nu", 1.0))
    mfg        = pde_cfg.get("manufactured_solution", {})
    out_cfg    = oracle_cfg["output"]["grid"]
    nx, ny     = out_cfg["nx"], out_cfg["ny"]

    if mfg.get("u") and mfg.get("p") is not None:
        # ── Manufactured: compute f = -ν∆u + ∇p symbolically ──────────────
        u_strs = mfg["u"]
        p_str  = str(mfg["p"])
        sx, sy = sp.symbols("x y", real=True)
        loc    = {"x": sx, "y": sy, "pi": sp.pi}
        u_sym  = [sp.sympify(s, locals=loc) for s in u_strs]
        p_sym  = sp.sympify(p_str, locals=loc)

        f_syms = []
        for u_i, coord in zip(u_sym, [sx, sy]):
            lap    = sp.diff(u_i, sx, 2) + sp.diff(u_i, sy, 2)
            grad_p = sp.diff(p_sym, coord)
            f_syms.append(sp.simplify(-nu * lap + grad_p))

        f1_str = str(f_syms[0])
        f2_str = str(f_syms[1])
        u1_str, u2_str = u_strs[0], u_strs[1]

        data_section = f"""\
    # ── PDE PARAMETERS (INLINED) ─────────────────────────────────────────
    nu = {nu}  # kinematic viscosity

    # BODY FORCE (PROVIDED — f = -ν∆u_exact + ∇p_exact)
    #   u_exact  = [{u1_str!r}, {u2_str!r}]
    #   p_exact  = {p_str!r}
    #   f1 = {f1_str!r}
    #   f2 = {f2_str!r}
    f_body = _manufactured_stokes_f(msh, {f1_str!r}, {f2_str!r})

    # BOUNDARY CONDITIONS (PROVIDED — velocity = u_exact on all ∂Ω)
    bc_u  = _stokes_exact_bc(msh, W, V_col, {u1_str!r}, {u2_str!r})
    bc_p  = _pressure_pin(msh, W)   # pin pressure at (0,0) to remove null space
    bcs   = [bc_u] + ([bc_p] if bc_p is not None else [])"""
    else:
        # ── No-exact: inline source_term and BC config ─────────────────────
        src = pde_cfg.get("source_term", ["0.0", "0.0"])
        if isinstance(src, (list, tuple)):
            f1_str, f2_str = str(src[0]), str(src[1])
        else:
            f1_str, f2_str = str(src), "0.0"

        bc_raw = oracle_cfg.get("bc", {}).get("dirichlet", [])
        if isinstance(bc_raw, dict):
            bc_raw = [bc_raw]
        bc_repr = repr(bc_raw)

        data_section = f"""\
    # ── PDE PARAMETERS (INLINED) ─────────────────────────────────────────
    nu = {nu}  # kinematic viscosity

    # BODY FORCE (PROVIDED — given as source term)
    #   f = [{f1_str!r}, {f2_str!r}]
    f_body = _manufactured_stokes_f(msh, {f1_str!r}, {f2_str!r})

    # BOUNDARY CONDITIONS (PROVIDED — inlined from case configuration)
    _bc_data = {bc_repr}
    bcs_u = _stokes_bc_from_config(msh, W, V_col, _bc_data)
    bc_p  = _pressure_pin(msh, W)   # pin pressure at (0,0) to remove null space
    bcs   = bcs_u + ([bc_p] if bc_p is not None else [])"""

    return f'''def solve(case_spec: dict) -> dict:
    from basix.ufl import element as _bel, mixed_element as _bmix

    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N        = None   # mesh resolution (32–64 typical; Stokes is memory-intensive)
    deg_u    = None   # velocity element degree — hint: 2  (must be > deg_p)
    deg_p    = None   # pressure element degree — hint: 1
    ksp_type = None   # e.g. "minres" or "gmres"
    pc_type  = None   # e.g. "hypre" or "lu"
    rtol     = None   # e.g. 1e-8

    # ── MESH & MIXED FUNCTION SPACE  (PROVIDED) ──────────────────────────
    # Taylor-Hood pair: P_deg_u (velocity) × P_deg_p (pressure)
    msh  = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    W    = fem.functionspace(msh, _bmix([
        _bel("Lagrange", cell, deg_u, shape=(gdim,)),
        _bel("Lagrange", cell, deg_p),
    ]))
    V_col, _ = W.sub(0).collapse()

{data_section}

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # Steady Stokes:  -ν ∇²u + ∇p = f_body,   ∇·u = 0
    # Mixed weak form (Taylor-Hood):
    #   ν ∫ ∇u:∇v dx  −  ∫ p ∇·v dx  −  ∫ q ∇·u dx  =  ∫ f_body·v dx
    (u_t, p_t) = ufl.TrialFunctions(W)
    (v,   q  ) = ufl.TestFunctions(W)

    a = ...   # TODO: bilinear form a(u_t, p_t; v, q)
    L = ...   # TODO: linear form  L(v)

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    wh  = LinearProblem(a, L, bcs=bcs, petsc_options={{
        "ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol,
    }}, petsc_options_prefix="p2_stokes_").solve()
    u_h = wh.sub(0).collapse()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    return {{
        "u": _sample_vector_magnitude(u_h, {nx}, {ny}),
        "solver_info": {{"mesh_resolution": N, "element_degree": deg_u,
                        "pressure_degree": deg_p,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol,
                        "nu": {nu}}},
    }}
'''


# ─────────────────────────────────────────────────────────────────────────────
# Navier-Stokes: dynamic skeleton (PDE data inlined at prompt-gen time)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ns_skeleton(case: Dict) -> str:
    """Generate a Navier-Stokes skeleton with body force and BCs pre-filled.
    Manufactured cases: f = (u·∇)u - ν∆u + ∇p inlined; BC = u_exact on all ∂Ω.
    No-exact cases: source_term and BC config inlined.
    """
    import sympy as sp

    oracle_cfg = case["oracle_config"]
    pde_cfg    = oracle_cfg["pde"]
    params     = pde_cfg.get("pde_params", {})
    nu         = float(params.get("nu", 1.0))
    mfg        = pde_cfg.get("manufactured_solution", {})
    out_cfg    = oracle_cfg["output"]["grid"]
    nx, ny     = out_cfg["nx"], out_cfg["ny"]

    if mfg.get("u") and mfg.get("p") is not None:
        # ── Manufactured: compute f = (u·∇)u - ν∆u + ∇p symbolically ─────
        u_strs = mfg["u"]
        p_str  = str(mfg.get("p", "0"))
        sx, sy = sp.symbols("x y", real=True)
        loc    = {"x": sx, "y": sy, "pi": sp.pi}
        u_sym  = [sp.sympify(s, locals=loc) for s in u_strs]
        p_sym  = sp.sympify(p_str, locals=loc)

        f_syms = []
        for i, (u_i, coord) in enumerate(zip(u_sym, [sx, sy])):
            conv   = sum(u_sym[j] * sp.diff(u_i, [sx, sy][j]) for j in range(2))
            lap    = sp.diff(u_i, sx, 2) + sp.diff(u_i, sy, 2)
            grad_p = sp.diff(p_sym, coord)
            f_syms.append(sp.simplify(conv - nu * lap + grad_p))

        f1_str = str(f_syms[0])
        f2_str = str(f_syms[1])
        u1_str, u2_str = u_strs[0], u_strs[1]

        data_section = f"""\
    # ── PDE PARAMETERS (INLINED) ─────────────────────────────────────────
    nu = {nu}  # kinematic viscosity

    # BODY FORCE (PROVIDED — f = (u·∇)u_exact - ν∆u_exact + ∇p_exact)
    #   u_exact  = [{u1_str!r}, {u2_str!r}]
    #   p_exact  = {p_str!r}
    #   f1 = {f1_str!r}
    #   f2 = {f2_str!r}
    f_body = _manufactured_stokes_f(msh, {f1_str!r}, {f2_str!r})

    # BOUNDARY CONDITIONS (PROVIDED — velocity = u_exact on all ∂Ω)
    bc_u  = _stokes_exact_bc(msh, W, V_col, {u1_str!r}, {u2_str!r})
    bc_p  = _pressure_pin(msh, W)   # pin pressure at (0,0) to remove null space
    bcs   = [bc_u] + ([bc_p] if bc_p is not None else [])"""
    else:
        # ── No-exact: inline source_term and BC config ─────────────────────
        src = pde_cfg.get("source_term", ["0.0", "0.0"])
        if isinstance(src, (list, tuple)):
            f1_str, f2_str = str(src[0]), str(src[1])
        else:
            f1_str, f2_str = str(src), "0.0"

        bc_raw = oracle_cfg.get("bc", {}).get("dirichlet", [])
        if isinstance(bc_raw, dict):
            bc_raw = [bc_raw]
        bc_repr = repr(bc_raw)

        data_section = f"""\
    # ── PDE PARAMETERS (INLINED) ─────────────────────────────────────────
    nu = {nu}  # kinematic viscosity

    # BODY FORCE (PROVIDED — given as source term)
    #   f = [{f1_str!r}, {f2_str!r}]
    f_body = _manufactured_stokes_f(msh, {f1_str!r}, {f2_str!r})

    # BOUNDARY CONDITIONS (PROVIDED — inlined from case configuration)
    _bc_data = {bc_repr}
    bcs_u = _stokes_bc_from_config(msh, W, V_col, _bc_data)
    bc_p  = _pressure_pin(msh, W)   # pin pressure at (0,0) to remove null space
    bcs   = bcs_u + ([bc_p] if bc_p is not None else [])"""

    return f'''def solve(case_spec: dict) -> dict:
    from basix.ufl import element as _bel, mixed_element as _bmix
    from dolfinx.nls.petsc import NewtonSolver as _NewtonSolver

    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N             = None   # mesh resolution (24–48 typical; NS is expensive)
    deg_u         = None   # velocity element degree — hint: 2
    deg_p         = None   # pressure element degree — hint: 1
    newton_rtol   = None   # Newton convergence tolerance (e.g. 1e-8)
    newton_max_it = None   # max Newton iterations (e.g. 30–50)

    # ── MESH & MIXED FUNCTION SPACE  (PROVIDED) ──────────────────────────
    # Taylor-Hood pair: P_deg_u (velocity) × P_deg_p (pressure)
    msh  = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    W    = fem.functionspace(msh, _bmix([
        _bel("Lagrange", cell, deg_u, shape=(gdim,)),
        _bel("Lagrange", cell, deg_p),
    ]))
    V_col, _ = W.sub(0).collapse()

{data_section}

    # ── NONLINEAR RESIDUAL FORM  (YOUR CODE) ─────────────────────────────
    # Steady Navier-Stokes:  (u·∇)u − ν ∇²u + ∇p = f_body,   ∇·u = 0
    # Newton form — define residual F, Jacobian J = dF/dw is provided:
    #   F = ∫ (u·∇u)·v dx + ν ∫ ∇u:∇v dx − ∫ p ∇·v dx − ∫ q ∇·u dx − ∫ f·v dx
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = ...   # TODO: nonlinear residual form (see hint above)
    J = ufl.derivative(F, w)   # automatic Jacobian — provided

    # ── SOLVE  (PROVIDED — Newton solver) ────────────────────────────────
    _prob   = NonlinearProblem(F, w, bcs=bcs, J=J,
                               petsc_options_prefix="p2_ns_")
    _solver = _NewtonSolver(MPI.COMM_WORLD, _prob)
    _solver.rtol   = newton_rtol
    _solver.max_it = newton_max_it
    _solver.solve(w)
    w.x.scatter_forward()
    u_h = w.sub(0).collapse()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    return {{
        "u": _sample_vector_magnitude(u_h, {nx}, {ny}),
        "solver_info": {{"mesh_resolution": N, "element_degree": deg_u,
                        "pressure_degree": deg_p,
                        "ksp_type": "newton_internal", "pc_type": "newton_internal",
                        "rtol": newton_rtol, "newton_rtol": newton_rtol,
                        "newton_max_it": newton_max_it, "nu": {nu}}},
    }}
'''


# ─────────────────────────────────────────────────────────────────────────────
# Linear Elasticity: dynamic skeleton (PDE data inlined at prompt-gen time)
# ─────────────────────────────────────────────────────────────────────────────

def _build_le_skeleton(case: Dict) -> str:
    """Generate a linear elasticity skeleton with body force and BCs pre-filled.
    Manufactured cases: f = -div(σ(u_exact)) inlined; BC = u_exact on all ∂Ω.
    No-exact cases: source_term and BC config inlined.
    """
    import sympy as sp

    oracle_cfg = case["oracle_config"]
    pde_cfg    = oracle_cfg["pde"]
    params     = pde_cfg.get("pde_params", {})
    mfg        = pde_cfg.get("manufactured_solution", {})
    out_cfg    = oracle_cfg["output"]["grid"]
    nx, ny     = out_cfg["nx"], out_cfg["ny"]

    # Lamé constants
    if "lambda" in params and "mu" in params:
        lam    = float(params["lambda"])
        mu     = float(params["mu"])
        E, nu  = None, None
    else:
        E      = float(params.get("E", 1.0))
        nu     = float(params.get("nu", 0.3))
        mu     = E / (2.0 * (1.0 + nu))
        lam    = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    param_comment = (f"# E={E}, ν={nu} → λ={lam:.6g}, μ={mu:.6g}"
                     if E is not None else f"# λ={lam:.6g}, μ={mu:.6g}")

    if mfg.get("u"):
        # ── Manufactured: compute f = -div(σ(u_exact)) symbolically ───────
        u_strs = mfg["u"]
        sx, sy = sp.symbols("x y", real=True)
        loc    = {"x": sx, "y": sy, "pi": sp.pi}
        u1 = sp.sympify(u_strs[0], locals=loc)
        u2 = sp.sympify(u_strs[1], locals=loc)

        du1x = sp.diff(u1, sx);  du1y = sp.diff(u1, sy)
        du2x = sp.diff(u2, sx);  du2y = sp.diff(u2, sy)
        eps11 = du1x;            eps22 = du2y
        eps12 = sp.Rational(1, 2) * (du1y + du2x)
        tr_e  = eps11 + eps22

        lam_s, mu_s = sp.sympify(lam), sp.sympify(mu)
        sig11 = 2 * mu_s * eps11 + lam_s * tr_e
        sig22 = 2 * mu_s * eps22 + lam_s * tr_e
        sig12 = 2 * mu_s * eps12

        f1 = sp.simplify(-(sp.diff(sig11, sx) + sp.diff(sig12, sy)))
        f2 = sp.simplify(-(sp.diff(sig12, sx) + sp.diff(sig22, sy)))

        f1_str, f2_str   = str(f1), str(f2)
        u1_str, u2_str   = u_strs[0], u_strs[1]

        data_section = f"""\
    # ── PDE PARAMETERS (INLINED) ─────────────────────────────────────────
    {param_comment}
    lam = {lam:.10g}
    mu  = {mu:.10g}

    # BODY FORCE (PROVIDED — f = -div(σ(u_exact)), σ = 2μ ε + λ tr(ε) I)
    #   u_exact = [{u1_str!r}, {u2_str!r}]
    #   f1 = {f1_str!r}
    #   f2 = {f2_str!r}
    f_body = _manufactured_stokes_f(msh, {f1_str!r}, {f2_str!r})

    # BOUNDARY CONDITIONS (PROVIDED — u = u_exact on all ∂Ω)
    bcs = [_le_exact_bc(msh, V, {u1_str!r}, {u2_str!r})]"""
    else:
        # ── No-exact: inline source_term and BC config ─────────────────────
        src = pde_cfg.get("source_term", ["0.0", "0.0"])
        if isinstance(src, (list, tuple)):
            f1_str, f2_str = str(src[0]), str(src[1])
        else:
            f1_str, f2_str = str(src), "0.0"

        bc_raw = oracle_cfg.get("bc", {}).get("dirichlet", [])
        if isinstance(bc_raw, dict):
            bc_raw = [bc_raw]
        bc_repr = repr(bc_raw)

        data_section = f"""\
    # ── PDE PARAMETERS (INLINED) ─────────────────────────────────────────
    {param_comment}
    lam = {lam:.10g}
    mu  = {mu:.10g}

    # BODY FORCE (PROVIDED — given as source term)
    #   f = [{f1_str!r}, {f2_str!r}]
    f_body = _manufactured_stokes_f(msh, {f1_str!r}, {f2_str!r})

    # BOUNDARY CONDITIONS (PROVIDED — inlined from case configuration)
    _bc_data = {bc_repr}
    bcs = _le_bc_from_config(msh, V, _bc_data)"""

    return f'''def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N        = None   # mesh resolution (32–64 typical)
    degree   = None   # element degree — hint: P2 (avoids volumetric locking near ν → 0.5)
    ksp_type = None   # e.g. "cg"
    pc_type  = None   # e.g. "hypre"
    rtol     = None   # e.g. 1e-8

    # ── MESH & FUNCTION SPACE  (PROVIDED) ────────────────────────────────
    msh  = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    gdim = msh.geometry.dim
    V    = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))

{data_section}

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # Linear elasticity:  -∇·σ(u) = f_body  in Ω
    #   σ(u) = 2μ ε(u) + λ tr(ε(u)) I,   ε(u) = sym(∇u)
    # Weak form:   ∫ σ(u):ε(v) dx  =  ∫ f_body·v dx   ∀v
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    def eps(w):   return ufl.sym(ufl.grad(w))
    def sigma(w): return ...   # TODO: 2*mu*eps(w) + lam*ufl.tr(eps(w))*ufl.Identity(gdim)

    a = ...   # TODO: ufl.inner(sigma(u_t), eps(v)) * ufl.dx
    L = ...   # TODO: ufl.inner(f_body, v) * ufl.dx

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    u_h = LinearProblem(a, L, bcs=bcs, petsc_options={{
        "ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol,
    }}, petsc_options_prefix="p2_le_").solve()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    return {{
        "u": _sample_vector_magnitude(u_h, {nx}, {ny}),
        "solver_info": {{"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol,
                        "lam": lam, "mu": mu}},
    }}
'''


_CONVDIFF_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N        = 64
    degree   = 1
    ksp_type = "gmres"
    pc_type  = "ilu"
    rtol     = 1e-8
    use_supg = True    # set True for high Péclet (Pe > 1); harmless otherwise

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))
    x   = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    # NOTE: case_spec["oracle_config"]["pde"]["pde_params"] contains:
    #   'epsilon': diffusion coefficient ε
    #   'beta': convection velocity [β_x, β_y]
    # Check case_spec["oracle_config"]["pde"].get("time") for transient variant.
    #
    # Helpers: _manufactured_f_and_bc, _f_from_str, _bc_from_str

    pde_cfg = case_spec["oracle_config"]["pde"]
    params  = pde_cfg.get("pde_params", {})
    epsilon = float(params.get("epsilon", 0.01))
    beta_v  = params.get("beta", [1.0, 1.0])
    beta    = ufl.as_vector([float(beta_v[0]), float(beta_v[1])])

    # TODO: define f_h and bc
    f_h = ...   # TODO
    bc  = ...   # TODO

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # PDE:  -ε ∇²u + β·∇u = f   in Ω,   u = g on ∂Ω
    # Standard weak form:  ε ∫ ∇u·∇v dx + ∫ (β·∇u) v dx = ∫ f v dx
    # SUPG stabilization (for high Pe): adds τ_SUPG * (β·∇u) * (β·∇v) terms
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # TODO: bilinear form (with optional SUPG)
    L = ...   # TODO: linear form

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    u_h = LinearProblem(a, L, bcs=[bc], petsc_options={
        "ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol,
    }).solve()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["oracle_config"]["output"]["grid"]
    return {
        "u": _sample_scalar(u_h, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol},
    }
'''

_STOKES_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N        = 32       # mesh resolution (Stokes is expensive; 32–64 typical)
    deg_u    = 2        # velocity element degree  — MUST be > deg_p for inf-sup
    deg_p    = 1        # pressure element degree
    ksp_type = "gmres"
    pc_type  = "lu"
    rtol     = 1e-8

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    from basix.ufl import element as _bel, mixed_element as _bmix
    msh  = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    # Taylor-Hood: P{deg_u}/P{deg_p}
    W = fem.functionspace(msh, _bmix([
        _bel("Lagrange", cell, deg_u, shape=(gdim,)),
        _bel("Lagrange", cell, deg_p),
    ]))
    V_col, _ = W.sub(0).collapse()
    x = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    # case_spec['pde']['pde_params']['nu'] — kinematic viscosity
    # case_spec['pde']['manufactured_solution'] — {'u': [...], 'p': ...} (may be present)
    # case_spec['pde']['source_term'] — body force as list ['fx', 'fy'] (may be present)
    #
    # For manufactured Stokes:
    #   f = -ν ∇²u_exact + ∇p_exact  (compute symbolically from strings)

    pde_cfg = case_spec["pde"]
    nu = float(pde_cfg.get("pde_params", {}).get("nu", 1.0))

    # TODO: define body force f as ufl.as_vector([fx, fy])
    f_body = ...   # TODO: e.g. ufl.as_vector([0.0, 0.0])

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # Stokes:  -ν ∇²u + ∇p = f,  ∇·u = 0
    # Weak form (mixed):
    #   ν ∫ ∇u:∇v dx - ∫ p ∇·v dx + ∫ q ∇·u dx = ∫ f·v dx
    (u_t, p_t) = ufl.TrialFunctions(W)
    (v, q)     = ufl.TestFunctions(W)

    a = ...   # TODO: bilinear form
    L = ...   # TODO: linear form

    # ── BOUNDARY CONDITIONS  (YOUR CODE) ─────────────────────────────────
    # No-slip on walls: u = 0 (or manufactured u_exact)
    # Hint: use _bc_vec_zero(msh, V_col) for homogeneous BC then lift to W.sub(0)
    fdim  = msh.topology.dim - 1
    facets = _dmesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_col), fdim, facets)
    u_bc   = fem.Function(V_col); u_bc.x.array[:] = 0.0  # TODO: replace with exact BC if available
    bc_u   = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    bcs    = [bc_u]
    # TODO: add pressure pin or manufactured pressure BC if needed

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    wh = LinearProblem(a, L, bcs=bcs, petsc_options={
        "ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol,
    }).solve()
    u_h = wh.sub(0).collapse()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["output"]["grid"]
    return {
        "u": _sample_vector_magnitude(u_h, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": deg_u,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol},
    }
'''

_NS_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N              = 32
    deg_u          = 2
    deg_p          = 1
    newton_rtol    = 1e-8
    newton_max_it  = 30

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    from basix.ufl import element as _bel, mixed_element as _bmix
    msh  = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    cell = msh.topology.cell_name()
    gdim = msh.geometry.dim
    W = fem.functionspace(msh, _bmix([
        _bel("Lagrange", cell, deg_u, shape=(gdim,)),
        _bel("Lagrange", cell, deg_p),
    ]))
    V_col, _ = W.sub(0).collapse()
    x = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    pde_cfg = case_spec["pde"]
    nu = float(pde_cfg.get("pde_params", {}).get("nu", 1.0))
    # TODO: define body force f_body as ufl.as_vector([fx, fy])
    f_body = ...   # TODO

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # Navier-Stokes (steady):  (u·∇)u - ν ∇²u + ∇p = f,  ∇·u = 0
    # Newton linearisation: F(w) = 0, J = dF/dw
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    F = ...   # TODO: nonlinear residual form
    J = ufl.derivative(F, w)   # Jacobian (automatic differentiation — provided)

    # ── BOUNDARY CONDITIONS  (YOUR CODE) ─────────────────────────────────
    fdim   = msh.topology.dim - 1
    facets = _dmesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_col), fdim, facets)
    u_bc   = fem.Function(V_col); u_bc.x.array[:] = 0.0  # TODO: set exact BC
    bcs    = [fem.dirichletbc(u_bc, dofs_u, W.sub(0))]

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    from dolfinx.nls.petsc import NewtonSolver
    from dolfinx.fem.petsc import NonlinearProblem as _NLP
    problem = _NLP(F, w, bcs=bcs, J=J)
    solver  = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.rtol = newton_rtol
    solver.max_it = newton_max_it
    solver.solve(w)
    w.x.scatter_forward()
    u_h = w.sub(0).collapse()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["output"]["grid"]
    return {
        "u": _sample_vector_magnitude(u_h, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": deg_u,
                        "ksp_type": "gmres", "pc_type": "lu", "rtol": newton_rtol},
    }
'''

_ELASTICITY_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N        = 64
    degree   = 2        # P2 avoids locking for most ν; increase if ν → 0.5
    ksp_type = "cg"
    pc_type  = "hypre"
    rtol     = 1e-8

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    msh  = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    gdim = msh.geometry.dim
    V    = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))
    x    = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    # case_spec['pde']['pde_params']:
    #   'E', 'nu'  (Young's modulus + Poisson ratio)  or  'lambda', 'mu' (Lamé)
    # case_spec['pde']['source_term']: list of 2 strings ['fx', 'fy'] or single string
    # case_spec['pde']['manufactured_solution']['u']: list ['ux','uy'] or string

    pde_cfg = case_spec["pde"]
    params  = pde_cfg.get("pde_params", {})
    E_val   = params.get("E",  None)
    nu_val  = params.get("nu", None)
    lam_val = params.get("lambda", None)
    mu_val  = params.get("mu",     None)

    # TODO: compute Lamé constants mu, lam from E,nu  or  use lam,mu directly
    mu  = ...   # TODO
    lam = ...   # TODO

    # TODO: define body force f as ufl.as_vector([...])
    f_body = ...   # TODO

    # TODO: define Dirichlet BC (displacement on ∂Ω)
    # Hint: for homogeneous  →  _bc_vec_zero(msh, V)
    bc = ...   # TODO

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # -∇·σ(u) = f,   σ = 2μ ε(u) + λ tr(ε(u)) I,   ε(u) = sym(∇u)
    # Weak form:  ∫ σ(u):ε(v) dx = ∫ f·v dx   ∀ v
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    def eps(w): return ufl.sym(ufl.grad(w))
    def sigma(w): return ...   # TODO: 2μ ε(w) + λ tr(ε(w)) I

    a = ...   # TODO: ∫ σ(u):ε(v) dx
    L = ...   # TODO: ∫ f·v dx

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    u_h = LinearProblem(a, L, bcs=[bc], petsc_options={
        "ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol,
        "pc_hypre_type": "boomeramg",
    }).solve()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["output"]["grid"]
    return {
        "u": _sample_vector_magnitude(u_h, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol},
    }
'''

_DARCY_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    # Use pressure formulation (standard Poisson): -∇·(κ ∇p) = f
    N        = 64
    degree   = 1
    ksp_type = "cg"
    pc_type  = "hypre"
    rtol     = 1e-8

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))
    x   = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    pde_cfg    = case_spec["pde"]
    kappa_spec = pde_cfg.get("coefficients", {}).get("kappa", {"type": "constant", "value": 1.0})
    kappa      = _kappa_from_spec(msh, kappa_spec)

    if "manufactured_solution" in pde_cfg and "u" in pde_cfg["manufactured_solution"]:
        f_h, bc = _manufactured_f_and_bc(msh, V, pde_cfg, kappa_spec)
    else:
        f_str = pde_cfg.get("source_term", "0.0")
        bc_str = pde_cfg.get("bc", {}).get("dirichlet", {}).get("value", "0.0")
        f_h = _f_from_str(msh, f_str)
        bc  = _bc_from_str(msh, V, bc_str)

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # Pressure form: -∇·(κ ∇p) = f
    u_t = ufl.TrialFunction(V)
    v   = ufl.TestFunction(V)

    a = ...   # TODO: identical to Poisson but for pressure p
    L = ...   # TODO

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    u_h = LinearProblem(a, L, bcs=[bc], petsc_options={
        "ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol,
    }).solve()

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["output"]["grid"]
    return {
        "u": _sample_scalar(u_h, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": rtol},
    }
'''

_RXNDIFF_SKELETON = '''
def solve(case_spec: dict) -> dict:
    # ── NUMERICAL PARAMETERS  (YOUR CHOICES) ─────────────────────────────
    N             = 64
    degree        = 1
    newton_rtol   = 1e-8
    newton_max_it = 25
    ksp_type      = "gmres"
    pc_type       = "ilu"

    # ── SETUP  (PROVIDED) ────────────────────────────────────────────────
    msh = _dmesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V   = fem.functionspace(msh, ("Lagrange", degree))
    x   = ufl.SpatialCoordinate(msh)

    # ── PDE DATA  (YOUR CODE) ────────────────────────────────────────────
    # case_spec['pde']['pde_params']:
    #   'epsilon': diffusion coefficient ε
    #   'reaction': reaction term string, e.g. 'u*(1-u)' or 'u**3'
    # If time parameters present, add time-stepping.
    pde_cfg = case_spec["pde"]
    params  = pde_cfg.get("pde_params", {})
    epsilon = float(params.get("epsilon", 1.0))
    rxn_str = str(params.get("reaction", "0"))

    # TODO: define f_h (source term) and bc
    f_h = ...   # TODO
    bc  = ...   # TODO

    # ── VARIATIONAL FORM  (YOUR CODE) ────────────────────────────────────
    # PDE:  -ε ∇²u + R(u) = f   in Ω
    # R(u) may be nonlinear (e.g. u³). Use Newton if so.
    # Nonlinear residual form:  F(u) = ε ∫ ∇u·∇v dx + ∫ R(u) v dx - ∫ f v dx = 0
    uh = fem.Function(V)    # iterate
    v  = ufl.TestFunction(V)

    # TODO: define reaction R_u as a UFL expression in uh
    # e.g. for rxn_str == 'u*(1-u)': R_u = uh * (1 - uh)
    R_u = ...   # TODO

    F = ...   # TODO: nonlinear residual
    J = ufl.derivative(F, uh)   # Jacobian (provided — automatic)

    # ── SOLVE  (PROVIDED) ────────────────────────────────────────────────
    from dolfinx.nls.petsc import NewtonSolver
    from dolfinx.fem.petsc import NonlinearProblem as _NLP
    problem = _NLP(F, uh, bcs=[bc], J=J)
    solver  = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.rtol   = newton_rtol
    solver.max_it = newton_max_it
    solver.solve(uh)

    # ── OUTPUT  (PROVIDED) ───────────────────────────────────────────────
    out = case_spec["output"]["grid"]
    return {
        "u": _sample_scalar(uh, out["nx"], out["ny"]),
        "solver_info": {"mesh_resolution": N, "element_degree": degree,
                        "ksp_type": ksp_type, "pc_type": pc_type, "rtol": newton_rtol},
    }
'''

# Map PDE type → skeleton
# Static skeletons (complex PDE types not yet fully templatized)
_SKELETONS_STATIC = {
    "darcy":             _DARCY_SKELETON,
}

# Dynamic skeleton builders (PDE data inlined at prompt-gen time)
_SKELETON_BUILDERS = {
    "poisson":               _build_poisson_skeleton,
    "heat":                  _build_heat_skeleton,
    "helmholtz":             _build_helmholtz_skeleton,
    "biharmonic":            _build_biharmonic_skeleton,
    "convection_diffusion":  _build_convdiff_skeleton,
    "reaction_diffusion":    _build_rxndiff_skeleton,
    "stokes":                _build_stokes_skeleton,
    "navier_stokes":         _build_ns_skeleton,
    "linear_elasticity":     _build_le_skeleton,
}

# All supported PDE types
_SUPPORTED_PDE_TYPES = set(_SKELETONS_STATIC) | set(_SKELETON_BUILDERS)


def _get_skeleton(pde_type: str, case: Dict) -> str:
    """Return the skeleton code for a given PDE type and case."""
    if pde_type in _SKELETON_BUILDERS:
        return _SKELETON_BUILDERS[pde_type](case)
    return _SKELETONS_STATIC.get(pde_type, _build_poisson_skeleton(case))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_template_prompt(case: Dict, oracle_info: Optional[Dict] = None) -> str:
    """
    Generate a template-guided prompt for the P2 (API Decoupling) experiment.

    The prompt contains:
      1. Problem statement (same PDE description as standard prompt)
      2. Provided utility functions (sampling, BC helpers, expression parsing)
      3. A PDE-specific skeleton with:
           - Provided sections: mesh/space setup, solver call, output
           - TODO sections:     variational form, BCs, PDE data extraction
         The model fills in ONLY the TODO sections.

    This isolates numerical reasoning ability from DOLFINx API knowledge.
    """
    case_id  = case["id"]
    pde_cfg  = case["oracle_config"]["pde"]
    pde_type = pde_cfg["type"]

    # ── Header: problem statement ──────────────────────────────────────
    from pdebench.core.prompt_builder import EQUATION_TEMPLATES, format_coefficient
    if pde_type == "convection_diffusion" and "time" in pde_cfg:
        eq_tmpl = EQUATION_TEMPLATES.get("convection_diffusion_transient",
                                         EQUATION_TEMPLATES["convection_diffusion"])
    else:
        eq_tmpl = EQUATION_TEMPLATES.get(pde_type, EQUATION_TEMPLATES["poisson"])

    prompt = f"""# Task: Solve {eq_tmpl['title']}  [Template-Guided Mode]

## Problem Description

{eq_tmpl['equation']}

{eq_tmpl['description']}

**Case ID:** {case_id}
"""

    # PDE math metadata
    math_type = case.get("pde_classification", {}).get("math_type", [])
    if math_type:
        prompt += f"\n**Math Type:** {', '.join(math_type)}\n"

    manufactured = pde_cfg.get("manufactured_solution", {})
    if "u" in manufactured:
        prompt += f"\n**Manufactured Solution:** u = {manufactured['u']}\n"
        if pde_type in ["stokes", "navier_stokes"]:
            prompt += f"**Manufactured Pressure:** p = {manufactured.get('p', 'N/A')}\n"
    else:
        src = pde_cfg.get("source_term")
        if src:
            prompt += f"\n**Source Term:** f = {src}\n"
        ic = pde_cfg.get("initial_condition")
        if ic:
            prompt += f"**Initial Condition:** u₀ = {ic}\n"

    coefficients = pde_cfg.get("coefficients", {})
    if coefficients:
        prompt += "\n**Coefficients:**\n"
        for name, coeff in coefficients.items():
            prompt += f"- κ = {format_coefficient(coeff)}\n"

    if pde_type in ["convection_diffusion"]:
        params = pde_cfg.get("pde_params", {})
        epsilon = params.get("epsilon", 0.01)
        beta = params.get("beta", [1.0, 1.0])
        beta_norm = (beta[0]**2 + beta[1]**2)**0.5 if isinstance(beta, list) else float(beta)
        peclet = beta_norm / epsilon if epsilon > 0 else float("inf")
        prompt += f"\n**Convection-Diffusion Parameters:**\n- ε = {epsilon}\n- β = {beta}\n- Péclet ≈ {peclet:.1f}\n"
        if peclet > 10:
            prompt += "⚠️  High Péclet — SUPG stabilization strongly recommended.\n"

    if pde_type in ["stokes", "navier_stokes"]:
        nu = pde_cfg.get("pde_params", {}).get("nu", 1.0)
        prompt += f"\n**Viscosity:** ν = {nu}\n"

    if pde_type == "helmholtz":
        k = pde_cfg.get("pde_params", {}).get("k",
            pde_cfg.get("pde_params", {}).get("wave_number", 10.0))
        prompt += f"\n**Wavenumber:** k = {k}\n"

    if pde_type == "linear_elasticity":
        params = pde_cfg.get("pde_params", {})
        E  = params.get("E")
        nu = params.get("nu")
        lam = params.get("lambda")
        mu  = params.get("mu")
        if E is not None and nu is not None:
            prompt += f"\n**Material Parameters:** E = {E}, ν = {nu}\n"
        elif lam is not None and mu is not None:
            prompt += f"\n**Material Parameters:** λ = {lam}, μ = {mu}\n"

    if "time" in pde_cfg:
        tc = pde_cfg["time"]
        prompt += (f"\n**Time Parameters:** t_end={tc.get('t_end',1.0)}, "
                   f"dt_suggested={tc.get('dt',0.01)}, "
                   f"scheme={tc.get('scheme','backward_euler')}\n")

    if oracle_info:
        eval_cfg = case.get("evaluation_config", {})
        at = eval_cfg.get("accuracy_tolerance", eval_cfg.get("tolerance", 1.2))
        tt = eval_cfg.get("time_tolerance",     eval_cfg.get("tolerance", 1.2))
        min_err = 1e-6
        target_err  = max(oracle_info.get("error", 0.0) * at, min_err)
        target_time = oracle_info.get("time", 0.0) * tt
        prompt += f"""
---

**Pass/Fail Criteria:**
- Accuracy: error ≤ {target_err:.2e}
- Time: wall_time_sec ≤ {target_time:.3f}s
"""

    # ── Experiment framing ─────────────────────────────────────────────
    prompt += """
---

## Your Task (Template-Guided Mode)

**All DOLFINx API boilerplate is provided below** — mesh creation, function space
setup, point-sampling output, and solver invocation are already written for you.

You only need to fill in the sections marked `# TODO`:
1. **Variational form** — the bilinear form `a(u,v)` and linear form `L(v)`
2. **Boundary conditions** — read BC values from `case_spec` using the provided helpers
3. **Numerical parameters** — choose mesh resolution, element degree, solver type, tolerance

This focuses purely on mathematical / numerical reasoning, not API syntax.

**Return the complete, runnable Python code (fill in all TODOs).**
"""

    # ── Utility code + skeleton ────────────────────────────────────────
    skeleton = _get_skeleton(pde_type, case)
    prompt += f"""
---

## Provided Utilities + Code Skeleton

```python
{_UTILITY_CODE}
{skeleton}
```
"""

    return prompt


def is_template_supported(pde_type: str) -> bool:
    """Return True if a template exists for this PDE type."""
    return pde_type in _SUPPORTED_PDE_TYPES
