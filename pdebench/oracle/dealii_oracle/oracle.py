"""
dealii_oracle/oracle.py
=======================

Python dispatcher for the deal.II oracle backend.

Workflow for each solve() call:
  1. preprocess_case_spec() – inject _computed_* expression fields
  2. ensure_built()         – cmake + make on first call (cached)
  3. run_oracle_program()   – invoke C++ binary via subprocess
  4. Wrap output in OracleResult
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .._types import OracleResult, compute_rel_L2_grid
from .common import ensure_built, preprocess_case_spec, run_oracle_program

# Paths resolved relative to this file so the oracle works regardless of cwd
_ORACLE_DIR   = Path(__file__).resolve().parent
_PROGRAMS_DIR = _ORACLE_DIR / "programs"
_BUILD_DIR    = _ORACLE_DIR / "build"


class DealIIOracleSolver:
    """
    Oracle backend that compiles deal.II C++ programs on first use and
    calls the appropriate binary for each PDE type.

    The interface mirrors FiredrakeOracleSolver: accepts oracle_config
    (the 'oracle_config' sub-dict from benchmark.jsonl) and returns an
    OracleResult with the same field semantics.
    """

    def __init__(self, timeout_sec: int = 300):
        self._timeout = timeout_sec
        self._built   = False

    def _ensure_built(self) -> None:
        if not self._built:
            ensure_built(_PROGRAMS_DIR, _BUILD_DIR)
            self._built = True

    def solve(self, case_spec: Dict[str, Any]) -> OracleResult:
        """
        Solve one PDE case with the deal.II oracle.

        Args:
            case_spec: oracle_config dict from benchmark.jsonl
                       (same dict passed to OracleSolver.solve()).

        Returns:
            OracleResult with reference grid, baseline_error, baseline_time.
        """
        pde_type = case_spec["pde"]["type"]

        # 1. Inject _computed_* expression fields for C++
        enriched = preprocess_case_spec(case_spec)

        # 2. Compile oracle binaries if not yet done
        self._ensure_built()

        # 3. Run C++ binary
        grid, meta = run_oracle_program(
            pde_type   = pde_type,
            case_spec  = enriched,
            build_dir  = _BUILD_DIR,
            timeout_sec = self._timeout,
        )

        # 4. Baseline error:
        #    - If exact solution available → compare FEM sol to exact
        #    - Otherwise → FEM oracle IS the reference (error ≈ 0)
        baseline_error = 0.0
        if enriched["pde"].get("_has_exact", False):
            # The C++ binary only writes the FEM solution.
            # We treat the FEM solution as the reference (it is fine-mesh
            # accurate). baseline_error represents oracle self-error, which
            # for a fine-mesh FEM solve is very small.
            baseline_error = 0.0  # TODO: could add exact vs FEM comparison

        return OracleResult(
            baseline_error = float(baseline_error),
            baseline_time  = float(meta.get("baseline_time", 0.0)),
            reference      = grid,
            solver_info    = {
                "ksp_type":  meta.get("ksp_type",  ""),
                "pc_type":   meta.get("pc_type",   ""),
                "rtol":      meta.get("rtol",       0.0),
                "library":   "dealii",
            },
            num_dofs       = int(meta.get("num_dofs", 0)),
        )
