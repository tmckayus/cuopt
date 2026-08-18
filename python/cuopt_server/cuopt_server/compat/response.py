# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map a cuOpt ``Solution`` to the legacy HTTP JSON envelope.

Mirrors ``create_solution`` / ``make_response`` used by the cuopt_server
LP path so GET ``/cuopt/solution/{id}`` from the HTTP→gRPC shim matches
the legacy REST shape.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cuopt.linear_programming.solution.solution import (
    LPTerminationStatus,
    MILPTerminationStatus,
)

from cuopt_server.utils.solver import make_response

_SOLUTION_STATUSES = (
    LPTerminationStatus.Optimal,
    LPTerminationStatus.IterationLimit,
    LPTerminationStatus.TimeLimit,
    MILPTerminationStatus.Optimal,
    MILPTerminationStatus.FeasibleFound,
)


def _call(attr):
    try:
        return attr()
    except AttributeError:
        return None


def _tolist(value):
    if value is None:
        return None
    return value.tolist() if hasattr(value, "tolist") else value


def _extract_pdlpwarmstart_data(data):
    if data is None:
        return None
    return {
        "current_primal_solution": data.current_primal_solution,
        "current_dual_solution": data.current_dual_solution,
        "initial_primal_average": data.initial_primal_average,
        "initial_dual_average": data.initial_dual_average,
        "current_ATY": data.current_ATY,
        "sum_primal_solutions": data.sum_primal_solutions,
        "sum_dual_solutions": data.sum_dual_solutions,
        "last_restart_duality_gap_primal_solution": (
            data.last_restart_duality_gap_primal_solution
        ),
        "last_restart_duality_gap_dual_solution": (
            data.last_restart_duality_gap_dual_solution
        ),
        "initial_primal_weight": data.initial_primal_weight,
        "initial_step_size": data.initial_step_size,
        "total_pdlp_iterations": data.total_pdlp_iterations,
        "total_pdhg_iterations": data.total_pdhg_iterations,
        "last_candidate_kkt_score": data.last_candidate_kkt_score,
        "last_restart_kkt_score": data.last_restart_kkt_score,
        "sum_solution_weight": data.sum_solution_weight,
        "iterations_since_last_restart": data.iterations_since_last_restart,
    }


def _set_solved_by(solution: Dict[str, Any], sol) -> None:
    """Emit whichever field name the installed cuopt/server pair uses.

    Older trees expose ``get_solved_by_pdlp`` (bool ``solved_by_pdlp``);
    newer ones expose ``get_solved_by`` (enum ``solved_by``).
    """
    if hasattr(sol, "get_solved_by_pdlp"):
        solution["solved_by_pdlp"] = _call(sol.get_solved_by_pdlp)
        return
    solved_by = _call(getattr(sol, "get_solved_by", lambda: None))
    solution["solved_by"] = getattr(solved_by, "name", solved_by)


def create_solver_response(sol) -> Dict[str, Any]:
    """Build ``{status, solution}`` matching legacy ``create_solution``."""
    solution: Dict[str, Any] = {}
    status = sol.get_termination_status()
    # Legacy always records termination_reason in notes (may be empty).
    notes: List[str] = [sol.get_termination_reason()]

    if status in _SOLUTION_STATUSES:
        solution["problem_category"] = sol.get_problem_category().name
        solution["primal_solution"] = _tolist(_call(sol.get_primal_solution))
        solution["dual_solution"] = _tolist(_call(sol.get_dual_solution))
        solution["primal_objective"] = _call(sol.get_primal_objective)
        solution["dual_objective"] = _call(sol.get_dual_objective)
        solution["solver_time"] = sol.get_solve_time()
        _set_solved_by(solution, sol)
        solution["vars"] = sol.get_vars()
        lp_stats = _call(sol.get_lp_stats)
        solution["lp_statistics"] = {} if lp_stats is None else lp_stats
        solution["reduced_cost"] = _tolist(_call(sol.get_reduced_cost))
        milp_stats = _call(sol.get_milp_stats)
        solution["milp_statistics"] = {} if milp_stats is None else milp_stats
        # Built then stripped before return (legacy SolverBinaryResponse).
        solution["pdlpwarmstart_data"] = _extract_pdlpwarmstart_data(
            _call(sol.get_pdlp_warm_start_data)
        )

    return {"status": status.name, "solution": solution}, notes


def solution_to_legacy_response(
    sol,
    *,
    req_id: str = "",
    warnings: Optional[List[str]] = None,
    total_solve_time: float = 0,
) -> Dict[str, Any]:
    """Full legacy GET ``/cuopt/solution/{id}`` body for a completed solve."""
    solver_res, notes = create_solver_response(sol)
    # Warm-start blob is served from a separate endpoint on legacy; strip it.
    sol_body = solver_res.get("solution") or {}
    sol_body.pop("pdlpwarmstart_data", None)

    envelope = make_response(
        {"solver_response": solver_res},
        warnings=warnings or [],
        notes=notes,
        reqId=req_id,
        total_solve_time=total_solve_time,
    )
    return envelope
