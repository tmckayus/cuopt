# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert legacy REST LP/MILP JSON dicts to DataModel / SolverSettings and back.

These helpers let Python callers load the same JSON (or msgpack-decoded dict)
that ``cuopt_server`` accepts, submit it with the gRPC client, and map a
:class:`~cuopt.linear_programming.solution.Solution` to the legacy GET
``/cuopt/solution/{id}`` envelope.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from cuopt.linear_programming.data_model import DataModel
from cuopt.linear_programming.solution.solution import (
    LPTerminationStatus,
    MILPTerminationStatus,
)
from cuopt.linear_programming.solver_settings import (
    SolverSettings,
    solver_params,
)

JsonInput = Union[str, Mapping[str, Any]]

_SOLUTION_STATUSES = (
    LPTerminationStatus.Optimal,
    LPTerminationStatus.IterationLimit,
    LPTerminationStatus.TimeLimit,
    MILPTerminationStatus.Optimal,
    MILPTerminationStatus.FeasibleFound,
)


def _load_mapping(data: JsonInput) -> Mapping[str, Any]:
    if isinstance(data, Mapping):
        return data
    if isinstance(data, str):
        if os.path.isfile(data):
            with open(data, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(data)
    raise TypeError(
        f"Unsupported input type {type(data)!r}; expected dict, JSON string, "
        "or path"
    )


def _as_array(value, dtype=None):
    if value is None:
        return None
    if isinstance(value, list):
        if any(x in ("inf", "ninf") for x in value):
            value = [
                np.inf if x == "inf" else -np.inf if x == "ninf" else x
                for x in value
            ]
        return np.array(value) if dtype is None else np.array(value, dtype)
    return value


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    return value if isinstance(value, Mapping) else {}


def _coerce_settings_for_grpc(
    solver_settings: SolverSettings,
) -> SolverSettings:
    """Bool solver flags become 0/1 so the gRPC path accepts them."""
    for name, value in list(solver_settings.settings_dict.items()):
        if isinstance(value, bool):
            solver_settings.settings_dict[name] = int(value)
    return solver_settings


def _fill_data_model(payload: Mapping[str, Any]) -> DataModel:
    data_model = DataModel()
    csr = _section(payload, "csr_constraint_matrix")
    if not csr:
        raise ValueError("legacy LP dict is missing csr_constraint_matrix")
    data_model.set_csr_constraint_matrix(
        _as_array(csr.get("values"), np.float64),
        _as_array(csr.get("indices"), np.int32),
        _as_array(csr.get("offsets"), np.int32),
    )

    constraint_bounds = _section(payload, "constraint_bounds")
    bounds = _as_array(constraint_bounds.get("bounds"), np.float64)
    if bounds is not None:
        data_model.set_constraint_bounds(bounds)
    types = _as_array(constraint_bounds.get("types"))
    if types is not None and len(types):
        data_model.set_row_types(types)
    upper = _as_array(constraint_bounds.get("upper_bounds"), np.float64)
    if upper is not None and len(upper):
        data_model.set_constraint_upper_bounds(upper)
    lower = _as_array(constraint_bounds.get("lower_bounds"), np.float64)
    if lower is not None and len(lower):
        data_model.set_constraint_lower_bounds(lower)

    objective = _section(payload, "objective_data")
    coefficients = _as_array(objective.get("coefficients"), np.float64)
    if coefficients is not None:
        data_model.set_objective_coefficients(coefficients)
    if objective.get("scalability_factor") is not None:
        data_model.set_objective_scaling_factor(
            objective["scalability_factor"]
        )
    if objective.get("offset") is not None:
        data_model.set_objective_offset(objective["offset"])

    variable_bounds = _section(payload, "variable_bounds")
    v_upper = _as_array(variable_bounds.get("upper_bounds"), np.float64)
    if v_upper is not None:
        data_model.set_variable_upper_bounds(v_upper)
    v_lower = _as_array(variable_bounds.get("lower_bounds"), np.float64)
    if v_lower is not None:
        data_model.set_variable_lower_bounds(v_lower)

    initial = _section(payload, "initial_solution")
    primal = _as_array(initial.get("primal"), np.float64)
    if primal is not None:
        data_model.set_initial_primal_solution(primal)
    dual = _as_array(initial.get("dual"), np.float64)
    if dual is not None:
        data_model.set_initial_dual_solution(dual)

    if payload.get("maximize") is not None:
        data_model.set_maximize(payload["maximize"])
    if payload.get("variable_types") is not None:
        data_model.set_variable_types(_as_array(payload["variable_types"]))
    if payload.get("variable_names") is not None:
        data_model.set_variable_names(payload["variable_names"])
    return data_model


def _fill_solver_settings(
    payload: Mapping[str, Any],
    warmstart_data=None,
) -> Tuple[SolverSettings, List[str]]:
    warnings: List[str] = []
    solver_settings = SolverSettings()
    solver_config = _section(payload, "solver_config")
    if not solver_config and warmstart_data is None:
        return _coerce_settings_for_grpc(solver_settings), warnings

    tolerances = _section(solver_config, "tolerances")
    if tolerances.get("optimality") is not None:
        solver_settings.set_optimality_tolerance(tolerances["optimality"])
    for param in solver_params:
        if param.endswith("tolerance") or param in tolerances:
            param_value = tolerances.get(param)
            if param_value is None:
                param_value = solver_config.get(param)
        else:
            param_value = solver_config.get(param)
        if param_value is not None and param_value != "":
            if isinstance(param_value, bool):
                param_value = int(param_value)
            solver_settings.set_parameter(param, param_value)

    if solver_config.get("time_limit") is not None:
        solver_settings.set_parameter(
            "time_limit", solver_config["time_limit"]
        )
    if solver_config.get("iteration_limit") is not None:
        solver_settings.set_parameter(
            "iteration_limit", solver_config["iteration_limit"]
        )
    if warmstart_data is not None:
        solver_settings.set_pdlp_warm_start_data(warmstart_data)
    if solver_config.get("user_problem_file"):
        warnings.append("solver config user_problem_file ignored")
    if solver_config.get("solution_file"):
        warnings.append("solver config solution_file ignored")
    return _coerce_settings_for_grpc(solver_settings), warnings


def dict_to_datamodel(
    data: JsonInput,
    *,
    warmstart_data=None,
    return_warnings: bool = False,
):
    """Convert a legacy REST LP/MILP dict to ``(DataModel, SolverSettings)``.

    ``data`` may be a mapping, a JSON string, or a path to a JSON file.
    Lists are converted to numpy arrays; the strings ``"inf"`` / ``"ninf"``
    become IEEE infinities.

    Parameters
    ----------
    data : dict or str
        Legacy cuOpt REST payload (the body of POST ``/cuopt/request``).
    warmstart_data : optional
        PDLP warm-start blob passed to
        :meth:`SolverSettings.set_pdlp_warm_start_data`.
    return_warnings : bool, default False
        If True, also return ignored-field warnings.

    Returns
    -------
    data_model, solver_settings
        Ready for ``cuopt.linear_programming.Solve`` or
        ``cuopt.grpc.linear_programming.Client.submit``.
    """
    payload = _load_mapping(data)
    data_model = _fill_data_model(payload)
    solver_settings, warnings = _fill_solver_settings(
        payload, warmstart_data=warmstart_data
    )
    if return_warnings:
        return data_model, solver_settings, warnings
    return data_model, solver_settings


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
    if hasattr(sol, "get_solved_by_pdlp"):
        solution["solved_by_pdlp"] = _call(sol.get_solved_by_pdlp)
        return
    solved_by = _call(getattr(sol, "get_solved_by", lambda: None))
    solution["solved_by"] = getattr(solved_by, "name", solved_by)


def _legacy_envelope(
    response: Mapping[str, Any],
    *,
    warnings: Optional[List[str]] = None,
    notes: Optional[List[str]] = None,
    req_id: str = "",
    total_solve_time: float = 0,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"response": dict(response)}
    if total_solve_time:
        body["response"]["total_solve_time"] = total_solve_time
    if req_id:
        body["reqId"] = req_id
    if warnings:
        body["warnings"] = warnings
    if notes:
        body["notes"] = notes
    return body


def dict_from_solution(
    sol,
    *,
    req_id: str = "",
    warnings: Optional[List[str]] = None,
    total_solve_time: float = 0,
) -> Dict[str, Any]:
    """Map a cuOpt ``Solution`` to the legacy GET ``/cuopt/solution/{id}`` body.

    Matches the envelope produced by ``cuopt_server`` (``reqId``,
    ``response.solver_response``, ``vars``, list-encoded arrays). The PDLP
    warm-start blob is omitted; legacy serves it from a separate endpoint.

    Parameters
    ----------
    sol : cuopt.linear_programming.solution.Solution
        Result from a local ``Solve`` or ``Client.result``.
    req_id : str, optional
        Copied to the ``reqId`` field when non-empty.
    warnings : list of str, optional
    total_solve_time : float, optional
        Wall time in seconds; omitted from the envelope when 0.
    """
    solution: Dict[str, Any] = {}
    status = sol.get_termination_status()
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

    solver_res = {"status": status.name, "solution": solution}
    return _legacy_envelope(
        {"solver_response": solver_res},
        warnings=warnings,
        notes=notes,
        req_id=req_id,
        total_solve_time=total_solve_time,
    )
