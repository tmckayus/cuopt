# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for legacy HTTP solution envelope mapping."""

from types import SimpleNamespace

import numpy as np

from cuopt.linear_programming.solution.solution import LPTerminationStatus
from cuopt_server.compat.response import (
    create_solver_response,
    solution_to_legacy_response,
)


class _FakeCategory:
    name = "LP"


class _FakeSol:
    def get_termination_status(self):
        return LPTerminationStatus.Optimal

    def get_termination_reason(self):
        return "Optimal"

    def get_problem_category(self):
        return _FakeCategory()

    def get_primal_solution(self):
        return np.array([1.0, 2.0])

    def get_dual_solution(self):
        return np.array([0.5])

    def get_primal_objective(self):
        return 3.0

    def get_dual_objective(self):
        return 3.0

    def get_solve_time(self):
        return 0.01

    def get_solved_by_pdlp(self):
        return True

    def get_vars(self):
        return {"x": 1.0, "y": 2.0}

    def get_lp_stats(self):
        return {"nb_iterations": 1}

    def get_reduced_cost(self):
        return np.array([0.0, 0.0])

    def get_milp_stats(self):
        return None

    def get_pdlp_warm_start_data(self):
        return SimpleNamespace(
            current_primal_solution=[1.0],
            current_dual_solution=[0.5],
            initial_primal_average=[1.0],
            initial_dual_average=[0.5],
            current_ATY=[0.0],
            sum_primal_solutions=[1.0],
            sum_dual_solutions=[0.5],
            last_restart_duality_gap_primal_solution=[1.0],
            last_restart_duality_gap_dual_solution=[0.5],
            initial_primal_weight=1.0,
            initial_step_size=0.1,
            total_pdlp_iterations=1,
            total_pdhg_iterations=1,
            last_candidate_kkt_score=0.0,
            last_restart_kkt_score=0.0,
            sum_solution_weight=1.0,
            iterations_since_last_restart=1,
        )


def test_create_solver_response_optimal():
    res, notes = create_solver_response(_FakeSol())
    assert res["status"] == "Optimal"
    assert res["solution"]["primal_objective"] == 3.0
    assert res["solution"]["primal_solution"] == [1.0, 2.0]
    assert res["solution"]["dual_solution"] == [0.5]
    assert "pdlpwarmstart_data" in res["solution"]
    assert notes == ["Optimal"]


def test_legacy_envelope_strips_warmstart():
    body = solution_to_legacy_response(
        _FakeSol(), req_id="abc", total_solve_time=0.02
    )
    assert body["reqId"] == "abc"
    assert "warnings" not in body or body.get("warnings") == []
    solver = body["response"]["solver_response"]
    assert solver["status"] == "Optimal"
    assert body["response"]["total_solve_time"] == 0.02
    sol = solver["solution"]
    assert sol["primal_objective"] == 3.0
    assert sol["vars"] == {"x": 1.0, "y": 2.0}
    assert "pdlpwarmstart_data" not in sol
    assert "problem_category" in sol
    assert "lp_statistics" in sol
    assert "milp_statistics" in sol
