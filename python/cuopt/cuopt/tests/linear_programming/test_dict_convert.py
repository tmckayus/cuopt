# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dict_to_datamodel / dict_from_solution."""

import copy
import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np

from cuopt.linear_programming.dict_convert import (
    dict_from_solution,
    dict_to_datamodel,
)
from cuopt.linear_programming.solution.solution import LPTerminationStatus

LP_EXAMPLE = {
    "csr_constraint_matrix": {
        "offsets": [0, 2, 4],
        "indices": [0, 1, 0, 1],
        "values": [3.0, 4.0, 2.7, 10.1],
    },
    "constraint_bounds": {
        "upper_bounds": [5.4, 4.9],
        "lower_bounds": ["ninf", "ninf"],
    },
    "objective_data": {
        "coefficients": [0.2, 0.1],
        "scalability_factor": 1.0,
        "offset": 0.0,
    },
    "variable_bounds": {
        "upper_bounds": ["inf", "inf"],
        "lower_bounds": [0.0, 0.0],
    },
    "maximize": False,
    "variable_names": ["x", "y"],
    "solver_config": {"tolerances": {"optimality": 0.0001}, "time_limit": 5},
}


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

    def get_solved_by(self):
        return SimpleNamespace(name="PDLP")

    def get_vars(self):
        return {"x": 1.0, "y": 2.0}

    def get_lp_stats(self):
        return {"nb_iterations": 1}

    def get_reduced_cost(self):
        return np.array([0.0, 0.0])

    def get_milp_stats(self):
        return None

    def get_pdlp_warm_start_data(self):
        return SimpleNamespace(current_primal_solution=[1.0])


def test_dict_to_datamodel_from_mapping():
    dm, settings = dict_to_datamodel(copy.deepcopy(LP_EXAMPLE))
    assert len(dm.get_objective_coefficients()) == 2
    assert np.allclose(dm.get_objective_coefficients(), [0.2, 0.1])
    assert np.isinf(dm.get_variable_upper_bounds()).all()
    assert np.isneginf(dm.get_constraint_lower_bounds()).all()
    names = dm.get_variable_names()
    assert list(names) == ["x", "y"]
    assert settings.get_parameter("time_limit") == 5
    assert settings.get_parameter("absolute_primal_tolerance") == 0.0001


def test_dict_to_datamodel_from_json_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as fh:
        json.dump(LP_EXAMPLE, fh)
        path = fh.name
    try:
        dm, _settings = dict_to_datamodel(path)
        assert len(dm.get_objective_coefficients()) == 2
    finally:
        os.unlink(path)


def test_dict_from_solution_envelope():
    body = dict_from_solution(_FakeSol(), req_id="abc", total_solve_time=0.02)
    assert body["reqId"] == "abc"
    solver = body["response"]["solver_response"]
    assert solver["status"] == "Optimal"
    assert body["response"]["total_solve_time"] == 0.02
    sol = solver["solution"]
    assert sol["primal_objective"] == 3.0
    assert sol["primal_solution"] == [1.0, 2.0]
    assert sol["vars"] == {"x": 1.0, "y": 2.0}
    assert sol["solved_by"] == "PDLP"
    assert "pdlpwarmstart_data" not in sol
    assert sol["lp_statistics"] == {"nb_iterations": 1}
    assert sol["milp_statistics"] == {}
    assert body["notes"] == ["Optimal"]
