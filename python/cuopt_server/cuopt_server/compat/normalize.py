# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize legacy LP/MILP JSON lists into typed numpy arrays."""

from __future__ import annotations

import copy
from typing import Any, Mapping, MutableMapping

import numpy as np

from cuopt_server.utils.linear_programming.data_definition import LPData

_TMAP = {
    "csr_constraint_matrix": {
        "offsets": (True, np.int32),
        "indices": (True, np.int32),
        "values": (True, np.float64),
    },
    "constraint_bounds": {
        "bounds": (True, np.float64),
        "upper_bounds": (True, np.float64),
        "lower_bounds": (True, np.float64),
        "types": (True, "U1"),
    },
    "initial_solution": {
        "primal": (True, np.float64),
        "dual": (True, np.float64),
    },
    "objective_data": {
        "coefficients": (True, np.float64),
    },
    "variable_bounds": {
        "upper_bounds": (True, np.float64),
        "lower_bounds": (True, np.float64),
    },
    "variable_types": (True, "U1"),
}


def _modify(value, dtype=None):
    if isinstance(value, list):
        if "inf" in value or "ninf" in value:
            value = [
                np.inf if x == "inf" else -np.inf if x == "ninf" else x
                for x in value
            ]
        if dtype is None:
            return np.array(value)
        return np.array(value, dtype)
    return value


def _apply_dict(
    data: MutableMapping[str, Any], tmap: Mapping[str, Any]
) -> None:
    for key, value in list(data.items()):
        if isinstance(value, dict) and key in tmap:
            _apply_dict(value, tmap[key])
        elif key in tmap and tmap[key][0]:
            data[key] = _modify(value, tmap[key][1])


def normalize_lp_dict(data: Mapping[str, Any]) -> dict:
    """Return a deep-copied dict with list fields converted to numpy arrays."""
    out = copy.deepcopy(dict(data))
    _apply_dict(out, _TMAP)
    return out


def normalize_lp_data(lp_data: LPData) -> LPData:
    """In-place normalize of an :class:`LPData` instance (lists → numpy)."""
    csr = lp_data.csr_constraint_matrix
    csr.indices = _modify(csr.indices, np.int32)
    csr.offsets = _modify(csr.offsets, np.int32)
    csr.values = _modify(csr.values, np.float64)

    cb = lp_data.constraint_bounds
    if cb is not None:
        cb.bounds = _modify(cb.bounds, np.float64)
        cb.upper_bounds = _modify(cb.upper_bounds, np.float64)
        cb.lower_bounds = _modify(cb.lower_bounds, np.float64)
        cb.types = _modify(cb.types)

    if lp_data.initial_solution is not None:
        lp_data.initial_solution.primal = _modify(
            lp_data.initial_solution.primal, np.float64
        )
        lp_data.initial_solution.dual = _modify(
            lp_data.initial_solution.dual, np.float64
        )

    if lp_data.objective_data is not None:
        lp_data.objective_data.coefficients = _modify(
            lp_data.objective_data.coefficients, np.float64
        )

    if lp_data.variable_bounds is not None:
        lp_data.variable_bounds.upper_bounds = _modify(
            lp_data.variable_bounds.upper_bounds, np.float64
        )
        lp_data.variable_bounds.lower_bounds = _modify(
            lp_data.variable_bounds.lower_bounds, np.float64
        )

    if lp_data.variable_types is not None:
        lp_data.variable_types = _modify(lp_data.variable_types)

    return lp_data
