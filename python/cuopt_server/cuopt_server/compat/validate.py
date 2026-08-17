# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Essential legacy LP/MILP JSON validation (no FastAPI / HTTP coupling)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np

from cuopt_server.utils.linear_programming.data_definition import LPData


class LegacyJsonValidationError(ValueError):
    """Raised when legacy JSON fails essential structural checks."""


def _is_empty(value) -> bool:
    return value is None or len(value) == 0


def _ok(msg: str = "ok") -> Tuple[bool, str]:
    return True, msg


def _bad(msg: str) -> Tuple[bool, str]:
    return False, msg


def validate_csr_matrix(csr_data) -> Tuple[bool, str]:
    if _is_empty(csr_data.indices) or _is_empty(csr_data.values):
        return _bad("CSR indices and values must be non-empty")
    if _is_empty(csr_data.offsets):
        return _bad("CSR offsets must be non-empty")
    if np.min(csr_data.indices) < 0:
        return _bad("indices values must be greater than or equal to 0")
    if np.min(csr_data.offsets) < 0:
        return _bad("offset values must be greater than or equal to 0")
    if len(csr_data.indices) != len(csr_data.values):
        return _bad("Length of values array must be equal to indices array")
    return _ok("Valid CSR Matrix")


def validate_constraint_bounds(constraint_bounds) -> Tuple[bool, str]:
    if constraint_bounds is None:
        return _bad("constraint_bounds is required")
    if _is_empty(constraint_bounds.upper_bounds) or _is_empty(
        constraint_bounds.lower_bounds
    ):
        if _is_empty(constraint_bounds.types):
            return _bad(
                "Either Row types or upper and lower bounds must be provided"
            )
        if any(
            row_type not in ["E", "G", "L"]
            for row_type in constraint_bounds.types
        ):
            return _bad("Row types must be E, L or G")
        return _ok("Valid constraint bounds")
    if len(constraint_bounds.upper_bounds) != len(
        constraint_bounds.lower_bounds
    ):
        return _bad(
            "Size of constraint upper bounds must be same as constaint lower bounds"
        )
    return _ok("Valid constraint bounds")


def validate_variable_bounds(LP_data) -> Tuple[bool, str]:
    variable_bounds = LP_data.variable_bounds
    coeff = LP_data.objective_data.coefficients
    if variable_bounds is None:
        return _ok("Valid variable bounds")
    if (
        variable_bounds.upper_bounds is not None
        and variable_bounds.lower_bounds is not None
    ):
        if len(variable_bounds.upper_bounds) != len(
            variable_bounds.lower_bounds
        ):
            return _bad(
                "Size of variable upper bounds must be same as variable lower bounds"
            )
    elif variable_bounds.upper_bounds is not None:
        if len(variable_bounds.upper_bounds) != len(coeff):
            return _bad(
                "Size of variable upper bounds must be same as size of objective coefficients"
            )
    elif variable_bounds.lower_bounds is not None:
        if len(variable_bounds.lower_bounds) != len(coeff):
            return _bad(
                "Size of variable lower bounds must be same as size of objective coefficients"
            )
    return _ok("Valid variable bounds")


def validate_initial_solution(LP_data) -> Tuple[bool, str]:
    initial_solution = LP_data.initial_solution
    if initial_solution is None:
        return _ok("Valid initial solution")
    objective_data = LP_data.objective_data
    constraint_bounds = LP_data.constraint_bounds
    if initial_solution.primal is not None:
        if len(initial_solution.primal) != len(objective_data.coefficients):
            return _bad(
                "Size of initial solution must be same as size of objective coefficients"
            )
    if initial_solution.dual is not None:
        n_constraints = (
            len(constraint_bounds.bounds)
            if constraint_bounds.bounds is not None
            else 0
        )
        if constraint_bounds.upper_bounds is not None and len(
            constraint_bounds.upper_bounds
        ):
            n_constraints = len(constraint_bounds.upper_bounds)
        if len(initial_solution.dual) != n_constraints:
            return _bad(
                "Size of initial dual solution must be same as size of constraint bounds"
            )
    return _ok("Valid initial solution")


def validate_lp_data(lp_data: LPData) -> None:
    """Run essential checks on a normalized :class:`LPData` instance.

    Raises
    ------
    LegacyJsonValidationError
        On the first failed check.
    """
    checks = (
        validate_csr_matrix(lp_data.csr_constraint_matrix),
        validate_constraint_bounds(lp_data.constraint_bounds),
        validate_variable_bounds(lp_data),
        validate_initial_solution(lp_data),
    )
    for ok, msg in checks:
        if not ok:
            raise LegacyJsonValidationError(msg)


def validate_lp_dict(
    data: Mapping[str, Any],
    *,
    parse: bool = True,
) -> Optional[LPData]:
    """Validate a legacy LP/MILP JSON object.

    Parameters
    ----------
    data : mapping
        Dict shaped like the legacy OpenAPI ``LPData`` schema.
    parse : bool
        If True (default), also run Pydantic ``LPData.parse_obj`` for
        shape/type checks before semantic checks. If False, ``data`` must
        already be an :class:`LPData` (or this raises).

    Returns
    -------
    LPData or None
        The parsed :class:`LPData` when ``parse`` is True, else None.

    Raises
    ------
    LegacyJsonValidationError
        On failure.
    Exception
        From Pydantic if the payload shape is wrong.
    """
    if isinstance(data, LPData):
        validate_lp_data(data)
        return data

    if not parse:
        raise TypeError(
            "validate_lp_dict(parse=False) requires an LPData instance"
        )

    try:
        lp_data = LPData.parse_obj(dict(data))
    except Exception as exc:  # pydantic ValidationError / TypeError
        raise LegacyJsonValidationError(str(exc)) from exc

    # Semantic checks need arrays; normalize lightly for inf/ninf lists.
    from cuopt_server.compat.normalize import normalize_lp_data

    normalize_lp_data(lp_data)
    validate_lp_data(lp_data)
    return lp_data
