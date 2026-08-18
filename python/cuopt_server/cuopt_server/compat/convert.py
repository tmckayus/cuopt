# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert legacy LP/MILP JSON to DataModel + SolverSettings (no validation)."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Tuple, Union

from cuopt.linear_programming.data_model import DataModel
from cuopt.linear_programming.solver_settings import SolverSettings

from cuopt_server.compat.normalize import normalize_lp_data
from cuopt_server.utils.linear_programming.data_definition import LPData
from cuopt_server.utils.linear_programming.solver import (
    create_data_model,
    create_solver,
)

JsonInput = Union[str, Mapping[str, Any], LPData]


def _load_mapping(data: JsonInput) -> Mapping[str, Any] | LPData:
    if isinstance(data, LPData):
        return data
    if isinstance(data, Mapping):
        return data
    if isinstance(data, str):
        if os.path.isfile(data):
            with open(data, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(data)
    raise TypeError(
        f"Unsupported input type {type(data)!r}; expected dict, LPData, "
        "JSON string, or path"
    )


def parse_lp_data(data: JsonInput) -> LPData:
    """Parse legacy JSON into :class:`LPData` and normalize arrays.

    Does **not** run semantic validation. Call
    :func:`cuopt_server.compat.validate.validate_lp_data` separately if needed.
    """
    loaded = _load_mapping(data)
    if isinstance(loaded, LPData):
        lp_data = loaded
    else:
        payload = dict(loaded)
        # Historical helper dropped variable_names when loading from file;
        # keep names when present so clients can round-trip.
        lp_data = LPData.parse_obj(payload)
    return normalize_lp_data(lp_data)


def _coerce_settings_for_grpc(
    solver_settings: SolverSettings,
) -> SolverSettings:
    """Grpc ``prepare_solver_settings`` rejects Python bools for int params.

    Legacy ``SolverConfig`` Pydantic defaults are bools (e.g. ``mip_scaling=True``).
    Local string-based setters tolerate ``str(True)``; the gRPC path does not.
    """
    for name, value in list(solver_settings.settings_dict.items()):
        if isinstance(value, bool):
            solver_settings.settings_dict[name] = int(value)
    return solver_settings


def lp_data_to_datamodel(
    lp_data: LPData,
    *,
    warmstart_data=None,
    return_warnings: bool = False,
):
    """Convert a normalized :class:`LPData` to ``(DataModel, SolverSettings)``.

    With ``return_warnings``, also returns the warning list the legacy server
    collects while building the data model and solver settings.
    """
    dm_warnings, data_model = create_data_model(lp_data)
    cs_warnings, solver_settings = create_solver(lp_data, warmstart_data)
    settings = _coerce_settings_for_grpc(solver_settings)
    if return_warnings:
        return data_model, settings, [*dm_warnings, *cs_warnings]
    return data_model, settings


def json_to_datamodel(
    data: JsonInput,
    *,
    warmstart_data=None,
    validate: bool = False,
) -> Tuple[DataModel, SolverSettings]:
    """Convert legacy LP/MILP JSON to ``(DataModel, SolverSettings)``.

    Validation is off by default for trusted client-side producers. The HTTP
    shim should pass ``validate=True`` (or call
    :func:`~cuopt_server.compat.validate.validate_lp_data` separately).
    """
    lp_data = parse_lp_data(data)
    if validate:
        from cuopt_server.compat.validate import validate_lp_data

        validate_lp_data(lp_data)
    return lp_data_to_datamodel(lp_data, warmstart_data=warmstart_data)
