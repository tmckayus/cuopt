# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy JSON compatibility helpers for LP/MILP.

Three separable pieces:

1. :mod:`cuopt_server.compat.validate` — essential structural checks
2. :mod:`cuopt_server.compat.convert` — JSON/dict → DataModel + SolverSettings
3. :mod:`cuopt_server.compat.shim` — thin HTTP→gRPC FastAPI pass-through

Client-side Python callers can use :func:`json_to_datamodel` without
validation; the HTTP shim validates by default.
"""

from cuopt_server.compat.codec import decode_request_body, encode_payload
from cuopt_server.compat.convert import (
    json_to_datamodel,
    lp_data_to_datamodel,
    parse_lp_data,
)
from cuopt_server.compat.normalize import normalize_lp_dict
from cuopt_server.compat.validate import (
    LegacyJsonValidationError,
    validate_lp_data,
    validate_lp_dict,
)

__all__ = [
    "LegacyJsonValidationError",
    "decode_request_body",
    "encode_payload",
    "json_to_datamodel",
    "lp_data_to_datamodel",
    "normalize_lp_dict",
    "parse_lp_data",
    "validate_lp_data",
    "validate_lp_dict",
]
