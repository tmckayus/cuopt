# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy JSON compatibility helpers for LP/MILP.

The package contains three server-specific pieces:

1. :mod:`cuopt_server.compat.validate` — essential structural checks
2. :mod:`cuopt_server.compat.codec` — legacy HTTP payload encoding
3. :mod:`cuopt_server.compat.shim` — thin HTTP→gRPC FastAPI pass-through

The shared DataModel/SolverSettings/Solution converters are public APIs in
``cuopt.linear_programming``. The HTTP shim validates by default, then calls
those same converters.
"""

from cuopt_server.compat.codec import decode_request_body, encode_payload
from cuopt_server.compat.normalize import normalize_lp_dict
from cuopt_server.compat.validate import (
    LegacyJsonValidationError,
    parse_lp_data,
    validate_lp_data,
    validate_lp_dict,
)

__all__ = [
    "LegacyJsonValidationError",
    "decode_request_body",
    "encode_payload",
    "normalize_lp_dict",
    "parse_lp_data",
    "validate_lp_data",
    "validate_lp_dict",
]
