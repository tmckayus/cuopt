# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request body codecs matching the legacy cuOpt REST server.

Supported Content-Types (same as ``job_queue.deserialize``):

- ``application/json``
- ``application/vnd.msgpack`` (with ``msgpack_numpy`` patch)
- ``application/zlib`` (zlib-compressed JSON)
- ``application/octet-stream`` (restricted pickle)
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional, Tuple

from fastapi import HTTPException

from cuopt_server.utils.job_queue import (
    deserialize,
    mime_json,
    mime_msgpack,
    mime_pickle,
    mime_zlib,
)

SUPPORTED_CONTENT_TYPES = (
    mime_json,
    mime_msgpack,
    mime_zlib,
    mime_pickle,
)


def normalize_content_type(header: Optional[str]) -> str:
    """Strip parameters (e.g. ``charset=utf-8``) and default to JSON."""
    if not header:
        return mime_json
    ctype = header.split(";", 1)[0].strip().lower()
    if not ctype:
        return mime_json
    return ctype


def decode_request_body(content_type: Optional[str], body: bytes) -> Any:
    """Decode a raw HTTP body the same way the legacy server does.

    Raises
    ------
    HTTPException
        415 for unsupported types, 422 on decode failure (same status codes
        as the legacy path).
    """
    ctype = normalize_content_type(content_type)
    if ctype not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported Content-Type {ctype!r}; "
                f"supported: {list(SUPPORTED_CONTENT_TYPES)}"
            ),
        )
    return deserialize(ctype, body)


def encode_payload(
    data: Mapping[str, Any],
    content_type: str = mime_json,
) -> Tuple[bytes, str]:
    """Encode a dict for bench / client use. Returns ``(body, content_type)``."""
    import json
    import zlib

    import msgpack
    import msgpack_numpy

    msgpack_numpy.patch()

    ctype = normalize_content_type(content_type)
    if ctype == mime_json:
        return json.dumps(data).encode("utf-8"), mime_json
    if ctype == mime_zlib:
        raw = json.dumps(data).encode("utf-8")
        return zlib.compress(raw, zlib.Z_BEST_SPEED), mime_zlib
    if ctype == mime_msgpack:
        return msgpack.dumps(data), mime_msgpack
    if ctype == mime_pickle:
        import pickle

        return pickle.dumps(
            data, protocol=pickle.HIGHEST_PROTOCOL
        ), mime_pickle
    raise ValueError(f"Unsupported content type for encode: {ctype}")


def unwrap_managed_envelope(data: Any) -> Any:
    """If body is ``{action, data, ...}``, return the inner ``data``."""
    if (
        isinstance(data, MutableMapping)
        and "data" in data
        and "csr_constraint_matrix" not in data
    ):
        return data["data"]
    return data
