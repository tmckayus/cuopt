# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Codec round-trip tests for legacy content types."""

import copy

import pytest
from fastapi import HTTPException

from cuopt_server.compat.codec import (
    decode_request_body,
    encode_payload,
    unwrap_managed_envelope,
)
from cuopt_server.utils.job_queue import (
    mime_json,
    mime_msgpack,
    mime_pickle,
    mime_zlib,
)
from cuopt_server.utils.linear_programming.data_definition import (
    lp_example_data,
)


@pytest.mark.parametrize(
    "mime",
    [mime_json, mime_msgpack, mime_zlib, mime_pickle],
)
def test_encode_decode_roundtrip(mime):
    payload = copy.deepcopy(lp_example_data)
    body, ctype = encode_payload(payload, mime)
    assert ctype == mime
    assert isinstance(body, (bytes, bytearray))
    decoded = decode_request_body(ctype, body)
    assert (
        decoded["csr_constraint_matrix"]["offsets"]
        == payload["csr_constraint_matrix"]["offsets"]
    )
    assert (
        decoded["objective_data"]["coefficients"]
        == payload["objective_data"]["coefficients"]
    )


def test_unsupported_content_type():
    with pytest.raises(HTTPException) as ei:
        decode_request_body("text/plain", b"{}")
    assert ei.value.status_code == 415


def test_unwrap_managed_envelope():
    inner = copy.deepcopy(lp_example_data)
    wrapped = {"action": "cuOpt_LP", "data": inner, "client_version": "26.08"}
    assert unwrap_managed_envelope(wrapped) is inner
    assert unwrap_managed_envelope(inner) is inner
