# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin FastAPI HTTP→gRPC pass-through for legacy LP/MILP JSON.

No job queue: each request converts JSON → DataModel and calls the Python
gRPC ``Client``. Concurrent requests are handled by the ASGI server.

Environment:
  CUOPT_GRPC_HOST   (default: 127.0.0.1)
  CUOPT_GRPC_PORT   (default: 5001)
  CUOPT_SHIM_VALIDATE (default: 1) — set 0 to skip semantic validation

Run::

  python -m cuopt_server.compat.shim --http-port 5000 --grpc-host 127.0.0.1 --grpc-port 5001
"""

from __future__ import annotations

import argparse
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from cuopt.grpc.linear_programming import Client, GrpcError, JobStatus

from cuopt_server.compat.codec import (
    decode_request_body,
    unwrap_managed_envelope,
)
from cuopt_server.compat.convert import lp_data_to_datamodel, parse_lp_data
from cuopt_server.compat.response import solution_to_legacy_response
from cuopt_server.compat.validate import (
    LegacyJsonValidationError,
    validate_lp_data,
)
from cuopt_server.utils.job_queue import check_client_version

_MAX_TRACKED_JOBS = 4096

# Map gRPC JobStatus names → legacy GET /cuopt/request/{id} strings.
_LEGACY_STATUS = {
    JobStatus.QUEUED: "queued",
    JobStatus.PROCESSING: "running",
    JobStatus.COMPLETED: "completed",
    JobStatus.FAILED: "aborted",
    JobStatus.CANCELLED: "aborted",
}

_DEFAULT_HOST = os.environ.get("CUOPT_GRPC_HOST", "127.0.0.1")
_DEFAULT_PORT = int(os.environ.get("CUOPT_GRPC_PORT", "5001"))
_VALIDATE = os.environ.get("CUOPT_SHIM_VALIDATE", "1") not in (
    "0",
    "false",
    "False",
)


def _make_client(host: str, port: int) -> Client:
    try:
        return Client(host, port, tls=False)
    except TypeError:
        return Client(host, port)


def create_app(
    grpc_host: str = _DEFAULT_HOST,
    grpc_port: int = _DEFAULT_PORT,
    *,
    validate: bool = _VALIDATE,
) -> FastAPI:
    state: dict[str, Any] = {}

    # Warnings are produced at submit time but reported with the solution,
    # so keep them per job (bounded; oldest evicted).
    job_warnings: "OrderedDict[str, list[str]]" = OrderedDict()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["client"] = _make_client(grpc_host, grpc_port)
        state["validate"] = validate
        state["grpc_host"] = grpc_host
        state["grpc_port"] = grpc_port
        yield
        state.pop("client", None)
        job_warnings.clear()

    app = FastAPI(
        title="cuOpt HTTP→gRPC shim",
        version="0.1.0",
        lifespan=lifespan,
    )

    def client() -> Client:
        c = state.get("client")
        if c is None:
            raise HTTPException(503, "gRPC client not ready")
        return c

    def remember_warnings(job_id: str, warnings: list[str]) -> None:
        job_warnings[job_id] = warnings
        while len(job_warnings) > _MAX_TRACKED_JOBS:
            job_warnings.popitem(last=False)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "grpc_host": state.get("grpc_host"),
            "grpc_port": state.get("grpc_port"),
            "validate": state.get("validate"),
        }

    @app.post("/cuopt/request")
    async def submit_request(request: Request, timings: bool = False):
        """Accept legacy LP/MILP body; return ``reqId`` (= gRPC job_id).

        Content-Types match the legacy server: JSON, msgpack, zlib, pickle.
        Pass ``?timings=1`` for server-side stage timings; the default body is
        exactly ``{"reqId": ...}`` because clients treat a single-key ``reqId``
        response as "still pending".
        """
        t0 = time.perf_counter()
        raw = await request.body()
        stage_ms: dict[str, float] = {}

        t_dec = time.perf_counter()
        decoded = decode_request_body(request.headers.get("content-type"), raw)
        body = unwrap_managed_envelope(decoded)
        stage_ms["decode_ms"] = (time.perf_counter() - t_dec) * 1000.0
        stage_ms["body_bytes"] = float(len(raw))

        client_version = request.headers.get("CLIENT-VERSION")
        if client_version is None and isinstance(decoded, dict):
            client_version = decoded.get("client_version")
        warnings = check_client_version(client_version or "")

        try:
            t1 = time.perf_counter()
            lp_data = parse_lp_data(body)
            stage_ms["parse_ms"] = (time.perf_counter() - t1) * 1000.0

            if state.get("validate", True):
                t2 = time.perf_counter()
                validate_lp_data(lp_data)
                stage_ms["validate_ms"] = (time.perf_counter() - t2) * 1000.0

            t3 = time.perf_counter()
            data_model, settings, convert_warnings = lp_data_to_datamodel(
                lp_data, return_warnings=True
            )
            warnings.extend(convert_warnings)
            stage_ms["convert_ms"] = (time.perf_counter() - t3) * 1000.0

            t4 = time.perf_counter()
            job_id = client().submit(data_model, settings)
            stage_ms["submit_ms"] = (time.perf_counter() - t4) * 1000.0
            remember_warnings(job_id, warnings)
        except LegacyJsonValidationError as exc:
            raise HTTPException(400, str(exc)) from exc
        except GrpcError as exc:
            raise HTTPException(502, f"gRPC submit failed: {exc}") from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                400, f"conversion/submit failed: {exc}"
            ) from exc

        stage_ms["total_ms"] = (time.perf_counter() - t0) * 1000.0
        body_out: dict[str, Any] = {"reqId": job_id}
        if timings:
            body_out["timings_ms"] = stage_ms
        return JSONResponse(body_out)

    @app.get("/cuopt/request/{job_id}")
    def check_status(job_id: str):
        """Legacy returns a bare status string: queued|running|completed|aborted."""
        try:
            status = client().status(job_id)
        except GrpcError as exc:
            raise HTTPException(502, str(exc)) from exc
        if status == JobStatus.NOT_FOUND:
            raise HTTPException(404, f"job {job_id} does not exist")
        return _LEGACY_STATUS.get(status, status.name.lower())

    @app.get("/cuopt/solution/{job_id}")
    def get_solution(job_id: str):
        """Full legacy solution envelope (or ``{"reqId"}`` while pending)."""
        try:
            status = client().status(job_id)
            if status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                # Legacy polls with HTTP 200 and body {"reqId": ...} only.
                return {"reqId": job_id}
            if status == JobStatus.NOT_FOUND:
                raise HTTPException(404, f"job {job_id} does not exist")
            if status in (JobStatus.FAILED, JobStatus.CANCELLED):
                raise HTTPException(
                    400, f"job {job_id} ended with {status.name}"
                )
            sol = client().result(job_id)
        except HTTPException:
            raise
        except GrpcError as exc:
            raise HTTPException(502, str(exc)) from exc

        if sol is None:
            return {"reqId": job_id}

        try:
            total_solve_time = float(sol.get_solve_time())
        except Exception:
            total_solve_time = 0.0

        return solution_to_legacy_response(
            sol,
            req_id=job_id,
            warnings=job_warnings.get(job_id) or [],
            total_solve_time=total_solve_time,
        )

    def _delete_job(job_id: str):
        try:
            client().delete(job_id)
        except GrpcError as exc:
            raise HTTPException(502, str(exc)) from exc
        job_warnings.pop(job_id, None)

    @app.delete("/cuopt/request/{job_id}")
    def delete_request(job_id: str):
        _delete_job(job_id)
        return {"reqId": job_id, "deleted": True}

    @app.delete("/cuopt/solution/{job_id}")
    def delete_solution(job_id: str):
        """Legacy alias: DELETE /cuopt/solution/{id} returns empty 200."""
        _delete_job(job_id)
        return Response(status_code=200)

    return app


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="cuOpt HTTP→gRPC shim")
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=5000)
    parser.add_argument("--grpc-host", default=_DEFAULT_HOST)
    parser.add_argument("--grpc-port", type=int, default=_DEFAULT_PORT)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip semantic JSON validation (trusted producers)",
    )
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app(
        args.grpc_host,
        args.grpc_port,
        validate=not args.no_validate,
    )
    uvicorn.run(
        app, host=args.http_host, port=args.http_port, log_level="info"
    )


if __name__ == "__main__":
    main()
