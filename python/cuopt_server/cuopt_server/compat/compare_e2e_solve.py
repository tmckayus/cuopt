# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""End-to-end solve comparison: legacy HTTP vs HTTP shim+gRPC vs direct gRPC.

Measures wall time until a solution (or error) is available. Supports a timed
loop with per-run and average CSVs.

Paths:
  legacy       — POST msgpack to cuopt_server, poll /cuopt/solution (Accept msgpack)
  shim         — POST msgpack to thin HTTP shim → gRPC, poll /cuopt/solution (Accept msgpack)
  client /
  client-native — msgpack/dict → DataModel → Client.submit/wait/result
                  (native gRPC; no problem JSON). Optional
                  ``--client-map-solution`` adds Solution→legacy dict (map_ms).
  client-native+map / client-map
               — same as client-native, then Solution→legacy dict (no JSON dumps)
  client-json  — JSON text → parse → DataModel → gRPC → legacy JSON result
                  (approximates a cuopt_sh-style JSON in / JSON out client)

Example (single pass)::

  python -m cuopt_server.compat.compare_e2e_solve \\
    --msgpack-file /path/to/L2CTA3D.numpy.msgpack \\
    --time-limit 30 --order client-native --grpc-port 18601

Example (loop + CSV)::

  python -m cuopt_server.compat.compare_e2e_solve \\
    --msgpack-file /path/to/L2CTA3D.numpy.msgpack \\
    --time-limit 30 --iterations 5 --warmup \\
    --legacy-url http://127.0.0.1:18600 \\
    --shim-url http://127.0.0.1:18602 \\
    --grpc-host 127.0.0.1 --grpc-port 18601 \\
    --order legacy,shim,client-native+map,client-json \\
    --csv /tmp/e2e_runs.csv --csv-summary /tmp/e2e_avg.csv

Warmup (default on) runs until the first successful solve per path and is
not recorded. Timed iterations then collect ``--iterations`` successful
runs; failures are retried and omitted from the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


CSV_FIELDS = [
    "iteration",
    "path",
    "ok",
    "convert_ms",
    "decode_ms",
    "validate_ms",
    "post_ms",
    "submit_ms",
    "wait_ms",
    "result_ms",
    "poll_ms",
    "get_wall_ms",
    "grpc_status_ms",
    "grpc_result_ms",
    "map_ms",
    "json_encode_ms",
    "response_bytes",
    "total_wall_ms",
    "primal_objective",
    "status",
    "req_id",
    "error",
]


def _load_msgpack(path: str) -> Tuple[Dict[str, Any], bytes]:
    import msgpack
    import msgpack_numpy

    msgpack_numpy.patch()
    with open(path, "rb") as f:
        raw = f.read()
    data = msgpack.loads(raw, strict_map_key=False)
    return data, raw


def _encode_msgpack(data: Dict[str, Any]) -> bytes:
    import msgpack
    import msgpack_numpy

    msgpack_numpy.patch()
    return msgpack.dumps(data)


def _parse_timings_header(headers) -> Optional[Dict[str, float]]:
    raw = headers.get("X-Cuopt-Timings") or headers.get("x-cuopt-timings")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    out: Dict[str, float] = {}
    for k, v in parsed.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


MIME_MSGPACK = "application/vnd.msgpack"


def _decode_response_body(
    raw: bytes, content_type: Optional[str]
) -> Dict[str, Any]:
    if not raw:
        return {}
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if "msgpack" in ctype:
        import msgpack
        import msgpack_numpy

        msgpack_numpy.patch()
        data = msgpack.loads(raw, strict_map_key=False)
        return data if isinstance(data, dict) else {"_payload": data}
    if "zlib" in ctype:
        import zlib

        raw = zlib.decompress(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
        return payload if isinstance(payload, dict) else {"_payload": payload}
    except Exception:
        return {"_raw": raw[:500].decode("utf-8", errors="replace")}


def _http(
    method: str,
    url: str,
    *,
    body: Optional[bytes] = None,
    content_type: Optional[str] = None,
    accept: str = MIME_MSGPACK,
    timeout: float = 3600.0,
) -> Tuple[int, Dict[str, Any], float, Dict[str, Any]]:
    headers = {"Accept": accept}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(
        url, data=body, headers=headers, method=method
    )
    t0 = time.perf_counter()
    meta: Dict[str, Any] = {
        "response_bytes": 0,
        "content_type": None,
        "timings_ms": None,
    }
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
            meta["content_type"] = resp.headers.get("Content-Type")
            meta["timings_ms"] = _parse_timings_header(resp.headers)
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
        meta["content_type"] = (
            e.headers.get("Content-Type") if e.headers else None
        )
        if e.headers:
            meta["timings_ms"] = _parse_timings_header(e.headers)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    meta["response_bytes"] = len(raw) if raw else 0
    payload = _decode_response_body(raw, meta["content_type"])
    return status, payload, elapsed_ms, meta


def _poll_legacy(
    base: str,
    req_id: str,
    timeout_s: float,
    *,
    timings: bool = False,
    accept: str = MIME_MSGPACK,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    url = f"{base.rstrip('/')}/cuopt/solution/{req_id}"
    if timings:
        url += "?timings=1"
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    last: Dict[str, Any] = {}
    last_meta: Dict[str, Any] = {}
    last_get_ms = 0.0
    while time.perf_counter() < deadline:
        status, last, last_get_ms, last_meta = _http(
            "GET", url, accept=accept, timeout=600
        )
        if status >= 400:
            break
        # Done: full envelope, or FastAPI error detail.
        if "response" in last or "error" in last:
            break
        if (
            isinstance(last, dict)
            and "detail" in last
            and "response" not in last
        ):
            last = dict(last)
            last["error"] = last["detail"]
            break
        time.sleep(0.25)
    info = {
        "get_wall_ms": last_get_ms,
        "response_bytes": last_meta.get("response_bytes"),
        "content_type": last_meta.get("content_type"),
        "timings_ms": last_meta.get("timings_ms") or {},
    }
    return last, (time.perf_counter() - t0) * 1000.0, info


def _poll_shim(
    base: str, req_id: str, timeout_s: float, *, accept: str = MIME_MSGPACK
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    # Shim returns the same legacy envelope as cuopt_server.
    # Timings stay in X-Cuopt-Timings so Accept can remain msgpack.
    return _poll_legacy(base, req_id, timeout_s, timings=True, accept=accept)


def _extract_legacy_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "error" in result:
        out["error"] = result["error"]
        return out
    if "detail" in result and "response" not in result:
        out["error"] = result["detail"]
        return out
    try:
        sol = result["response"]["solver_response"]["solution"]
        out["status"] = result["response"]["solver_response"].get("status")
        out["primal_objective"] = sol.get("primal_objective")
        out["solver_time"] = sol.get("solver_time")
    except Exception as exc:
        out["parse_error"] = str(exc)
        out["keys"] = list(result.keys())
    return out


def _extract_shim_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return _extract_legacy_summary(result)


def is_success(result: Dict[str, Any]) -> bool:
    """True only when the run produced a usable primal objective."""
    if not result.get("ok"):
        return False
    summary = result.get("summary") or {}
    if summary.get("error") or summary.get("parse_error"):
        return False
    return summary.get("primal_objective") is not None


def run_path(
    name: str,
    base_url: str,
    body: bytes,
    *,
    mode: str,
    poll_timeout_s: float,
    delete: bool,
) -> Dict[str, Any]:
    print(f"\n=== {name} ({base_url}) ===")
    post_url = f"{base_url.rstrip('/')}/cuopt/request"
    if mode == "shim":
        # Shim omits stage timings by default to stay byte-compatible with
        # clients that treat a lone reqId as "pending".
        post_url += "?timings=1"
    t_all = time.perf_counter()
    status, post_resp, post_ms, _post_meta = _http(
        "POST",
        post_url,
        body=body,
        content_type="application/vnd.msgpack",
        timeout=poll_timeout_s,
    )
    print(f"  POST status={status}  wall={post_ms:.0f} ms")
    if status >= 400:
        print(f"  POST failed: {post_resp}")
        return {
            "name": name,
            "ok": False,
            "post_ms": post_ms,
            "error": post_resp,
            "summary": {},
            "server_timings_ms": {},
            "get_timings_ms": {},
        }

    req_id = post_resp.get("reqId")
    if not req_id:
        print(f"  no reqId in response: {post_resp}")
        return {
            "name": name,
            "ok": False,
            "post_ms": post_ms,
            "error": post_resp,
            "summary": {},
            "server_timings_ms": {},
            "get_timings_ms": {},
        }

    print(f"  reqId={req_id}")
    if post_resp.get("timings_ms"):
        print(f"  server POST timings_ms={post_resp['timings_ms']}")

    if mode == "legacy":
        result, poll_ms, get_info = _poll_legacy(
            base_url, req_id, poll_timeout_s, timings=False
        )
        summary = _extract_legacy_summary(result)
        del_url = f"{base_url.rstrip('/')}/cuopt/solution/{req_id}"
    else:
        result, poll_ms, get_info = _poll_shim(
            base_url, req_id, poll_timeout_s
        )
        summary = _extract_shim_summary(result)
        del_url = f"{base_url.rstrip('/')}/cuopt/solution/{req_id}"

    get_timings = dict(get_info.get("timings_ms") or {})
    if get_info.get("response_bytes") is not None:
        get_timings.setdefault("response_bytes", get_info["response_bytes"])
    if get_info.get("get_wall_ms") is not None:
        get_timings["get_wall_ms"] = get_info["get_wall_ms"]

    total_ms = (time.perf_counter() - t_all) * 1000.0
    print(f"  poll_ms={poll_ms:.0f}  total_wall_ms={total_ms:.0f}")
    if get_timings:
        rb = get_timings.get("response_bytes")
        rb_note = (
            f"{rb / (1024 * 1024):.1f} MiB"
            if isinstance(rb, (int, float)) and rb
            else "n/a"
        )
        parts = [
            f"wall={get_timings.get('get_wall_ms', float('nan')):.0f} ms",
            f"bytes={rb_note}",
        ]
        if get_timings.get("status_ms") is not None:
            parts.append(f"status={get_timings['status_ms']:.0f} ms")
        if get_timings.get("result_ms") is not None:
            parts.append(f"grpc_result={get_timings['result_ms']:.0f} ms")
        if get_timings.get("map_ms") is not None:
            parts.append(f"map={get_timings['map_ms']:.0f} ms")
        enc_ms = get_timings.get("encode_ms")
        if enc_ms is None:
            enc_ms = get_timings.get("json_encode_ms")
        if enc_ms is not None:
            parts.append(f"encode={enc_ms:.0f} ms")
        ct = get_info.get("content_type")
        if ct:
            parts.append(f"ctype={ct}")
        print("  GET breakdown: " + "  ".join(parts))
    print(f"  summary={summary}")

    if delete:
        try:
            _http("DELETE", del_url, timeout=60)
        except Exception as exc:
            print(f"  delete warning: {exc}")

    ok = (
        status < 400
        and "error" not in summary
        and "parse_error" not in summary
        and summary.get("primal_objective") is not None
    )
    return {
        "name": name,
        "ok": ok,
        "req_id": req_id,
        "post_ms": post_ms,
        "poll_ms": poll_ms,
        "total_wall_ms": total_ms,
        "server_timings_ms": post_resp.get("timings_ms") or {},
        "get_timings_ms": get_timings,
        "summary": summary,
    }


def _jsonable(obj: Any) -> Any:
    """Convert numpy arrays (and nested structures) into JSON-serializable form."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return obj


def _encode_problem_json(payload: Dict[str, Any]) -> bytes:
    return json.dumps(_jsonable(payload)).encode("utf-8")


def _estimate_problem_json_bytes(payload: Dict[str, Any]) -> int:
    """Rough upper bound for JSON text size (avoids OOMing huge LPs)."""
    try:
        nnz = len(payload["csr_constraint_matrix"]["values"])
        ncols = len(payload["objective_data"]["coefficients"])
        nrows = len(payload["csr_constraint_matrix"]["offsets"]) - 1
    except Exception:
        return 0
    # ~20 ASCII bytes per numeric token is typical for float JSON text.
    return int((nnz * 3 + ncols + nrows) * 20)


_MAX_PROBLEM_JSON_BYTES = 1_500_000_000  # ~1.5 GiB; skip encode past this


def run_grpc_client_direct(
    payload: Dict[str, Any],
    *,
    grpc_host: str,
    grpc_port: int,
    poll_timeout_s: float,
    delete: bool,
    validate: bool = False,
    name: str = "client-native",
    map_solution: bool = False,
) -> Dict[str, Any]:
    """Native gRPC path: in-memory dict → DataModel → submit/wait/result.

    Does **not** encode the problem or solution as JSON. Use
    :func:`run_grpc_client_json` for a cuopt_sh-style JSON round-trip.
    With ``map_solution``, times Solution→legacy dict as ``map_ms`` (no dumps).
    """
    from cuopt.grpc.linear_programming import Client, JobStatus
    from cuopt.linear_programming import (
        toDataModelAndSettings,
        toDictFromSolution,
    )
    from cuopt_server.compat.validate import validate_lp_dict

    print(f"\n=== {name} (gRPC {grpc_host}:{grpc_port}) ===")
    if map_solution:
        print("  map_solution=True (Solution → legacy dict after result)")
    try:
        client = Client(grpc_host, grpc_port, tls=False)
    except TypeError:
        client = Client(grpc_host, grpc_port)

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    if validate:
        validate_lp_dict(payload)
    data_model, settings = toDataModelAndSettings(payload)
    timings["convert_ms"] = (time.perf_counter() - t0) * 1000.0
    print(f"  convert_ms={timings['convert_ms']:.1f} (validate={validate})")

    t1 = time.perf_counter()
    job_id = client.submit(data_model, settings)
    timings["submit_ms"] = (time.perf_counter() - t1) * 1000.0
    print(f"  submit_ms={timings['submit_ms']:.1f}  job_id={job_id}")

    t2 = time.perf_counter()
    status = client.wait(job_id, timeout=poll_timeout_s)
    timings["wait_ms"] = (time.perf_counter() - t2) * 1000.0
    print(f"  wait_ms={timings['wait_ms']:.1f}  status={status}")

    summary: Dict[str, Any] = {"status": getattr(status, "name", str(status))}
    if status == JobStatus.COMPLETED:
        t3 = time.perf_counter()
        sol = client.result(job_id)
        timings["result_ms"] = (time.perf_counter() - t3) * 1000.0
        try:
            summary["primal_objective"] = float(sol.get_primal_objective())
        except Exception:
            summary["primal_objective"] = None
        print(f"  result_ms={timings['result_ms']:.1f}")
        if map_solution:
            t_map = time.perf_counter()
            envelope = toDictFromSolution(sol)
            timings["map_ms"] = (time.perf_counter() - t_map) * 1000.0
            print(f"  map_ms={timings['map_ms']:.1f}")
            try:
                summary["primal_objective"] = envelope["response"][
                    "solver_response"
                ]["solution"].get("primal_objective")
                summary["status"] = envelope["response"][
                    "solver_response"
                ].get("status", summary["status"])
            except Exception:
                pass
    elif status in (JobStatus.FAILED, JobStatus.CANCELLED):
        summary["error"] = f"job ended with {status.name}"

    total_ms = (time.perf_counter() - t_all) * 1000.0
    print(f"  total_wall_ms={total_ms:.0f}")
    print(f"  summary={summary}")
    print(f"  timings={timings}")

    if delete:
        try:
            client.delete(job_id)
        except Exception as exc:
            print(f"  delete warning: {exc}")

    return {
        "name": name,
        "ok": (
            "error" not in summary
            and summary.get("primal_objective") is not None
        ),
        "req_id": job_id,
        "post_ms": timings.get("convert_ms", 0.0)
        + timings.get("submit_ms", 0.0),
        "poll_ms": timings.get("wait_ms", 0.0)
        + timings.get("result_ms", 0.0)
        + timings.get("map_ms", 0.0),
        "total_wall_ms": total_ms,
        "server_timings_ms": timings,
        "get_timings_ms": {},
        "summary": summary,
    }


def run_grpc_client_json(
    payload: Dict[str, Any],
    *,
    json_body: Optional[bytes],
    json_file: str,
    grpc_host: str,
    grpc_port: int,
    poll_timeout_s: float,
    delete: bool,
    validate: bool = False,
) -> Dict[str, Any]:
    """JSON→gRPC→JSON path approximating a cuopt_sh-style client.

    If ``json_file`` is provided, each iteration times ``json.load`` directly
    from that file. Otherwise, if ``json_body`` is provided, it times
    ``json.loads``. As a last resort it starts from the in-memory msgpack dict.

    Stages timed into ``total_wall_ms``:
      [decode JSON] → convert → submit → wait → result → map → json.dumps
    """
    from cuopt.grpc.linear_programming import Client, JobStatus
    from cuopt.linear_programming import (
        toDataModelAndSettings,
        toDictFromSolution,
    )
    from cuopt_server.compat.validate import validate_lp_dict

    name = "client-json"
    print(f"\n=== {name} (gRPC {grpc_host}:{grpc_port}) ===")
    if json_file:
        print(f"  loading problem JSON file each iteration: {json_file}")
    elif json_body is not None:
        print(
            f"  problem JSON size={len(json_body) / (1024 * 1024):.1f} MiB "
            "(pre-encoded once; decode timed per iteration)"
        )
    else:
        print(
            "  problem JSON skipped (too large); starting from in-memory "
            "dict after msgpack load — result still mapped+encoded to JSON"
        )
    try:
        client = Client(grpc_host, grpc_port, tls=False)
    except TypeError:
        client = Client(grpc_host, grpc_port)

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    if json_file:
        t_dec = time.perf_counter()
        with open(json_file, "r", encoding="utf-8") as f:
            problem = json.load(f)
        timings["decode_ms"] = (time.perf_counter() - t_dec) * 1000.0
        print(f"  json.load_ms={timings['decode_ms']:.1f}")
    elif json_body is not None:
        t_dec = time.perf_counter()
        problem = json.loads(json_body.decode("utf-8"))
        timings["decode_ms"] = (time.perf_counter() - t_dec) * 1000.0
        print(f"  decode_ms={timings['decode_ms']:.1f}")
    else:
        problem = payload
        timings["decode_ms"] = 0.0

    # Apply the same per-run settings used by the msgpack-backed paths.
    solver_config = problem.get("solver_config") or {}
    if not isinstance(solver_config, dict):
        solver_config = {}
    solver_config = dict(solver_config)
    solver_config["time_limit"] = payload["solver_config"]["time_limit"]
    solver_config.setdefault("tolerances", {"optimality": 1e-4})
    solver_config["log_to_console"] = False
    problem["solver_config"] = solver_config

    t0 = time.perf_counter()
    if validate:
        validate_lp_dict(problem)
    data_model, settings = toDataModelAndSettings(problem)
    timings["convert_ms"] = (time.perf_counter() - t0) * 1000.0
    print(f"  convert_ms={timings['convert_ms']:.1f} (validate={validate})")

    t1 = time.perf_counter()
    job_id = client.submit(data_model, settings)
    timings["submit_ms"] = (time.perf_counter() - t1) * 1000.0
    print(f"  submit_ms={timings['submit_ms']:.1f}  job_id={job_id}")

    t2 = time.perf_counter()
    status = client.wait(job_id, timeout=poll_timeout_s)
    timings["wait_ms"] = (time.perf_counter() - t2) * 1000.0
    print(f"  wait_ms={timings['wait_ms']:.1f}  status={status}")

    summary: Dict[str, Any] = {"status": getattr(status, "name", str(status))}
    if status == JobStatus.COMPLETED:
        t3 = time.perf_counter()
        sol = client.result(job_id)
        timings["result_ms"] = (time.perf_counter() - t3) * 1000.0
        print(f"  result_ms={timings['result_ms']:.1f}")

        t_map = time.perf_counter()
        envelope = toDictFromSolution(sol)
        timings["map_ms"] = (time.perf_counter() - t_map) * 1000.0

        t_enc = time.perf_counter()
        raw_out = json.dumps(envelope).encode("utf-8")
        timings["json_encode_ms"] = (time.perf_counter() - t_enc) * 1000.0
        timings["response_bytes"] = float(len(raw_out))

        try:
            summary["primal_objective"] = envelope["response"][
                "solver_response"
            ]["solution"].get("primal_objective")
            summary["status"] = envelope["response"]["solver_response"].get(
                "status", summary["status"]
            )
            summary["solver_time"] = envelope["response"]["solver_response"][
                "solution"
            ].get("solver_time")
        except Exception:
            try:
                summary["primal_objective"] = float(sol.get_primal_objective())
            except Exception:
                summary["primal_objective"] = None

        print(
            f"  map_ms={timings['map_ms']:.1f}  "
            f"json_encode_ms={timings['json_encode_ms']:.1f}  "
            f"response_bytes="
            f"{timings['response_bytes'] / (1024 * 1024):.1f} MiB"
        )
    elif status in (JobStatus.FAILED, JobStatus.CANCELLED):
        summary["error"] = f"job ended with {status.name}"

    total_ms = (time.perf_counter() - t_all) * 1000.0
    print(f"  total_wall_ms={total_ms:.0f}")
    print(f"  summary={summary}")
    print(f"  timings={timings}")

    if delete:
        try:
            client.delete(job_id)
        except Exception as exc:
            print(f"  delete warning: {exc}")

    return {
        "name": name,
        "ok": (
            "error" not in summary
            and summary.get("primal_objective") is not None
        ),
        "req_id": job_id,
        "post_ms": (
            timings.get("decode_ms", 0.0)
            + timings.get("convert_ms", 0.0)
            + timings.get("submit_ms", 0.0)
        ),
        "poll_ms": (
            timings.get("wait_ms", 0.0)
            + timings.get("result_ms", 0.0)
            + timings.get("map_ms", 0.0)
            + timings.get("json_encode_ms", 0.0)
        ),
        "total_wall_ms": total_ms,
        "server_timings_ms": timings,
        "get_timings_ms": {
            "map_ms": timings.get("map_ms"),
            "json_encode_ms": timings.get("json_encode_ms"),
            "response_bytes": timings.get("response_bytes"),
            "result_ms": timings.get("result_ms"),
        },
        "summary": summary,
    }


def flatten_result(result: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    st = result.get("server_timings_ms") or {}
    gt = result.get("get_timings_ms") or {}
    summary = result.get("summary") or {}
    err = result.get("error")
    if isinstance(err, dict):
        err = json.dumps(err)[:500]
    elif err is None and "error" in summary:
        err = str(summary.get("error"))[:500]

    wait_ms = st.get("wait_ms")
    if wait_ms is None:
        wait_ms = result.get("poll_ms")

    # Prefer client result_ms; fall back to shim GET grpc result.
    result_ms = st.get("result_ms")
    if result_ms is None:
        result_ms = gt.get("result_ms")

    map_ms = gt.get("map_ms")
    if map_ms is None:
        map_ms = st.get("map_ms")
    json_encode_ms = gt.get("encode_ms")
    if json_encode_ms is None:
        json_encode_ms = gt.get("json_encode_ms")
    if json_encode_ms is None:
        json_encode_ms = st.get("encode_ms")
    if json_encode_ms is None:
        json_encode_ms = st.get("json_encode_ms")
    response_bytes = gt.get("response_bytes")
    if response_bytes is None:
        response_bytes = st.get("response_bytes")

    return {
        "iteration": iteration,
        "path": result.get("name"),
        "ok": bool(result.get("ok")),
        "convert_ms": st.get("convert_ms"),
        "decode_ms": st.get("decode_ms"),
        "validate_ms": st.get("validate_ms"),
        "post_ms": result.get("post_ms"),
        "submit_ms": st.get("submit_ms"),
        "wait_ms": wait_ms,
        "result_ms": result_ms,
        "poll_ms": result.get("poll_ms"),
        "get_wall_ms": gt.get("get_wall_ms"),
        "grpc_status_ms": gt.get("status_ms"),
        "grpc_result_ms": gt.get("result_ms"),
        "map_ms": map_ms,
        # CSV column kept; value is response encode (msgpack or JSON).
        "json_encode_ms": json_encode_ms,
        "response_bytes": response_bytes,
        "total_wall_ms": result.get("total_wall_ms"),
        "primal_objective": summary.get("primal_objective"),
        "status": summary.get("status"),
        "req_id": result.get("req_id"),
        "error": err,
    }


def _mean(vals: List[float]) -> float:
    return statistics.mean(vals) if vals else float("nan")


def _stdev(vals: List[float]) -> float:
    return statistics.stdev(vals) if len(vals) >= 2 else float("nan")


def write_runs_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in CSV_FIELDS})


def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_path.setdefault(str(row["path"]), []).append(row)

    fields = [
        "path",
        "n",
        "ok_n",
        "mean_total_s",
        "stdev_total_s",
        "mean_post_s",
        "stdev_post_s",
        "mean_poll_s",
        "stdev_poll_s",
        "mean_get_wall_ms",
        "mean_grpc_status_ms",
        "mean_grpc_result_ms",
        "mean_map_ms",
        "mean_json_encode_ms",
        "mean_response_bytes",
        "mean_convert_ms",
        "mean_submit_ms",
        "mean_wait_ms",
        "mean_result_ms",
        "mean_primal_objective",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for pname, prows in by_path.items():
            # CSV only contains successful runs; keep ok_n for compatibility.
            ok_rows = [r for r in prows if r.get("ok", True)]
            totals = [
                float(r["total_wall_ms"]) / 1000.0
                for r in ok_rows
                if r.get("total_wall_ms") is not None
            ]
            posts = [
                float(r["post_ms"]) / 1000.0
                for r in ok_rows
                if r.get("post_ms") is not None
            ]
            polls = [
                float(r["poll_ms"]) / 1000.0
                for r in ok_rows
                if r.get("poll_ms") is not None
            ]

            def col(key: str) -> List[float]:
                return [
                    float(r[key])
                    for r in ok_rows
                    if r.get(key) is not None
                    and not (isinstance(r[key], float) and math.isnan(r[key]))
                ]

            w.writerow(
                {
                    "path": pname,
                    "n": len(ok_rows),
                    "ok_n": len(ok_rows),
                    "mean_total_s": _mean(totals),
                    "stdev_total_s": _stdev(totals),
                    "mean_post_s": _mean(posts),
                    "stdev_post_s": _stdev(posts),
                    "mean_poll_s": _mean(polls),
                    "stdev_poll_s": _stdev(polls),
                    "mean_get_wall_ms": _mean(col("get_wall_ms")),
                    "mean_grpc_status_ms": _mean(col("grpc_status_ms")),
                    "mean_grpc_result_ms": _mean(col("grpc_result_ms")),
                    "mean_map_ms": _mean(col("map_ms")),
                    "mean_json_encode_ms": _mean(col("json_encode_ms")),
                    "mean_response_bytes": _mean(col("response_bytes")),
                    "mean_convert_ms": _mean(col("convert_ms")),
                    "mean_submit_ms": _mean(col("submit_ms")),
                    "mean_wait_ms": _mean(col("wait_ms")),
                    "mean_result_ms": _mean(col("result_ms")),
                    "mean_primal_objective": _mean(col("primal_objective")),
                }
            )


def _print_comparison(results: List[Dict[str, Any]]) -> None:
    print("\n=== comparison ===")
    print(
        f"{'path':14s}  {'submit_s':>8s}  {'wait_s':>8s}  "
        f"{'total_s':>8s}  notes"
    )
    for r in results:
        notes = ""
        st = r.get("server_timings_ms") or {}
        gt = r.get("get_timings_ms") or {}
        if "decode_ms" in st:
            notes += f" decode_ms={st['decode_ms']:.0f}"
        if "convert_ms" in st:
            notes += f" convert_ms={st['convert_ms']:.0f}"
        if "submit_ms" in st:
            notes += f" submit_ms={st['submit_ms']:.0f}"
        if "wait_ms" in st:
            notes += f" wait_ms={st['wait_ms']:.0f}"
        if "result_ms" in st:
            notes += f" result_ms={st['result_ms']:.0f}"
        if gt.get("result_ms") is not None and "result_ms" not in st:
            notes += f" grpc_result_ms={gt['result_ms']:.0f}"
        map_ms = gt.get("map_ms", st.get("map_ms"))
        if map_ms is not None:
            notes += f" map_ms={map_ms:.0f}"
        enc_ms = gt.get("encode_ms", st.get("encode_ms"))
        if enc_ms is None:
            enc_ms = gt.get("json_encode_ms", st.get("json_encode_ms"))
        if enc_ms is not None:
            notes += f" encode_ms={enc_ms:.0f}"
        response_bytes = gt.get("response_bytes", st.get("response_bytes"))
        if response_bytes is not None:
            notes += f" resp_MiB={float(response_bytes) / (1024 * 1024):.1f}"
        s = r.get("summary") or {}
        if s.get("solver_time") is not None:
            notes += f" solver_time={s['solver_time']}"
        if s.get("primal_objective") is not None:
            notes += f" obj={s['primal_objective']}"
        if s.get("status"):
            notes += f" status={s['status']}"
        print(
            f"{r['name']:14s}  {r.get('post_ms', float('nan')) / 1000:8.2f}  "
            f"{r.get('poll_ms', float('nan')) / 1000:8.2f}  "
            f"{r.get('total_wall_ms', float('nan')) / 1000:8.2f}  {notes}"
        )


def _print_averages(rows: List[Dict[str, Any]]) -> None:
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_path.setdefault(str(row["path"]), []).append(row)
    print("\n=== averages (successful runs only) ===")
    print(
        f"{'path':14s}  {'n_ok':>5s}  {'mean_total_s':>12s}  "
        f"{'stdev_s':>8s}  {'mean_post_s':>11s}  {'mean_poll_s':>11s}"
    )
    for pname, prows in by_path.items():
        ok_rows = [r for r in prows if r.get("ok")]
        totals = [
            float(r["total_wall_ms"]) / 1000.0
            for r in ok_rows
            if r.get("total_wall_ms") is not None
        ]
        posts = [
            float(r["post_ms"]) / 1000.0
            for r in ok_rows
            if r.get("post_ms") is not None
        ]
        polls = [
            float(r["poll_ms"]) / 1000.0
            for r in ok_rows
            if r.get("poll_ms") is not None
        ]
        print(
            f"{pname:14s}  {len(ok_rows):5d}  {_mean(totals):12.2f}  "
            f"{_stdev(totals):8.2f}  {_mean(posts):11.2f}  {_mean(polls):11.2f}"
        )


def _run_one_order(
    order: List[str],
    *,
    data: Dict[str, Any],
    body: bytes,
    json_body: bytes,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for name in order:
        if name == "legacy":
            if not args.legacy_url:
                print("skip legacy (no --legacy-url)")
                continue
            results.append(
                run_path(
                    "legacy",
                    args.legacy_url,
                    body,
                    mode="legacy",
                    poll_timeout_s=args.poll_timeout,
                    delete=not args.no_delete,
                )
            )
        elif name == "shim":
            if not args.shim_url:
                print("skip shim (no --shim-url)")
                continue
            results.append(
                run_path(
                    "shim+grpc",
                    args.shim_url,
                    body,
                    mode="shim",
                    poll_timeout_s=args.poll_timeout,
                    delete=not args.no_delete,
                )
            )
        elif name in (
            "client",
            "client-native",
            "client-direct",
            "client-native+map",
            "client-map",
        ):
            if not args.grpc_port:
                print(f"skip {name} (no --grpc-port)")
                continue
            map_solution = name in ("client-native+map", "client-map") or (
                getattr(args, "client_map_solution", False)
                and name in ("client", "client-native", "client-direct")
            )
            path_name = (
                "client-native+map" if map_solution else "client-native"
            )
            results.append(
                run_grpc_client_direct(
                    data,
                    grpc_host=args.grpc_host,
                    grpc_port=args.grpc_port,
                    poll_timeout_s=args.poll_timeout,
                    delete=not args.no_delete,
                    validate=args.validate_client,
                    name=path_name,
                    map_solution=map_solution,
                )
            )
        elif name == "client-json":
            if not args.grpc_port:
                print("skip client-json (no --grpc-port)")
                continue
            results.append(
                run_grpc_client_json(
                    data,
                    json_body=json_body if json_body else None,
                    json_file=args.json_file,
                    grpc_host=args.grpc_host,
                    grpc_port=args.grpc_port,
                    poll_timeout_s=args.poll_timeout,
                    delete=not args.no_delete,
                    validate=args.validate_client,
                )
            )
        else:
            print(f"unknown order entry: {name}")
    return results


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--msgpack-file",
        default="/home/tmckay/create_data/L2CTA3D.numpy.msgpack",
    )
    p.add_argument(
        "--json-file",
        default="",
        help=(
            "Existing legacy problem JSON file for client-json. The file is "
            "opened and parsed with json.load during every timed iteration."
        ),
    )
    p.add_argument("--time-limit", type=float, default=30.0)
    p.add_argument("--legacy-url", default="")
    p.add_argument("--shim-url", default="")
    p.add_argument("--grpc-host", default="127.0.0.1")
    p.add_argument("--grpc-port", type=int, default=0)
    p.add_argument("--poll-timeout", type=float, default=600.0)
    p.add_argument("--no-delete", action="store_true")
    p.add_argument(
        "--validate-client",
        action="store_true",
        help="Run semantic validation on gRPC client paths (off by default)",
    )
    p.add_argument(
        "--order",
        default="legacy,shim,client-native",
        help=(
            "Comma list: legacy,shim,client-native|client,"
            "client-native+map|client-map,client-json"
        ),
    )
    p.add_argument(
        "--client-map-solution",
        action="store_true",
        help=(
            "On client-native, also map Solution→legacy dict (map_ms). "
            "Prefer --order client-native+map so both can run in one pass."
        ),
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of successful timed runs to record per path in --order",
    )
    p.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run untimed warmup until first success before timed iterations "
        "(default: on)",
    )
    p.add_argument(
        "--warmup-max-attempts",
        type=int,
        default=10,
        help="Max attempts to obtain a successful warmup run",
    )
    p.add_argument(
        "--max-fail-retries",
        type=int,
        default=10,
        help="Extra failed timed attempts allowed before giving up on a path",
    )
    p.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Seconds to sleep between individual path runs",
    )
    p.add_argument(
        "--csv",
        default="",
        help="Write per-run timings CSV to this path",
    )
    p.add_argument(
        "--csv-summary",
        default="",
        help="Write per-path averages CSV (defaults to <csv>.summary.csv)",
    )
    args = p.parse_args(argv)

    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")

    print(f"Loading {args.msgpack_file} ...")
    data, _raw = _load_msgpack(args.msgpack_file)
    nnz = len(data["csr_constraint_matrix"]["values"])
    nrows = len(data["csr_constraint_matrix"]["offsets"]) - 1
    ncols = len(data["objective_data"]["coefficients"])
    print(f"  rows={nrows} cols={ncols} nnz={nnz}")

    sc = data.get("solver_config") or {}
    if not isinstance(sc, dict):
        sc = {}
    sc = dict(sc)
    sc["time_limit"] = args.time_limit
    sc.setdefault("tolerances", {"optimality": 1e-4})
    sc["log_to_console"] = False
    data["solver_config"] = sc

    order = [x.strip() for x in args.order.split(",") if x.strip()]
    need_http_body = any(x in ("legacy", "shim") for x in order)
    need_json_body = any(x == "client-json" for x in order)
    if args.json_file and not os.path.isfile(args.json_file):
        raise SystemExit(f"--json-file does not exist: {args.json_file}")
    body = b""
    json_body = b""
    if need_http_body:
        print(f"Re-encoding msgpack with time_limit={args.time_limit}s ...")
        t0 = time.perf_counter()
        body = _encode_msgpack(data)
        print(
            f"  encoded {len(body) / (1024**2):.1f} MiB in "
            f"{(time.perf_counter() - t0) * 1000:.0f} ms"
        )
    if need_json_body and args.json_file:
        print(
            f"client-json will json.load existing file each iteration: "
            f"{args.json_file} "
            f"({os.path.getsize(args.json_file) / (1024**2):.1f} MiB)"
        )
    elif need_json_body:
        est = _estimate_problem_json_bytes(data)
        print(f"client-json problem JSON estimate ≈ {est / (1024**2):.1f} MiB")
        if est > _MAX_PROBLEM_JSON_BYTES:
            print(
                "  skipping problem JSON encode/decode "
                f"(estimate exceeds {_MAX_PROBLEM_JSON_BYTES / (1024**2):.0f} MiB); "
                "result→JSON still timed"
            )
            json_body = b""
        else:
            print("Encoding problem JSON for client-json path ...")
            t0 = time.perf_counter()
            json_body = _encode_problem_json(data)
            print(
                f"  encoded {len(json_body) / (1024**2):.1f} MiB in "
                f"{(time.perf_counter() - t0) * 1000:.0f} ms "
                "(encode cost excluded from timed iterations)"
            )

    all_rows: List[Dict[str, Any]] = []
    for path_name in order:
        print(f"\n########## path={path_name} ##########")
        if args.warmup:
            warmed = False
            for w_attempt in range(1, args.warmup_max_attempts + 1):
                print(
                    f"\n--- warmup {path_name} "
                    f"attempt {w_attempt}/{args.warmup_max_attempts} "
                    f"(not timed) ---"
                )
                results = _run_one_order(
                    [path_name],
                    data=data,
                    body=body,
                    json_body=json_body,
                    args=args,
                )
                _print_comparison(results)
                if results and is_success(results[0]):
                    print(
                        f"  warmup OK for {path_name} — "
                        "discarding timings, starting timed loop"
                    )
                    warmed = True
                    if args.cooldown > 0:
                        time.sleep(args.cooldown)
                    break
                print(f"  warmup FAILED for {path_name} — retrying")
                if args.cooldown > 0:
                    time.sleep(args.cooldown)
            if not warmed:
                print(
                    f"ERROR: warmup never succeeded for {path_name} "
                    f"after {args.warmup_max_attempts} attempts; skipping"
                )
                continue

        recorded = 0
        failures = 0
        attempt = 0
        max_attempts = args.iterations + args.max_fail_retries
        while recorded < args.iterations and attempt < max_attempts:
            attempt += 1
            print(
                f"\n########## timed {path_name} "
                f"{recorded + 1}/{args.iterations} "
                f"(attempt {attempt}) ##########"
            )
            results = _run_one_order(
                [path_name],
                data=data,
                body=body,
                json_body=json_body,
                args=args,
            )
            _print_comparison(results)
            if not results:
                failures += 1
                continue
            r = results[0]
            if is_success(r):
                recorded += 1
                all_rows.append(flatten_result(r, recorded))
                print(f"  recorded timed run {recorded}/{args.iterations}")
            else:
                failures += 1
                print("  FAILED — not recorded")
            if args.cooldown > 0:
                time.sleep(args.cooldown)

        if recorded < args.iterations:
            print(
                f"WARNING: only recorded {recorded}/{args.iterations} "
                f"successful timed runs for {path_name} "
                f"({failures} failures)"
            )

    _print_averages(all_rows)

    csv_path = args.csv
    summary_path = args.csv_summary
    if csv_path:
        write_runs_csv(csv_path, all_rows)
        print(f"\nWrote per-run CSV: {csv_path}")
        if not summary_path:
            if csv_path.endswith(".csv"):
                summary_path = csv_path[:-4] + ".summary.csv"
            else:
                summary_path = csv_path + ".summary.csv"
    if summary_path:
        write_summary_csv(summary_path, all_rows)
        print(f"Wrote averages CSV: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
