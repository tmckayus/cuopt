# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Micro-benchmark for legacy JSON validate / convert / gRPC submit / HTTP shim.

Examples::

  # Large synthetic payload
  python -m cuopt_server.compat.bench_timing --rows 8000 --cols 8000 --skip-grpc

  # Pre-built msgpack (e.g. L2CTA3D)
  python -m cuopt_server.compat.bench_timing \\
      --msgpack-file /home/tmckay/create_data/L2CTA3D.numpy.msgpack \\
      --no-deepcopy --warmup 0 --repeats 1 \\
      --grpc-port 5001 --http-url http://127.0.0.1:5000 --http-raw
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from cuopt_server.compat.codec import encode_payload
from cuopt_server.utils.job_queue import (
    mime_json,
    mime_msgpack,
    mime_pickle,
    mime_zlib,
)

_CONTENT_TYPES = {
    "json": mime_json,
    "msgpack": mime_msgpack,
    "zlib": mime_zlib,
    "pickle": mime_pickle,
}


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = int(round((p / 100.0) * (len(ys) - 1)))
    return ys[k]


def _report(name: str, samples_ms: List[float], extra: str = "") -> None:
    suffix = f"  {extra}" if extra else ""
    print(
        f"{name:32s}  n={len(samples_ms):3d}  "
        f"mean={statistics.mean(samples_ms):8.3f} ms  "
        f"p50={_pct(samples_ms, 50):8.3f}  "
        f"p95={_pct(samples_ms, 95):8.3f}  "
        f"min={min(samples_ms):8.3f}  max={max(samples_ms):8.3f}"
        f"{suffix}"
    )


def _time_it(fn: Callable[[], Any], warmup: int, repeats: int) -> List[float]:
    for _ in range(warmup):
        fn()
    out: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def make_synthetic_lp(
    rows: int,
    cols: int,
    *,
    nnz_per_row: int = 10,
    seed: int = 0,
    time_limit: float = 1.0,
) -> Dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    k = min(nnz_per_row, cols)
    indptr = [0]
    indices: List[int] = []
    values: List[float] = []
    for _ in range(rows):
        c = rng.choice(cols, size=k, replace=False)
        indices.extend(int(x) for x in c)
        values.extend(float(x) for x in rng.normal(size=k))
        indptr.append(len(indices))
    return {
        "csr_constraint_matrix": {
            "offsets": indptr,
            "indices": indices,
            "values": values,
        },
        "constraint_bounds": {
            "upper_bounds": [1.0] * rows,
            "lower_bounds": ["ninf"] * rows,
        },
        "objective_data": {
            "coefficients": [float(x) for x in rng.normal(size=cols)],
            "scalability_factor": 1.0,
            "offset": 0.0,
        },
        "variable_bounds": {
            "upper_bounds": ["inf"] * cols,
            "lower_bounds": [0.0] * cols,
        },
        "maximize": False,
        "solver_config": {
            "tolerances": {"optimality": 1e-4},
            "time_limit": time_limit,
        },
    }


def load_payload_from_mps(
    path: str, time_limit: float = 1.0
) -> Dict[str, Any]:
    try:
        import cuopt_mps_parser
    except ImportError as exc:
        raise SystemExit(
            "cuopt_mps_parser is not installed; use --msgpack-file / "
            "--json-file / --rows/--cols"
        ) from exc

    model = cuopt_mps_parser.ParseMps(path)
    data = cuopt_mps_parser.toDict(model, json=True)
    data.setdefault("solver_config", {})
    data["solver_config"]["time_limit"] = time_limit
    data["solver_config"].setdefault("tolerances", {"optimality": 1e-4})
    return data


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-file", help="Legacy LP/MILP JSON file")
    parser.add_argument(
        "--msgpack-file",
        help="Pre-serialized msgpack (list JSON or msgpack_numpy arrays)",
    )
    parser.add_argument("--mps", help="MPS file (needs cuopt_mps_parser)")
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--cols", type=int, default=0)
    parser.add_argument("--nnz-per-row", type=int, default=10)
    parser.add_argument("--time-limit", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--grpc-host", default="127.0.0.1")
    parser.add_argument("--grpc-port", type=int, default=5001)
    parser.add_argument("--http-url", default="")
    parser.add_argument(
        "--content-types",
        default="json,msgpack,zlib",
        help="Used when re-encoding; with --msgpack-file defaults to msgpack",
    )
    parser.add_argument(
        "--http-raw",
        action="store_true",
        help="POST --msgpack-file bytes as-is (recommended for huge payloads)",
    )
    parser.add_argument("--skip-grpc", action="store_true")
    parser.add_argument("--skip-http", action="store_true")
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--no-delete-jobs", action="store_true")
    parser.add_argument(
        "--no-deepcopy",
        action="store_true",
        help="Reuse payload across iterations (required for multi-GB problems)",
    )
    parser.add_argument("--http-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    raw_http_body: Optional[bytes] = None
    payload: Dict[str, Any]

    if args.msgpack_file:
        import msgpack
        import msgpack_numpy

        msgpack_numpy.patch()
        t0 = time.perf_counter()
        with open(args.msgpack_file, "rb") as f:
            raw_http_body = f.read()
        disk_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        payload = msgpack.loads(raw_http_body, strict_map_key=False)
        decode_ms = (time.perf_counter() - t1) * 1000.0
        source = f"msgpack-file:{args.msgpack_file}"
        print(
            f"disk read {disk_ms:.0f} ms ({len(raw_http_body) / (1024**2):.1f} MiB); "
            f"msgpack decode {decode_ms:.0f} ms"
        )
        args.http_raw = True
        args.content_types = "msgpack"
        args.no_deepcopy = True
    elif args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        source = f"json-file:{args.json_file}"
    elif args.mps:
        payload = load_payload_from_mps(args.mps, time_limit=args.time_limit)
        source = f"mps:{args.mps}"
    elif args.rows > 0 and args.cols > 0:
        payload = make_synthetic_lp(
            args.rows,
            args.cols,
            nnz_per_row=args.nnz_per_row,
            time_limit=args.time_limit,
        )
        source = f"synthetic {args.rows}x{args.cols}"
    else:
        payload = make_synthetic_lp(
            5000, 5000, nnz_per_row=10, time_limit=args.time_limit
        )
        source = "synthetic 5000x5000 (default)"

    def _maybe_copy(p: Dict[str, Any]) -> Dict[str, Any]:
        return p if args.no_deepcopy else copy.deepcopy(p)

    csr = payload["csr_constraint_matrix"]
    nnz = len(csr["values"])
    nrows = len(csr["offsets"]) - 1
    ncols = len(payload["objective_data"]["coefficients"])
    if raw_http_body is not None:
        size_note = f"msgpack≈{len(raw_http_body) / (1024 * 1024):.2f} MiB"
    else:
        try:
            size_note = f"json≈{len(json.dumps(payload).encode('utf-8')) / (1024 * 1024):.2f} MiB"
        except Exception:
            size_note = "size=n/a"
    print(
        f"payload: {source}  rows={nrows} cols={ncols} nnz={nnz}  {size_note}"
    )

    from cuopt.linear_programming import toDataModelAndSettings
    from cuopt_server.compat.validate import parse_lp_data, validate_lp_data

    def _convert(validate: bool):
        problem = _maybe_copy(payload)
        if validate:
            lp = parse_lp_data(problem)
            validate_lp_data(lp)
        return toDataModelAndSettings(problem)

    if not args.skip_local:
        print("=== local library timings ===")
        samples = _time_it(
            lambda: parse_lp_data(_maybe_copy(payload)),
            args.warmup,
            args.repeats,
        )
        _report("parse+normalize", samples)

        def _validate() -> None:
            lp = parse_lp_data(_maybe_copy(payload))
            validate_lp_data(lp)

        samples = _time_it(_validate, args.warmup, args.repeats)
        _report("parse+validate", samples)

        samples = _time_it(
            lambda: _convert(validate=False),
            args.warmup,
            args.repeats,
        )
        _report("json→DataModel (no val)", samples)

        samples = _time_it(
            lambda: _convert(validate=True),
            args.warmup,
            args.repeats,
        )
        _report("json→DataModel (validate)", samples)

    encoded: Dict[str, Tuple[bytes, str]] = {}
    if raw_http_body is not None and args.http_raw:
        encoded["msgpack"] = (raw_http_body, mime_msgpack)
        print("=== encode sizes ===")
        print(
            f"  msgpack  {len(raw_http_body) / (1024 * 1024):8.2f} MiB  "
            f"(raw file, no re-encode)"
        )
    elif not args.msgpack_file:
        print("=== encode sizes ===")
        for name, mime in _CONTENT_TYPES.items():
            t0 = time.perf_counter()
            body, ctype = encode_payload(payload, mime)
            ms = (time.perf_counter() - t0) * 1000.0
            encoded[name] = (body, ctype)
            print(
                f"  {name:8s}  {len(body) / (1024 * 1024):8.2f} MiB  "
                f"encode={ms:7.2f} ms  ({ctype})"
            )

    if not args.skip_grpc:
        print("=== gRPC Client.submit timings ===")
        from cuopt.grpc.linear_programming import Client

        try:
            client = Client(args.grpc_host, args.grpc_port, tls=False)
        except TypeError:
            client = Client(args.grpc_host, args.grpc_port)

        job_ids: List[str] = []

        # Convert once, then time submit alone (isolates gRPC hop)
        print("  (building DataModel once for submit-only timing...)")
        t0 = time.perf_counter()
        dm, settings = _convert(validate=True)
        convert_once_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  one-shot convert+validate: {convert_once_ms:.1f} ms")

        def _submit_only() -> str:
            jid = client.submit(dm, settings)
            job_ids.append(jid)
            return jid

        samples = _time_it(_submit_only, args.warmup, args.repeats)
        _report("Client.submit only (gRPC hop)", samples)

        if not args.no_delete_jobs:
            for jid in job_ids:
                try:
                    client.delete(jid)
                except Exception:
                    pass

    if not args.skip_http and args.http_url:
        print("=== HTTP shim timings ===")
        import urllib.error
        import urllib.request

        url = args.http_url.rstrip("/") + "/cuopt/request"
        ctypes = [
            c.strip()
            for c in args.content_types.split(",")
            if c.strip() in encoded
        ]
        http_job_ids: List[str] = []

        for name in ctypes:
            body, ctype = encoded[name]

            def _http_submit(body=body, ctype=ctype):
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": ctype},
                    method="POST",
                )
                with urllib.request.urlopen(
                    req, timeout=args.http_timeout
                ) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            def _http_timed(body=body, ctype=ctype):
                data = _http_submit(body=body, ctype=ctype)
                http_job_ids.append(data["reqId"])
                return data

            samples = _time_it(_http_timed, args.warmup, args.repeats)
            _report(
                f"HTTP POST ({name})",
                samples,
                extra=f"body={len(body) / (1024 * 1024):.2f} MiB",
            )
            try:
                last = _http_submit()
                http_job_ids.append(last["reqId"])
                if "timings_ms" in last:
                    print(
                        f"  last server-side timings_ms ({name}):",
                        last["timings_ms"],
                    )
            except urllib.error.URLError as exc:
                print(f"  (could not fetch server timings for {name}: {exc})")

        if not args.no_delete_jobs:
            from cuopt.grpc.linear_programming import Client

            try:
                client = Client(args.grpc_host, args.grpc_port, tls=False)
            except TypeError:
                client = Client(args.grpc_host, args.grpc_port)
            for jid in http_job_ids:
                try:
                    client.delete(jid)
                except Exception:
                    pass
    elif not args.skip_http and not args.http_url:
        print("(skip HTTP: pass --http-url to exercise the shim)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
