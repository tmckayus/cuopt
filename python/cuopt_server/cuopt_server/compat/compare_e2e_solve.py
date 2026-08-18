# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""End-to-end solve comparison: legacy HTTP vs HTTP shim+gRPC vs direct gRPC client.

Measures wall time until a solution (or error) is available. Supports a timed
loop with per-run and average CSVs.

Paths:
  legacy  — POST msgpack to cuopt_server, poll /cuopt/solution
  shim    — POST msgpack to thin HTTP shim → gRPC, poll /cuopt/solution
  client  — in-process msgpack/dict → DataModel → Client.submit/wait/result
            (no HTTP)

Example (single pass)::

  python -m cuopt_server.compat.compare_e2e_solve \\
    --msgpack-file /path/to/L2CTA3D.numpy.msgpack \\
    --time-limit 30 --order client --grpc-port 18601

Example (loop + CSV)::

  python -m cuopt_server.compat.compare_e2e_solve \\
    --msgpack-file /path/to/L2CTA3D.numpy.msgpack \\
    --time-limit 30 --iterations 5 --warmup \\
    --legacy-url http://127.0.0.1:18600 \\
    --shim-url http://127.0.0.1:18602 \\
    --grpc-host 127.0.0.1 --grpc-port 18601 \\
    --order legacy,shim,client \\
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


def _http(
    method: str,
    url: str,
    *,
    body: Optional[bytes] = None,
    content_type: Optional[str] = None,
    timeout: float = 3600.0,
) -> Tuple[int, Dict[str, Any], float]:
    headers = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(
        url, data=body, headers=headers, method=method
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except Exception:
        payload = {"_raw": raw[:500].decode("utf-8", errors="replace")}
    return status, payload, elapsed_ms


def _poll_legacy(
    base: str, req_id: str, timeout_s: float
) -> Tuple[Dict[str, Any], float]:
    url = f"{base.rstrip('/')}/cuopt/solution/{req_id}"
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    last: Dict[str, Any] = {}
    while time.perf_counter() < deadline:
        status, last, _ = _http("GET", url, timeout=60)
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
    return last, (time.perf_counter() - t0) * 1000.0


def _poll_shim(
    base: str, req_id: str, timeout_s: float
) -> Tuple[Dict[str, Any], float]:
    # Shim now returns the same legacy envelope as cuopt_server.
    return _poll_legacy(base, req_id, timeout_s)


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
    status, post_resp, post_ms = _http(
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
        }

    print(f"  reqId={req_id}")
    if post_resp.get("timings_ms"):
        print(f"  server timings_ms={post_resp['timings_ms']}")

    if mode == "legacy":
        result, poll_ms = _poll_legacy(base_url, req_id, poll_timeout_s)
        summary = _extract_legacy_summary(result)
        del_url = f"{base_url.rstrip('/')}/cuopt/solution/{req_id}"
    else:
        result, poll_ms = _poll_shim(base_url, req_id, poll_timeout_s)
        summary = _extract_shim_summary(result)
        del_url = f"{base_url.rstrip('/')}/cuopt/solution/{req_id}"

    total_ms = (time.perf_counter() - t_all) * 1000.0
    print(f"  poll_ms={poll_ms:.0f}  total_wall_ms={total_ms:.0f}")
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
        "summary": summary,
    }


def run_grpc_client_direct(
    payload: Dict[str, Any],
    *,
    grpc_host: str,
    grpc_port: int,
    poll_timeout_s: float,
    delete: bool,
    validate: bool = False,
) -> Dict[str, Any]:
    """Client-side convert → gRPC Client.submit/wait/result (no HTTP)."""
    from cuopt.grpc.linear_programming import Client, JobStatus
    from cuopt_server.compat.convert import json_to_datamodel

    print(f"\n=== client-direct (gRPC {grpc_host}:{grpc_port}) ===")
    try:
        client = Client(grpc_host, grpc_port, tls=False)
    except TypeError:
        client = Client(grpc_host, grpc_port)

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    data_model, settings = json_to_datamodel(payload, validate=validate)
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
        "name": "client-direct",
        "ok": (
            "error" not in summary
            and summary.get("primal_objective") is not None
        ),
        "req_id": job_id,
        "post_ms": timings.get("convert_ms", 0.0)
        + timings.get("submit_ms", 0.0),
        "poll_ms": timings.get("wait_ms", 0.0) + timings.get("result_ms", 0.0),
        "total_wall_ms": total_ms,
        "server_timings_ms": timings,
        "summary": summary,
    }


def flatten_result(result: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    st = result.get("server_timings_ms") or {}
    summary = result.get("summary") or {}
    err = result.get("error")
    if isinstance(err, dict):
        err = json.dumps(err)[:500]
    elif err is None and "error" in summary:
        err = str(summary.get("error"))[:500]

    wait_ms = st.get("wait_ms")
    if wait_ms is None:
        wait_ms = result.get("poll_ms")

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
        "result_ms": st.get("result_ms"),
        "poll_ms": result.get("poll_ms"),
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
        if "convert_ms" in st:
            notes += f" convert_ms={st['convert_ms']:.0f}"
        if "submit_ms" in st:
            notes += f" submit_ms={st['submit_ms']:.0f}"
        if "wait_ms" in st:
            notes += f" wait_ms={st['wait_ms']:.0f}"
        if "result_ms" in st:
            notes += f" result_ms={st['result_ms']:.0f}"
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
        elif name == "client":
            if not args.grpc_port:
                print("skip client (no --grpc-port)")
                continue
            results.append(
                run_grpc_client_direct(
                    data,
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
        help="Run semantic validation on client-direct path (off by default)",
    )
    p.add_argument(
        "--order",
        default="legacy,shim,client",
        help="Comma list: legacy,shim,client",
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
    body = b""
    if need_http_body:
        print(f"Re-encoding msgpack with time_limit={args.time_limit}s ...")
        t0 = time.perf_counter()
        body = _encode_msgpack(data)
        print(
            f"  encoded {len(body) / (1024**2):.1f} MiB in "
            f"{(time.perf_counter() - t0) * 1000:.0f} ms"
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
                    [path_name], data=data, body=body, args=args
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
                [path_name], data=data, body=body, args=args
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
