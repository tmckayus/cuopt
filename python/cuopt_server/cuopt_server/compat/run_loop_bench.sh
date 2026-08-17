#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run timed e2e loop for legacy / shim / client paths and write CSVs.
#
# On a single GPU with large instances, legacy HTTP and gRPC cannot both
# hold a large RMM pool at once. This script therefore runs each path in
# ORDER sequentially: start only the servers that path needs, run all
# iterations for that path, tear down, free GPU, then move on.
#
# Required:
#   MSGPACK_FILE   path to numpy-msgpack LP (e.g. L2CTA3D.numpy.msgpack)
#
# Optional env (defaults shown):
#   ITERATIONS=5
#   TIME_LIMIT=30
#   ORDER=legacy,shim,client
#   CUOPT_GIGABYTES_PER_PROC=6
#   MAX_MESSAGE_MB=1024
#   LEGACY_PORT=18600 GRPC_PORT=18601 SHIM_PORT=18602
#   OUT_DIR=./bench_out
#   COOLDOWN=2
#   POLL_TIMEOUT=600

set -euo pipefail

export PYTHONUNBUFFERED=1

MSGPACK_FILE="${MSGPACK_FILE:?Set MSGPACK_FILE to the numpy msgpack LP path}"
ITERATIONS="${ITERATIONS:-5}"
TIME_LIMIT="${TIME_LIMIT:-30}"
ORDER="${ORDER:-legacy,shim,client}"
CUOPT_GIGABYTES_PER_PROC="${CUOPT_GIGABYTES_PER_PROC:-6}"
MAX_MESSAGE_MB="${MAX_MESSAGE_MB:-1024}"
LEGACY_PORT="${LEGACY_PORT:-18600}"
GRPC_PORT="${GRPC_PORT:-18601}"
SHIM_PORT="${SHIM_PORT:-18602}"
OUT_DIR="${OUT_DIR:-./bench_out}"
COOLDOWN="${COOLDOWN:-2}"
POLL_TIMEOUT="${POLL_TIMEOUT:-600}"

LEGACY_URL="http://127.0.0.1:${LEGACY_PORT}"
SHIM_URL="http://127.0.0.1:${SHIM_PORT}"

mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
CSV="${OUT_DIR}/e2e_runs_${STAMP}.csv"
SUMMARY="${OUT_DIR}/e2e_avg_${STAMP}.csv"
LOG_DIR="${OUT_DIR}/logs_${STAMP}"
mkdir -p "${LOG_DIR}"

PIDS=()

free_gpu() {
  echo "Stopping servers / freeing GPU..."
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  PIDS=()
  fuser -k "${LEGACY_PORT}/tcp" 2>/dev/null || true
  fuser -k "${GRPC_PORT}/tcp" 2>/dev/null || true
  fuser -k "${SHIM_PORT}/tcp" 2>/dev/null || true
  # leftover workers / solvers
  pkill -9 -f "cuopt_grpc_server --port ${GRPC_PORT}" 2>/dev/null || true
  pkill -9 -f "cuopt_service --port ${LEGACY_PORT}" 2>/dev/null || true
  pkill -9 -f "compat.shim .*--http-port ${SHIM_PORT}" 2>/dev/null || true
  sleep 3
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader || true
  fi
}

cleanup() {
  free_gpu
}
trap cleanup EXIT

wait_tcp() {
  local host="$1" port="$2" name="$3" tries="${4:-60}"
  for _ in $(seq 1 "${tries}"); do
    if python -c "import socket; socket.create_connection(('${host}', ${port}), 1)" 2>/dev/null; then
      echo "${name} up on ${host}:${port}"
      return 0
    fi
    sleep 0.5
  done
  echo "ERROR: ${name} did not come up on ${host}:${port}" >&2
  return 1
}

wait_http() {
  local url="$1" name="$2" tries="${3:-90}"
  for _ in $(seq 1 "${tries}"); do
    if python -c "import urllib.request; urllib.request.urlopen('${url}', timeout=2)" 2>/dev/null; then
      echo "${name} healthy at ${url}"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: ${name} health check failed: ${url}" >&2
  return 1
}

start_legacy() {
  echo "Starting legacy HTTP server on ${LEGACY_PORT}..."
  nohup python -m cuopt_server.cuopt_service --port "${LEGACY_PORT}" \
    >"${LOG_DIR}/legacy.log" 2>&1 &
  PIDS+=($!)
  wait_http "${LEGACY_URL}/cuopt/health" "legacy"
}

start_grpc() {
  echo "Starting gRPC server on ${GRPC_PORT}..."
  nohup env CUOPT_GIGABYTES_PER_PROC="${CUOPT_GIGABYTES_PER_PROC}" \
    cuopt_grpc_server --port "${GRPC_PORT}" --workers 1 \
    --max-message-mb "${MAX_MESSAGE_MB}" \
    >"${LOG_DIR}/grpc.log" 2>&1 &
  PIDS+=($!)
  wait_tcp 127.0.0.1 "${GRPC_PORT}" "grpc"
}

start_shim() {
  echo "Starting HTTP→gRPC shim on ${SHIM_PORT}..."
  nohup python -m cuopt_server.compat.shim \
    --http-host 127.0.0.1 --http-port "${SHIM_PORT}" \
    --grpc-host 127.0.0.1 --grpc-port "${GRPC_PORT}" \
    >"${LOG_DIR}/shim.log" 2>&1 &
  PIDS+=($!)
  wait_http "${SHIM_URL}/health" "shim"
}

run_path_loop() {
  local path="$1"
  local tmp_csv="${LOG_DIR}/runs_${path}.csv"
  local tmp_sum="${LOG_DIR}/avg_${path}.csv"
  local args=(
    --msgpack-file "${MSGPACK_FILE}"
    --time-limit "${TIME_LIMIT}"
    --iterations "${ITERATIONS}"
    --cooldown "${COOLDOWN}"
    --poll-timeout "${POLL_TIMEOUT}"
    --order "${path}"
    --csv "${tmp_csv}"
    --csv-summary "${tmp_sum}"
  )
  case "${path}" in
    legacy)
      args+=(--legacy-url "${LEGACY_URL}")
      ;;
    shim)
      args+=(--shim-url "${SHIM_URL}" --grpc-host 127.0.0.1 --grpc-port "${GRPC_PORT}")
      ;;
    client)
      args+=(--grpc-host 127.0.0.1 --grpc-port "${GRPC_PORT}")
      ;;
    *)
      echo "ERROR: unknown path '${path}'" >&2
      return 1
      ;;
  esac
  echo "===== path=${path} iterations=${ITERATIONS} ====="
  python -m cuopt_server.compat.compare_e2e_solve "${args[@]}" \
    | tee -a "${LOG_DIR}/compare.log"
}

merge_csvs() {
  python - "${CSV}" "${SUMMARY}" "${LOG_DIR}" <<'PY'
import csv, sys
from pathlib import Path

out_runs, out_avg, log_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
run_files = sorted(log_dir.glob("runs_*.csv"))
if not run_files:
    raise SystemExit("no per-path run CSVs to merge")

rows = []
fields = None
for f in run_files:
    with f.open() as fh:
        r = csv.DictReader(fh)
        fields = r.fieldnames
        rows.extend(list(r))

with open(out_runs, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# Recompute summary via the compare module helpers
from cuopt_server.compat.compare_e2e_solve import write_summary_csv

# coerce types for averaging
typed = []
for row in rows:
    t = dict(row)
    t["ok"] = str(row.get("ok", "")).lower() in ("1", "true", "yes")
    for k in (
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
    ):
        v = row.get(k)
        if v is None or v == "":
            t[k] = None
        else:
            try:
                t[k] = float(v)
            except ValueError:
                t[k] = None
    typed.append(t)

write_summary_csv(out_avg, typed)
print(f"merged {len(rows)} rows -> {out_runs}")
print(f"summary -> {out_avg}")
PY
}

echo "MSGPACK_FILE=${MSGPACK_FILE}"
echo "ITERATIONS=${ITERATIONS} TIME_LIMIT=${TIME_LIMIT} ORDER=${ORDER}"
echo "CUOPT_GIGABYTES_PER_PROC=${CUOPT_GIGABYTES_PER_PROC}"
echo "CSV=${CSV}"
echo "SUMMARY=${SUMMARY}"
echo "(paths run sequentially with GPU freed between them)"

IFS=',' read -ra _ORDER_PARTS <<< "${ORDER}"
for raw in "${_ORDER_PARTS[@]}"; do
  path="$(echo "${raw}" | tr -d '[:space:]')"
  [[ -z "${path}" ]] && continue
  free_gpu
  case "${path}" in
    legacy)
      start_legacy
      run_path_loop legacy
      ;;
    shim)
      start_grpc
      start_shim
      run_path_loop shim
      ;;
    client)
      start_grpc
      run_path_loop client
      ;;
    *)
      echo "ERROR: unknown ORDER entry '${path}'" >&2
      exit 1
      ;;
  esac
done

free_gpu
merge_csvs

echo
echo "Done."
echo "  per-run CSV : ${CSV}"
echo "  averages CSV: ${SUMMARY}"
echo "  logs        : ${LOG_DIR}"
cat "${SUMMARY}"
