#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start legacy HTTP + gRPC + HTTP→gRPC shim, run timed loop, write CSVs, tear down.
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

need_legacy=0
need_shim=0
need_grpc=0
IFS=',' read -ra _ORDER_PARTS <<< "${ORDER}"
for p in "${_ORDER_PARTS[@]}"; do
  p="$(echo "$p" | tr -d '[:space:]')"
  case "$p" in
    legacy) need_legacy=1 ;;
    shim) need_shim=1; need_grpc=1 ;;
    client) need_grpc=1 ;;
  esac
done

PIDS=()
cleanup() {
  echo "Cleaning up servers..."
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  # workers may outlive parent briefly
  fuser -k "${LEGACY_PORT}/tcp" 2>/dev/null || true
  fuser -k "${GRPC_PORT}/tcp" 2>/dev/null || true
  fuser -k "${SHIM_PORT}/tcp" 2>/dev/null || true
  sleep 2
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

echo "MSGPACK_FILE=${MSGPACK_FILE}"
echo "ITERATIONS=${ITERATIONS} TIME_LIMIT=${TIME_LIMIT} ORDER=${ORDER}"
echo "CUOPT_GIGABYTES_PER_PROC=${CUOPT_GIGABYTES_PER_PROC}"
echo "CSV=${CSV}"
echo "SUMMARY=${SUMMARY}"

if [[ "${need_legacy}" -eq 1 ]]; then
  echo "Starting legacy HTTP server on ${LEGACY_PORT}..."
  nohup python -m cuopt_server.cuopt_service --port "${LEGACY_PORT}" \
    >"${LOG_DIR}/legacy.log" 2>&1 &
  PIDS+=($!)
  wait_http "${LEGACY_URL}/cuopt/health" "legacy"
fi

if [[ "${need_grpc}" -eq 1 ]]; then
  echo "Starting gRPC server on ${GRPC_PORT}..."
  nohup env CUOPT_GIGABYTES_PER_PROC="${CUOPT_GIGABYTES_PER_PROC}" \
    cuopt_grpc_server --port "${GRPC_PORT}" --workers 1 \
    --max-message-mb "${MAX_MESSAGE_MB}" \
    >"${LOG_DIR}/grpc.log" 2>&1 &
  PIDS+=($!)
  wait_tcp 127.0.0.1 "${GRPC_PORT}" "grpc"
fi

if [[ "${need_shim}" -eq 1 ]]; then
  echo "Starting HTTP→gRPC shim on ${SHIM_PORT}..."
  nohup python -m cuopt_server.compat.shim \
    --http-host 127.0.0.1 --http-port "${SHIM_PORT}" \
    --grpc-host 127.0.0.1 --grpc-port "${GRPC_PORT}" \
    >"${LOG_DIR}/shim.log" 2>&1 &
  PIDS+=($!)
  wait_http "${SHIM_URL}/health" "shim"
fi

ARGS=(
  --msgpack-file "${MSGPACK_FILE}"
  --time-limit "${TIME_LIMIT}"
  --iterations "${ITERATIONS}"
  --cooldown "${COOLDOWN}"
  --poll-timeout "${POLL_TIMEOUT}"
  --order "${ORDER}"
  --csv "${CSV}"
  --csv-summary "${SUMMARY}"
)
if [[ "${need_legacy}" -eq 1 ]]; then
  ARGS+=(--legacy-url "${LEGACY_URL}")
fi
if [[ "${need_shim}" -eq 1 ]]; then
  ARGS+=(--shim-url "${SHIM_URL}")
fi
if [[ "${need_grpc}" -eq 1 ]]; then
  ARGS+=(--grpc-host 127.0.0.1 --grpc-port "${GRPC_PORT}")
fi

echo "Running compare loop..."
python -m cuopt_server.compat.compare_e2e_solve "${ARGS[@]}" | tee "${LOG_DIR}/compare.log"

echo
echo "Done."
echo "  per-run CSV : ${CSV}"
echo "  averages CSV: ${SUMMARY}"
echo "  logs        : ${LOG_DIR}"
