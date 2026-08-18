# HTTP ↔ gRPC shim + e2e bench (personal branch only)

Experimental `cuopt_server.compat` package: legacy JSON validation/convert,
thin HTTP→gRPC shim, and timed e2e compare (legacy / shim / client-native /
client-json).

**Not intended for upstream PR.**

The shim speaks the legacy REST shape, so `cuopt_sh` / `CuOptServiceSelfHostClient`
work against it unchanged: `POST /cuopt/request` returns `{"reqId": ...}`,
`GET /cuopt/solution/{id}` returns the full `response.solver_response` envelope
(and a bare `{"reqId"}` while the job is pending). Add `?timings=1` to the POST
to get server-side stage timings in the JSON body; add `?timings=1` to the GET
to get result-path timings in the `X-Cuopt-Timings` header
(`status_ms`, `result_ms`, `map_ms`, `json_encode_ms`, `response_bytes`).
Timings are off by default because clients treat a single-key `reqId` POST body
as "still pending".

On a single GPU with large instances, the harness runs each path in
`ORDER` **sequentially** (start servers → N iterations → tear down / free
GPU → next path). Legacy HTTP and gRPC both need a large RMM pool and
cannot share an ~8 GiB GPU concurrently.

Each path does an **untimed warmup** until the first successful solve, then
records only successful timed iterations (failures are retried / skipped,
not written to CSV).

 Keep on a personal fork branch
(`feature/http-grpc-shim` on `github.com/tmckayus/cuopt`).

## Prerequisites

- Working cuOpt install with `cuopt_grpc_server` and Python packages
  (`cuopt`, `cuopt_server`, `msgpack`, `msgpack_numpy`, `fastapi`, `uvicorn`)
- This branch checked out (so `python/cuopt_server/.../compat` is on
  `PYTHONPATH`, or install editable)
- Problem file: numpy msgpack LP, e.g. `L2CTA3D.numpy.msgpack` (~616 MiB).
  Copy it to the target machine separately (not in git).
- Enough GPU memory for the instance (L2CTA3D needs
  `CUOPT_GIGABYTES_PER_PROC=6` on an ~8 GiB GPU)

## Cut-and-paste: loop bench on another machine

```bash
# --- once: get the branch ---
git clone git@github.com:tmckayus/cuopt.git
cd cuopt
git fetch origin feature/http-grpc-shim
git checkout feature/http-grpc-shim

# Activate your cuOpt env (example)
# conda activate cuopt   # or: source /path/to/.cuopt_env2/bin/activate

# Prefer this tree's cuopt_server (compat + bool coerce) over site-packages
export PYTHONPATH="$(pwd)/python/cuopt_server:${PYTHONPATH:-}"
export PATH="$(dirname $(which python)):${PATH}"   # ensure cuopt_grpc_server is found

# --- data (copy from the other machine; adjust path) ---
export MSGPACK_FILE=/path/to/L2CTA3D.numpy.msgpack
test -f "$MSGPACK_FILE"

# --- knobs ---
export ITERATIONS=5
export TIME_LIMIT=30
# Paths: legacy | shim | client-native (alias: client) | client-json
export ORDER=legacy,shim,client-native,client-json
export CUOPT_GIGABYTES_PER_PROC=6
export MAX_MESSAGE_MB=1024
export OUT_DIR="$(pwd)/bench_out"
export COOLDOWN=2
export WARMUP=1   # untimed successful run first; set 0 to disable

# --- run (starts servers, loops, writes CSV, tears down) ---
chmod +x python/cuopt_server/cuopt_server/compat/run_loop_bench.sh
python/cuopt_server/cuopt_server/compat/run_loop_bench.sh
```

Outputs under `OUT_DIR`:

- `e2e_runs_<stamp>.csv` — one row per **successful** timed iteration
- `e2e_avg_<stamp>.csv` — mean/stdev per path
- `logs_<stamp>/` — legacy / grpc / shim / compare logs

## Client-only (no HTTP servers)

```bash
export PYTHONPATH="$(pwd)/python/cuopt_server:${PYTHONPATH:-}"
export CUOPT_GIGABYTES_PER_PROC=6
cuopt_grpc_server --port 18601 --workers 1 --max-message-mb 1024 &
sleep 2

# Native gRPC (no problem/result JSON) and/or JSON round-trip client
python -m cuopt_server.compat.compare_e2e_solve \
  --msgpack-file "$MSGPACK_FILE" \
  --time-limit 30 \
  --iterations 5 \
  --order client-native,client-json \
  --grpc-host 127.0.0.1 --grpc-port 18601 \
  --csv ./bench_out/client_runs.csv \
  --csv-summary ./bench_out/client_avg.csv

fuser -k 18601/tcp 2>/dev/null || true
```

`client-native` is dict→DataModel→gRPC→Solution (no JSON). `client-json`
adds result→legacy JSON encode (and problem JSON decode when the problem is
small enough; huge LPs skip problem JSON to avoid OOM and still time the
result JSON).

## Manual single-pass compare

Same module without `--iterations` (defaults to 1). Start servers yourself,
then pass `--legacy-url` / `--shim-url` / `--grpc-port` as needed.
