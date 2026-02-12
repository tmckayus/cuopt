# C API MPS Benchmark

Runs the same workflow as `run_mps_benchmarks.py` using the **C API** only (no Python). Produces CSV in the same format so you can use `compare_benchmark_results.py` and avoid Python timing jitter.

## Build

From the **repo root**:

```bash
./benchmarks/linear_programming/c_api_benchmark/build.sh
```

Or from the cpp build directory:

```bash
cd cpp/build
cmake ..
make c_api_mps_benchmark
```

Binary: `cpp/build/c_api_mps_benchmark`

## Usage

```text
c_api_mps_benchmark <mps_directory> <output_csv> [OPTIONS]
```

Options:

- `--iterations N`   Number of runs per file, results averaged (default: 5)
- `--time-limit SEC` Solver time limit per solve
- `--method M`       0=Concurrent, 1=PDLP, 2=DualSimplex, 3=Barrier (default: 0)

## Example

```bash
# From cpp/build after building
./c_api_mps_benchmark /path/to/netlib_mps ./results_capi.csv --iterations 5

# Compare with a Python-run baseline (no Python jitter in the C API run)
python compare_benchmark_results.py netlib_python.csv results_capi.csv
```

## CSV output

Same columns as `run_mps_benchmarks.py`: `file`, `status`, `objective_value`, `wall_time_seconds`, `solver_time_seconds`, `solved_by_method`, and the optional `cython_*` columns. For C API runs, `solved_by_method` and all `cython_*` columns are left empty.
