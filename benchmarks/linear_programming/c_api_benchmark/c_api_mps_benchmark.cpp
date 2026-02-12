/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * C API MPS benchmark: same workflow as run_mps_benchmarks.py but without Python.
 * Produces CSV in the same format for use with compare_benchmark_results.py.
 *
 * Build: from cpp/build: cmake .. && make c_api_mps_benchmark
 * Or: ./build.sh from this directory (see build.sh).
 *
 * Usage:
 *   c_api_mps_benchmark <mps_directory> <output_csv> [--iterations N] [--time-limit SEC]
 *
 * CSV columns: file, status, objective_value, wall_time_seconds, solver_time_seconds,
 *              solved_by_method, cython_total_sec, cython_problem_creation_sec,
 *              cython_solve_sec, cython_solution_creation_sec,
 *              c_api_total_sec, c_api_problem_creation_sec, c_api_solve_sec,
 * c_api_solution_creation_sec (solved_by_method and cython_* are empty for C API runs; c_api_* are
 * filled by this benchmark.)
 */

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/cuopt_c.h>

#include <dirent.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static const char* termination_status_to_status_string(cuopt_int_t status)
{
  switch (status) {
    case CUOPT_TERIMINATION_STATUS_OPTIMAL: return "Optimal";
    case CUOPT_TERIMINATION_STATUS_INFEASIBLE: return "Infeasible";
    case CUOPT_TERIMINATION_STATUS_UNBOUNDED: return "Unbounded";
    case CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT: return "IterationLimit";
    case CUOPT_TERIMINATION_STATUS_TIME_LIMIT: return "TimeLimit";
    case CUOPT_TERIMINATION_STATUS_NUMERICAL_ERROR: return "NumericalError";
    case CUOPT_TERIMINATION_STATUS_PRIMAL_FEASIBLE: return "PrimalFeasible";
    case CUOPT_TERIMINATION_STATUS_FEASIBLE_FOUND: return "FeasibleFound";
    case CUOPT_TERIMINATION_STATUS_CONCURRENT_LIMIT: return "ConcurrentLimit";
    case CUOPT_TERIMINATION_STATUS_WORK_LIMIT: return "WorkLimit";
    default: return "Unknown";
  }
}

static std::vector<std::string> find_mps_files(const std::string& directory)
{
  std::vector<std::string> out;
  DIR* dir = opendir(directory.c_str());
  if (!dir) return out;

  struct dirent* ent;
  while ((ent = readdir(dir)) != nullptr) {
    std::string name(ent->d_name);
    if (name == "." || name == "..") continue;
    size_t n = name.size();
    if (n >= 4 && (name.compare(n - 4, 4, ".mps") == 0 || name.compare(n - 4, 4, ".MPS") == 0)) {
      out.push_back(directory + "/" + name);
    }
  }
  closedir(dir);
  std::sort(out.begin(), out.end());
  return out;
}

static std::string basename(const std::string& path)
{
  size_t i = path.find_last_of("/\\");
  return i == std::string::npos ? path : path.substr(i + 1);
}

struct RunResult {
  std::string status;
  double objective_value;
  double wall_time_sec;
  double solver_time_sec;
  double c_api_problem_creation_sec;
  double c_api_solve_sec;
  double c_api_solution_creation_sec;
  double c_api_total_sec;
  bool ok;
};

static RunResult run_one(const std::string& mps_path, double time_limit, int method)
{
  RunResult r;
  r.objective_value             = std::numeric_limits<double>::quiet_NaN();
  r.wall_time_sec               = 0.0;
  r.solver_time_sec             = std::numeric_limits<double>::quiet_NaN();
  r.c_api_problem_creation_sec  = std::numeric_limits<double>::quiet_NaN();
  r.c_api_solve_sec             = std::numeric_limits<double>::quiet_NaN();
  r.c_api_solution_creation_sec = std::numeric_limits<double>::quiet_NaN();
  r.c_api_total_sec             = std::numeric_limits<double>::quiet_NaN();
  r.status                      = "Error";
  r.ok                          = false;

  cuOptOptimizationProblem problem = nullptr;
  cuOptSolverSettings settings     = nullptr;
  cuOptSolution solution           = nullptr;

  cuopt_int_t st = cuOptReadProblem(mps_path.c_str(), &problem);
  if (st != CUOPT_SUCCESS) {
    if (st == CUOPT_MPS_FILE_ERROR)
      r.status = "Error(MPS_FILE)";
    else if (st == CUOPT_MPS_PARSE_ERROR)
      r.status = "Error(MPS_PARSE)";
    return r;
  }

  st = cuOptCreateSolverSettings(&settings);
  if (st != CUOPT_SUCCESS) {
    cuOptDestroyProblem(&problem);
    return r;
  }

  st = cuOptSetIntegerParameter(settings, CUOPT_METHOD, method);
  if (st != CUOPT_SUCCESS) {
    cuOptDestroyProblem(&problem);
    cuOptDestroySolverSettings(&settings);
    return r;
  }
  st = cuOptSetIntegerParameter(settings, CUOPT_PRESOLVE, CUOPT_PRESOLVE_OFF);
  if (st != CUOPT_SUCCESS) {
    cuOptDestroyProblem(&problem);
    cuOptDestroySolverSettings(&settings);
    return r;
  }
  if (time_limit > 0 && time_limit < 1e30) {
    st = cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, static_cast<cuopt_float_t>(time_limit));
    if (st != CUOPT_SUCCESS) {
      cuOptDestroyProblem(&problem);
      cuOptDestroySolverSettings(&settings);
      return r;
    }
  }

  auto wall_start = std::chrono::steady_clock::now();
  st              = cuOptSolve(problem, settings, &solution);
  auto wall_end   = std::chrono::steady_clock::now();
  r.wall_time_sec = std::chrono::duration<double>(wall_end - wall_start).count();

  if (st != CUOPT_SUCCESS) {
    r.status = "Error";
    cuOptDestroyProblem(&problem);
    cuOptDestroySolverSettings(&settings);
    cuOptDestroySolution(&solution);
    return r;
  }

  cuopt_int_t term_status  = -1;
  cuopt_float_t solve_time = 0.0f;
  cuopt_float_t obj_val    = 0.0f;

  cuOptGetTerminationStatus(solution, &term_status);
  cuOptGetSolveTime(solution, &solve_time);
  cuOptGetObjectiveValue(solution, &obj_val);

  cuopt_float_t pc = 0.0f, sol = 0.0f, solc = 0.0f, tot = 0.0f;
  if (cuOptGetLastSolveTimings(&pc, &sol, &solc, &tot) == CUOPT_SUCCESS) {
    r.c_api_problem_creation_sec  = static_cast<double>(pc);
    r.c_api_solve_sec             = static_cast<double>(sol);
    r.c_api_solution_creation_sec = static_cast<double>(solc);
    r.c_api_total_sec             = static_cast<double>(tot);
  }

  r.status          = termination_status_to_status_string(term_status);
  r.solver_time_sec = static_cast<double>(solve_time);
  r.objective_value = static_cast<double>(obj_val);
  r.ok              = true;

  cuOptDestroyProblem(&problem);
  cuOptDestroySolverSettings(&settings);
  cuOptDestroySolution(&solution);
  return r;
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <mps_directory> <output_csv> [--iterations N] [--time-limit SEC] [--method M]\n"
              << "  method: 0=Concurrent, 1=PDLP, 2=DualSimplex, 3=Barrier (default 0)\n";
    return 1;
  }

  std::string mps_dir    = argv[1];
  std::string output_csv = argv[2];
  int iterations         = 5;
  double time_limit      = std::numeric_limits<double>::infinity();
  int method             = CUOPT_METHOD_CONCURRENT;

  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      iterations = atoi(argv[++i]);
      if (iterations < 1) iterations = 1;
    } else if (strcmp(argv[i], "--time-limit") == 0 && i + 1 < argc) {
      time_limit = atof(argv[++i]);
    } else if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
      method = atoi(argv[++i]);
    }
  }

  auto files = find_mps_files(mps_dir);
  if (files.empty()) {
    std::cerr << "No MPS files found in " << mps_dir << "\n";
    return 1;
  }

  std::ofstream out(output_csv);
  if (!out) {
    std::cerr << "Cannot open " << output_csv << " for writing\n";
    return 1;
  }

  // CSV header: same as run_mps_benchmarks.py plus C API stage timings
  out
    << "file,status,objective_value,wall_time_seconds,solver_time_seconds,solved_by_method,"
    << "cython_total_sec,cython_problem_creation_sec,cython_solve_sec,cython_solution_creation_sec,"
    << "c_api_total_sec,c_api_problem_creation_sec,c_api_solve_sec,c_api_solution_creation_sec\n";

  std::cout << "Found " << files.size() << " MPS files. " << iterations
            << " iteration(s) per file.\n";

  for (size_t idx = 0; idx < files.size(); idx++) {
    const std::string& path = files[idx];
    std::string name        = basename(path);

    double sum_wall = 0.0, sum_solver = 0.0, sum_obj = 0.0;
    double sum_c_api_pc = 0.0, sum_c_api_solve = 0.0, sum_c_api_solc = 0.0, sum_c_api_total = 0.0;
    int valid       = 0;
    int c_api_count = 0;
    std::string last_status;
    double last_obj = std::numeric_limits<double>::quiet_NaN();

    for (int it = 0; it < iterations; it++) {
      RunResult r = run_one(path, time_limit, method);
      last_status = r.status;
      if (r.ok) {
        sum_wall += r.wall_time_sec;
        sum_solver += r.solver_time_sec;
        sum_obj += r.objective_value;
        last_obj = r.objective_value;
        if (!std::isnan(r.c_api_total_sec)) {
          sum_c_api_pc += r.c_api_problem_creation_sec;
          sum_c_api_solve += r.c_api_solve_sec;
          sum_c_api_solc += r.c_api_solution_creation_sec;
          sum_c_api_total += r.c_api_total_sec;
          c_api_count++;
        }
        valid++;
      }
    }

    double avg_wall = valid > 0 ? sum_wall / iterations : 0.0;
    double avg_solver =
      valid > 0 ? sum_solver / iterations : std::numeric_limits<double>::quiet_NaN();
    double avg_obj = valid > 0 ? sum_obj / iterations : last_obj;
    double avg_c_api_pc =
      (c_api_count > 0) ? sum_c_api_pc / c_api_count : std::numeric_limits<double>::quiet_NaN();
    double avg_c_api_solve =
      (c_api_count > 0) ? sum_c_api_solve / c_api_count : std::numeric_limits<double>::quiet_NaN();
    double avg_c_api_solc =
      (c_api_count > 0) ? sum_c_api_solc / c_api_count : std::numeric_limits<double>::quiet_NaN();
    double avg_c_api_total =
      (c_api_count > 0) ? sum_c_api_total / c_api_count : std::numeric_limits<double>::quiet_NaN();

    out << name << "," << last_status << ",";
    if (valid > 0 && !std::isnan(avg_obj))
      out << std::setprecision(15) << avg_obj;
    else
      out << "nan";
    out << "," << std::setprecision(10) << avg_wall << ",";
    if (valid > 0 && !std::isnan(avg_solver)) out << avg_solver;
    out << ",,,,,";  // solved_by_method and cython_* empty
    if (c_api_count > 0 && !std::isnan(avg_c_api_total)) {
      out << std::setprecision(10) << avg_c_api_total << "," << avg_c_api_pc << ","
          << avg_c_api_solve << "," << avg_c_api_solc;
    } else {
      out << ",,,,";  // c_api_total_sec, c_api_problem_creation_sec, c_api_solve_sec,
                      // c_api_solution_creation_sec
    }
    out << "\n";
    out.flush();

    std::cout << "[" << (idx + 1) << "/" << files.size() << "] " << name << " " << last_status
              << " obj=" << avg_obj << " wall=" << avg_wall << "s solver=" << avg_solver << "s\n";
  }

  std::cout << "Results written to " << output_csv << "\n";
  return 0;
}
