/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/optimization_problem_solution_interface.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <memory>
#include <mps_parser/data_model_view.hpp>
#include <raft/core/handle.hpp>
#include <string>
#include <utility>
#include <vector>

namespace cuopt {
namespace cython {

// aggregate for call_solve() return type
// to be exposed to cython:
struct solver_ret_t {
  linear_programming::problem_category_t problem_type;

  // Solution interface pointers (only one will be non-null based on problem_type)
  linear_programming::lp_solution_interface_t<int, double>* lp_solution;
  linear_programming::mip_solution_interface_t<int, double>* mip_solution;

  // NOTE: No destructor - Python/Cython will manage solution pointer lifecycle using helper
  // functions
};

// Helper functions for Python/Cython to delete solution pointers
// (Cython has trouble with templated pointer deletion)
inline void delete_lp_solution(linear_programming::lp_solution_interface_t<int, double>* ptr)
{
  delete ptr;
}

inline void delete_mip_solution(linear_programming::mip_solution_interface_t<int, double>* ptr)
{
  delete ptr;
}

// Wrapper functions to expose the API to cython.

// Call solve_lp and return solution interface pointer
linear_programming::lp_solution_interface_t<int, double>* call_solve_lp(
  linear_programming::optimization_problem_interface_t<int, double>* problem_interface,
  linear_programming::pdlp_solver_settings_t<int, double>& solver_settings,
  bool is_batch_mode = false);

// Call solve_mip and return solution interface pointer
linear_programming::mip_solution_interface_t<int, double>* call_solve_mip(
  linear_programming::optimization_problem_interface_t<int, double>* problem_interface,
  linear_programming::mip_solver_settings_t<int, double>& solver_settings);

// Main solve entry point from Python
std::unique_ptr<solver_ret_t> call_solve(cuopt::mps_parser::data_model_view_t<int, double>*,
                                         linear_programming::solver_settings_t<int, double>*,
                                         unsigned int flags = cudaStreamNonBlocking,
                                         bool is_batch_mode = false);

std::pair<std::vector<std::unique_ptr<solver_ret_t>>, double> call_batch_solve(
  std::vector<cuopt::mps_parser::data_model_view_t<int, double>*>,
  linear_programming::solver_settings_t<int, double>*);

}  // namespace cython
}  // namespace cuopt
