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
#include <rmm/device_buffer.hpp>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace cuopt {
namespace cython {

// GPU-backed LP solution struct (uses device memory)
struct linear_programming_ret_t {
  std::unique_ptr<rmm::device_buffer> primal_solution_;
  std::unique_ptr<rmm::device_buffer> dual_solution_;
  std::unique_ptr<rmm::device_buffer> reduced_cost_;
  /* -- PDLP Warm Start Data -- */
  std::unique_ptr<rmm::device_buffer> current_primal_solution_;
  std::unique_ptr<rmm::device_buffer> current_dual_solution_;
  std::unique_ptr<rmm::device_buffer> initial_primal_average_;
  std::unique_ptr<rmm::device_buffer> initial_dual_average_;
  std::unique_ptr<rmm::device_buffer> current_ATY_;
  std::unique_ptr<rmm::device_buffer> sum_primal_solutions_;
  std::unique_ptr<rmm::device_buffer> sum_dual_solutions_;
  std::unique_ptr<rmm::device_buffer> last_restart_duality_gap_primal_solution_;
  std::unique_ptr<rmm::device_buffer> last_restart_duality_gap_dual_solution_;
  double initial_primal_weight_;
  double initial_step_size_;
  int total_pdlp_iterations_;
  int total_pdhg_iterations_;
  double last_candidate_kkt_score_;
  double last_restart_kkt_score_;
  double sum_solution_weight_;
  int iterations_since_last_restart_;
  /* -- /PDLP Warm Start Data -- */

  linear_programming::pdlp_termination_status_t termination_status_;
  error_type_t error_status_;
  std::string error_message_;

  /*Termination stats*/
  double l2_primal_residual_;
  double l2_dual_residual_;
  double primal_objective_;
  double dual_objective_;
  double gap_;
  int nb_iterations_;
  double solve_time_;
  bool solved_by_pdlp_;
};

// CPU-backed LP solution struct (uses host memory)
struct cpu_linear_programming_ret_t {
  std::vector<double> primal_solution_;
  std::vector<double> dual_solution_;
  std::vector<double> reduced_cost_;
  /* -- PDLP Warm Start Data -- */
  std::vector<double> current_primal_solution_;
  std::vector<double> current_dual_solution_;
  std::vector<double> initial_primal_average_;
  std::vector<double> initial_dual_average_;
  std::vector<double> current_ATY_;
  std::vector<double> sum_primal_solutions_;
  std::vector<double> sum_dual_solutions_;
  std::vector<double> last_restart_duality_gap_primal_solution_;
  std::vector<double> last_restart_duality_gap_dual_solution_;
  double initial_primal_weight_;
  double initial_step_size_;
  int total_pdlp_iterations_;
  int total_pdhg_iterations_;
  double last_candidate_kkt_score_;
  double last_restart_kkt_score_;
  double sum_solution_weight_;
  int iterations_since_last_restart_;
  /* -- /PDLP Warm Start Data -- */

  linear_programming::pdlp_termination_status_t termination_status_;
  error_type_t error_status_;
  std::string error_message_;

  /*Termination stats*/
  double l2_primal_residual_;
  double l2_dual_residual_;
  double primal_objective_;
  double dual_objective_;
  double gap_;
  int nb_iterations_;
  double solve_time_;
  bool solved_by_pdlp_;
};

// GPU-backed MIP solution struct (uses device memory)
struct mip_ret_t {
  std::unique_ptr<rmm::device_buffer> solution_;

  linear_programming::mip_termination_status_t termination_status_;
  error_type_t error_status_;
  std::string error_message_;

  /*Termination stats*/
  double objective_;
  double mip_gap_;
  double solution_bound_;
  double total_solve_time_;
  double presolve_time_;
  double max_constraint_violation_;
  double max_int_violation_;
  double max_variable_bound_violation_;
  int nodes_;
  int simplex_iterations_;
};

// CPU-backed MIP solution struct (uses host memory)
struct cpu_mip_ret_t {
  std::vector<double> solution_;

  linear_programming::mip_termination_status_t termination_status_;
  error_type_t error_status_;
  std::string error_message_;

  /*Termination stats*/
  double objective_;
  double mip_gap_;
  double solution_bound_;
  double total_solve_time_;
  double presolve_time_;
  double max_constraint_violation_;
  double max_int_violation_;
  double max_variable_bound_violation_;
  int nodes_;
  int simplex_iterations_;
};

// Aggregate for call_solve() return type
// Uses std::variant to hold either GPU or CPU solution structs
struct solver_ret_t {
  linear_programming::problem_category_t problem_type;
  std::variant<linear_programming_ret_t, cpu_linear_programming_ret_t> lp_ret;
  std::variant<mip_ret_t, cpu_mip_ret_t> mip_ret;
};

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
