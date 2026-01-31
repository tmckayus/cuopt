/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <utilities/logger.hpp>

namespace cuopt::linear_programming {

// ============================================================================
// Remote execution stubs (placeholder implementations)
// ============================================================================

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_remote(
  cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  CUOPT_LOG_ERROR("solve_lp_remote (CPU problem) stub called - remote solving not yet implemented");

  // TODO: Implement actual remote LP solving via gRPC
  // For now, throw an error instead of trying to create a fake solution
  // (which would require CUDA initialization on a potentially CPU-only host)
  throw cuopt::logic_error(
    "Remote LP solving is not yet implemented. "
    "Please use local solving (unset CUOPT_REMOTE_HOST/PORT) or wait for remote execution support.",
    cuopt::error_type_t::RuntimeError);
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                                          mip_solver_settings_t<i_t, f_t> const& settings)
{
  CUOPT_LOG_ERROR(
    "solve_mip_remote (CPU problem) stub called - remote solving not yet implemented");

  // TODO: Implement actual remote MIP solving via gRPC
  // For now, throw an error instead of trying to create a fake solution
  // (which would require CUDA initialization on a potentially CPU-only host)
  throw cuopt::logic_error(
    "Remote MIP solving is not yet implemented. "
    "Please use local solving (unset CUOPT_REMOTE_HOST/PORT) or wait for remote execution support.",
    cuopt::error_type_t::RuntimeError);
}

// ============================================================================
// Remote execution for GPU problems (converts to CPU then calls CPU remote)
// ============================================================================

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_remote(
  gpu_optimization_problem_t<i_t, f_t>& gpu_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  CUOPT_LOG_INFO("solve_lp_remote (GPU problem) - converting to CPU for remote execution");

  // Convert GPU problem to CPU problem (copies device data to host)
  auto cpu_problem = cpu_optimization_problem_t<i_t, f_t>(nullptr);  // No CUDA resources for remote

  // Copy scalar properties
  cpu_problem.set_maximize(gpu_problem.get_sense());
  cpu_problem.set_objective_offset(gpu_problem.get_objective_offset());
  cpu_problem.set_problem_category(gpu_problem.get_problem_category());

  // Copy names
  cpu_problem.set_problem_name(gpu_problem.get_problem_name());
  cpu_problem.set_objective_name(gpu_problem.get_objective_name());
  cpu_problem.set_variable_names(gpu_problem.get_variable_names());
  cpu_problem.set_row_names(gpu_problem.get_row_names());

  // Copy objective coefficients
  auto obj_coeffs = gpu_problem.get_objective_coefficients_host();
  if (!obj_coeffs.empty()) {
    cpu_problem.set_objective_coefficients(obj_coeffs.data(), obj_coeffs.size());
  }

  // Copy constraint matrix (CSR format)
  auto matrix_values  = gpu_problem.get_constraint_matrix_values_host();
  auto matrix_indices = gpu_problem.get_constraint_matrix_indices_host();
  auto matrix_offsets = gpu_problem.get_constraint_matrix_offsets_host();
  if (!matrix_values.empty()) {
    cpu_problem.set_csr_constraint_matrix(matrix_values.data(),
                                          matrix_values.size(),
                                          matrix_indices.data(),
                                          matrix_indices.size(),
                                          matrix_offsets.data(),
                                          matrix_offsets.size());
  }

  // Copy constraint bounds
  auto constraint_lb = gpu_problem.get_constraint_lower_bounds_host();
  auto constraint_ub = gpu_problem.get_constraint_upper_bounds_host();
  if (!constraint_lb.empty()) {
    cpu_problem.set_constraint_lower_bounds(constraint_lb.data(), constraint_lb.size());
  }
  if (!constraint_ub.empty()) {
    cpu_problem.set_constraint_upper_bounds(constraint_ub.data(), constraint_ub.size());
  }

  // Copy variable bounds
  auto var_lb = gpu_problem.get_variable_lower_bounds_host();
  auto var_ub = gpu_problem.get_variable_upper_bounds_host();
  if (!var_lb.empty()) { cpu_problem.set_variable_lower_bounds(var_lb.data(), var_lb.size()); }
  if (!var_ub.empty()) { cpu_problem.set_variable_upper_bounds(var_ub.data(), var_ub.size()); }

  // Copy variable types
  auto var_types = gpu_problem.get_variable_types_host();
  if (!var_types.empty()) { cpu_problem.set_variable_types(var_types.data(), var_types.size()); }

  // Copy quadratic objective if present
  if (gpu_problem.has_quadratic_objective()) {
    auto quad_offsets = gpu_problem.get_quadratic_objective_offsets();
    auto quad_indices = gpu_problem.get_quadratic_objective_indices();
    auto quad_values  = gpu_problem.get_quadratic_objective_values();
    cpu_problem.set_quadratic_objective_matrix(quad_values.data(),
                                               quad_values.size(),
                                               quad_indices.data(),
                                               quad_indices.size(),
                                               quad_offsets.data(),
                                               quad_offsets.size());
  }

  // Call CPU remote solver
  return solve_lp_remote(cpu_problem, settings);
}

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(gpu_optimization_problem_t<i_t, f_t>& gpu_problem,
                                          mip_solver_settings_t<i_t, f_t> const& settings)
{
  CUOPT_LOG_INFO("solve_mip_remote (GPU problem) - converting to CPU for remote execution");

  // Convert GPU problem to CPU problem (copies device data to host)
  auto cpu_problem = cpu_optimization_problem_t<i_t, f_t>(nullptr);  // No CUDA resources for remote

  // Copy scalar properties
  cpu_problem.set_maximize(gpu_problem.get_sense());
  cpu_problem.set_objective_offset(gpu_problem.get_objective_offset());
  cpu_problem.set_problem_category(gpu_problem.get_problem_category());

  // Copy names
  cpu_problem.set_problem_name(gpu_problem.get_problem_name());
  cpu_problem.set_objective_name(gpu_problem.get_objective_name());
  cpu_problem.set_variable_names(gpu_problem.get_variable_names());
  cpu_problem.set_row_names(gpu_problem.get_row_names());

  // Copy objective coefficients
  auto obj_coeffs = gpu_problem.get_objective_coefficients_host();
  if (!obj_coeffs.empty()) {
    cpu_problem.set_objective_coefficients(obj_coeffs.data(), obj_coeffs.size());
  }

  // Copy constraint matrix (CSR format)
  auto matrix_values  = gpu_problem.get_constraint_matrix_values_host();
  auto matrix_indices = gpu_problem.get_constraint_matrix_indices_host();
  auto matrix_offsets = gpu_problem.get_constraint_matrix_offsets_host();
  if (!matrix_values.empty()) {
    cpu_problem.set_csr_constraint_matrix(matrix_values.data(),
                                          matrix_values.size(),
                                          matrix_indices.data(),
                                          matrix_indices.size(),
                                          matrix_offsets.data(),
                                          matrix_offsets.size());
  }

  // Copy constraint bounds
  auto constraint_lb = gpu_problem.get_constraint_lower_bounds_host();
  auto constraint_ub = gpu_problem.get_constraint_upper_bounds_host();
  if (!constraint_lb.empty()) {
    cpu_problem.set_constraint_lower_bounds(constraint_lb.data(), constraint_lb.size());
  }
  if (!constraint_ub.empty()) {
    cpu_problem.set_constraint_upper_bounds(constraint_ub.data(), constraint_ub.size());
  }

  // Copy variable bounds
  auto var_lb = gpu_problem.get_variable_lower_bounds_host();
  auto var_ub = gpu_problem.get_variable_upper_bounds_host();
  if (!var_lb.empty()) { cpu_problem.set_variable_lower_bounds(var_lb.data(), var_lb.size()); }
  if (!var_ub.empty()) { cpu_problem.set_variable_upper_bounds(var_ub.data(), var_ub.size()); }

  // Copy variable types
  auto var_types = gpu_problem.get_variable_types_host();
  if (!var_types.empty()) { cpu_problem.set_variable_types(var_types.data(), var_types.size()); }

  // Copy quadratic objective if present
  if (gpu_problem.has_quadratic_objective()) {
    auto quad_offsets = gpu_problem.get_quadratic_objective_offsets();
    auto quad_indices = gpu_problem.get_quadratic_objective_indices();
    auto quad_values  = gpu_problem.get_quadratic_objective_values();
    cpu_problem.set_quadratic_objective_matrix(quad_values.data(),
                                               quad_values.size(),
                                               quad_indices.data(),
                                               quad_indices.size(),
                                               quad_offsets.data(),
                                               quad_offsets.size());
  }

  // Call CPU remote solver
  return solve_mip_remote(cpu_problem, settings);
}

// Explicit template instantiations for remote execution stubs
template optimization_problem_solution_t<int, double> solve_lp_remote(
  cpu_optimization_problem_t<int, double>&, pdlp_solver_settings_t<int, double> const&);

template mip_solution_t<int, double> solve_mip_remote(cpu_optimization_problem_t<int, double>&,
                                                      mip_solver_settings_t<int, double> const&);

template optimization_problem_solution_t<int, double> solve_lp_remote(
  gpu_optimization_problem_t<int, double>&, pdlp_solver_settings_t<int, double> const&);

template mip_solution_t<int, double> solve_mip_remote(gpu_optimization_problem_t<int, double>&,
                                                      mip_solver_settings_t<int, double> const&);

}  // namespace cuopt::linear_programming
