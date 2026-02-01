/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/gpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <utilities/logger.hpp>

namespace cuopt::linear_programming {

// ============================================================================
// Remote execution stubs (placeholder implementations)
// ============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
  cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  CUOPT_LOG_INFO(
    "solve_lp_remote (CPU problem) stub called - returning dummy solution for testing");

  // TODO: Implement actual remote LP solving via gRPC
  // For now, return a dummy solution with fake data (allows testing the full flow)
  i_t n_vars        = cpu_problem.get_n_variables();
  i_t n_constraints = cpu_problem.get_n_constraints();

  std::vector<f_t> primal_solution(n_vars, 0.0);
  std::vector<f_t> dual_solution(n_constraints, 0.0);
  std::vector<f_t> reduced_cost(n_vars, 0.0);

  // Create fake warm start data with recognizable non-zero values for testing
  std::vector<f_t> current_primal_ws(n_vars, 1.1);       // Fill with 1.1
  std::vector<f_t> current_dual_ws(n_constraints, 2.2);  // Fill with 2.2
  std::vector<f_t> initial_primal_avg_ws(n_vars, 3.3);
  std::vector<f_t> initial_dual_avg_ws(n_constraints, 4.4);
  std::vector<f_t> current_ATY_ws(n_vars, 5.5);
  std::vector<f_t> sum_primal_ws(n_vars, 6.6);
  std::vector<f_t> sum_dual_ws(n_constraints, 7.7);
  std::vector<f_t> last_restart_primal_ws(n_vars, 8.8);
  std::vector<f_t> last_restart_dual_ws(n_constraints, 9.9);

  auto solution = std::make_unique<cpu_lp_solution_t<i_t, f_t>>(
    std::move(primal_solution),
    std::move(dual_solution),
    std::move(reduced_cost),
    pdlp_termination_status_t::Optimal,  // Fake optimal status
    0.0,                                 // Primal objective (zero solution)
    0.0,                                 // Dual objective (zero solution)
    0.01,                                // Dummy solve time
    0.001,                               // l2_primal_residual
    0.002,                               // l2_dual_residual
    0.003,                               // gap
    42,                                  // num_iterations
    true,                                // solved_by_pdlp
    std::move(current_primal_ws),
    std::move(current_dual_ws),
    std::move(initial_primal_avg_ws),
    std::move(initial_dual_avg_ws),
    std::move(current_ATY_ws),
    std::move(sum_primal_ws),
    std::move(sum_dual_ws),
    std::move(last_restart_primal_ws),
    std::move(last_restart_dual_ws),
    99.1,  // initial_primal_weight
    99.2,  // initial_step_size
    100,   // total_pdlp_iterations
    200,   // total_pdhg_iterations
    99.3,  // last_candidate_kkt_score
    99.4,  // last_restart_kkt_score
    99.5,  // sum_solution_weight
    10     // iterations_since_last_restart
  );

  return solution;
}

template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
  cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  mip_solver_settings_t<i_t, f_t> const& settings)
{
  CUOPT_LOG_INFO(
    "solve_mip_remote (CPU problem) stub called - returning dummy solution for testing");

  // TODO: Implement actual remote MIP solving via gRPC
  // For now, return a dummy solution with fake data (allows testing the full flow)
  i_t n_vars = cpu_problem.get_n_variables();

  std::vector<f_t> solution(n_vars, 0.0);
  auto mip_solution = std::make_unique<cpu_mip_solution_t<i_t, f_t>>(
    std::move(solution),
    mip_termination_status_t::Optimal,  // Fake optimal status
    0.0,                                // Objective value (zero solution)
    0.0,                                // MIP gap
    0.0,                                // Solution bound
    0.01,                               // Total solve time
    0.0,                                // Presolve time
    0.0,                                // Max constraint violation
    0.0,                                // Max int violation
    0.0,                                // Max variable bound violation
    0,                                  // Number of nodes
    0);                                 // Number of simplex iterations

  return mip_solution;
}

// ============================================================================
// Remote execution for GPU problems (converts to CPU then calls CPU remote)
// ============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
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

  // Call CPU remote solver (returns unique_ptr<cpu_lp_solution_t>)
  auto cpu_solution_interface = solve_lp_remote(cpu_problem, settings);

  // Convert CPU solution back to GPU solution (since we started with a GPU problem)
  auto* cpu_solution_ptr = dynamic_cast<cpu_lp_solution_t<i_t, f_t>*>(cpu_solution_interface.get());
  if (!cpu_solution_ptr) {
    throw cuopt::logic_error("Failed to cast CPU solution interface to cpu_lp_solution_t",
                             cuopt::error_type_t::RuntimeError);
  }

  // Convert to GPU solution and wrap in interface
  // Use the per-thread default stream for the conversion
  auto gpu_solution = cpu_solution_ptr->to_gpu_solution(rmm::cuda_stream_per_thread);
  return std::make_unique<gpu_lp_solution_t<i_t, f_t>>(std::move(gpu_solution));
}

template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
  gpu_optimization_problem_t<i_t, f_t>& gpu_problem,
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

  // Call CPU remote solver (returns unique_ptr<cpu_mip_solution_t>)
  auto cpu_solution_interface = solve_mip_remote(cpu_problem, settings);

  // Convert CPU solution back to GPU solution (since we started with a GPU problem)
  auto* cpu_solution_ptr =
    dynamic_cast<cpu_mip_solution_t<i_t, f_t>*>(cpu_solution_interface.get());
  if (!cpu_solution_ptr) {
    throw cuopt::logic_error("Failed to cast CPU solution interface to cpu_mip_solution_t",
                             cuopt::error_type_t::RuntimeError);
  }

  // Convert to GPU solution and wrap in interface
  // Use the per-thread default stream for the conversion
  auto gpu_solution = cpu_solution_ptr->to_gpu_solution(rmm::cuda_stream_per_thread);
  return std::make_unique<gpu_mip_solution_t<i_t, f_t>>(std::move(gpu_solution));
}

// Explicit template instantiations for remote execution stubs
template std::unique_ptr<lp_solution_interface_t<int, double>> solve_lp_remote(
  cpu_optimization_problem_t<int, double>&, pdlp_solver_settings_t<int, double> const&);

template std::unique_ptr<mip_solution_interface_t<int, double>> solve_mip_remote(
  cpu_optimization_problem_t<int, double>&, mip_solver_settings_t<int, double> const&);

template std::unique_ptr<lp_solution_interface_t<int, double>> solve_lp_remote(
  gpu_optimization_problem_t<int, double>&, pdlp_solver_settings_t<int, double> const&);

template std::unique_ptr<mip_solution_interface_t<int, double>> solve_mip_remote(
  gpu_optimization_problem_t<int, double>&, mip_solver_settings_t<int, double> const&);

}  // namespace cuopt::linear_programming
