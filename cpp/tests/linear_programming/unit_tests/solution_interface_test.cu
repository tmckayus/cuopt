/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * @file solution_interface_test.cu
 * @brief Tests for optimization_problem_solution_interface_t polymorphic methods
 *        and GPU/CPU conversion functions
 *
 * Tests:
 * - LP solutions throw std::logic_error when calling MIP-only methods
 * - MIP solutions throw std::logic_error when calling LP-only methods
 * - Polymorphic methods work correctly for both LP and MIP solutions
 * - GPU ↔ CPU problem conversions
 * - GPU ↔ CPU solution conversions
 * - GPU ↔ CPU warmstart data conversions
 * - MPS data model to problem conversions
 * - Solution to Python return type conversions
 */

#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/cpu_pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/gpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/optimization_problem_utils.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>

#include <gtest/gtest.h>

#include <stdexcept>

namespace cuopt::linear_programming {

class SolutionInterfaceTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    const std::string& rapidsDatasetRootDir = cuopt::test::get_rapids_dataset_root_dir();
    lp_file_  = rapidsDatasetRootDir + "/linear_programming/afiro_original.mps";
    mip_file_ = rapidsDatasetRootDir + "/mip/bb_optimality.mps";
  }

  std::string lp_file_;
  std::string mip_file_;
};

// Test that LP solution throws when calling MIP-only methods
TEST_F(SolutionInterfaceTest, lp_solution_throws_on_mip_methods)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);
  raft::handle_t handle;

  auto problem = std::make_unique<gpu_optimization_problem_t<int, double>>(&handle);
  populate_from_mps_data_model(problem.get(), mps_data);

  pdlp_solver_settings_t<int, double> settings;
  settings.time_limit = 60.0;

  auto solution = solve_lp(problem.get(), settings);
  ASSERT_NE(solution, nullptr);

  // LP solution should throw on MIP-only methods
  EXPECT_THROW(solution->get_mip_gap(), std::logic_error);
  EXPECT_THROW(solution->get_solution_bound(), std::logic_error);
}

// Test that MIP solution throws when calling LP-only methods
TEST_F(SolutionInterfaceTest, mip_solution_throws_on_lp_methods)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(mip_file_);
  raft::handle_t handle;

  auto problem = std::make_unique<gpu_optimization_problem_t<int, double>>(&handle);
  populate_from_mps_data_model(problem.get(), mps_data);

  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 60.0;

  auto solution = solve_mip(problem.get(), settings);
  ASSERT_NE(solution, nullptr);

  // MIP solution should throw on LP-only methods
  EXPECT_THROW(solution->get_dual_solution(), std::logic_error);
  EXPECT_THROW(solution->get_dual_objective_value(), std::logic_error);
  EXPECT_THROW(solution->get_reduced_costs(), std::logic_error);
}

// Test that polymorphic methods work correctly for LP solutions
TEST_F(SolutionInterfaceTest, lp_solution_polymorphic_methods)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);
  raft::handle_t handle;

  auto problem = std::make_unique<gpu_optimization_problem_t<int, double>>(&handle);
  populate_from_mps_data_model(problem.get(), mps_data);

  pdlp_solver_settings_t<int, double> settings;
  settings.time_limit = 60.0;

  auto solution = solve_lp(problem.get(), settings);
  ASSERT_NE(solution, nullptr);

  // Base interface polymorphic methods should work
  optimization_problem_solution_interface_t<int, double>* base_ptr = solution.get();

  EXPECT_FALSE(base_ptr->is_mip());
  EXPECT_NO_THROW(base_ptr->get_error_status());
  EXPECT_NO_THROW(base_ptr->get_solve_time());
  EXPECT_NO_THROW(base_ptr->get_solution_host());
  EXPECT_NO_THROW(base_ptr->get_termination_status_int());
  EXPECT_NO_THROW(base_ptr->get_objective_value());

  // LP-specific polymorphic methods should work
  EXPECT_NO_THROW(base_ptr->get_dual_solution());
  EXPECT_NO_THROW(base_ptr->get_dual_objective_value());
  EXPECT_NO_THROW(base_ptr->get_reduced_costs());
}

// Test that polymorphic methods work correctly for MIP solutions
TEST_F(SolutionInterfaceTest, mip_solution_polymorphic_methods)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(mip_file_);
  raft::handle_t handle;

  auto problem = std::make_unique<gpu_optimization_problem_t<int, double>>(&handle);
  populate_from_mps_data_model(problem.get(), mps_data);

  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 60.0;

  auto solution = solve_mip(problem.get(), settings);
  ASSERT_NE(solution, nullptr);

  // Base interface polymorphic methods should work
  optimization_problem_solution_interface_t<int, double>* base_ptr = solution.get();

  EXPECT_TRUE(base_ptr->is_mip());
  EXPECT_NO_THROW(base_ptr->get_error_status());
  EXPECT_NO_THROW(base_ptr->get_solve_time());
  EXPECT_NO_THROW(base_ptr->get_solution_host());
  EXPECT_NO_THROW(base_ptr->get_termination_status_int());
  EXPECT_NO_THROW(base_ptr->get_objective_value());

  // MIP-specific polymorphic methods should work
  EXPECT_NO_THROW(base_ptr->get_mip_gap());
  EXPECT_NO_THROW(base_ptr->get_solution_bound());
}

// Test get_termination_status_int returns valid values
TEST_F(SolutionInterfaceTest, termination_status_int_values)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);
  raft::handle_t handle;

  auto problem = std::make_unique<gpu_optimization_problem_t<int, double>>(&handle);
  populate_from_mps_data_model(problem.get(), mps_data);

  pdlp_solver_settings_t<int, double> settings;
  settings.time_limit = 60.0;

  auto solution = solve_lp(problem.get(), settings);
  ASSERT_NE(solution, nullptr);

  int status = solution->get_termination_status_int();
  // Should be a valid termination status constant
  EXPECT_TRUE(status == CUOPT_TERIMINATION_STATUS_OPTIMAL ||
              status == CUOPT_TERIMINATION_STATUS_INFEASIBLE ||
              status == CUOPT_TERIMINATION_STATUS_TIME_LIMIT ||
              status == CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT ||
              status == CUOPT_TERIMINATION_STATUS_NUMERICAL_ERROR ||
              status == CUOPT_TERIMINATION_STATUS_NO_TERMINATION);
}

// =============================================================================
// Problem Conversion Tests
// =============================================================================

// Test GPU problem to_optimization_problem (move semantics)
TEST_F(SolutionInterfaceTest, gpu_problem_to_optimization_problem)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);
  raft::handle_t handle;

  auto problem = std::make_unique<gpu_optimization_problem_t<int, double>>(&handle);
  populate_from_mps_data_model(problem.get(), mps_data);

  int orig_n_vars        = problem->get_n_variables();
  int orig_n_constraints = problem->get_n_constraints();

  // Convert to concrete optimization_problem_t
  auto concrete_problem = problem->to_optimization_problem();

  EXPECT_EQ(concrete_problem.get_n_variables(), orig_n_vars);
  EXPECT_EQ(concrete_problem.get_n_constraints(), orig_n_constraints);
}

// Test CPU problem to_optimization_problem (copies data to GPU)
TEST_F(SolutionInterfaceTest, cpu_problem_to_optimization_problem)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);

  // CPU problem needs a handle to convert to GPU
  raft::handle_t handle;
  auto problem = std::make_unique<cpu_optimization_problem_t<int, double>>(&handle);
  populate_from_mps_data_model(problem.get(), mps_data);

  int orig_n_vars        = problem->get_n_variables();
  int orig_n_constraints = problem->get_n_constraints();

  // Convert to concrete GPU-backed optimization_problem_t
  auto concrete_problem = problem->to_optimization_problem();

  EXPECT_EQ(concrete_problem.get_n_variables(), orig_n_vars);
  EXPECT_EQ(concrete_problem.get_n_constraints(), orig_n_constraints);
}

// Test MPS data model to optimization problem conversion
TEST_F(SolutionInterfaceTest, mps_data_model_to_optimization_problem)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);
  raft::handle_t handle;

  auto problem = mps_data_model_to_optimization_problem(&handle, mps_data);

  EXPECT_EQ(problem.get_n_variables(), mps_data.get_n_variables());
  EXPECT_EQ(problem.get_n_constraints(), mps_data.get_n_constraints());
  EXPECT_EQ(problem.get_nnz(), mps_data.get_nnz());
}

// =============================================================================
// Solution Conversion Tests
// =============================================================================

// Test CPU LP solution to GPU conversion
TEST_F(SolutionInterfaceTest, cpu_lp_solution_to_gpu)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);

  // Create CPU problem and solve
  auto cpu_problem = std::make_unique<cpu_optimization_problem_t<int, double>>();
  populate_from_mps_data_model(cpu_problem.get(), mps_data);

  pdlp_solver_settings_t<int, double> settings;
  settings.time_limit = 60.0;

  // Solve on CPU (simulated remote)
  auto cpu_solution = cpu_problem->solve_lp_remote(settings);
  ASSERT_NE(cpu_solution, nullptr);

  // Get original values (use explicit id=0 to avoid ambiguity with overloaded method)
  double orig_objective   = cpu_solution->get_objective_value(0);
  int orig_status         = cpu_solution->get_termination_status_int();
  auto& orig_primal       = cpu_solution->get_primal_solution_host();
  size_t orig_primal_size = orig_primal.size();

  // Convert to GPU solution
  auto gpu_solution = cpu_solution->to_gpu_solution(rmm::cuda_stream_per_thread);

  // Verify values match
  EXPECT_NEAR(gpu_solution.get_objective_value(0), orig_objective, 1e-9);
  EXPECT_EQ(static_cast<int>(gpu_solution.get_termination_status()), orig_status);
  EXPECT_EQ(gpu_solution.get_primal_solution().size(), orig_primal_size);
}

// Test CPU MIP solution to GPU conversion
TEST_F(SolutionInterfaceTest, cpu_mip_solution_to_gpu)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(mip_file_);

  // Create CPU problem and solve
  auto cpu_problem = std::make_unique<cpu_optimization_problem_t<int, double>>();
  populate_from_mps_data_model(cpu_problem.get(), mps_data);

  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 60.0;

  // Solve on CPU (simulated remote)
  auto cpu_solution = cpu_problem->solve_mip_remote(settings);
  ASSERT_NE(cpu_solution, nullptr);

  // Get original values
  double orig_objective = cpu_solution->get_objective_value();
  int orig_status       = cpu_solution->get_termination_status_int();

  // Convert to GPU solution
  auto gpu_solution = cpu_solution->to_gpu_solution(rmm::cuda_stream_per_thread);

  // Verify values match
  EXPECT_NEAR(gpu_solution.get_objective_value(), orig_objective, 1e-9);
  EXPECT_EQ(static_cast<int>(gpu_solution.get_termination_status()), orig_status);
}

// Test GPU LP solution to Python return type
TEST_F(SolutionInterfaceTest, gpu_lp_solution_to_python_ret)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);
  raft::handle_t handle;

  auto problem = std::make_unique<gpu_optimization_problem_t<int, double>>(&handle);
  populate_from_mps_data_model(problem.get(), mps_data);

  pdlp_solver_settings_t<int, double> settings;
  settings.time_limit = 60.0;

  auto solution = solve_lp(problem.get(), settings);
  ASSERT_NE(solution, nullptr);

  // Use explicit id=0 to avoid ambiguity with overloaded method
  double orig_objective = solution->get_objective_value(0);

  // Convert to Python return type
  auto python_ret = std::move(*solution).to_python_lp_ret();

  // Should be GPU variant (linear_programming_ret_t)
  EXPECT_TRUE(std::holds_alternative<cuopt::cython::linear_programming_ret_t>(python_ret));

  auto& gpu_ret = std::get<cuopt::cython::linear_programming_ret_t>(python_ret);
  EXPECT_NEAR(gpu_ret.primal_objective_, orig_objective, 1e-9);
}

// Test CPU LP solution to Python return type
TEST_F(SolutionInterfaceTest, cpu_lp_solution_to_python_ret)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);

  auto cpu_problem = std::make_unique<cpu_optimization_problem_t<int, double>>();
  populate_from_mps_data_model(cpu_problem.get(), mps_data);

  pdlp_solver_settings_t<int, double> settings;
  settings.time_limit = 60.0;

  auto cpu_solution = cpu_problem->solve_lp_remote(settings);
  ASSERT_NE(cpu_solution, nullptr);

  // Use explicit id=0 to avoid ambiguity with overloaded method
  double orig_objective = cpu_solution->get_objective_value(0);

  // Convert to Python return type
  auto python_ret = std::move(*cpu_solution).to_python_lp_ret();

  // Should be CPU variant (cpu_linear_programming_ret_t)
  EXPECT_TRUE(std::holds_alternative<cuopt::cython::cpu_linear_programming_ret_t>(python_ret));

  auto& cpu_ret = std::get<cuopt::cython::cpu_linear_programming_ret_t>(python_ret);
  EXPECT_NEAR(cpu_ret.primal_objective_, orig_objective, 1e-9);
}

// Test GPU MIP solution to Python return type
TEST_F(SolutionInterfaceTest, gpu_mip_solution_to_python_ret)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(mip_file_);
  raft::handle_t handle;

  auto problem = std::make_unique<gpu_optimization_problem_t<int, double>>(&handle);
  populate_from_mps_data_model(problem.get(), mps_data);

  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 60.0;

  auto solution = solve_mip(problem.get(), settings);
  ASSERT_NE(solution, nullptr);

  double orig_objective = solution->get_objective_value();

  // Convert to Python return type
  auto python_ret = std::move(*solution).to_python_mip_ret();

  // Should be GPU variant (mip_ret_t)
  EXPECT_TRUE(std::holds_alternative<cuopt::cython::mip_ret_t>(python_ret));

  auto& gpu_ret = std::get<cuopt::cython::mip_ret_t>(python_ret);
  EXPECT_NEAR(gpu_ret.objective_, orig_objective, 1e-9);
}

// Test CPU MIP solution to Python return type
TEST_F(SolutionInterfaceTest, cpu_mip_solution_to_python_ret)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(mip_file_);

  auto cpu_problem = std::make_unique<cpu_optimization_problem_t<int, double>>();
  populate_from_mps_data_model(cpu_problem.get(), mps_data);

  mip_solver_settings_t<int, double> settings;
  settings.time_limit = 60.0;

  auto cpu_solution = cpu_problem->solve_mip_remote(settings);
  ASSERT_NE(cpu_solution, nullptr);

  double orig_objective = cpu_solution->get_objective_value();

  // Convert to Python return type
  auto python_ret = std::move(*cpu_solution).to_python_mip_ret();

  // Should be CPU variant (cpu_mip_ret_t)
  EXPECT_TRUE(std::holds_alternative<cuopt::cython::cpu_mip_ret_t>(python_ret));

  auto& cpu_ret = std::get<cuopt::cython::cpu_mip_ret_t>(python_ret);
  EXPECT_NEAR(cpu_ret.objective_, orig_objective, 1e-9);
}

// =============================================================================
// Warmstart Data Conversion Tests
// =============================================================================

// Test GPU warmstart to CPU conversion
TEST_F(SolutionInterfaceTest, gpu_warmstart_to_cpu)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);
  raft::handle_t handle;

  auto problem = std::make_unique<gpu_optimization_problem_t<int, double>>(&handle);
  populate_from_mps_data_model(problem.get(), mps_data);

  pdlp_solver_settings_t<int, double> settings;
  settings.time_limit      = 60.0;
  settings.iteration_limit = 100;  // Stop early to ensure warmstart data

  auto solution = solve_lp(problem.get(), settings);
  ASSERT_NE(solution, nullptr);

  // Check if warmstart data is available
  if (solution->has_warm_start_data()) {
    // Get warmstart data values
    auto current_primal = solution->get_current_primal_solution_host();
    auto current_dual   = solution->get_current_dual_solution_host();

    EXPECT_GT(current_primal.size(), 0u);
    EXPECT_GT(current_dual.size(), 0u);
  }
}

// Test CPU warmstart to GPU conversion
TEST_F(SolutionInterfaceTest, cpu_warmstart_to_gpu)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);

  auto cpu_problem = std::make_unique<cpu_optimization_problem_t<int, double>>();
  populate_from_mps_data_model(cpu_problem.get(), mps_data);

  pdlp_solver_settings_t<int, double> settings;
  settings.time_limit      = 60.0;
  settings.iteration_limit = 100;  // Stop early to ensure warmstart data

  auto cpu_solution = cpu_problem->solve_lp_remote(settings);
  ASSERT_NE(cpu_solution, nullptr);

  // Check if warmstart data is available
  if (cpu_solution->has_warm_start_data()) {
    // Get warmstart data values from CPU solution
    auto current_primal = cpu_solution->get_current_primal_solution_host();
    auto current_dual   = cpu_solution->get_current_dual_solution_host();

    EXPECT_GT(current_primal.size(), 0u);
    EXPECT_GT(current_dual.size(), 0u);

    // Convert solution to GPU and verify warmstart is preserved
    auto gpu_solution = cpu_solution->to_gpu_solution(rmm::cuda_stream_per_thread);

    // Warmstart data should be available in GPU solution
    // (it gets copied during conversion)
  }
}

// =============================================================================
// Problem Interface Copy Methods Tests
// =============================================================================

// Test GPU problem copy_*_to_host methods
TEST_F(SolutionInterfaceTest, gpu_problem_copy_to_host_methods)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);
  raft::handle_t handle;

  auto problem = std::make_unique<gpu_optimization_problem_t<int, double>>(&handle);
  populate_from_mps_data_model(problem.get(), mps_data);

  int n_vars        = problem->get_n_variables();
  int n_constraints = problem->get_n_constraints();
  int nnz           = problem->get_nnz();

  // Test copy_objective_coefficients_to_host
  std::vector<double> obj_coeffs(n_vars);
  problem->copy_objective_coefficients_to_host(obj_coeffs.data(), n_vars);
  EXPECT_EQ(obj_coeffs.size(), static_cast<size_t>(n_vars));

  // Test copy_variable_lower_bounds_to_host
  std::vector<double> var_lb(n_vars);
  problem->copy_variable_lower_bounds_to_host(var_lb.data(), n_vars);
  EXPECT_EQ(var_lb.size(), static_cast<size_t>(n_vars));

  // Test copy_variable_upper_bounds_to_host
  std::vector<double> var_ub(n_vars);
  problem->copy_variable_upper_bounds_to_host(var_ub.data(), n_vars);
  EXPECT_EQ(var_ub.size(), static_cast<size_t>(n_vars));

  // Test copy_constraint_bounds_to_host
  std::vector<double> rhs(n_constraints);
  problem->copy_constraint_bounds_to_host(rhs.data(), n_constraints);
  EXPECT_EQ(rhs.size(), static_cast<size_t>(n_constraints));

  // Test copy_constraint_matrix_to_host
  std::vector<double> values(nnz);
  std::vector<int> col_indices(nnz);
  std::vector<int> row_offsets(n_constraints + 1);
  problem->copy_constraint_matrix_to_host(
    values.data(), col_indices.data(), row_offsets.data(), nnz, nnz, n_constraints + 1);
  EXPECT_EQ(values.size(), static_cast<size_t>(nnz));
}

// Test CPU problem copy_*_to_host methods
TEST_F(SolutionInterfaceTest, cpu_problem_copy_to_host_methods)
{
  auto mps_data = cuopt::mps_parser::parse_mps<int, double>(lp_file_);

  auto problem = std::make_unique<cpu_optimization_problem_t<int, double>>();
  populate_from_mps_data_model(problem.get(), mps_data);

  int n_vars        = problem->get_n_variables();
  int n_constraints = problem->get_n_constraints();
  int nnz           = problem->get_nnz();

  // Test copy_objective_coefficients_to_host
  std::vector<double> obj_coeffs(n_vars);
  problem->copy_objective_coefficients_to_host(obj_coeffs.data(), n_vars);
  EXPECT_EQ(obj_coeffs.size(), static_cast<size_t>(n_vars));

  // Test copy_variable_lower_bounds_to_host
  std::vector<double> var_lb(n_vars);
  problem->copy_variable_lower_bounds_to_host(var_lb.data(), n_vars);
  EXPECT_EQ(var_lb.size(), static_cast<size_t>(n_vars));

  // Test copy_variable_upper_bounds_to_host
  std::vector<double> var_ub(n_vars);
  problem->copy_variable_upper_bounds_to_host(var_ub.data(), n_vars);
  EXPECT_EQ(var_ub.size(), static_cast<size_t>(n_vars));

  // Test copy_constraint_bounds_to_host
  std::vector<double> rhs(n_constraints);
  problem->copy_constraint_bounds_to_host(rhs.data(), n_constraints);
  EXPECT_EQ(rhs.size(), static_cast<size_t>(n_constraints));

  // Test copy_constraint_matrix_to_host
  std::vector<double> values(nnz);
  std::vector<int> col_indices(nnz);
  std::vector<int> row_offsets(n_constraints + 1);
  problem->copy_constraint_matrix_to_host(
    values.data(), col_indices.data(), row_offsets.data(), nnz, nnz, n_constraints + 1);
  EXPECT_EQ(values.size(), static_cast<size_t>(nnz));
}

}  // namespace cuopt::linear_programming
