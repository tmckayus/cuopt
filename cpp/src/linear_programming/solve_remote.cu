/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/cpu_pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/gpu_optimization_problem.hpp>
#include <cuopt/linear_programming/gpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/grpc_client.hpp>
#include <utilities/logger.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace cuopt::linear_programming {

// ============================================================================
// Helper function to get gRPC server address from environment variables
// ============================================================================

static std::string get_grpc_server_address()
{
  const char* host = std::getenv("CUOPT_REMOTE_HOST");
  const char* port = std::getenv("CUOPT_REMOTE_PORT");

  if (host == nullptr || port == nullptr) {
    throw std::runtime_error(
      "Remote execution enabled but CUOPT_REMOTE_HOST and/or CUOPT_REMOTE_PORT not set");
  }

  return std::string(host) + ":" + std::string(port);
}

// ============================================================================
// Remote execution via gRPC
// ============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
  cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  // Initialize logger for remote execution path (ref-counted, safe to call multiple times)
  init_logger_t log(settings.log_file, settings.log_to_console);

  CUOPT_LOG_INFO("solve_lp_remote (CPU problem) - connecting to gRPC server");

  // Build gRPC client configuration
  grpc_client_config_t config;
  config.server_address = get_grpc_server_address();

  // Configure log streaming based on settings
  if (settings.log_to_console) {
    config.stream_logs  = true;
    config.log_callback = [](const std::string& line) { std::cout << line << std::endl; };
  }

  // Create client and connect
  grpc_client_t client(config);
  if (!client.connect()) {
    throw std::runtime_error("Failed to connect to gRPC server: " + client.get_last_error());
  }

  CUOPT_LOG_INFO("solve_lp_remote - connected to %s, submitting problem",
                 config.server_address.c_str());

  // Call the remote solver
  auto result = client.solve_lp(cpu_problem, settings);

  if (!result.success) {
    throw std::runtime_error("Remote LP solve failed: " + result.error_message);
  }

  CUOPT_LOG_INFO("solve_lp_remote - solve completed successfully");

  return std::move(result.solution);
}

template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
  cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  mip_solver_settings_t<i_t, f_t> const& settings)
{
  // Initialize logger for remote execution path (ref-counted, safe to call multiple times)
  init_logger_t log(settings.log_file, settings.log_to_console);

  CUOPT_LOG_INFO("solve_mip_remote (CPU problem) - connecting to gRPC server");

  // Build gRPC client configuration
  grpc_client_config_t config;
  config.server_address = get_grpc_server_address();

  // Configure log streaming based on settings
  if (settings.log_to_console) {
    config.stream_logs  = true;
    config.log_callback = [](const std::string& line) { std::cout << line << std::endl; };
  }

  // Check if user has set incumbent callbacks
  auto mip_callbacks   = settings.get_mip_callbacks();
  bool has_incumbents  = !mip_callbacks.empty();
  bool enable_tracking = has_incumbents;

  // Set up incumbent callback forwarding if user has callbacks
  if (has_incumbents) {
    config.incumbent_callback = [&mip_callbacks](int64_t index,
                                                 double objective,
                                                 const std::vector<double>& solution) -> bool {
      // Forward incumbent to all user callbacks
      for (auto* callback : mip_callbacks) {
        if (callback != nullptr) {
          // Only SET_SOLUTION callbacks are relevant for incumbents - these receive
          // solutions from the solver. GET_SOLUTION callbacks are for providing
          // initial solutions to the solver, not for receiving incumbents.
          if (callback->get_type() == internals::base_solution_callback_type::SET_SOLUTION) {
            auto* set_callback = static_cast<internals::set_solution_callback_t*>(callback);
            // Copy solution to non-const buffer for callback interface
            std::vector<double> solution_copy = solution;
            double obj_copy                   = objective;
            set_callback->set_solution(solution_copy.data(), &obj_copy);
          }
        }
      }
      // Return true to continue solving (don't cancel)
      return true;
    };
  }

  // Create client and connect
  grpc_client_t client(config);
  if (!client.connect()) {
    throw std::runtime_error("Failed to connect to gRPC server: " + client.get_last_error());
  }

  CUOPT_LOG_INFO("solve_mip_remote - connected to %s, submitting problem (incumbents=%s)",
                 config.server_address.c_str(),
                 enable_tracking ? "enabled" : "disabled");

  // Call the remote solver
  auto result = client.solve_mip(cpu_problem, settings, enable_tracking);

  if (!result.success) {
    throw std::runtime_error("Remote MIP solve failed: " + result.error_message);
  }

  CUOPT_LOG_INFO("solve_mip_remote - solve completed successfully");

  return std::move(result.solution);
}

// ============================================================================
// Remote execution for GPU problems (converts to CPU then calls CPU remote)
// ============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
  optimization_problem_t<i_t, f_t>& problem, pdlp_solver_settings_t<i_t, f_t> const& settings)
{
  // Initialize logger for remote execution path (ref-counted, safe to call multiple times)
  init_logger_t log(settings.log_file, settings.log_to_console);

  CUOPT_LOG_INFO("solve_lp_remote (GPU problem) - converting to CPU for remote execution");

  // Convert GPU problem to CPU problem using the encapsulated conversion method
  auto cpu_problem = problem.to_cpu_optimization_problem();

  // Call CPU remote solver (returns unique_ptr<lp_solution_interface_t>)
  auto cpu_solution_interface = solve_lp_remote(*cpu_problem, settings);

  // Convert CPU solution back to GPU solution (since we started with a GPU problem)
  auto gpu_solution = cpu_solution_interface->to_gpu_solution(rmm::cuda_stream_per_thread);
  return std::make_unique<gpu_lp_solution_t<i_t, f_t>>(std::move(gpu_solution));
}

template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
  optimization_problem_t<i_t, f_t>& problem, mip_solver_settings_t<i_t, f_t> const& settings)
{
  // Initialize logger for remote execution path (ref-counted, safe to call multiple times)
  init_logger_t log(settings.log_file, settings.log_to_console);

  CUOPT_LOG_INFO("solve_mip_remote (GPU problem) - converting to CPU for remote execution");

  // Convert GPU problem to CPU problem using the encapsulated conversion method
  auto cpu_problem = problem.to_cpu_optimization_problem();

  // Call CPU remote solver (returns unique_ptr<mip_solution_interface_t>)
  auto cpu_solution_interface = solve_mip_remote(*cpu_problem, settings);

  // Convert CPU solution back to GPU solution (since we started with a GPU problem)
  auto gpu_solution = cpu_solution_interface->to_gpu_solution(rmm::cuda_stream_per_thread);
  return std::make_unique<gpu_mip_solution_t<i_t, f_t>>(std::move(gpu_solution));
}

// Explicit template instantiations for remote execution stubs
template std::unique_ptr<lp_solution_interface_t<int, double>> solve_lp_remote(
  cpu_optimization_problem_t<int, double>&, pdlp_solver_settings_t<int, double> const&);

template std::unique_ptr<mip_solution_interface_t<int, double>> solve_mip_remote(
  cpu_optimization_problem_t<int, double>&, mip_solver_settings_t<int, double> const&);

template std::unique_ptr<lp_solution_interface_t<int, double>> solve_lp_remote(
  optimization_problem_t<int, double>&, pdlp_solver_settings_t<int, double> const&);

template std::unique_ptr<mip_solution_interface_t<int, double>> solve_mip_remote(
  optimization_problem_t<int, double>&, mip_solver_settings_t<int, double> const&);

}  // namespace cuopt::linear_programming
