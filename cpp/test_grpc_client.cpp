/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file test_grpc_client.cpp
 * @brief CLI tool for testing cuopt_grpc_server
 *
 * This is a thin CLI wrapper around grpc_client_t. All transport logic is
 * encapsulated in grpc_client_t - this file has no direct gRPC knowledge.
 *
 * Supports multiple modes:
 *   solve    - Submit problem and wait for result
 *   submit   - Submit problem and return job ID
 *   status   - Check status of a job by ID
 *   result   - Get solution for a job by ID
 *   cancel   - Cancel a job by ID
 *   delete   - Delete a job and its data
 *   logs     - Stream log lines for a job
 *   incumbent - Get incumbent solutions for a MIP job
 */

#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/optimization_problem_utils.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/grpc_client.hpp>
#include <mps_parser/parser.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

using namespace cuopt::linear_programming;

// =============================================================================
// Helper Functions
// =============================================================================

std::string read_file(const std::string& path)
{
  std::ifstream file(path);
  if (!file) { return ""; }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

// =============================================================================
// Problem Creation
// =============================================================================

cpu_optimization_problem_t<int32_t, double> create_simple_lp_problem()
{
  cpu_optimization_problem_t<int32_t, double> problem;

  std::vector<double> c = {1.0, 1.0};
  problem.set_objective_coefficients(c.data(), 2);
  problem.set_maximize(true);

  std::vector<double> A_values   = {1.0, 2.0};
  std::vector<int32_t> A_indices = {0, 1};
  std::vector<int32_t> A_offsets = {0, 2};
  problem.set_csr_constraint_matrix(A_values.data(), 2, A_indices.data(), 2, A_offsets.data(), 2);

  std::vector<double> var_lb = {0.0, 0.0};
  std::vector<double> var_ub = {std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::infinity()};
  problem.set_variable_lower_bounds(var_lb.data(), 2);
  problem.set_variable_upper_bounds(var_ub.data(), 2);

  std::vector<double> con_lb = {-std::numeric_limits<double>::infinity()};
  std::vector<double> con_ub = {4.0};
  problem.set_constraint_lower_bounds(con_lb.data(), 1);
  problem.set_constraint_upper_bounds(con_ub.data(), 1);

  problem.set_problem_name("simple_2var_lp");
  return problem;
}

cpu_optimization_problem_t<int32_t, double> load_mps_problem(const std::string& mps_path)
{
  std::cout << "Loading MPS file: " << mps_path << "\n";
  auto start = std::chrono::steady_clock::now();

  auto mps_data = cuopt::mps_parser::parse_mps<int32_t, double>(mps_path);

  auto parse_end = std::chrono::steady_clock::now();
  auto parse_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(parse_end - start).count();

  std::cout << "  Parsed in " << parse_ms << " ms\n";
  std::cout << "  Variables: " << mps_data.get_n_variables() << "\n";
  std::cout << "  Constraints: " << mps_data.get_n_constraints() << "\n";
  std::cout << "  Nonzeros: " << mps_data.get_constraint_matrix_values().size() << "\n";

  bool is_mip = false;
  for (char vt : mps_data.get_variable_types()) {
    if (vt == 'I' || vt == 'B') {
      is_mip = true;
      break;
    }
  }
  std::cout << "  Problem type: " << (is_mip ? "MIP" : "LP") << "\n";

  cpu_optimization_problem_t<int32_t, double> problem;
  populate_from_mps_data_model(&problem, mps_data);

  auto convert_end = std::chrono::steady_clock::now();
  auto convert_ms =
    std::chrono::duration_cast<std::chrono::milliseconds>(convert_end - parse_end).count();
  std::cout << "  Converted in " << convert_ms << " ms\n";

  return problem;
}

bool problem_is_mip(const cpu_optimization_problem_t<int32_t, double>& problem)
{
  auto var_types = problem.get_variable_types_host();
  for (auto vt : var_types) {
    if (vt == var_t::INTEGER) { return true; }
  }
  return false;
}

// =============================================================================
// CLI
// =============================================================================

void print_usage(const char* prog_name)
{
  std::cout << "Usage: " << prog_name << " [options] <server> <mode> [mode_options]\n\n"
            << "Options (must come before server):\n"
            << "  --tls                        Enable TLS encryption\n"
            << "  --tls-root <path>            Path to root CA certificate (PEM)\n"
            << "  --tls-cert <path>            Path to client certificate (PEM) for mTLS\n"
            << "  --tls-key <path>             Path to client private key (PEM) for mTLS\n"
            << "  --mps <path>                 Load problem from MPS file instead of test problem\n"
            << "  --time-limit <seconds>       Solver time limit (default: 60)\n"
            << "  --use-wait                   Use WaitForCompletion RPC instead of polling\n"
            << "\nModes:\n"
            << "  solve                        Submit problem and wait for result\n"
            << "  submit                       Submit problem and return job ID\n"
            << "  status <job_id>              Check status of a job\n"
            << "  result <job_id>              Get LP solution for a job\n"
            << "  mip-result <job_id>          Get MIP solution for a job\n"
            << "  cancel <job_id>              Cancel a running job\n"
            << "  delete <job_id>              Delete a job and its data\n"
            << "  logs <job_id>                Stream log lines for a job\n"
            << "  incumbent <job_id> [index]   Get incumbent solutions (default index=0)\n"
            << "\nExamples:\n"
            << "  " << prog_name << " localhost:9112 solve\n"
            << "  " << prog_name << " --mps problem.mps localhost:9112 solve\n"
            << "  " << prog_name << " localhost:9112 status abc-123\n";
}

// =============================================================================
// Mode Implementations
// =============================================================================

int mode_solve(grpc_client_t& client, const std::string& mps_path, double time_limit)
{
  std::cout << "=== Loading problem ===\n";
  cpu_optimization_problem_t<int32_t, double> problem;
  bool is_mip = false;

  if (mps_path.empty()) {
    problem = create_simple_lp_problem();
    std::cout << "Problem: maximize x + y, subject to x + 2y <= 4, x,y >= 0\n";
    std::cout << "Expected: x=4, y=0, objective=4\n\n";
  } else {
    problem = load_mps_problem(mps_path);
    is_mip  = problem_is_mip(problem);
    std::cout << "\n";
  }

  std::cout << "=== Solving " << (is_mip ? "MIP" : "LP") << " ===\n";

  if (is_mip) {
    mip_solver_settings_t<int32_t, double> settings;
    settings.time_limit = time_limit;
    auto result         = client.solve_mip(problem, settings, false);

    if (!result.success) {
      std::cerr << "Solve failed: " << result.error_message << "\n";
      return 1;
    }

    std::cout << "\n=== MIP Solution ===\n";
    std::cout << "Objective: " << result.solution->get_objective_value() << "\n";
    auto primal = result.solution->get_solution_host();
    std::cout << "Solution size: " << primal.size() << " variables\n";
  } else {
    pdlp_solver_settings_t<int32_t, double> settings;
    settings.log_to_console = true;
    settings.time_limit     = time_limit;
    auto result             = client.solve_lp(problem, settings);

    if (!result.success) {
      std::cerr << "Solve failed: " << result.error_message << "\n";
      return 1;
    }

    std::cout << "\n=== LP Solution ===\n";
    std::cout << "Primal objective: " << result.solution->get_objective_value() << "\n";
    std::cout << "Dual objective: " << result.solution->get_dual_objective_value() << "\n";
    auto primal = result.solution->get_primal_solution_host();
    std::cout << "Solution size: " << primal.size() << " variables\n";
  }

  std::cout << "\nSolve completed successfully!\n";
  return 0;
}

int mode_submit(grpc_client_t& client, const std::string& mps_path, double time_limit)
{
  cpu_optimization_problem_t<int32_t, double> problem;
  bool is_mip = false;

  if (mps_path.empty()) {
    std::cout << "Creating test LP problem...\n";
    problem = create_simple_lp_problem();
  } else {
    problem = load_mps_problem(mps_path);
    is_mip  = problem_is_mip(problem);
  }

  std::cout << "Submitting " << (is_mip ? "MIP" : "LP") << " job...\n";
  submit_result_t result;

  if (is_mip) {
    mip_solver_settings_t<int32_t, double> settings;
    settings.time_limit = time_limit;
    result              = client.submit_mip(problem, settings, false);
  } else {
    pdlp_solver_settings_t<int32_t, double> settings;
    settings.log_to_console = true;
    settings.time_limit     = time_limit;
    result                  = client.submit_lp(problem, settings);
  }

  if (!result.success) {
    std::cerr << "Submit failed: " << result.error_message << "\n";
    return 1;
  }

  std::cout << "JOB_ID=" << result.job_id << "\n";
  return 0;
}

int mode_status(grpc_client_t& client, const std::string& job_id)
{
  auto result = client.check_status(job_id);

  if (!result.success) {
    std::cerr << "Status check failed: " << result.error_message << "\n";
    return 1;
  }

  std::cout << "JOB_ID=" << job_id << "\n";
  std::cout << "STATUS=" << job_status_to_string(result.status) << "\n";
  if (!result.message.empty()) { std::cout << "MESSAGE=" << result.message << "\n"; }
  if (result.result_size_bytes > 0) {
    std::cout << "RESULT_SIZE=" << result.result_size_bytes << " bytes\n";
  }
  return 0;
}

int mode_result_lp(grpc_client_t& client, const std::string& job_id)
{
  auto result = client.get_lp_result<int32_t, double>(job_id);

  if (!result.success) {
    std::cerr << "Get result failed: " << result.error_message << "\n";
    return 1;
  }

  std::cout << "JOB_ID=" << job_id << "\n";
  std::cout << "Primal objective: " << result.solution->get_objective_value() << "\n";
  std::cout << "Dual objective: " << result.solution->get_dual_objective_value() << "\n";
  auto primal = result.solution->get_primal_solution_host();
  std::cout << "Solution size: " << primal.size() << " variables\n";
  return 0;
}

int mode_result_mip(grpc_client_t& client, const std::string& job_id)
{
  auto result = client.get_mip_result<int32_t, double>(job_id);

  if (!result.success) {
    std::cerr << "Get result failed: " << result.error_message << "\n";
    return 1;
  }

  std::cout << "JOB_ID=" << job_id << "\n";
  std::cout << "Objective: " << result.solution->get_objective_value() << "\n";
  auto primal = result.solution->get_solution_host();
  std::cout << "Solution size: " << primal.size() << " variables\n";
  return 0;
}

int mode_cancel(grpc_client_t& client, const std::string& job_id)
{
  auto result = client.cancel_job(job_id);

  std::cout << "JOB_ID=" << job_id << "\n";
  std::cout << "CANCELLED=" << (result.success ? "true" : "false") << "\n";
  std::cout << "JOB_STATUS=" << job_status_to_string(result.job_status) << "\n";
  if (!result.message.empty()) { std::cout << "MESSAGE=" << result.message << "\n"; }
  if (!result.error_message.empty()) { std::cout << "ERROR=" << result.error_message << "\n"; }
  return result.success ? 0 : 1;
}

int mode_delete(grpc_client_t& client, const std::string& job_id)
{
  bool success = client.delete_job(job_id);

  std::cout << "JOB_ID=" << job_id << "\n";
  std::cout << "DELETED=" << (success ? "true" : "false") << "\n";
  if (!success) { std::cout << "ERROR=" << client.get_last_error() << "\n"; }
  return success ? 0 : 1;
}

int mode_logs(grpc_client_t& client, const std::string& job_id)
{
  std::cout << "Streaming logs for job " << job_id << "...\n\n";

  bool success = client.stream_logs(job_id, 0, [](const std::string& line, bool job_complete) {
    std::string trimmed = line;
    if (!trimmed.empty() && trimmed.back() == '\n') trimmed.pop_back();
    if (!trimmed.empty()) { std::cout << "[LOG] " << trimmed << "\n"; }
    if (job_complete) { std::cout << "\n[Job complete]\n"; }
    return true;  // Continue streaming
  });

  return success ? 0 : 1;
}

int mode_incumbent(grpc_client_t& client, const std::string& job_id, int64_t from_index)
{
  auto result = client.get_incumbents(job_id, from_index, 10);

  if (!result.success) {
    std::cerr << "Get incumbents failed: " << result.error_message << "\n";
    return 1;
  }

  std::cout << "JOB_ID=" << job_id << "\n";
  std::cout << "INCUMBENTS=" << result.incumbents.size() << "\n";
  std::cout << "NEXT_INDEX=" << result.next_index << "\n";
  std::cout << "JOB_COMPLETE=" << (result.job_complete ? "true" : "false") << "\n";

  for (const auto& inc : result.incumbents) {
    std::cout << "\n  Incumbent " << inc.index << ": objective=" << inc.objective
              << ", vars=" << inc.assignment.size() << "\n";
  }

  return 0;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
  // Parse options
  bool enable_tls = false;
  bool use_wait   = false;
  std::string tls_root_path;
  std::string tls_cert_path;
  std::string tls_key_path;
  std::string mps_path;
  double time_limit = 60.0;

  int arg_idx = 1;
  while (arg_idx < argc && argv[arg_idx][0] == '-') {
    std::string arg = argv[arg_idx];
    if (arg == "--tls") {
      enable_tls = true;
      arg_idx++;
    } else if (arg == "--use-wait") {
      use_wait = true;
      arg_idx++;
    } else if (arg == "--tls-root" && arg_idx + 1 < argc) {
      tls_root_path = argv[++arg_idx];
      arg_idx++;
    } else if (arg == "--tls-cert" && arg_idx + 1 < argc) {
      tls_cert_path = argv[++arg_idx];
      arg_idx++;
    } else if (arg == "--tls-key" && arg_idx + 1 < argc) {
      tls_key_path = argv[++arg_idx];
      arg_idx++;
    } else if (arg == "--mps" && arg_idx + 1 < argc) {
      mps_path = argv[++arg_idx];
      arg_idx++;
    } else if (arg == "--time-limit" && arg_idx + 1 < argc) {
      time_limit = std::stod(argv[++arg_idx]);
      arg_idx++;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "ERROR: Unknown option '" << arg << "'\n\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (argc - arg_idx < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string server_address = argv[arg_idx++];
  std::string mode           = argv[arg_idx++];

  // Configure client
  grpc_client_config_t config;
  config.server_address  = server_address;
  config.timeout_seconds = static_cast<int>(time_limit) + 120;
  config.enable_tls      = enable_tls;
  config.use_wait        = use_wait;

  if (!tls_root_path.empty()) {
    config.tls_root_certs = read_file(tls_root_path);
    if (config.tls_root_certs.empty()) {
      std::cerr << "ERROR: Failed to read TLS root cert: " << tls_root_path << "\n";
      return 1;
    }
  }
  if (!tls_cert_path.empty()) {
    config.tls_client_cert = read_file(tls_cert_path);
    if (config.tls_client_cert.empty()) {
      std::cerr << "ERROR: Failed to read TLS client cert: " << tls_cert_path << "\n";
      return 1;
    }
  }
  if (!tls_key_path.empty()) {
    config.tls_client_key = read_file(tls_key_path);
    if (config.tls_client_key.empty()) {
      std::cerr << "ERROR: Failed to read TLS client key: " << tls_key_path << "\n";
      return 1;
    }
  }

  // Enable log streaming for solve mode
  if (mode == "solve") {
    config.stream_logs  = true;
    config.log_callback = [](const std::string& line) {
      std::string trimmed = line;
      if (!trimmed.empty() && trimmed.back() == '\n') trimmed.pop_back();
      if (!trimmed.empty()) { std::cout << "[LOG] " << trimmed << "\n"; }
    };

    // Enable incumbent callback for MIP problems
    config.incumbent_callback =
      [](int64_t index, double objective, const std::vector<double>& solution) {
        std::cout << "[INCUMBENT] #" << index << ": objective=" << objective
                  << ", vars=" << solution.size() << "\n";
        return true;  // Continue solving
      };
  }

  if (enable_tls) {
    std::cout << "TLS enabled" << (tls_cert_path.empty() ? "" : " with client cert (mTLS)") << "\n";
  }
  if (use_wait) { std::cout << "Using WaitForCompletion RPC (instead of polling)\n"; }

  // Create client and connect
  grpc_client_t client(config);

  std::cout << "Connecting to " << server_address << "...\n";
  if (!client.connect()) {
    std::cerr << "ERROR: " << client.get_last_error() << "\n";
    return 1;
  }
  std::cout << "Connected.\n\n";

  // Dispatch to mode
  if (mode == "solve") {
    return mode_solve(client, mps_path, time_limit);
  } else if (mode == "submit") {
    return mode_submit(client, mps_path, time_limit);
  } else if (mode == "status") {
    if (arg_idx >= argc) {
      std::cerr << "ERROR: status mode requires job_id\n";
      return 1;
    }
    return mode_status(client, argv[arg_idx]);
  } else if (mode == "result") {
    if (arg_idx >= argc) {
      std::cerr << "ERROR: result mode requires job_id\n";
      return 1;
    }
    return mode_result_lp(client, argv[arg_idx]);
  } else if (mode == "mip-result") {
    if (arg_idx >= argc) {
      std::cerr << "ERROR: mip-result mode requires job_id\n";
      return 1;
    }
    return mode_result_mip(client, argv[arg_idx]);
  } else if (mode == "cancel") {
    if (arg_idx >= argc) {
      std::cerr << "ERROR: cancel mode requires job_id\n";
      return 1;
    }
    return mode_cancel(client, argv[arg_idx]);
  } else if (mode == "delete") {
    if (arg_idx >= argc) {
      std::cerr << "ERROR: delete mode requires job_id\n";
      return 1;
    }
    return mode_delete(client, argv[arg_idx]);
  } else if (mode == "logs") {
    if (arg_idx >= argc) {
      std::cerr << "ERROR: logs mode requires job_id\n";
      return 1;
    }
    return mode_logs(client, argv[arg_idx]);
  } else if (mode == "incumbent") {
    if (arg_idx >= argc) {
      std::cerr << "ERROR: incumbent mode requires job_id\n";
      return 1;
    }
    std::string job_id = argv[arg_idx++];
    int64_t from_index = (arg_idx < argc) ? std::stoll(argv[arg_idx]) : 0;
    return mode_incumbent(client, job_id, from_index);
  } else {
    std::cerr << "ERROR: Unknown mode '" << mode << "'\n\n";
    print_usage(argv[0]);
    return 1;
  }
}
