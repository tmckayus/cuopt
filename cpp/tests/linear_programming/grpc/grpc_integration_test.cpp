/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file grpc_integration_test.cpp
 * @brief Integration tests for gRPC client-server communication
 *
 * These tests start a real cuopt_grpc_server process and verify end-to-end
 * communication, including error handling paths.
 *
 * Environment variables:
 *   CUOPT_GRPC_SERVER_PATH - Path to cuopt_grpc_server binary (default: looks in build dir)
 *   CUOPT_TEST_PORT_BASE   - Base port for test servers (default: 19000)
 *
 * Note: These tests require GPU access for actual solves. On systems without GPU,
 * only connectivity and protocol tests will run.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/optimization_problem_utils.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/grpc_client.hpp>
#include <mps_parser/parser.hpp>

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>

using namespace cuopt::linear_programming;

namespace {

/**
 * @brief Helper class to manage server process lifecycle
 */
class ServerProcess {
 public:
  ServerProcess() : pid_(-1), port_(0) {}

  ~ServerProcess() { stop(); }

  /**
   * @brief Set TLS configuration for health checks
   * @param root_certs PEM-encoded root CA certificates
   * @param client_cert Optional client certificate for mTLS
   * @param client_key Optional client key for mTLS
   */
  void set_tls_config(const std::string& root_certs,
                      const std::string& client_cert = "",
                      const std::string& client_key  = "")
  {
    tls_root_certs_  = root_certs;
    tls_client_cert_ = client_cert;
    tls_client_key_  = client_key;
  }

  /**
   * @brief Start the server process
   * @param port Port to listen on
   * @param extra_args Additional command-line arguments
   * @return true if server started successfully
   */
  bool start(int port, const std::vector<std::string>& extra_args = {})
  {
    port_ = port;

    // Find server binary
    std::string server_path = find_server_binary();
    if (server_path.empty()) {
      std::cerr << "Could not find cuopt_grpc_server binary\n";
      return false;
    }

    pid_ = fork();
    if (pid_ < 0) {
      std::cerr << "fork() failed\n";
      return false;
    }

    if (pid_ == 0) {
      // Child process - exec server
      std::vector<const char*> args;
      args.push_back(server_path.c_str());
      args.push_back("--port");
      std::string port_str = std::to_string(port);
      args.push_back(port_str.c_str());
      args.push_back("--workers");
      args.push_back("1");

      for (const auto& arg : extra_args) {
        args.push_back(arg.c_str());
      }
      args.push_back(nullptr);

      // Redirect stdout/stderr to log file
      std::string log_file = "/tmp/cuopt_test_server_" + std::to_string(port) + ".log";
      int fd               = open(log_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd >= 0) {
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        close(fd);
      }

      execv(server_path.c_str(), const_cast<char**>(args.data()));
      _exit(127);  // exec failed
    }

    // Parent process - wait for server to be ready
    return wait_for_ready(10000);  // 10 second timeout
  }

  /**
   * @brief Stop the server process
   */
  void stop()
  {
    if (pid_ > 0) {
      // Send SIGTERM
      kill(pid_, SIGTERM);

      // Wait with timeout
      int status;
      int wait_ms = 0;
      while (wait_ms < 5000) {
        int ret = waitpid(pid_, &status, WNOHANG);
        if (ret != 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_ms += 100;
      }

      // Force kill if still running
      if (waitpid(pid_, &status, WNOHANG) == 0) {
        kill(pid_, SIGKILL);
        waitpid(pid_, &status, 0);
      }

      pid_ = -1;
    }
  }

  /**
   * @brief Get the server port
   */
  int port() const { return port_; }

  /**
   * @brief Check if server is running
   */
  bool is_running() const
  {
    if (pid_ <= 0) return false;
    return kill(pid_, 0) == 0;
  }

 private:
  /**
   * @brief Search for executable in PATH environment variable
   * @param name Executable name to find
   * @return Full path to executable, or empty string if not found
   */
  std::string find_in_path(const std::string& name)
  {
    const char* path_env = std::getenv("PATH");
    if (!path_env) return "";

    std::string path_str(path_env);
    std::string::size_type start = 0;
    std::string::size_type end;

    while ((end = path_str.find(':', start)) != std::string::npos || start < path_str.size()) {
      std::string dir;
      if (end != std::string::npos) {
        dir   = path_str.substr(start, end - start);
        start = end + 1;
      } else {
        dir   = path_str.substr(start);
        start = path_str.size();
      }

      if (dir.empty()) continue;

      std::string full_path = dir + "/" + name;
      if (access(full_path.c_str(), X_OK) == 0) { return full_path; }
    }

    return "";
  }

  std::string find_server_binary()
  {
    // Check environment variable first (set by CMake tests)
    const char* env_path = std::getenv("CUOPT_GRPC_SERVER_PATH");
    if (env_path && access(env_path, X_OK) == 0) { return env_path; }

    // Check if cuopt_grpc_server is in PATH (installed location, like cuopt_cli)
    // This is the expected path in CI after installation
    std::string path_result = find_in_path("cuopt_grpc_server");
    if (!path_result.empty()) { return path_result; }

    // Fallback: try common build directories (for local development)
    std::vector<std::string> paths = {
      "./cuopt_grpc_server",
      "../cuopt_grpc_server",
      "../../cuopt_grpc_server",
      "./build/cuopt_grpc_server",
      "../build/cuopt_grpc_server",
    };

    for (const auto& path : paths) {
      if (access(path.c_str(), X_OK) == 0) { return path; }
    }

    return "";
  }

  bool wait_for_ready(int timeout_ms)
  {
    auto start = std::chrono::steady_clock::now();

    while (true) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);

      if (elapsed.count() >= timeout_ms) { return false; }

      // Try to connect with appropriate TLS settings
      grpc_client_config_t config;
      config.server_address = "localhost:" + std::to_string(port_);

      if (!tls_root_certs_.empty()) {
        config.enable_tls      = true;
        config.tls_root_certs  = tls_root_certs_;
        config.tls_client_cert = tls_client_cert_;
        config.tls_client_key  = tls_client_key_;
      }

      grpc_client_t client(config);

      if (client.connect()) { return true; }

      // Check if process died
      int status;
      if (waitpid(pid_, &status, WNOHANG) != 0) {
        std::cerr << "Server process died during startup\n";
        return false;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  pid_t pid_;
  int port_;
  std::string tls_root_certs_;
  std::string tls_client_cert_;
  std::string tls_client_key_;
};

/**
 * @brief Get a unique port for testing
 */
int get_test_port()
{
  static std::atomic<int> port_counter{0};

  int base_port        = 19000;
  const char* env_base = std::getenv("CUOPT_TEST_PORT_BASE");
  if (env_base) { base_port = std::atoi(env_base); }

  return base_port + port_counter.fetch_add(1);
}

}  // anonymous namespace

/**
 * @brief Test fixture for gRPC integration tests
 */
class GrpcIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override { port_ = get_test_port(); }

  void TearDown() override { server_.stop(); }

  /**
   * @brief Start server with default settings
   */
  bool start_server(const std::vector<std::string>& extra_args = {})
  {
    return server_.start(port_, extra_args);
  }

  /**
   * @brief Create a connected client with optional custom config
   *
   * Server address and poll interval are always set to test defaults.
   * Pass a config to customize other settings (callbacks, timeouts, TLS, etc.)
   */
  std::unique_ptr<grpc_client_t> create_client(grpc_client_config_t config = {})
  {
    // Always set test defaults for server address and fast polling
    config.server_address   = "localhost:" + std::to_string(port_);
    config.poll_interval_ms = 100;  // Fast polling for tests

    // Set reasonable test timeout if not customized (default is -1 meaning "not set")
    if (config.timeout_seconds == -1) { config.timeout_seconds = 60; }

    auto client = std::make_unique<grpc_client_t>(config);
    if (!client->connect()) { return nullptr; }
    return client;
  }

  /**
   * @brief Get path to a test dataset file using RAPIDS_DATASET_ROOT_DIR
   */
  std::string get_test_data_path(const std::string& subdir, const std::string& filename)
  {
    const char* env_var      = std::getenv("RAPIDS_DATASET_ROOT_DIR");
    std::string dataset_root = env_var ? env_var : "./datasets";
    return dataset_root + "/" + subdir + "/" + filename;
  }

  std::string get_test_lp_path(const std::string& filename)
  {
    return get_test_data_path("linear_programming", filename);
  }

  std::string get_test_mip_path(const std::string& filename)
  {
    return get_test_data_path("mip", filename);
  }

  /**
   * @brief Load an optimization problem from an MPS file
   *
   * The MPS parser automatically detects LP vs MIP based on variable types in the file.
   */
  cpu_optimization_problem_t<int32_t, double> load_problem_from_mps(const std::string& mps_path)
  {
    auto mps_data = cuopt::mps_parser::parse_mps<int32_t, double>(mps_path);
    cpu_optimization_problem_t<int32_t, double> problem(nullptr);
    populate_from_mps_data_model(&problem, mps_data);
    return problem;
  }

  /**
   * @brief Create a simple MIP problem that is guaranteed feasible
   *
   * Problem: Minimize x + 2y
   *          Subject to: x + y >= 1
   *          x, y ∈ {0, 1} (binary)
   *
   * Optimal solution: x=1, y=0, objective=1
   */
  cpu_optimization_problem_t<int32_t, double> create_simple_mip()
  {
    cpu_optimization_problem_t<int32_t, double> problem(nullptr);

    // Objective: minimize x + 2y
    std::vector<double> c = {1.0, 2.0};
    problem.set_objective_coefficients(c.data(), 2);
    problem.set_maximize(false);

    // Constraint: x + y >= 1  (i.e., 1 <= x + y <= inf)
    std::vector<double> A_values   = {1.0, 1.0};
    std::vector<int32_t> A_indices = {0, 1};
    std::vector<int32_t> A_offsets = {0, 2};
    problem.set_csr_constraint_matrix(A_values.data(), 2, A_indices.data(), 2, A_offsets.data(), 2);

    // Variable bounds: 0 <= x, y <= 1 (binary)
    std::vector<double> var_lb = {0.0, 0.0};
    std::vector<double> var_ub = {1.0, 1.0};
    problem.set_variable_lower_bounds(var_lb.data(), 2);
    problem.set_variable_upper_bounds(var_ub.data(), 2);

    // Variable types: both binary (INTEGER with bounds [0,1])
    std::vector<var_t> var_types = {var_t::INTEGER, var_t::INTEGER};
    problem.set_variable_types(var_types.data(), 2);

    // Constraint bounds: x + y >= 1
    std::vector<double> con_lb = {1.0};
    std::vector<double> con_ub = {1e20};
    problem.set_constraint_lower_bounds(con_lb.data(), 1);
    problem.set_constraint_upper_bounds(con_ub.data(), 1);

    return problem;
  }

  ServerProcess server_;
  int port_;
};

// =============================================================================
// Connectivity Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, ServerStartsAndAcceptsConnections)
{
  ASSERT_TRUE(start_server()) << "Failed to start server";
  ASSERT_TRUE(server_.is_running());

  auto client = create_client();
  ASSERT_NE(client, nullptr) << "Failed to connect to server";
  EXPECT_TRUE(client->is_connected());
}

TEST_F(GrpcIntegrationTest, ConnectToNonexistentServer)
{
  // Don't start server
  grpc_client_config_t config;
  config.server_address = "localhost:" + std::to_string(port_);

  grpc_client_t client(config);
  EXPECT_FALSE(client.connect());
  EXPECT_FALSE(client.get_last_error().empty());
}

// =============================================================================
// Job Status Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, CheckStatus_JobNotFound)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  auto status = client->check_status("nonexistent-job-id");
  EXPECT_TRUE(status.success);
  EXPECT_EQ(status.status, job_status_t::NOT_FOUND);
}

// =============================================================================
// Submit and Solve Tests (require GPU)
// =============================================================================

TEST_F(GrpcIntegrationTest, SubmitLPWithManualPolling)
{
  // This test demonstrates manual polling with check_status for users who
  // want to implement custom polling logic (e.g., with progress reporting)
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;

  // Step 1: Submit job
  auto submit_result = client->submit_lp(problem, settings);
  ASSERT_TRUE(submit_result.success) << submit_result.error_message;
  std::string job_id = submit_result.job_id;
  EXPECT_FALSE(job_id.empty());

  // Step 2: Poll with check_status until completion
  job_status_t final_status = job_status_t::QUEUED;
  int poll_count            = 0;
  const int max_polls       = 60;  // 30 seconds max (500ms intervals)

  while (poll_count < max_polls) {
    auto status_result = client->check_status(job_id);
    ASSERT_TRUE(status_result.success) << status_result.error_message;

    final_status = status_result.status;

    // Check if job is done
    if (final_status == job_status_t::COMPLETED || final_status == job_status_t::FAILED ||
        final_status == job_status_t::CANCELLED) {
      break;
    }

    ++poll_count;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  EXPECT_EQ(final_status, job_status_t::COMPLETED)
    << "Job did not complete, status: " << job_status_to_string(final_status);

  // Step 3: Get result
  auto result = client->get_lp_result<int32_t, double>(job_id);
  EXPECT_TRUE(result.success) << result.error_message;
  EXPECT_NE(result.solution, nullptr);

  if (result.solution) { EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0); }
}

TEST_F(GrpcIntegrationTest, SolveLPWithWaitRPC)
{
  ASSERT_TRUE(start_server());

  grpc_client_config_t config;
  config.use_wait = true;  // Use WaitForCompletion RPC

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);
}

// =============================================================================
// MIP Solve Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, SolveMIPBlocking)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  // Simple MIP: min x + 2y, s.t. x + y >= 1, x,y binary
  // Optimal: x=1, y=0, objective=1
  auto problem = create_simple_mip();

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;

  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);

  // Trust the solver - just verify it reported optimal with expected objective
  EXPECT_EQ(result.solution->get_termination_status(), mip_termination_status_t::Optimal);
  EXPECT_NEAR(result.solution->get_objective_value(), 1.0, 0.01);
}

// =============================================================================
// Cancellation Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, CancelNonexistentJob)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  auto result = client->cancel_job("nonexistent-job-id");
  // Cancelling nonexistent job should report the job as NOT_FOUND
  EXPECT_EQ(result.job_status, job_status_t::NOT_FOUND);
}

// =============================================================================
// Delete Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, DeleteNonexistentJob)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  // Deleting a non-existent job should return false with an error message.
  // This prevents silent failures where user thinks wrong job was deleted.
  bool deleted = client->delete_job("nonexistent-job-id");
  EXPECT_FALSE(deleted);
  EXPECT_FALSE(client->get_last_error().empty());
}

// =============================================================================
// Streaming Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, StreamLogs_JobNotFound)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  bool callback_called = false;
  bool result          = client->stream_logs(
    "nonexistent-job-id", 0, [&callback_called](const std::string& line, bool complete) {
      callback_called = true;
      return true;
    });

  // For non-existent job, server should return error status
  // No log messages should be sent, so callback should not be called
  EXPECT_FALSE(callback_called);
  EXPECT_FALSE(result);
}

// =============================================================================
// Error Recovery Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, ClientReconnectsAfterServerRestart)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);
  EXPECT_TRUE(client->is_connected());

  // Verify client works before restart
  auto status_before = client->check_status("test-job");
  EXPECT_TRUE(status_before.success);

  // Stop server
  server_.stop();
  EXPECT_FALSE(server_.is_running());

  // RPC should fail while server is down
  auto status_down = client->check_status("test-job");
  EXPECT_FALSE(status_down.success);

  // Restart server
  ASSERT_TRUE(start_server());

  // gRPC channels auto-reconnect on next RPC - original client should work again
  auto status_after = client->check_status("test-job");
  EXPECT_TRUE(status_after.success)
    << "Original client should auto-reconnect: " << status_after.error_message;
}

TEST_F(GrpcIntegrationTest, ClientHandlesServerCrashDuringSolve)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  // Submit a long-running problem
  std::string mps_path = get_test_mip_path("neos5-free-bound.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 120.0;  // Long time limit

  // Submit the job
  auto submit_result = client->submit_mip(problem, settings);
  ASSERT_TRUE(submit_result.success);
  std::string job_id = submit_result.job_id;

  // Give it a moment to start processing
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Kill the server while job is running
  server_.stop();

  // Trying to check status should fail gracefully with connection error
  auto status_result = client->check_status(job_id);
  EXPECT_FALSE(status_result.success);
  EXPECT_FALSE(status_result.error_message.empty());
  // gRPC typically reports "unavailable" or "failed" for connection issues
  bool has_connection_error =
    status_result.error_message.find("unavailable") != std::string::npos ||
    status_result.error_message.find("Unavailable") != std::string::npos ||
    status_result.error_message.find("failed") != std::string::npos ||
    status_result.error_message.find("Failed") != std::string::npos;
  EXPECT_TRUE(has_connection_error)
    << "Expected connection error, got: " << status_result.error_message;
}

TEST_F(GrpcIntegrationTest, ClientTimeoutConfiguration)
{
  ASSERT_TRUE(start_server());

  // Create client with very short timeout - neos5-free-bound.mps takes much longer
  grpc_client_config_t config;
  config.timeout_seconds  = 1;    // 1 second timeout
  config.poll_interval_ms = 100;  // Fast polling to hit timeout quickly

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  // Submit a problem that takes much longer than 1 second to solve
  std::string mps_path = get_test_mip_path("neos5-free-bound.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;  // Solver has long time limit

  // Use submit + manual wait so we have job_id for cleanup
  auto submit_result = client->submit_mip(problem, settings);
  ASSERT_TRUE(submit_result.success);
  std::string job_id = submit_result.job_id;

  // Poll until timeout - with 1 second timeout, this should fail quickly
  auto start     = std::chrono::steady_clock::now();
  bool completed = false;
  while (!completed) {
    auto elapsed =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
    if (elapsed.count() >= config.timeout_seconds) break;

    auto status = client->check_status(job_id);
    if (status.status == job_status_t::COMPLETED || status.status == job_status_t::FAILED) {
      completed = true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Should have timed out (not completed within 1 second)
  EXPECT_FALSE(completed) << "Complex MIP should not complete in 1 second";

  // Clean up - cancel the still-running job
  client->cancel_job(job_id);
}

TEST_F(GrpcIntegrationTest, MultipleSequentialSolves)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  // Solve multiple problems sequentially to verify no resource leaks
  for (int i = 0; i < 3; ++i) {
    std::string mps_path = get_test_lp_path("afiro_original.mps");
    auto problem         = load_problem_from_mps(mps_path);

    pdlp_solver_settings_t<int32_t, double> settings;
    settings.time_limit = 30.0;

    auto result = client->solve_lp(problem, settings);
    EXPECT_TRUE(result.success) << "Solve #" << i << " failed: " << result.error_message;
    EXPECT_NE(result.solution, nullptr);
    EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);
  }
}

TEST_F(GrpcIntegrationTest, GetResultForNonexistentJob)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  // Try to get result for a job that doesn't exist
  auto result = client->get_lp_result<int32_t, double>("nonexistent-job-12345");
  EXPECT_FALSE(result.success);
}

TEST_F(GrpcIntegrationTest, CancelRunningJob)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  // Submit a long-running problem
  std::string mps_path = get_test_mip_path("neos5-free-bound.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 120.0;

  // Submit the job
  auto submit_result = client->submit_mip(problem, settings);
  ASSERT_TRUE(submit_result.success);
  std::string job_id = submit_result.job_id;

  // Give it a moment to start
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Cancel the job
  auto cancel_result = client->cancel_job(job_id);

  // Should report cancelled status (or completed if it finished quickly)
  EXPECT_TRUE(cancel_result.job_status == job_status_t::CANCELLED ||
              cancel_result.job_status == job_status_t::COMPLETED ||
              cancel_result.job_status == job_status_t::PROCESSING);
}

// =============================================================================
// Streaming Upload/Download Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, StreamingUpload_LargeProblem)
{
  // Start server with small max message size to force streaming
  // 1MB limit will force streaming for cod105_max.mps (2.6MB)
  ASSERT_TRUE(start_server({"--max-message-mb", "1"}));

  grpc_client_config_t config;
  config.timeout_seconds  = 120;         // Longer timeout for large problem
  config.chunk_size_bytes = 256 * 1024;  // 256KB chunks

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  // Load cod105_max.mps (2.6MB) - will require streaming upload
  std::string mps_path = get_test_mip_path("cod105_max.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;  // Limit solve time

  // This should automatically use streaming upload
  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;
  // We don't check the objective since we're limiting time
}

// =============================================================================
// Streaming Robustness Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, StreamingUpload_ServerRestart)
{
  // Test that client handles server restart during/between streaming operations
  ASSERT_TRUE(start_server({"--max-message-mb", "1"}));
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  // First solve should work
  std::string mps_path = get_test_mip_path("sudoku.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result1 = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result1.success) << result1.error_message;

  // Restart server
  server_.stop();
  ASSERT_TRUE(start_server({"--max-message-mb", "1"}));

  // Create new client and solve again
  auto client2 = create_client();
  ASSERT_NE(client2, nullptr);

  auto result2 = client2->solve_mip(problem, settings, false);
  EXPECT_TRUE(result2.success) << result2.error_message;
}

TEST_F(GrpcIntegrationTest, StreamingUpload_VerySmallChunks)
{
  // Test streaming with very small chunk size (stress test chunking logic)
  ASSERT_TRUE(start_server({"--max-message-mb", "1"}));

  grpc_client_config_t config;
  config.timeout_seconds  = 120;
  config.chunk_size_bytes = 64 * 1024;  // Very small 64KB chunks

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_mip_path("sudoku.mps");  // 250KB
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;

  // This should work despite small chunk sizes
  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;
}

// =============================================================================
// Log Streaming Tests (on running jobs)
// =============================================================================

TEST_F(GrpcIntegrationTest, SolveMIPWithLogCallback)
{
  // Test using log_callback in config for automatic log streaming during solve
  ASSERT_TRUE(start_server());

  std::vector<std::string> received_logs;
  std::mutex log_mutex;

  grpc_client_config_t config;
  config.timeout_seconds = 120;
  config.stream_logs     = true;  // Enable automatic log streaming
  config.log_callback    = [&](const std::string& line) {
    std::lock_guard<std::mutex> lock(log_mutex);
    received_logs.push_back(line);
    std::cout << "[Log] " << line << std::endl;
  };

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_mip_path("bb_optimality.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit     = 30.0;
  settings.log_to_console = true;  // Enable logging on server side

  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;

  // Log callback may or may not receive logs depending on timing
  std::cout << "[Test] Received " << received_logs.size() << " log lines" << std::endl;
}

// =============================================================================
// Incumbent Callback Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, IncumbentCallbacks_MIP)
{
  ASSERT_TRUE(start_server());

  // Track incumbent callbacks
  std::vector<double> incumbent_objectives;
  std::mutex incumbent_mutex;

  grpc_client_config_t config;
  config.timeout_seconds = 120;
  // Set incumbent callback in config
  config.incumbent_callback =
    [&](int64_t index, double objective, const std::vector<double>& solution) {
      std::lock_guard<std::mutex> lock(incumbent_mutex);
      incumbent_objectives.push_back(objective);
      std::cout << "[Test] Received incumbent #" << index << " with objective: " << objective
                << std::endl;
      return true;  // Continue solving
    };

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  // Use neos5-free-bound.mps which is known to generate incumbents
  std::string mps_path = get_test_mip_path("neos5-free-bound.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  // Solve with incumbents enabled
  auto result = client->solve_mip(problem, settings, true);
  EXPECT_TRUE(result.success) << result.error_message;
  EXPECT_NE(result.solution, nullptr);

  // Check if we received any incumbents
  // Note: For some problems/time limits, we may not get intermediate incumbents
  std::cout << "[Test] Received " << incumbent_objectives.size() << " incumbent(s)" << std::endl;

  // If we got incumbents, verify they're improving (for minimization)
  if (incumbent_objectives.size() > 1) {
    for (size_t i = 1; i < incumbent_objectives.size(); ++i) {
      // Each incumbent should be <= previous (improving)
      EXPECT_LE(incumbent_objectives[i], incumbent_objectives[i - 1] + 1e-6);
    }
  }
}

TEST_F(GrpcIntegrationTest, IncumbentCallback_CancelsSolve)
{
  ASSERT_TRUE(start_server());

  int callback_count = 0;
  grpc_client_config_t config;
  config.timeout_seconds = 120;
  // Return false after receiving 2 incumbents to cancel the solve
  config.incumbent_callback = [&](int64_t index, double objective, const std::vector<double>&) {
    callback_count++;
    std::cout << "[Test] Incumbent #" << index << " obj=" << objective << " (will cancel after 2)"
              << std::endl;
    return callback_count < 2;  // Return false on 2nd incumbent to cancel
  };

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  // Use neos5-free-bound.mps which generates multiple incumbents
  std::string mps_path = get_test_mip_path("neos5-free-bound.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;

  auto start  = std::chrono::steady_clock::now();
  auto result = client->solve_mip(problem, settings, true);
  auto elapsed =
    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);

  // Solve should have been cancelled (either success=false or terminated early)
  // Key verification: should complete much faster than time_limit
  EXPECT_LT(elapsed.count(), 30) << "Solve should have cancelled early, not run full time limit";
  std::cout << "[Test] Solve completed in " << elapsed.count() << "s with " << callback_count
            << " incumbent callbacks" << std::endl;
}

TEST_F(GrpcIntegrationTest, SolveInfeasibleLP)
{
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  // Create an infeasible LP: x >= 1 AND x <= 0
  cpu_optimization_problem_t<int32_t, double> problem(nullptr);

  // Single variable with conflicting bounds: x >= 1 AND x <= 0 (infeasible!)
  std::vector<double> var_lb = {1.0};  // x >= 1
  std::vector<double> var_ub = {0.0};  // x <= 0
  std::vector<double> obj    = {1.0};  // minimize x

  problem.set_variable_lower_bounds(var_lb.data(), 1);
  problem.set_variable_upper_bounds(var_ub.data(), 1);
  problem.set_objective_coefficients(obj.data(), 1);
  problem.set_maximize(false);

  // Empty constraint matrix (no constraints needed - bounds make it infeasible)
  std::vector<int32_t> offsets = {0};
  problem.set_csr_constraint_matrix(nullptr, 0, nullptr, 0, offsets.data(), 1);
  problem.set_constraint_lower_bounds(nullptr, 0);
  problem.set_constraint_upper_bounds(nullptr, 0);

  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result = client->solve_lp(problem, settings);

  // Should complete (not hang) but indicate non-optimal status
  // The solve itself should succeed in terms of RPC, but solution status indicates infeasibility
  if (result.success && result.solution) {
    // PDLP may return with a non-optimal termination status
    auto status = result.solution->get_termination_status();
    std::cout << "[Test] Infeasible LP returned status: " << static_cast<int>(status) << std::endl;
    // Should not be OPTIMAL
    EXPECT_NE(status, pdlp_termination_status_t::Optimal);
  }
  // Even if it fails, the test passes - we just want to ensure no hang/crash
}

TEST_F(GrpcIntegrationTest, ExplicitAsyncLPFlow)
{
  // Test the explicit async workflow: submit -> poll -> get_result
  ASSERT_TRUE(start_server());
  auto client = create_client();
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;

  // Step 1: Submit
  auto submit_result = client->submit_lp(problem, settings);
  ASSERT_TRUE(submit_result.success) << submit_result.error_message;
  ASSERT_FALSE(submit_result.job_id.empty());
  std::string job_id = submit_result.job_id;
  std::cout << "[Test] Submitted job: " << job_id << std::endl;

  // Step 2: Poll until complete
  job_status_t final_status = job_status_t::QUEUED;
  for (int i = 0; i < 120; ++i) {  // Max 60 seconds
    auto status = client->check_status(job_id);
    ASSERT_TRUE(status.success) << status.error_message;

    if (status.status == job_status_t::COMPLETED) {
      final_status = job_status_t::COMPLETED;
      std::cout << "[Test] Job completed after " << i * 500 << "ms" << std::endl;
      break;
    } else if (status.status == job_status_t::FAILED) {
      final_status = job_status_t::FAILED;
      FAIL() << "Job failed: " << status.message;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  ASSERT_EQ(final_status, job_status_t::COMPLETED) << "Job did not complete in time";

  // Step 3: Get result explicitly
  auto result = client->get_lp_result<int32_t, double>(job_id);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);

  // Step 4: Clean up
  bool deleted = client->delete_job(job_id);
  EXPECT_TRUE(deleted);
}

TEST_F(GrpcIntegrationTest, ConcurrentJobSubmission)
{
  // Test submitting multiple jobs and polling them concurrently
  // Server uses single worker, so jobs are queued and processed sequentially
  // but client should handle multiple outstanding jobs correctly
  ASSERT_TRUE(start_server());

  // Use two separate clients to simulate independent users
  auto client1 = create_client();
  auto client2 = create_client();
  ASSERT_NE(client1, nullptr);
  ASSERT_NE(client2, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;

  // Submit 3 jobs rapidly (2 from client1, 1 from client2)
  std::vector<std::pair<grpc_client_t*, std::string>> jobs;

  auto submit1 = client1->submit_lp(problem, settings);
  ASSERT_TRUE(submit1.success) << submit1.error_message;
  jobs.push_back({client1.get(), submit1.job_id});
  std::cout << "[Test] Client1 submitted job 1: " << submit1.job_id << std::endl;

  auto submit2 = client2->submit_lp(problem, settings);
  ASSERT_TRUE(submit2.success) << submit2.error_message;
  jobs.push_back({client2.get(), submit2.job_id});
  std::cout << "[Test] Client2 submitted job 2: " << submit2.job_id << std::endl;

  auto submit3 = client1->submit_lp(problem, settings);
  ASSERT_TRUE(submit3.success) << submit3.error_message;
  jobs.push_back({client1.get(), submit3.job_id});
  std::cout << "[Test] Client1 submitted job 3: " << submit3.job_id << std::endl;

  // Poll all jobs until complete
  std::vector<bool> completed(3, false);
  int completed_count = 0;

  for (int poll = 0; poll < 120 && completed_count < 3; ++poll) {  // Max 60 seconds
    for (size_t i = 0; i < jobs.size(); ++i) {
      if (completed[i]) continue;

      auto status = jobs[i].first->check_status(jobs[i].second);
      ASSERT_TRUE(status.success) << "Job " << i
                                  << " status check failed: " << status.error_message;

      if (status.status == job_status_t::COMPLETED) {
        completed[i] = true;
        completed_count++;
        std::cout << "[Test] Job " << i << " (" << jobs[i].second << ") completed" << std::endl;
      } else if (status.status == job_status_t::FAILED) {
        FAIL() << "Job " << i << " failed: " << status.message;
      }
    }

    if (completed_count < 3) { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }
  }

  ASSERT_EQ(completed_count, 3) << "Not all jobs completed in time";

  // Verify all results are correct
  for (size_t i = 0; i < jobs.size(); ++i) {
    auto result = jobs[i].first->get_lp_result<int32_t, double>(jobs[i].second);
    EXPECT_TRUE(result.success) << "Job " << i << " get_result failed: " << result.error_message;
    ASSERT_NE(result.solution, nullptr) << "Job " << i << " has null solution";
    EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0)
      << "Job " << i << " has wrong objective";
  }

  // Clean up all jobs
  for (size_t i = 0; i < jobs.size(); ++i) {
    bool deleted = jobs[i].first->delete_job(jobs[i].second);
    EXPECT_TRUE(deleted) << "Failed to delete job " << i;
  }

  std::cout << "[Test] All 3 concurrent jobs completed successfully" << std::endl;
}

// =============================================================================
// TLS Tests
// =============================================================================

class GrpcTlsIntegrationTest : public GrpcIntegrationTest {
 protected:
  std::string certs_dir_;
  bool generated_certs_ = false;

  void SetUp() override
  {
    GrpcIntegrationTest::SetUp();
    // Find certs directory using same env vars as CI (ci/test_self_hosted_service.sh)
    // CI sets: CERT_FOLDER, CUOPT_SSL_CERTFILE, CUOPT_SSL_KEYFILE, CLIENT_CERT

    // Option 1: CERT_FOLDER (matches CI script variable name)
    const char* cert_folder = std::getenv("CERT_FOLDER");
    if (cert_folder) {
      certs_dir_ = cert_folder;
      return;
    }

    // Option 2: Derive from CI's CUOPT_SSL_CERTFILE (parent directory)
    // CI sets: CUOPT_SSL_CERTFILE=${CERT_FOLDER}/server.crt
    const char* ssl_certfile = std::getenv("CUOPT_SSL_CERTFILE");
    if (ssl_certfile) {
      certs_dir_ = std::filesystem::path(ssl_certfile).parent_path().string();
      return;
    }

    // Option 3: Generate temporary certs for testing
    certs_dir_ = "/tmp/cuopt_test_certs_" + std::to_string(getpid());
    if (generate_test_certs()) {
      generated_certs_ = true;
    } else {
      certs_dir_ = "";
    }
  }

  void TearDown() override
  {
    GrpcIntegrationTest::TearDown();
    // Clean up generated certs
    if (generated_certs_ && !certs_dir_.empty()) { std::filesystem::remove_all(certs_dir_); }
  }

  bool generate_test_certs()
  {
    // Create temp directory
    std::filesystem::create_directories(certs_dir_);

    // Generate CA key and cert
    std::string ca_key = certs_dir_ + "/ca.key";
    std::string ca_crt = certs_dir_ + "/ca.crt";
    std::string cmd = "openssl req -x509 -newkey rsa:2048 -keyout " + ca_key + " -out " + ca_crt +
                      " -days 1 -nodes -subj '/CN=TestCA' 2>/dev/null";
    if (std::system(cmd.c_str()) != 0) return false;

    // Generate server key and CSR
    std::string server_key = certs_dir_ + "/server.key";
    std::string server_csr = certs_dir_ + "/server.csr";
    std::string server_crt = certs_dir_ + "/server.crt";
    cmd = "openssl req -newkey rsa:2048 -keyout " + server_key + " -out " + server_csr +
          " -nodes -subj '/CN=localhost' 2>/dev/null";
    if (std::system(cmd.c_str()) != 0) return false;

    // Sign server cert with CA
    cmd = "openssl x509 -req -in " + server_csr + " -CA " + ca_crt + " -CAkey " + ca_key +
          " -CAcreateserial -out " + server_crt + " -days 1 2>/dev/null";
    if (std::system(cmd.c_str()) != 0) return false;

    // Generate client key and CSR
    std::string client_key = certs_dir_ + "/client.key";
    std::string client_csr = certs_dir_ + "/client.csr";
    std::string client_crt = certs_dir_ + "/client.crt";
    cmd = "openssl req -newkey rsa:2048 -keyout " + client_key + " -out " + client_csr +
          " -nodes -subj '/CN=TestClient' 2>/dev/null";
    if (std::system(cmd.c_str()) != 0) return false;

    // Sign client cert with CA
    cmd = "openssl x509 -req -in " + client_csr + " -CA " + ca_crt + " -CAkey " + ca_key +
          " -CAcreateserial -out " + client_crt + " -days 1 2>/dev/null";
    if (std::system(cmd.c_str()) != 0) return false;

    return true;
  }

  bool certs_exist()
  {
    if (certs_dir_.empty()) return false;
    return std::filesystem::exists(certs_dir_ + "/server.crt") &&
           std::filesystem::exists(certs_dir_ + "/server.key") &&
           std::filesystem::exists(certs_dir_ + "/ca.crt");
  }

  bool client_certs_exist()
  {
    return certs_exist() && std::filesystem::exists(certs_dir_ + "/client.crt") &&
           std::filesystem::exists(certs_dir_ + "/client.key");
  }

  std::string read_file(const std::string& path)
  {
    std::ifstream file(path);
    if (!file) return "";
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }

  bool start_tls_server(bool require_client_cert = false)
  {
    std::vector<std::string> args = {"--tls",
                                     "--tls-cert",
                                     certs_dir_ + "/server.crt",
                                     "--tls-key",
                                     certs_dir_ + "/server.key",
                                     "--tls-root",
                                     certs_dir_ + "/ca.crt"};
    if (require_client_cert) { args.push_back("--require-client-cert"); }

    // Configure TLS for the health check connection
    std::string root_certs = read_file(certs_dir_ + "/ca.crt");
    std::string client_cert, client_key;
    if (require_client_cert) {
      client_cert = read_file(certs_dir_ + "/client.crt");
      client_key  = read_file(certs_dir_ + "/client.key");
    }
    server_.set_tls_config(root_certs, client_cert, client_key);

    return start_server(args);
  }

  std::unique_ptr<grpc_client_t> create_tls_client(bool with_client_cert = false)
  {
    grpc_client_config_t config;
    config.server_address  = "localhost:" + std::to_string(port_);
    config.timeout_seconds = 30;
    config.enable_tls      = true;
    config.tls_root_certs  = read_file(certs_dir_ + "/ca.crt");

    if (with_client_cert) {
      config.tls_client_cert = read_file(certs_dir_ + "/client.crt");
      config.tls_client_key  = read_file(certs_dir_ + "/client.key");
    }

    auto client = std::make_unique<grpc_client_t>(config);
    if (!client->connect()) { return nullptr; }
    return client;
  }
};

TEST_F(GrpcTlsIntegrationTest, TLS_BasicConnection)
{
  if (!certs_exist()) { GTEST_SKIP() << "TLS certificates not found in " << certs_dir_; }

  ASSERT_TRUE(start_tls_server(false));
  auto client = create_tls_client(false);
  ASSERT_NE(client, nullptr) << "Failed to connect with TLS";
  EXPECT_TRUE(client->is_connected());
}

TEST_F(GrpcTlsIntegrationTest, TLS_SolveLP)
{
  if (!certs_exist()) { GTEST_SKIP() << "TLS certificates not found"; }

  ASSERT_TRUE(start_tls_server(false));
  auto client = create_tls_client(false);
  ASSERT_NE(client, nullptr);

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 30.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;
  ASSERT_NE(result.solution, nullptr);
  EXPECT_NEAR(result.solution->get_objective_value(), -464.753, 1.0);
}

TEST_F(GrpcTlsIntegrationTest, mTLS_Connection)
{
  if (!client_certs_exist()) { GTEST_SKIP() << "mTLS certificates not found"; }

  // Start server requiring client certificates
  ASSERT_TRUE(start_tls_server(true));

  // Connect with client certificate
  auto client = create_tls_client(true);
  ASSERT_NE(client, nullptr) << "Failed to connect with mTLS";
  EXPECT_TRUE(client->is_connected());
}

TEST_F(GrpcTlsIntegrationTest, mTLS_RejectsClientWithoutCert)
{
  if (!client_certs_exist()) { GTEST_SKIP() << "mTLS certificates not found"; }

  // Start server requiring client certificates
  ASSERT_TRUE(start_tls_server(true));

  // Try to connect WITHOUT client certificate - should fail
  auto client = create_tls_client(false);
  EXPECT_EQ(client, nullptr) << "Server should reject client without certificate";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
