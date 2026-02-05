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

#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/optimization_problem_utils.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/grpc_client.hpp>
#include <mps_parser/parser.hpp>

#include "grpc_test_log_capture.hpp"

// For direct gRPC access in streaming tests
#include <cuopt_remote_service.grpc.pb.h>
#include <grpcpp/grpcpp.h>

// For building serialized request messages
#include <cuopt/linear_programming/utilities/grpc_service_mapper.hpp>

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <random>
#include <string>
#include <thread>

using namespace cuopt::linear_programming;
using cuopt::linear_programming::testing::GrpcTestLogCapture;

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

  /**
   * @brief Get the path to the server log file
   */
  std::string log_path() const
  {
    if (port_ <= 0) return "";
    return "/tmp/cuopt_test_server_" + std::to_string(port_) + ".log";
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
   * Always enables transfer hash logging for test verification
   */
  bool start_server(const std::vector<std::string>& extra_args = {})
  {
    // Always enable transfer hash for tests
    std::vector<std::string> args = {"--enable-transfer-hash"};
    args.insert(args.end(), extra_args.begin(), extra_args.end());
    return server_.start(port_, args);
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

    // Enable transfer hash for test verification
    config.enable_transfer_hash = true;

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
// Log Capture and Verification Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, LogCapture_ClientDebugLogs)
{
  // Test that client debug logs are captured via the debug_log_callback
  ASSERT_TRUE(start_server());

  GrpcTestLogCapture log_capture;

  grpc_client_config_t config;
  config.debug_log_callback = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  // Verify connection debug logs were captured
  EXPECT_TRUE(log_capture.client_log_contains("Connecting to"))
    << "Expected connection log. Captured logs:\n"
    << log_capture.get_client_logs();
  EXPECT_TRUE(log_capture.client_log_contains("Connected successfully"))
    << "Expected success log. Captured logs:\n"
    << log_capture.get_client_logs();
}

TEST_F(GrpcIntegrationTest, LogCapture_ClientConnectionFailure)
{
  // Test that connection failure logs are captured
  GrpcTestLogCapture log_capture;

  grpc_client_config_t config;
  config.server_address     = "localhost:" + std::to_string(port_);  // Server not started
  config.debug_log_callback = log_capture.client_callback();

  grpc_client_t client(config);
  EXPECT_FALSE(client.connect());

  // Verify failure logs were captured
  EXPECT_TRUE(log_capture.client_log_contains("Connection failed"))
    << "Expected failure log. Captured logs:\n"
    << log_capture.get_client_logs();
}

TEST_F(GrpcIntegrationTest, LogCapture_ServerLogs)
{
  // Test that server logs can be read from the log file
  ASSERT_TRUE(start_server());

  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_.log_path());
  log_capture.mark_test_start();  // Only capture logs from this point forward

  auto client = create_client();
  ASSERT_NE(client, nullptr);

  // Submit a simple problem to generate server logs
  auto problem = create_simple_mip();
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;

  // Give server a moment to flush logs
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Verify server logs contain expected entries (only from this test)
  std::string server_logs = log_capture.get_server_logs();
  EXPECT_FALSE(server_logs.empty()) << "Server logs should not be empty";

  // Server should log worker processing
  EXPECT_TRUE(log_capture.server_log_contains("[Worker"))
    << "Expected worker logs. Server log path: " << server_.log_path();
}

TEST_F(GrpcIntegrationTest, LogCapture_SubmitJobLogs)
{
  // Test that job submission logs are captured
  ASSERT_TRUE(start_server());

  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_.log_path());

  grpc_client_config_t config;
  config.debug_log_callback = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  // Mark test start AFTER connection (so we don't capture connection logs in server)
  log_capture.mark_test_start();

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);
  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;

  // Verify client captured the job submission logs
  EXPECT_TRUE(log_capture.client_log_contains("submit_or_upload"))
    << "Expected submit_or_upload log. Captured client logs:\n"
    << log_capture.get_client_logs();
  EXPECT_TRUE(log_capture.client_log_contains("job_id="))
    << "Expected job_id in logs. Captured client logs:\n"
    << log_capture.get_client_logs();

  // Verify we can use pattern matching for job_id format
  EXPECT_TRUE(log_capture.client_log_contains_pattern("job_id=[a-f0-9-]+"))
    << "Expected job_id pattern. Captured client logs:\n"
    << log_capture.get_client_logs();
}

TEST_F(GrpcIntegrationTest, LogCapture_WaitForServerLog)
{
  // Test the wait_for_server_log functionality
  ASSERT_TRUE(start_server());

  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_.log_path());

  grpc_client_config_t config;
  config.debug_log_callback = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  // Mark test start - only look for logs from this point forward
  log_capture.mark_test_start();

  // Submit a problem
  auto problem = create_simple_mip();
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto submit_result = client->submit_mip(problem, settings);
  ASSERT_TRUE(submit_result.success) << submit_result.error_message;

  // Wait for the server to log that it's processing the job (with timeout)
  // This will only find logs written after mark_test_start()
  bool found = log_capture.wait_for_server_log("Processing job", 10000, 100);
  EXPECT_TRUE(found) << "Server should log 'Processing job' within 10 seconds";
}

TEST_F(GrpcIntegrationTest, LogCapture_CountOccurrences)
{
  // Test log counting functionality
  GrpcTestLogCapture log_capture;

  // Add some test logs
  log_capture.add_client_log("Test message 1");
  log_capture.add_client_log("Test message 2");
  log_capture.add_client_log("Another test");
  log_capture.add_client_log("Test message 3");

  // Count occurrences
  EXPECT_EQ(log_capture.client_log_count("Test message"), 3);
  EXPECT_EQ(log_capture.client_log_count("Another"), 1);
  EXPECT_EQ(log_capture.client_log_count("Not found"), 0);
}

TEST_F(GrpcIntegrationTest, LogCapture_TestIsolation)
{
  // Test that mark_test_start() properly isolates logs between test phases
  ASSERT_TRUE(start_server());

  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_.log_path());

  grpc_client_config_t config;
  config.debug_log_callback = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  // Phase 1: Submit a job BEFORE marking test start
  auto problem = create_simple_mip();
  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result1 = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result1.success) << result1.error_message;

  // Give server time to log
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Now mark test start - should NOT see Phase 1 logs after this
  log_capture.mark_test_start();

  // Verify Phase 1 logs are NOT visible (because they're before the mark)
  EXPECT_FALSE(log_capture.server_log_contains("Processing job"))
    << "Should NOT see logs from before mark_test_start()";

  // Phase 2: Submit another job AFTER marking test start
  auto result2 = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result2.success) << result2.error_message;

  // Give server time to log
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Verify Phase 2 logs ARE visible
  EXPECT_TRUE(log_capture.server_log_contains("Processing job"))
    << "Should see logs from after mark_test_start()";

  // But we can still access all logs if needed for debugging
  std::string all_logs = log_capture.get_all_server_logs();
  // Count "Processing job" occurrences - should be at least 2 (from both phases)
  int count  = 0;
  size_t pos = 0;
  while ((pos = all_logs.find("Processing job", pos)) != std::string::npos) {
    ++count;
    pos += 14;  // length of "Processing job"
  }
  EXPECT_GE(count, 2) << "Should have at least 2 'Processing job' entries in full log";
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
    settings.time_limit = 10.0;

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
  settings.time_limit = 5.0;  // Short time - we're testing streaming, not solve accuracy

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
  settings.time_limit = 10.0;

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

TEST_F(GrpcIntegrationTest, StreamingUpload_OutOfOrderChunks)
{
  // Test that server properly handles chunks sent out of order by solving a real LP problem
  // We send chunks in non-sequential order and verify the solution is correct
  ASSERT_TRUE(start_server({"--max-message-mb", "1"}));

  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_.log_path());
  log_capture.mark_test_start();

  // Load and serialize afiro_original.mps problem with settings (as SolveLPRequest)
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);

  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;

  // Build and serialize the full SolveLPRequest (problem + settings)
  auto submit_request = build_lp_submit_request(problem, settings);
  const auto& lp_req  = submit_request.lp_request();
  std::string serialized_problem;
  ASSERT_TRUE(lp_req.SerializeToString(&serialized_problem));

  // Create a gRPC channel directly
  auto channel =
    grpc::CreateChannel("localhost:" + std::to_string(port_), grpc::InsecureChannelCredentials());
  auto stub = cuopt::remote::CuOptRemoteService::NewStub(channel);

  grpc::ClientContext context;
  auto stream = stub->UploadAndSubmit(&context);

  // 1. Send start message
  cuopt::remote::UploadJobRequest start_req;
  auto* start = start_req.mutable_start();
  start->set_problem_type(cuopt::remote::LP);
  start->set_resume(false);
  start->set_total_size(serialized_problem.size());

  ASSERT_TRUE(stream->Write(start_req));

  // 2. Read start ack
  cuopt::remote::UploadJobResponse start_resp;
  ASSERT_TRUE(stream->Read(&start_resp));
  ASSERT_TRUE(start_resp.has_ack())
    << "Expected ack, got error: "
    << (start_resp.has_error() ? start_resp.error().message() : "unknown");
  std::string upload_id = start_resp.ack().upload_id();

  // 3. Send chunks OUT OF ORDER
  // Use small chunks to ensure multiple chunks and meaningful reordering
  const size_t chunk_size = 1024;  // 1KB chunks for afiro (~5KB problem)
  std::vector<size_t> offsets;
  for (size_t i = 0; i < serialized_problem.size(); i += chunk_size) {
    offsets.push_back(i);
  }

  // Shuffle offsets to create out-of-order sequence
  // Reverse the order for a clearly non-sequential pattern
  std::reverse(offsets.begin(), offsets.end());

  std::cout << "[Test] Sending " << offsets.size() << " chunks in reverse order" << std::endl;

  for (size_t offset : offsets) {
    size_t n = std::min(chunk_size, serialized_problem.size() - offset);

    cuopt::remote::UploadJobRequest chunk_req;
    auto* chunk = chunk_req.mutable_chunk();
    chunk->set_upload_id(upload_id);
    chunk->set_offset(static_cast<int64_t>(offset));
    chunk->set_data(serialized_problem.data() + offset, n);

    ASSERT_TRUE(stream->Write(chunk_req)) << "Failed to write chunk at offset " << offset;

    cuopt::remote::UploadJobResponse chunk_resp;
    ASSERT_TRUE(stream->Read(&chunk_resp)) << "Failed to read ack for chunk at offset " << offset;

    ASSERT_FALSE(chunk_resp.has_error())
      << "Server rejected chunk at offset " << offset << ": " << chunk_resp.error().message();
    ASSERT_TRUE(chunk_resp.has_ack());
  }

  // 4. Send finish message
  cuopt::remote::UploadJobRequest finish_req;
  auto* finish = finish_req.mutable_finish();
  finish->set_upload_id(upload_id);
  ASSERT_TRUE(stream->Write(finish_req));

  // 5. Read finish response with job_id
  cuopt::remote::UploadJobResponse finish_resp;
  ASSERT_TRUE(stream->Read(&finish_resp));
  ASSERT_TRUE(finish_resp.has_submit()) << "Expected submit response";
  std::string job_id = finish_resp.submit().job_id();
  ASSERT_FALSE(job_id.empty()) << "Job ID should not be empty";

  stream->WritesDone();
  grpc::Status status = stream->Finish();
  ASSERT_TRUE(status.ok()) << "Stream finish failed: " << status.error_message();

  std::cout << "[Test] Out-of-order upload succeeded, job_id=" << job_id << std::endl;

  // 6. Wait for job completion and get result using a normal client
  grpc_client_config_t config;
  config.timeout_seconds = 60;
  auto client            = create_client(config);
  ASSERT_NE(client, nullptr);

  // Wait for completion
  auto wait_result = client->wait_for_completion(job_id);
  ASSERT_TRUE(wait_result.success) << "Wait failed: " << wait_result.error_message;
  ASSERT_EQ(wait_result.status, job_status_t::COMPLETED) << "Job did not complete successfully";

  // Get the result
  auto lp_result = client->get_lp_result<int32_t, double>(job_id);
  ASSERT_TRUE(lp_result.success) << "Get result failed: " << lp_result.error_message;
  ASSERT_TRUE(lp_result.solution != nullptr) << "Solution should not be null";

  // AFIRO optimal objective is approximately -464.75 (minimization problem)
  const double expected_objective = -464.75;
  const double tolerance          = 1.0;  // Allow some tolerance for solver differences

  std::cout << "[Test] Result objective: " << lp_result.solution->get_objective_value()
            << std::endl;
  std::cout << "[Test] Expected objective: " << expected_objective << std::endl;

  EXPECT_NEAR(lp_result.solution->get_objective_value(), expected_objective, tolerance)
    << "Objective value incorrect - server may have corrupted the problem data during "
       "out-of-order chunk reassembly";

  // Verify server logs show the upload was processed
  std::string server_logs = log_capture.get_server_logs();
  EXPECT_TRUE(server_logs.find("UploadAndSubmit") != std::string::npos)
    << "Server logs should mention UploadAndSubmit";
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
// Selective Retransmission Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, SelectiveRetransmission_UploadWithMissingChunks)
{
  // Test that server correctly requests retransmission of missing chunks
  // This test simulates chunk loss by sending an incomplete upload, then resending
  ASSERT_TRUE(start_server({"--max-message-mb", "1", "--verbose"}));

  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_.log_path());
  log_capture.mark_test_start();

  // Load and serialize afiro_original.mps problem
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);

  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;

  auto submit_request = build_lp_submit_request(problem, settings);
  const auto& lp_req  = submit_request.lp_request();
  std::string serialized_problem;
  ASSERT_TRUE(lp_req.SerializeToString(&serialized_problem));

  // Create gRPC channel directly
  auto channel =
    grpc::CreateChannel("localhost:" + std::to_string(port_), grpc::InsecureChannelCredentials());
  auto stub = cuopt::remote::CuOptRemoteService::NewStub(channel);

  grpc::ClientContext context;
  auto stream = stub->UploadAndSubmit(&context);

  // 1. Send start message
  cuopt::remote::UploadJobRequest start_req;
  auto* start = start_req.mutable_start();
  start->set_problem_type(cuopt::remote::LP);
  start->set_resume(false);
  start->set_total_size(serialized_problem.size());
  ASSERT_TRUE(stream->Write(start_req));

  cuopt::remote::UploadJobResponse start_resp;
  ASSERT_TRUE(stream->Read(&start_resp));
  ASSERT_TRUE(start_resp.has_ack());
  std::string upload_id = start_resp.ack().upload_id();

  // 2. Send only SOME chunks (skip every other chunk to create gaps)
  const size_t chunk_size = 512;  // Small chunks to ensure multiple
  size_t offset           = 0;
  std::vector<size_t> skipped_offsets;

  while (offset < serialized_problem.size()) {
    size_t n = std::min(chunk_size, serialized_problem.size() - offset);

    // Skip every other chunk
    bool skip = (offset / chunk_size) % 2 == 1;
    if (skip) {
      skipped_offsets.push_back(offset);
      offset += n;
      continue;
    }

    cuopt::remote::UploadJobRequest chunk_req;
    auto* chunk = chunk_req.mutable_chunk();
    chunk->set_upload_id(upload_id);
    chunk->set_offset(static_cast<int64_t>(offset));
    chunk->set_data(serialized_problem.data() + offset, n);

    ASSERT_TRUE(stream->Write(chunk_req));

    cuopt::remote::UploadJobResponse chunk_resp;
    ASSERT_TRUE(stream->Read(&chunk_resp));
    ASSERT_TRUE(chunk_resp.has_ack()) << "Expected ack, got error";

    offset += n;
  }

  std::cout << "[Test] Sent chunks with " << skipped_offsets.size() << " gaps" << std::endl;

  // 3. Send finish - server should request resend of missing chunks
  cuopt::remote::UploadJobRequest finish_req;
  finish_req.mutable_finish()->set_upload_id(upload_id);
  ASSERT_TRUE(stream->Write(finish_req));

  cuopt::remote::UploadJobResponse finish_resp;
  ASSERT_TRUE(stream->Read(&finish_resp));

  // Server should request resend of missing ranges
  ASSERT_TRUE(finish_resp.has_resend())
    << "Expected ResendRequest, got: "
    << (finish_resp.has_ack()      ? "ack"
        : finish_resp.has_submit() ? "submit"
        : finish_resp.has_error()  ? "error: " + finish_resp.error().message()
                                   : "unknown");

  std::cout << "[Test] Server requested resend of " << finish_resp.resend().missing_ranges_size()
            << " ranges" << std::endl;

  // 4. Resend the missing chunks
  for (const auto& range : finish_resp.resend().missing_ranges()) {
    int64_t range_offset = range.offset();
    int64_t range_end    = range_offset + range.size();

    while (range_offset < range_end) {
      size_t n =
        std::min(static_cast<size_t>(chunk_size), static_cast<size_t>(range_end - range_offset));

      cuopt::remote::UploadJobRequest chunk_req;
      auto* chunk = chunk_req.mutable_chunk();
      chunk->set_upload_id(upload_id);
      chunk->set_offset(range_offset);
      chunk->set_data(serialized_problem.data() + range_offset, n);

      ASSERT_TRUE(stream->Write(chunk_req));

      cuopt::remote::UploadJobResponse chunk_resp;
      ASSERT_TRUE(stream->Read(&chunk_resp));
      ASSERT_TRUE(chunk_resp.has_ack());

      range_offset += static_cast<int64_t>(n);
    }
  }

  // 5. Send finish again
  ASSERT_TRUE(stream->Write(finish_req));

  cuopt::remote::UploadJobResponse final_resp;
  ASSERT_TRUE(stream->Read(&final_resp));
  ASSERT_TRUE(final_resp.has_submit())
    << "Expected submit, got: "
    << (final_resp.has_resend()  ? "resend (still missing data)"
        : final_resp.has_error() ? "error: " + final_resp.error().message()
                                 : "unknown");

  std::string job_id = final_resp.submit().job_id();
  std::cout << "[Test] Upload with retransmission succeeded, job_id=" << job_id << std::endl;

  stream->WritesDone();
  ASSERT_TRUE(stream->Finish().ok());

  // 6. Verify the solution is correct
  grpc_client_config_t config;
  config.timeout_seconds = 60;
  auto client            = create_client(config);
  ASSERT_NE(client, nullptr);

  auto wait_result = client->wait_for_completion(job_id);
  ASSERT_TRUE(wait_result.success) << wait_result.error_message;
  ASSERT_EQ(wait_result.status, job_status_t::COMPLETED);

  auto lp_result = client->get_lp_result<int32_t, double>(job_id);
  ASSERT_TRUE(lp_result.success) << lp_result.error_message;
  ASSERT_TRUE(lp_result.solution != nullptr) << "Solution should not be null";

  const double expected_objective = -464.75;
  EXPECT_NEAR(lp_result.solution->get_objective_value(), expected_objective, 1.0)
    << "Incorrect result after retransmission - data may have been corrupted";

  // Check server logs for resend activity
  std::string server_logs = log_capture.get_server_logs();
  EXPECT_TRUE(server_logs.find("requesting resend") != std::string::npos ||
              server_logs.find("ResendRequest") != std::string::npos)
    << "Server logs should show resend activity";
}

TEST_F(GrpcIntegrationTest, SelectiveRetransmission_NormalUploadNoResend)
{
  // Test that normal complete uploads don't trigger unnecessary resends
  ASSERT_TRUE(start_server({"--max-message-mb", "0"}));  // Force streaming

  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_.log_path());

  grpc_client_config_t config;
  config.debug_log_callback = log_capture.client_callback();
  config.max_message_bytes  = 32 * 1024;  // Force streaming
  config.chunk_size_bytes   = 8 * 1024;

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  log_capture.mark_test_start();

  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);

  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;

  // Verify no resend requests occurred
  std::string client_logs = log_capture.get_client_logs();
  EXPECT_TRUE(client_logs.find("requested resend") == std::string::npos)
    << "No resend should occur for complete uploads";
  EXPECT_TRUE(client_logs.find("Requesting resend") == std::string::npos)
    << "No resend should occur for complete uploads";
}

// =============================================================================
// Timeout and Robustness Tests
// =============================================================================

TEST_F(GrpcIntegrationTest, ServerTimeout_AbandonedUpload)
{
  // Test that server correctly times out when client abandons upload mid-stream
  // Use short timeout (3 seconds) so test completes quickly

  ASSERT_TRUE(start_server({"--max-message-mb", "1", "--chunk-timeout", "3", "--verbose"}));

  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_.log_path());
  log_capture.mark_test_start();

  // Create gRPC channel directly for low-level control
  auto channel =
    grpc::CreateChannel("localhost:" + std::to_string(port_), grpc::InsecureChannelCredentials());
  auto stub = cuopt::remote::CuOptRemoteService::NewStub(channel);

  grpc::ClientContext context;
  auto stream = stub->UploadAndSubmit(&context);

  // 1. Send start message with a large total_size
  cuopt::remote::UploadJobRequest start_req;
  auto* start = start_req.mutable_start();
  start->set_problem_type(cuopt::remote::LP);
  start->set_resume(false);
  start->set_total_size(100000);  // Claim we'll send 100KB
  ASSERT_TRUE(stream->Write(start_req));

  cuopt::remote::UploadJobResponse start_resp;
  ASSERT_TRUE(stream->Read(&start_resp));
  ASSERT_TRUE(start_resp.has_ack());
  std::string upload_id = start_resp.ack().upload_id();
  std::cout << "[Test] Started upload " << upload_id << ", sending one chunk then abandoning"
            << std::endl;

  // 2. Send ONE chunk (10 bytes) then stop - don't send finish
  cuopt::remote::UploadJobRequest chunk_req;
  auto* chunk = chunk_req.mutable_chunk();
  chunk->set_upload_id(upload_id);
  chunk->set_offset(0);
  chunk->set_data("0123456789");  // Only 10 bytes of claimed 100KB
  ASSERT_TRUE(stream->Write(chunk_req));

  cuopt::remote::UploadJobResponse chunk_resp;
  ASSERT_TRUE(stream->Read(&chunk_resp));
  ASSERT_TRUE(chunk_resp.has_ack());

  std::cout << "[Test] Sent one chunk, now waiting for server timeout (~3 seconds)..." << std::endl;

  // 3. DON'T send finish - just abandon the stream
  // Try to read - server should eventually timeout and close the stream
  auto start_time = std::chrono::steady_clock::now();

  cuopt::remote::UploadJobResponse timeout_resp;
  bool got_response = stream->Read(&timeout_resp);

  auto elapsed =
    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time)
      .count();

  std::cout << "[Test] Server responded after " << elapsed
            << " seconds, got_response=" << (got_response ? "true" : "false") << std::endl;

  // Server should have closed the stream (Read returns false) or sent an error
  // The timeout should be around 3 seconds (not immediate, not much longer)
  EXPECT_GE(elapsed, 2) << "Server should wait at least 2 seconds before timeout";
  EXPECT_LE(elapsed, 10) << "Server should timeout within 10 seconds";

  // Clean up
  stream->WritesDone();
  grpc::Status status = stream->Finish();

  std::cout << "[Test] Stream finished with status: " << status.error_code() << " - "
            << status.error_message() << std::endl;

  // Check server logs for timeout message
  std::string server_logs = log_capture.get_server_logs();
  bool has_timeout_log    = server_logs.find("timeout") != std::string::npos ||
                         server_logs.find("Timeout") != std::string::npos ||
                         server_logs.find("TIMEOUT") != std::string::npos;

  std::cout << "[Test] Server logs mention timeout: " << (has_timeout_log ? "yes" : "no")
            << std::endl;
}

TEST_F(GrpcIntegrationTest, ServerTimeout_NoChunksSent)
{
  // Test that server times out when client sends start but never sends any chunks
  // This tests the "stalled at start" scenario

  ASSERT_TRUE(start_server({"--max-message-mb", "1", "--chunk-timeout", "3", "--verbose"}));

  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_.log_path());
  log_capture.mark_test_start();

  auto channel =
    grpc::CreateChannel("localhost:" + std::to_string(port_), grpc::InsecureChannelCredentials());
  auto stub = cuopt::remote::CuOptRemoteService::NewStub(channel);

  grpc::ClientContext context;
  auto stream = stub->UploadAndSubmit(&context);

  // Send start message only
  cuopt::remote::UploadJobRequest start_req;
  auto* start = start_req.mutable_start();
  start->set_problem_type(cuopt::remote::LP);
  start->set_resume(false);
  start->set_total_size(50000);  // Claim 50KB
  ASSERT_TRUE(stream->Write(start_req));

  cuopt::remote::UploadJobResponse start_resp;
  ASSERT_TRUE(stream->Read(&start_resp));
  ASSERT_TRUE(start_resp.has_ack());

  std::cout << "[Test] Sent start, waiting for server timeout (no chunks sent)..." << std::endl;

  // Don't send any chunks - wait for server to timeout
  auto start_time = std::chrono::steady_clock::now();

  cuopt::remote::UploadJobResponse timeout_resp;
  bool got_response = stream->Read(&timeout_resp);

  auto elapsed =
    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time)
      .count();

  std::cout << "[Test] Server responded after " << elapsed << " seconds" << std::endl;

  EXPECT_GE(elapsed, 2) << "Server should wait before timeout";
  EXPECT_LE(elapsed, 10) << "Server should timeout reasonably quickly";

  stream->WritesDone();
  stream->Finish();
}

TEST_F(GrpcIntegrationTest, SelectiveRetransmission_AllChunksMissing)
{
  // Test that server correctly requests retransmission when NO data chunks are sent
  // Client sends start + finish immediately, server should request ALL data

  ASSERT_TRUE(start_server({"--max-message-mb", "1", "--verbose"}));

  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_.log_path());
  log_capture.mark_test_start();

  // Load a small problem
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);

  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 60.0;

  auto submit_request = build_lp_submit_request(problem, settings);
  const auto& lp_req  = submit_request.lp_request();
  std::string serialized_problem;
  ASSERT_TRUE(lp_req.SerializeToString(&serialized_problem));

  auto channel =
    grpc::CreateChannel("localhost:" + std::to_string(port_), grpc::InsecureChannelCredentials());
  auto stub = cuopt::remote::CuOptRemoteService::NewStub(channel);

  grpc::ClientContext context;
  auto stream = stub->UploadAndSubmit(&context);

  // 1. Send start
  cuopt::remote::UploadJobRequest start_req;
  auto* start = start_req.mutable_start();
  start->set_problem_type(cuopt::remote::LP);
  start->set_resume(false);
  start->set_total_size(serialized_problem.size());
  ASSERT_TRUE(stream->Write(start_req));

  cuopt::remote::UploadJobResponse start_resp;
  ASSERT_TRUE(stream->Read(&start_resp));
  ASSERT_TRUE(start_resp.has_ack());
  std::string upload_id = start_resp.ack().upload_id();

  // 2. Send finish IMMEDIATELY without sending any chunks
  cuopt::remote::UploadJobRequest finish_req;
  finish_req.mutable_finish()->set_upload_id(upload_id);
  ASSERT_TRUE(stream->Write(finish_req));

  cuopt::remote::UploadJobResponse finish_resp;
  ASSERT_TRUE(stream->Read(&finish_resp));

  // Server should request resend of ALL data (entire range 0 to total_size)
  ASSERT_TRUE(finish_resp.has_resend())
    << "Expected ResendRequest for all missing data, got: "
    << (finish_resp.has_ack()      ? "ack"
        : finish_resp.has_submit() ? "submit"
        : finish_resp.has_error()  ? "error: " + finish_resp.error().message()
                                   : "unknown");

  const auto& resend = finish_resp.resend();
  ASSERT_GE(resend.missing_ranges_size(), 1) << "Should request at least one range";

  // Calculate total missing bytes requested
  int64_t total_missing = 0;
  for (const auto& range : resend.missing_ranges()) {
    total_missing += range.size();
    std::cout << "[Test] Server requested range: offset=" << range.offset()
              << " size=" << range.size() << std::endl;
  }

  EXPECT_EQ(total_missing, static_cast<int64_t>(serialized_problem.size()))
    << "Server should request all " << serialized_problem.size() << " bytes";

  // 3. Now actually send all the data
  const size_t chunk_size = 1024;
  for (const auto& range : resend.missing_ranges()) {
    int64_t offset = range.offset();
    int64_t end    = offset + range.size();

    while (offset < end) {
      size_t n = std::min(static_cast<size_t>(end - offset), chunk_size);

      cuopt::remote::UploadJobRequest chunk_req;
      auto* chunk = chunk_req.mutable_chunk();
      chunk->set_upload_id(upload_id);
      chunk->set_offset(offset);
      chunk->set_data(serialized_problem.data() + offset, n);
      ASSERT_TRUE(stream->Write(chunk_req));

      cuopt::remote::UploadJobResponse chunk_resp;
      ASSERT_TRUE(stream->Read(&chunk_resp));
      ASSERT_TRUE(chunk_resp.has_ack());

      offset += static_cast<int64_t>(n);
    }
  }

  // 4. Send finish again
  ASSERT_TRUE(stream->Write(finish_req));

  cuopt::remote::UploadJobResponse final_resp;
  ASSERT_TRUE(stream->Read(&final_resp));

  // Should now get submit response with job_id
  ASSERT_TRUE(final_resp.has_submit())
    << "Expected submit after resending all data, got: "
    << (final_resp.has_ack()      ? "ack"
        : final_resp.has_resend() ? "resend (still missing data)"
        : final_resp.has_error()  ? "error: " + final_resp.error().message()
                                  : "unknown");

  std::string job_id = final_resp.submit().job_id();
  std::cout << "[Test] Upload complete, job_id=" << job_id << std::endl;

  stream->WritesDone();
  stream->Finish();
}

TEST_F(GrpcIntegrationTest, ClientDisconnect_GracefulStreamClose)
{
  // Test that server handles graceful client disconnect (WritesDone + Finish)
  // This shouldn't cause any server errors or resource leaks

  ASSERT_TRUE(start_server({"--max-message-mb", "1", "--chunk-timeout", "3", "--verbose"}));

  GrpcTestLogCapture log_capture;
  log_capture.set_server_log_path(server_.log_path());
  log_capture.mark_test_start();

  auto channel =
    grpc::CreateChannel("localhost:" + std::to_string(port_), grpc::InsecureChannelCredentials());
  auto stub = cuopt::remote::CuOptRemoteService::NewStub(channel);

  grpc::ClientContext context;
  // Set deadline so Finish() doesn't block forever waiting for server
  context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
  auto stream = stub->UploadAndSubmit(&context);

  // Send start
  cuopt::remote::UploadJobRequest start_req;
  auto* start = start_req.mutable_start();
  start->set_problem_type(cuopt::remote::LP);
  start->set_resume(false);
  start->set_total_size(10000);
  ASSERT_TRUE(stream->Write(start_req));

  cuopt::remote::UploadJobResponse start_resp;
  ASSERT_TRUE(stream->Read(&start_resp));
  ASSERT_TRUE(start_resp.has_ack());

  std::cout << "[Test] Started upload, now gracefully closing stream..." << std::endl;

  // Gracefully close without sending all data
  stream->WritesDone();
  grpc::Status status = stream->Finish();

  std::cout << "[Test] Stream closed with status: " << status.error_code() << " - "
            << status.error_message() << std::endl;

  // Server should handle this gracefully - check logs for errors
  std::string server_logs = log_capture.get_server_logs();

  // Look for crash indicators (shouldn't find any)
  bool has_crash = server_logs.find("SEGV") != std::string::npos ||
                   server_logs.find("SIGABRT") != std::string::npos ||
                   server_logs.find("core dump") != std::string::npos;

  EXPECT_FALSE(has_crash) << "Server should not crash on graceful disconnect";

  // Server should still be responsive - try another operation
  grpc_client_config_t config;
  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  // Simple health check - server should respond
  auto status_result = client->check_status("nonexistent-job");
  EXPECT_TRUE(status_result.success) << "Server should still be responsive after client disconnect";
}

// =============================================================================
// Streaming Path Verification Tests (verify correct path via client logs)
// =============================================================================

TEST_F(GrpcIntegrationTest, VerifyUnaryUpload_SmallProblem)
{
  // Verify that small problems use unary upload (not streaming)
  ASSERT_TRUE(start_server());  // Default server with large message limit

  GrpcTestLogCapture log_capture;

  grpc_client_config_t config;
  config.debug_log_callback = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  log_capture.mark_test_start();

  // Use a small problem that should fit in a single message
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);

  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;

  // Verify unary path was used for upload
  EXPECT_TRUE(log_capture.client_log_contains("Attempting unary submit"))
    << "Should attempt unary submit for small problem. Logs:\n"
    << log_capture.get_client_logs();
  EXPECT_TRUE(log_capture.client_log_contains("Unary submit succeeded"))
    << "Should succeed with unary submit. Logs:\n"
    << log_capture.get_client_logs();
  EXPECT_FALSE(log_capture.client_log_contains("Starting streaming upload"))
    << "Should NOT use streaming upload for small problem. Logs:\n"
    << log_capture.get_client_logs();
}

TEST_F(GrpcIntegrationTest, VerifyStreaming_LargeProblemBothDirections)
{
  // Verify that large problems use streaming for BOTH upload AND download
  // Use a very small message limit to force streaming in both directions
  ASSERT_TRUE(start_server({"--max-message-mb", "0"}));  // Minimum limit

  GrpcTestLogCapture log_capture;

  grpc_client_config_t config;
  config.timeout_seconds    = 180;
  config.max_message_bytes  = 4 * 1024;  // 4KB - very small to force streaming for results too
  config.chunk_size_bytes   = 2 * 1024;  // 2KB chunks
  config.debug_log_callback = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  log_capture.mark_test_start();

  // Load cod105_max.mps (2.6MB) - much larger than 4KB limit
  // The solution will also be large enough to require streaming download
  std::string mps_path = get_test_mip_path("cod105_max.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 5.0;  // Short time - testing streaming, not solve accuracy

  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;

  std::string client_logs = log_capture.get_client_logs();

  // === Verify UPLOAD streaming ===
  bool upload_streaming_used =
    log_capture.client_log_contains("Starting streaming upload") ||
    log_capture.client_log_contains("falling back to streaming upload") ||
    log_capture.client_log_contains("Using streaming upload directly");
  EXPECT_TRUE(upload_streaming_used) << "Should use streaming upload for large problem. Logs:\n"
                                     << client_logs;

  EXPECT_TRUE(log_capture.client_log_contains("Streaming upload completed"))
    << "Streaming upload should complete. Logs:\n"
    << client_logs;

  EXPECT_TRUE(log_capture.client_log_contains_pattern("sent [0-9]+ chunks"))
    << "Should log number of chunks sent. Logs:\n"
    << client_logs;

  // === Verify DOWNLOAD streaming ===
  bool download_streaming_used =
    log_capture.client_log_contains("Starting streaming download") ||
    log_capture.client_log_contains("Using streaming download directly") ||
    log_capture.client_log_contains("falling back to streaming");
  EXPECT_TRUE(download_streaming_used) << "Should use streaming download for large result. Logs:\n"
                                       << client_logs;

  EXPECT_TRUE(log_capture.client_log_contains("Streaming download completed"))
    << "Streaming download should complete. Logs:\n"
    << client_logs;

  EXPECT_TRUE(log_capture.client_log_contains_pattern("received [0-9]+ chunks"))
    << "Should log number of chunks received. Logs:\n"
    << client_logs;

  // === Verify size negotiation logging ===
  EXPECT_TRUE(log_capture.client_log_contains("data_size="))
    << "Should log data size during upload. Logs:\n"
    << client_logs;
  EXPECT_TRUE(log_capture.client_log_contains("effective_max="))
    << "Should log effective max limit. Logs:\n"
    << client_logs;
}

TEST_F(GrpcIntegrationTest, VerifyUnaryDownload_SmallResult)
{
  // Verify that small results use unary download (not streaming)
  ASSERT_TRUE(start_server());  // Default server with large message limit

  GrpcTestLogCapture log_capture;

  grpc_client_config_t config;
  config.debug_log_callback = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  // Use a small problem with a small result
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);

  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;

  // Clear logs just before the solve to focus on download logs
  log_capture.mark_test_start();

  auto result = client->solve_lp(problem, settings);
  EXPECT_TRUE(result.success) << result.error_message;

  // Verify unary path was used for download
  EXPECT_TRUE(log_capture.client_log_contains("Attempting unary GetResult"))
    << "Should attempt unary GetResult for small result. Logs:\n"
    << log_capture.get_client_logs();
  EXPECT_TRUE(log_capture.client_log_contains("Unary GetResult succeeded"))
    << "Should succeed with unary GetResult. Logs:\n"
    << log_capture.get_client_logs();
  EXPECT_FALSE(log_capture.client_log_contains("Starting streaming download"))
    << "Should NOT use streaming download for small result. Logs:\n"
    << log_capture.get_client_logs();
}

TEST_F(GrpcIntegrationTest, VerifyMessageSizeNegotiation)
{
  // Verify that client logs the size negotiation details
  ASSERT_TRUE(start_server({"--max-message-mb", "1"}));

  GrpcTestLogCapture log_capture;

  grpc_client_config_t config;
  config.timeout_seconds    = 120;
  config.max_message_bytes  = 256 * 1024 * 1024;  // Client thinks 256MB is ok
  config.debug_log_callback = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  log_capture.mark_test_start();

  // Load a problem larger than server's 1MB limit
  std::string mps_path = get_test_mip_path("cod105_max.mps");
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 5.0;  // Short time - testing negotiation, not solve accuracy

  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;

  // Verify the logs show the size comparison
  EXPECT_TRUE(log_capture.client_log_contains("data_size=")) << "Should log data size. Logs:\n"
                                                             << log_capture.get_client_logs();
  EXPECT_TRUE(log_capture.client_log_contains("client_max="))
    << "Should log client max message size. Logs:\n"
    << log_capture.get_client_logs();
  EXPECT_TRUE(log_capture.client_log_contains("effective_max="))
    << "Should log effective max (considering both client and server limits). Logs:\n"
    << log_capture.get_client_logs();
}

TEST_F(GrpcIntegrationTest, VerifyStreamingFallbackOnResourceExhausted)
{
  // Verify that streaming is used when effective max is smaller than data size
  // Use very small limits to force streaming for the ~620KB problem
  ASSERT_TRUE(start_server({"--max-message-mb", "0"}));  // Minimum limit

  GrpcTestLogCapture log_capture;

  grpc_client_config_t config;
  config.timeout_seconds = 120;
  // Set client's max small so streaming is used directly
  config.max_message_bytes  = 64 * 1024;  // 64KB - smaller than the ~620KB problem
  config.chunk_size_bytes   = 32 * 1024;  // 32KB chunks
  config.debug_log_callback = log_capture.client_callback();

  auto client = create_client(config);
  ASSERT_NE(client, nullptr);

  log_capture.mark_test_start();

  // Load a problem larger than 64KB limit
  std::string mps_path = get_test_mip_path("cod105_max.mps");  // ~620KB
  auto problem         = load_problem_from_mps(mps_path);

  mip_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 5.0;  // Short time - testing streaming, not solve accuracy

  auto result = client->solve_mip(problem, settings, false);
  EXPECT_TRUE(result.success) << result.error_message;

  // Client should recognize that data_size > effective_max and use streaming directly
  bool streaming_used = log_capture.client_log_contains("falling back to streaming upload") ||
                        log_capture.client_log_contains("Using streaming upload directly") ||
                        log_capture.client_log_contains("Starting streaming upload");

  EXPECT_TRUE(streaming_used) << "Should use streaming (either fallback or direct). Logs:\n"
                              << log_capture.get_client_logs();

  EXPECT_TRUE(log_capture.client_log_contains("Streaming upload completed"))
    << "Streaming upload should complete successfully. Logs:\n"
    << log_capture.get_client_logs();
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
  settings.time_limit = 15.0;  // Enough time to generate some incumbents

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
  settings.time_limit = 10.0;

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

TEST_F(GrpcIntegrationTest, ConcurrentStreamingTransfers)
{
  // Test multiple clients performing streaming uploads/downloads simultaneously
  // This verifies server handles concurrent streaming without data corruption
  // Uses small message limits to force streaming even for small problems
  ASSERT_TRUE(start_server({"--max-message-mb", "0"}));  // Force streaming

  const int num_clients = 3;
  std::vector<std::unique_ptr<grpc_client_t>> clients;
  std::vector<std::future<bool>> futures;

  // Create clients with very small message limits to force streaming
  // afiro serialized is ~2.7KB, result is ~792 bytes
  // Use 512 byte limit to force streaming in both directions
  for (int i = 0; i < num_clients; ++i) {
    grpc_client_config_t config;
    config.timeout_seconds   = 30;
    config.max_message_bytes = 512;  // 512 bytes - force streaming for afiro
    config.chunk_size_bytes  = 256;  // 256 byte chunks
    auto client              = create_client(config);
    ASSERT_NE(client, nullptr) << "Failed to create client " << i;
    clients.push_back(std::move(client));
  }

  // Load test problem - use afiro for speed (small but verifiable)
  // afiro serialized is ~2.7KB, result is ~792 bytes - both will stream with 512B limit
  std::string mps_path = get_test_lp_path("afiro_original.mps");
  auto problem         = load_problem_from_mps(mps_path);

  pdlp_solver_settings_t<int32_t, double> settings;
  settings.time_limit = 10.0;  // Short timeout for speed

  // Launch concurrent streaming solves
  std::atomic<int> success_count{0};
  std::atomic<int> failure_count{0};

  auto solve_task = [&](int client_idx) -> bool {
    try {
      auto result = clients[client_idx]->solve_lp(problem, settings);
      if (result.success && result.solution != nullptr) {
        double obj = result.solution->get_objective_value();
        if (std::abs(obj - (-464.753)) < 1.0) {
          success_count++;
          return true;
        } else {
          std::cerr << "[Test] Client " << client_idx << " wrong objective: " << obj << std::endl;
          failure_count++;
          return false;
        }
      } else {
        std::cerr << "[Test] Client " << client_idx << " failed: " << result.error_message
                  << std::endl;
        failure_count++;
        return false;
      }
    } catch (const std::exception& e) {
      std::cerr << "[Test] Client " << client_idx << " exception: " << e.what() << std::endl;
      failure_count++;
      return false;
    }
  };

  // Launch all clients in parallel
  for (int i = 0; i < num_clients; ++i) {
    futures.push_back(std::async(std::launch::async, solve_task, i));
  }

  // Wait for all to complete
  for (int i = 0; i < num_clients; ++i) {
    bool result = futures[i].get();
    EXPECT_TRUE(result) << "Client " << i << " streaming solve failed";
  }

  EXPECT_EQ(success_count.load(), num_clients)
    << "Expected all " << num_clients << " clients to succeed, got " << success_count.load()
    << " successes, " << failure_count.load() << " failures";

  std::cout << "[Test] All " << num_clients << " concurrent streaming transfers completed"
            << std::endl;
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
  settings.time_limit = 10.0;

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
