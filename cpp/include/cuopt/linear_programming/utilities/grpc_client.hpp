/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Forward declarations for gRPC types (to avoid exposing gRPC headers in public API)
namespace grpc {
class Channel;
}

namespace cuopt::remote {
class CuOptRemoteService;
}

namespace cuopt::linear_programming {

// Forward declarations for test helper functions (implemented in grpc_client.cu)
void grpc_test_inject_mock_stub(class grpc_client_t& client, std::shared_ptr<void> stub);
void grpc_test_mark_as_connected(class grpc_client_t& client);

/**
 * @brief Configuration options for the gRPC client
 *
 * Streaming Configuration:
 * - For messages larger than max_message_bytes, the client automatically uses
 *   chunked streaming upload/download to handle arbitrarily large payloads.
 * - chunk_size_bytes controls the size of individual stream chunks (must be < max_message_bytes)
 * - max_upload_retries controls how many times to retry failed uploads with smaller chunks
 */
struct grpc_client_config_t {
  std::string server_address = "localhost:9112";
  int poll_interval_ms       = 500;  // How often to poll for status (when use_wait=false)
  int timeout_seconds        = -1;  // Max time to wait for job completion (-1 = use default 1 hour)
  bool use_wait              = false;  // Use WaitForCompletion RPC instead of polling CheckStatus
  bool stream_logs           = false;  // Whether to stream logs from server
  std::function<void(const std::string&)> log_callback = nullptr;  // Called for each log line

  // Debug log callback - receives internal client debug messages (for test verification)
  // This is separate from log_callback which receives solver logs from the server
  std::function<void(const std::string&)> debug_log_callback = nullptr;

  // Incumbent callback for MIP solves (called when new incumbent found)
  // Parameters: index, objective value, solution vector
  // Return false to cancel the solve
  // Note: Incumbent callbacks are only supported in polling mode (use_wait=false).
  // Callbacks are invoked on the main thread during the polling loop.
  std::function<bool(int64_t index, double objective, const std::vector<double>& solution)>
    incumbent_callback           = nullptr;
  int incumbent_poll_interval_ms = 1000;  // How often to poll for new incumbents

  // TLS configuration
  bool enable_tls = false;      // Enable TLS encryption
  std::string tls_root_certs;   // PEM-encoded root CA certificates (for verifying server)
  std::string tls_client_cert;  // PEM-encoded client certificate (for mTLS)
  std::string tls_client_key;   // PEM-encoded client private key (for mTLS)

  // Streaming configuration for large messages
  // Default max_message_bytes aligns with typical gRPC server defaults (256 MiB)
  int64_t max_message_bytes = 256LL * 1024 * 1024;
  // Default chunk size of 1 MiB - small enough to work with most configs
  int64_t chunk_size_bytes = 1LL * 1024 * 1024;
  // Number of upload retry attempts with reduced chunk size on failure
  int max_upload_retries = 2;
  // Stream upload timeout in milliseconds (per-upload, not per-chunk) - DEPRECATED
  // Use chunk_timeout_seconds instead for per-message timeouts
  int upload_timeout_ms = 300000;  // 5 minutes default

  // Per-chunk/message timeout in seconds for streaming operations
  // If no chunk, ack, or other message is received within this time, the stream is cancelled.
  // This prevents hanging when a peer stops responding mid-transfer.
  // Set to 0 to disable per-chunk timeouts (not recommended).
  int chunk_timeout_seconds = 60;  // 60 seconds default

  // Enable data integrity hash logging for streaming transfers (for testing/debugging)
  // When enabled, computes FNV-1a hash of transferred data and logs it.
  // Compare client upload hash with server received hash (and vice versa for downloads)
  // to verify data integrity after reassembly from chunks.
  // Default: false (disabled in production to avoid overhead on large transfers)
  bool enable_transfer_hash = false;
};

/**
 * @brief Job status enum (transport-agnostic)
 */
enum class job_status_t { QUEUED, PROCESSING, COMPLETED, FAILED, CANCELLED, NOT_FOUND };

/**
 * @brief Convert job status to string
 */
inline const char* job_status_to_string(job_status_t status)
{
  switch (status) {
    case job_status_t::QUEUED: return "QUEUED";
    case job_status_t::PROCESSING: return "PROCESSING";
    case job_status_t::COMPLETED: return "COMPLETED";
    case job_status_t::FAILED: return "FAILED";
    case job_status_t::CANCELLED: return "CANCELLED";
    case job_status_t::NOT_FOUND: return "NOT_FOUND";
    default: return "UNKNOWN";
  }
}

/**
 * @brief Result of a job status check
 */
struct job_status_result_t {
  bool success = false;
  std::string error_message;
  job_status_t status = job_status_t::NOT_FOUND;
  std::string message;
  int64_t result_size_bytes = 0;
};

/**
 * @brief Result of a submit operation (job ID)
 */
struct submit_result_t {
  bool success = false;
  std::string error_message;
  std::string job_id;
};

/**
 * @brief Result of a cancel operation
 */
struct cancel_result_t {
  bool success = false;
  std::string error_message;
  job_status_t job_status = job_status_t::NOT_FOUND;
  std::string message;
};

/**
 * @brief Incumbent solution entry
 */
struct incumbent_t {
  int64_t index    = 0;
  double objective = 0.0;
  std::vector<double> assignment;
};

/**
 * @brief Result of get incumbents operation
 */
struct incumbents_result_t {
  bool success = false;
  std::string error_message;
  std::vector<incumbent_t> incumbents;
  int64_t next_index = 0;
  bool job_complete  = false;
};

/**
 * @brief Result of a remote solve operation
 */
template <typename i_t, typename f_t>
struct remote_lp_result_t {
  bool success = false;
  std::string error_message;
  std::unique_ptr<cpu_lp_solution_t<i_t, f_t>> solution;
};

template <typename i_t, typename f_t>
struct remote_mip_result_t {
  bool success = false;
  std::string error_message;
  std::unique_ptr<cpu_mip_solution_t<i_t, f_t>> solution;
};

/**
 * @brief gRPC client for remote cuOpt solving
 *
 * This class provides a high-level interface for submitting optimization problems
 * to a remote cuopt_grpc_server and retrieving results. It handles:
 * - Connection management
 * - Job submission
 * - Status polling
 * - Optional log streaming
 * - Result retrieval and parsing
 *
 * Usage:
 * @code
 * grpc_client_t client("localhost:9112");
 * if (!client.connect()) { ... handle error ... }
 *
 * auto result = client.solve_lp(problem, settings);
 * if (result.success) {
 *   // Use result.solution
 * }
 * @endcode
 *
 * This class is designed to be used by:
 * - Test clients for validation
 * - solve_lp_remote() and solve_mip_remote() for production use
 */
class grpc_client_t {
  // Allow test helpers to access internal implementation for mock injection
  friend void grpc_test_inject_mock_stub(grpc_client_t&, std::shared_ptr<void>);
  friend void grpc_test_mark_as_connected(grpc_client_t&);

 public:
  /**
   * @brief Construct a gRPC client with configuration
   * @param config Client configuration options
   */
  explicit grpc_client_t(const grpc_client_config_t& config = grpc_client_config_t{});

  /**
   * @brief Construct a gRPC client with just server address
   * @param server_address Server address in "host:port" format
   */
  explicit grpc_client_t(const std::string& server_address);

  ~grpc_client_t();

  // Non-copyable, non-movable (due to atomic member and thread)
  grpc_client_t(const grpc_client_t&)            = delete;
  grpc_client_t& operator=(const grpc_client_t&) = delete;
  grpc_client_t(grpc_client_t&&)                 = delete;
  grpc_client_t& operator=(grpc_client_t&&)      = delete;

  /**
   * @brief Connect to the gRPC server
   * @return true if connection successful
   */
  bool connect();

  /**
   * @brief Check if connected to server
   */
  bool is_connected() const;

  /**
   * @brief Solve an LP problem remotely
   *
   * This is a blocking call that:
   * 1. Submits the problem to the server
   * 2. Polls for completion (with optional log streaming)
   * 3. Retrieves and parses the result
   *
   * @param problem The CPU optimization problem to solve
   * @param settings Solver settings
   * @return Result containing success status and solution (if successful)
   */
  template <typename i_t, typename f_t>
  remote_lp_result_t<i_t, f_t> solve_lp(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                        const pdlp_solver_settings_t<i_t, f_t>& settings);

  /**
   * @brief Solve a MIP problem remotely
   *
   * This is a blocking call that:
   * 1. Submits the problem to the server
   * 2. Polls for completion (with optional log streaming)
   * 3. Retrieves and parses the result
   *
   * @param problem The CPU optimization problem to solve
   * @param settings Solver settings
   * @param enable_incumbents Whether to enable incumbent solution streaming
   * @return Result containing success status and solution (if successful)
   */
  template <typename i_t, typename f_t>
  remote_mip_result_t<i_t, f_t> solve_mip(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                          const mip_solver_settings_t<i_t, f_t>& settings,
                                          bool enable_incumbents = false);

  // =========================================================================
  // Async Operations (for manual job management)
  // =========================================================================

  /**
   * @brief Submit an LP problem without waiting for result
   * @return Result containing job_id if successful
   */
  template <typename i_t, typename f_t>
  submit_result_t submit_lp(const cpu_optimization_problem_t<i_t, f_t>& problem,
                            const pdlp_solver_settings_t<i_t, f_t>& settings);

  /**
   * @brief Submit a MIP problem without waiting for result
   * @return Result containing job_id if successful
   */
  template <typename i_t, typename f_t>
  submit_result_t submit_mip(const cpu_optimization_problem_t<i_t, f_t>& problem,
                             const mip_solver_settings_t<i_t, f_t>& settings,
                             bool enable_incumbents = false);

  /**
   * @brief Check status of a submitted job
   * @param job_id The job ID to check
   * @return Status result including job state and optional result size
   */
  job_status_result_t check_status(const std::string& job_id);

  /**
   * @brief Wait for a job to complete (blocking)
   *
   * This is more efficient than polling check_status() but does not
   * return the result - call get_lp_result/get_mip_result afterward.
   *
   * @param job_id The job ID to wait for
   * @return Status result when job completes (COMPLETED, FAILED, or CANCELLED)
   */
  job_status_result_t wait_for_completion(const std::string& job_id);

  /**
   * @brief Get LP result for a completed job
   * @param job_id The job ID
   * @return Result containing solution if successful
   */
  template <typename i_t, typename f_t>
  remote_lp_result_t<i_t, f_t> get_lp_result(const std::string& job_id);

  /**
   * @brief Get MIP result for a completed job
   * @param job_id The job ID
   * @return Result containing solution if successful
   */
  template <typename i_t, typename f_t>
  remote_mip_result_t<i_t, f_t> get_mip_result(const std::string& job_id);

  /**
   * @brief Cancel a running job
   * @param job_id The job ID to cancel
   * @return Cancel result with status
   */
  cancel_result_t cancel_job(const std::string& job_id);

  /**
   * @brief Delete a job and its results from server
   * @param job_id The job ID to delete
   * @return true if deletion successful
   */
  bool delete_job(const std::string& job_id);

  /**
   * @brief Get incumbent solutions for a MIP job
   * @param job_id The job ID
   * @param from_index Start from this incumbent index
   * @param max_count Maximum number to return (0 = no limit)
   * @return Incumbents result
   */
  incumbents_result_t get_incumbents(const std::string& job_id,
                                     int64_t from_index = 0,
                                     int32_t max_count  = 0);

  /**
   * @brief Stream logs for a job (blocking until job completes or callback returns false)
   * @param job_id The job ID
   * @param from_byte Starting byte offset in log
   * @param callback Called for each log line; return false to stop streaming
   * @return true if streaming completed normally
   */
  bool stream_logs(const std::string& job_id,
                   int64_t from_byte,
                   std::function<bool(const std::string& line, bool job_complete)> callback);

  /**
   * @brief Get the last error message
   */
  const std::string& get_last_error() const { return last_error_; }

 private:
  struct impl_t;
  std::unique_ptr<impl_t> impl_;

  grpc_client_config_t config_;
  std::string last_error_;

  // Track server-reported max message size (may differ from our config)
  int64_t server_max_message_bytes_ = 0;

  // Internal helper for log streaming
  void start_log_streaming(const std::string& job_id);
  void stop_log_streaming();

  std::unique_ptr<std::thread> log_thread_;
  std::atomic<bool> stop_logs_{false};

  // =========================================================================
  // Streaming Upload/Download Support
  // =========================================================================

  /**
   * @brief Problem type enum for streaming upload
   */
  enum class streaming_problem_type_t { LP, MIP };

  /**
   * @brief Submit a job, automatically choosing unary or streaming based on size
   *
   * For small payloads (<= max_message_bytes), uses unary SubmitJob RPC.
   * For large payloads, uses chunked UploadAndSubmit streaming RPC.
   *
   * @param serialized_data Serialized protobuf data (SolveLPRequest or SolveMIPRequest)
   * @param problem_type Whether this is LP or MIP
   * @param job_id_out Output: the job ID assigned by server
   * @return true if submission successful
   */
  bool submit_or_upload(const std::vector<uint8_t>& serialized_data,
                        streaming_problem_type_t problem_type,
                        std::string& job_id_out);

  /**
   * @brief Upload large payload via chunked streaming with retry support
   *
   * Implements robust error handling:
   * - Per-chunk acknowledgment from server
   * - Adaptive chunk sizing based on server feedback
   * - Automatic retry with smaller chunks on failure
   *
   * @param data Raw bytes to upload
   * @param problem_type Whether this is LP or MIP
   * @param job_id_out Output: the job ID assigned by server
   * @param initial_chunk_size Starting chunk size (may be reduced on retry)
   * @return true if upload and submit successful
   */
  bool upload_and_submit(const uint8_t* data,
                         size_t size,
                         streaming_problem_type_t problem_type,
                         std::string& job_id_out,
                         int64_t initial_chunk_size);

  /**
   * @brief Get result, automatically choosing unary or streaming based on size
   *
   * First checks job status to get result size hint. If size exceeds
   * max_message_bytes, uses StreamResult; otherwise uses GetResult.
   *
   * @param job_id The job ID
   * @param result_data_out Output: serialized result (LPSolution or MIPSolution)
   * @return true if retrieval successful
   */
  bool get_result_or_stream(const std::string& job_id, std::vector<uint8_t>& result_data_out);

  /**
   * @brief Download result via chunked streaming
   *
   * Handles arbitrarily large results by reassembling chunks.
   * Implements error detection on each chunk.
   *
   * @param job_id The job ID
   * @param result_data_out Output: reassembled serialized result
   * @return true if download successful
   */
  bool stream_result(const std::string& job_id, std::vector<uint8_t>& result_data_out);

  /**
   * @brief Compute optimal chunk size based on known limits
   *
   * @param server_max Server-reported max message size (0 if unknown)
   * @param config_max Client-configured max message size
   * @param preferred Preferred chunk size
   * @return Safe chunk size that fits within limits
   */
  static int64_t compute_chunk_size(int64_t server_max, int64_t config_max, int64_t preferred);
};

}  // namespace cuopt::linear_programming
