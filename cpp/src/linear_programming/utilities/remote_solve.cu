/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <cuopt/linear_programming/utilities/remote_serialization.hpp>
#include <cuopt/linear_programming/utilities/remote_solve.hpp>
#include <utilities/logger.hpp>

#if CUOPT_ENABLE_GRPC
#include "remote_solve_grpc.hpp"
#endif

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace cuopt::linear_programming {

namespace {

// Message types for streaming protocol (must match server)
enum class MessageType : uint8_t {
  LOG_MESSAGE = 0,  // Log output from server
  SOLUTION    = 1,  // Final solution data
};

// Check if sync mode is enabled (default is async)
static bool use_sync_mode()
{
  const char* sync_env = std::getenv("CUOPT_REMOTE_USE_SYNC");
  return (sync_env != nullptr && std::string(sync_env) == "1");
}

static bool use_grpc_transport()
{
#if CUOPT_ENABLE_GRPC
  const char* transport = std::getenv("CUOPT_REMOTE_TRANSPORT");
  if (transport != nullptr && std::string(transport) == "socket") { return false; }
  return true;
#else
  return false;
#endif
}

template <typename f_t>
bool copy_incumbent_to_device(const std::vector<double>& host_assignment,
                              double host_objective,
                              f_t** d_assignment_out,
                              f_t** d_objective_out)
{
  *d_assignment_out = nullptr;
  *d_objective_out  = nullptr;
  if (host_assignment.empty()) { return false; }

  size_t n = host_assignment.size();
  std::vector<f_t> assignment(n);
  for (size_t i = 0; i < n; ++i) {
    assignment[i] = static_cast<f_t>(host_assignment[i]);
  }
  f_t objective = static_cast<f_t>(host_objective);

  if (cudaMalloc(reinterpret_cast<void**>(d_assignment_out), n * sizeof(f_t)) != cudaSuccess) {
    CUOPT_LOG_WARN("[remote_solve] Failed to cudaMalloc for incumbent assignment");
    return false;
  }
  if (cudaMalloc(reinterpret_cast<void**>(d_objective_out), sizeof(f_t)) != cudaSuccess) {
    CUOPT_LOG_WARN("[remote_solve] Failed to cudaMalloc for incumbent objective");
    cudaFree(*d_assignment_out);
    *d_assignment_out = nullptr;
    return false;
  }

  if (cudaMemcpy(*d_assignment_out, assignment.data(), n * sizeof(f_t), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    CUOPT_LOG_WARN("[remote_solve] Failed to cudaMemcpy incumbent assignment");
    cudaFree(*d_assignment_out);
    cudaFree(*d_objective_out);
    *d_assignment_out = nullptr;
    *d_objective_out  = nullptr;
    return false;
  }
  if (cudaMemcpy(*d_objective_out, &objective, sizeof(f_t), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    CUOPT_LOG_WARN("[remote_solve] Failed to cudaMemcpy incumbent objective");
    cudaFree(*d_assignment_out);
    cudaFree(*d_objective_out);
    *d_assignment_out = nullptr;
    *d_objective_out  = nullptr;
    return false;
  }

  return true;
}

template <typename f_t>
void invoke_incumbent_callbacks(
  const std::vector<cuopt::internals::base_solution_callback_t*>& callbacks,
  const std::vector<double>& assignment,
  double objective)
{
  f_t* d_assignment = nullptr;
  f_t* d_objective  = nullptr;
  if (!copy_incumbent_to_device<f_t>(assignment, objective, &d_assignment, &d_objective)) {
    return;
  }

  for (auto* cb : callbacks) {
    if (cb == nullptr) { continue; }
    if (cb->get_type() != cuopt::internals::base_solution_callback_type::GET_SOLUTION) { continue; }
    auto* get_cb = static_cast<cuopt::internals::get_solution_callback_t*>(cb);
    get_cb->get_solution(d_assignment, d_objective);
  }

  cudaDeviceSynchronize();
  cudaFree(d_assignment);
  cudaFree(d_objective);
}

/**
 * @brief Simple socket client for remote solve with streaming support
 */
class remote_client_t {
 public:
  remote_client_t(const std::string& host, int port) : host_(host), port_(port), sockfd_(-1) {}

  ~remote_client_t() { disconnect(); }

  bool connect()
  {
    if (sockfd_ >= 0) return true;

    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd_ < 0) {
      CUOPT_LOG_ERROR("[remote_solve] Failed to create socket");
      return false;
    }

    struct hostent* server = gethostbyname(host_.c_str());
    if (server == nullptr) {
      CUOPT_LOG_ERROR(std::string("[remote_solve] Unknown host: ") + host_);
      close(sockfd_);
      sockfd_ = -1;
      return false;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    addr.sin_port = htons(port_);

    if (::connect(sockfd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
      CUOPT_LOG_ERROR(std::string("[remote_solve] Failed to connect to ") + host_ + ":" +
                      std::to_string(port_));
      close(sockfd_);
      sockfd_ = -1;
      return false;
    }

    return true;
  }

  void disconnect()
  {
    if (sockfd_ >= 0) {
      close(sockfd_);
      sockfd_ = -1;
    }
  }

  bool send_request(const std::vector<uint8_t>& data)
  {
    if (sockfd_ < 0) return false;

    // Send size first (8 bytes for large problem support)
    uint64_t size = static_cast<uint64_t>(data.size());
    if (!write_all(&size, sizeof(size))) return false;
    if (!write_all(data.data(), data.size())) return false;
    return true;
  }

  /**
   * @brief Receive response with streaming log support.
   *
   * This method reads messages from the server. If the server sends LOG_MESSAGE
   * types, they are printed to the console. When a SOLUTION message is received,
   * the solution data is returned.
   *
   * @param data Output buffer for solution data
   * @param log_to_console If true, print received log messages
   * @return true if solution was received, false on error
   */
  bool receive_streaming_response(std::vector<uint8_t>& data, bool log_to_console)
  {
    if (sockfd_ < 0) return false;

    while (true) {
      // Read message type (1 byte)
      uint8_t msg_type;
      if (!read_all(&msg_type, 1)) {
        // If we can't read the message type, try legacy format
        // (server might not support streaming)
        return receive_response_legacy(data);
      }

      // Read payload size (8 bytes for large problem support)
      uint64_t payload_size;
      if (!read_all(&payload_size, sizeof(payload_size))) return false;

      // Sanity check - reject messages larger than 16GB
      if (payload_size > 16ULL * 1024 * 1024 * 1024) {
        CUOPT_LOG_ERROR(std::string("[remote_solve] Message too large: ") +
                        std::to_string(payload_size) + " bytes");
        return false;
      }

      if (static_cast<MessageType>(msg_type) == MessageType::LOG_MESSAGE) {
        // Read and display log message
        if (payload_size > 0) {
          std::vector<char> log_msg(payload_size + 1);
          if (!read_all(log_msg.data(), payload_size)) return false;
          log_msg[payload_size] = '\0';

          if (log_to_console) {
            // Print log message from server (already formatted)
            std::cout << log_msg.data() << std::flush;
          }
        }
      } else if (static_cast<MessageType>(msg_type) == MessageType::SOLUTION) {
        // Read solution data
        data.resize(payload_size);
        if (payload_size > 0) {
          if (!read_all(data.data(), payload_size)) return false;
        }
        return true;
      } else {
        CUOPT_LOG_WARN(std::string("[remote_solve] Unknown message type: ") +
                       std::to_string(static_cast<int>(msg_type)));
        // Skip unknown message
        if (payload_size > 0) {
          std::vector<uint8_t> skip_buf(payload_size);
          if (!read_all(skip_buf.data(), payload_size)) return false;
        }
      }
    }
  }

  // Legacy response format (non-streaming)
  bool receive_response(std::vector<uint8_t>& data) { return receive_response_legacy(data); }

 private:
  bool receive_response_legacy(std::vector<uint8_t>& data)
  {
    if (sockfd_ < 0) return false;

    // Read size first (8 bytes for large problem support)
    uint64_t size;
    if (!read_all(&size, sizeof(size))) return false;

    // Sanity check - reject responses larger than 16GB
    if (size > 16ULL * 1024 * 1024 * 1024) {
      CUOPT_LOG_ERROR(std::string("[remote_solve] Response too large: ") + std::to_string(size) +
                      " bytes");
      return false;
    }

    data.resize(size);
    if (!read_all(data.data(), size)) return false;
    return true;
  }

  bool write_all(const void* buf, size_t len)
  {
    const uint8_t* ptr = static_cast<const uint8_t*>(buf);
    size_t remaining   = len;
    while (remaining > 0) {
      ssize_t n = ::write(sockfd_, ptr, remaining);
      if (n <= 0) {
        CUOPT_LOG_ERROR("[remote_solve] Write failed");
        return false;
      }
      ptr += n;
      remaining -= n;
    }
    return true;
  }

  bool read_all(void* buf, size_t len)
  {
    uint8_t* ptr     = static_cast<uint8_t*>(buf);
    size_t remaining = len;
    while (remaining > 0) {
      ssize_t n = ::read(sockfd_, ptr, remaining);
      if (n <= 0) {
        CUOPT_LOG_ERROR("[remote_solve] Read failed");
        return false;
      }
      ptr += n;
      remaining -= n;
    }
    return true;
  }

  std::string host_;
  int port_;
  int sockfd_;
};

//============================================================================
// Async Mode Helpers
//============================================================================

template <typename i_t, typename f_t>
static std::pair<bool, std::string> submit_job(const std::string& host,
                                               int port,
                                               const std::vector<uint8_t>& request_data)
{
  remote_client_t client(host, port);
  if (!client.connect()) { return {false, "Failed to connect to server"}; }

  if (!client.send_request(request_data)) { return {false, "Failed to send request"}; }

  std::vector<uint8_t> response_data;
  if (!client.receive_response(response_data)) { return {false, "Failed to receive response"}; }

  auto serializer = get_serializer<i_t, f_t>();
  std::string job_id;
  std::string error_message;
  if (!serializer->deserialize_submit_response(response_data, job_id, error_message)) {
    return {false, error_message};
  }

  return {true, job_id};
}

/**
 * @brief Retrieve and display buffered logs from the server.
 *
 * @param host Server host
 * @param port Server port
 * @param job_id Job ID
 * @param frombyte Byte offset to start reading from
 * @return std::pair<bool, int64_t> - (job_exists, new_frombyte)
 */
template <typename i_t, typename f_t>
static std::pair<bool, int64_t> get_logs(const std::string& host,
                                         int port,
                                         const std::string& job_id,
                                         int64_t frombyte)
{
  remote_client_t client(host, port);
  if (!client.connect()) { return {false, frombyte}; }

  auto serializer   = get_serializer<i_t, f_t>();
  auto logs_request = serializer->serialize_get_logs_request(job_id, frombyte);

  if (!client.send_request(logs_request)) { return {false, frombyte}; }

  std::vector<uint8_t> response_data;
  if (!client.receive_response(response_data)) { return {false, frombyte}; }

  auto result = serializer->deserialize_logs_response(response_data);

  // Print any new log lines
  for (const auto& line : result.log_lines) {
    std::cout << line << "\n";
  }
  if (!result.log_lines.empty()) { std::cout.flush(); }

  return {result.job_exists, result.nbytes};
}

template <typename i_t, typename f_t>
static bool poll_until_complete(const std::string& host,
                                int port,
                                const std::string& job_id,
                                bool verbose)
{
  auto serializer    = get_serializer<i_t, f_t>();
  using job_status_t = typename remote_serializer_t<i_t, f_t>::job_status_t;

  int64_t log_frombyte = 0;  // Track position in log file

  while (true) {
    // Fetch and display any new log entries
    if (verbose) {
      auto [job_exists, new_frombyte] = get_logs<i_t, f_t>(host, port, job_id, log_frombyte);
      if (job_exists) { log_frombyte = new_frombyte; }
    }

    remote_client_t client(host, port);
    if (!client.connect()) {
      CUOPT_LOG_ERROR("[remote_solve] Failed to connect for status check");
      return false;
    }

    auto status_request = serializer->serialize_status_request(job_id);
    if (!client.send_request(status_request)) {
      CUOPT_LOG_ERROR("[remote_solve] Failed to send status request");
      return false;
    }

    std::vector<uint8_t> response_data;
    if (!client.receive_response(response_data)) {
      CUOPT_LOG_ERROR("[remote_solve] Failed to receive status response");
      return false;
    }

    auto status = serializer->deserialize_status_response(response_data);

    if (status == job_status_t::COMPLETED) {
      // Fetch any remaining log entries
      if (verbose) {
        get_logs<i_t, f_t>(host, port, job_id, log_frombyte);
        CUOPT_LOG_INFO(std::string("[remote_solve] Job ") + job_id + " completed");
      }
      return true;
    } else if (status == job_status_t::FAILED) {
      // Fetch any remaining log entries (may contain error info)
      if (verbose) { get_logs<i_t, f_t>(host, port, job_id, log_frombyte); }
      CUOPT_LOG_ERROR(std::string("[remote_solve] Job ") + job_id + " failed");
      return false;
    } else if (status == job_status_t::NOT_FOUND) {
      CUOPT_LOG_ERROR(std::string("[remote_solve] Job ") + job_id + " not found");
      return false;
    }

    // Job still queued or processing, wait and try again
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

template <typename i_t, typename f_t>
static std::pair<bool, std::vector<uint8_t>> get_result(const std::string& host,
                                                        int port,
                                                        const std::string& job_id)
{
  remote_client_t client(host, port);
  if (!client.connect()) { return {false, {}}; }

  auto serializer     = get_serializer<i_t, f_t>();
  auto result_request = serializer->serialize_get_result_request(job_id);

  if (!client.send_request(result_request)) { return {false, {}}; }

  std::vector<uint8_t> response_data;
  if (!client.receive_response(response_data)) { return {false, {}}; }

  return {true, response_data};
}

template <typename i_t, typename f_t>
static void delete_job(const std::string& host, int port, const std::string& job_id)
{
  remote_client_t client(host, port);
  if (!client.connect()) { return; }

  auto serializer     = get_serializer<i_t, f_t>();
  auto delete_request = serializer->serialize_delete_request(job_id);

  if (!client.send_request(delete_request)) { return; }

  std::vector<uint8_t> response_data;
  client.receive_response(response_data);  // Ignore result
}

template <typename i_t, typename f_t>
static cancel_job_result_t cancel_job_impl(const std::string& host,
                                           int port,
                                           const std::string& job_id)
{
  cancel_job_result_t result;
  result.success    = false;
  result.message    = "Unknown error";
  result.job_status = remote_job_status_t::NOT_FOUND;

  remote_client_t client(host, port);
  if (!client.connect()) {
    result.message = "Failed to connect to server";
    return result;
  }

  auto serializer     = get_serializer<i_t, f_t>();
  auto cancel_request = serializer->serialize_cancel_request(job_id);

  if (!client.send_request(cancel_request)) {
    result.message = "Failed to send cancel request";
    return result;
  }

  std::vector<uint8_t> response_data;
  if (!client.receive_response(response_data)) {
    result.message = "Failed to receive response";
    return result;
  }

  // Deserialize the cancel response
  auto cancel_result = serializer->deserialize_cancel_response(response_data);

  result.success = cancel_result.success;
  result.message = cancel_result.message;

  // Map serializer job_status_t to remote_job_status_t
  using serializer_status = typename remote_serializer_t<i_t, f_t>::job_status_t;
  switch (cancel_result.job_status) {
    case serializer_status::QUEUED: result.job_status = remote_job_status_t::QUEUED; break;
    case serializer_status::PROCESSING: result.job_status = remote_job_status_t::PROCESSING; break;
    case serializer_status::COMPLETED: result.job_status = remote_job_status_t::COMPLETED; break;
    case serializer_status::FAILED: result.job_status = remote_job_status_t::FAILED; break;
    case serializer_status::NOT_FOUND: result.job_status = remote_job_status_t::NOT_FOUND; break;
    case serializer_status::CANCELLED: result.job_status = remote_job_status_t::CANCELLED; break;
  }

  return result;
}

}  // namespace

//============================================================================
// LP Remote Solve
//============================================================================

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  const bool sync_mode = use_sync_mode();
  const bool grpc_mode = use_grpc_transport();

  CUOPT_LOG_INFO("[remote_solve] Solving LP remotely on %s:%d (%s via %s)",
                 config.host.c_str(),
                 config.port,
                 sync_mode ? "sync" : "async",
                 grpc_mode ? "gRPC" : "socket");

  // Log problem info (similar to local solve)
  if (settings.log_to_console) {
    auto n_rows = view.get_constraint_matrix_offsets().size() > 0
                    ? static_cast<i_t>(view.get_constraint_matrix_offsets().size()) - 1
                    : 0;
    auto n_cols = static_cast<i_t>(view.get_objective_coefficients().size());
    auto nnz    = static_cast<i_t>(view.get_constraint_matrix_values().size());
    CUOPT_LOG_INFO("Solving a problem with %d constraints, %d variables, and %d nonzeros (remote)",
                   n_rows,
                   n_cols,
                   nnz);
  }

  auto serializer = get_serializer<i_t, f_t>();

  if (grpc_mode) {
#if CUOPT_ENABLE_GRPC
    const std::string address = config.host + ":" + std::to_string(config.port);

    // Serialize as SolveLPRequest (server expects this protobuf, not AsyncRequest)
    std::vector<uint8_t> request_data = serializer->serialize_lp_request(view, settings);
    CUOPT_LOG_DEBUG(std::string("[remote_solve] Serialized LP request (gRPC): ") +
                    std::to_string(request_data.size()) + " bytes");

    std::string job_id;
    std::string err;
    if (!grpc_remote::upload_and_submit(address,
                                        grpc_remote::ProblemType::LP,
                                        request_data.data(),
                                        request_data.size(),
                                        job_id,
                                        err)) {
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "gRPC UploadAndSubmit failed: " + err, cuopt::error_type_t::RuntimeError));
    }

    // Optional realtime logs on client side
    volatile bool stop_logs = false;
    std::thread log_thread;
    if (settings.log_to_console) {
      log_thread =
        std::thread([&]() { grpc_remote::stream_logs_to_stdout(address, job_id, &stop_logs, ""); });
    }

    // Poll status until terminal, allowing log streaming and cancellation in other threads.
    std::string status;
    while (true) {
      std::string st_err;
      if (!grpc_remote::check_status(address, job_id, status, st_err)) {
        stop_logs = true;
        if (log_thread.joinable()) { log_thread.join(); }
        grpc_remote::delete_result(address, job_id);
        return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
          "gRPC CheckStatus failed: " + st_err, cuopt::error_type_t::RuntimeError));
      }

      if (status == "COMPLETED") { break; }
      if (status == "FAILED" || status == "CANCELLED" || status == "NOT_FOUND") {
        stop_logs = true;
        if (log_thread.joinable()) { log_thread.join(); }
        grpc_remote::delete_result(address, job_id);
        return optimization_problem_solution_t<i_t, f_t>(
          cuopt::logic_error("Remote job did not complete successfully (status=" + status + ")",
                             cuopt::error_type_t::RuntimeError));
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Retrieve result bytes via streaming (unlimited total size)
    std::vector<uint8_t> solution_bytes;
    std::string res_err;
    if (!grpc_remote::stream_result(address, job_id, solution_bytes, res_err)) {
      stop_logs = true;
      if (log_thread.joinable()) { log_thread.join(); }
      grpc_remote::delete_result(address, job_id);
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "gRPC StreamResult failed: " + res_err, cuopt::error_type_t::RuntimeError));
    }

    stop_logs = true;
    if (log_thread.joinable()) { log_thread.join(); }

    grpc_remote::delete_result(address, job_id);
    return serializer->deserialize_lp_solution(solution_bytes);
#else
    // Should be unreachable when grpc_mode==true
    (void)serializer;
#endif
  }

  if (sync_mode) {
    //=========================================================================
    // SYNC/BLOCKING MODE: Unified architecture
    //
    // Server-side: Job goes through queue, handled by worker process.
    // Client blocks until completion (server uses condition variable).
    // This enables cancellation for "sync" jobs and concurrent solves.
    //=========================================================================

    // Serialize as async request with blocking=true
    std::vector<uint8_t> request_data =
      serializer->serialize_async_lp_request(view, settings, true /* blocking */);
    CUOPT_LOG_DEBUG(std::string("[remote_solve] Serialized LP request (blocking): ") +
                    std::to_string(request_data.size()) + " bytes");

    // Connect and send
    remote_client_t client(config.host, config.port);
    if (!client.connect()) {
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "Failed to connect to remote server", cuopt::error_type_t::RuntimeError));
    }

    if (!client.send_request(request_data)) {
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "Failed to send request to remote server", cuopt::error_type_t::RuntimeError));
    }

    // Receive response (server blocks until job completes, then returns result)
    std::vector<uint8_t> response_data;
    if (!client.receive_response(response_data)) {
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "Failed to receive response from remote server", cuopt::error_type_t::RuntimeError));
    }

    CUOPT_LOG_DEBUG(std::string("[remote_solve] Received LP result (blocking): ") +
                    std::to_string(response_data.size()) + " bytes");

    // Deserialize solution from result response (same format as async GET_RESULT)
    return serializer->deserialize_lp_result_response(response_data);

  } else {
    //=========================================================================
    // ASYNC MODE: Submit → Poll → Get Result → Delete
    //=========================================================================

    // Serialize as async request with blocking=false
    std::vector<uint8_t> request_data =
      serializer->serialize_async_lp_request(view, settings, false /* blocking */);
    CUOPT_LOG_DEBUG(std::string("[remote_solve] Serialized LP request (async): ") +
                    std::to_string(request_data.size()) + " bytes");

    // Submit job
    auto [submit_ok, job_id_or_error] =
      submit_job<i_t, f_t>(config.host, config.port, request_data);
    if (!submit_ok) {
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "Job submission failed: " + job_id_or_error, cuopt::error_type_t::RuntimeError));
    }
    std::string job_id = job_id_or_error;
    CUOPT_LOG_INFO(std::string("[remote_solve] Job submitted, ID: ") + job_id);

    // Poll until complete
    if (!poll_until_complete<i_t, f_t>(config.host, config.port, job_id, settings.log_to_console)) {
      delete_job<i_t, f_t>(config.host, config.port, job_id);
      return optimization_problem_solution_t<i_t, f_t>(
        cuopt::logic_error("Job failed or not found", cuopt::error_type_t::RuntimeError));
    }

    // Get result
    auto [result_ok, result_data] = get_result<i_t, f_t>(config.host, config.port, job_id);
    if (!result_ok) {
      delete_job<i_t, f_t>(config.host, config.port, job_id);
      return optimization_problem_solution_t<i_t, f_t>(
        cuopt::logic_error("Failed to retrieve result", cuopt::error_type_t::RuntimeError));
    }

    // Delete job from server
    delete_job<i_t, f_t>(config.host, config.port, job_id);
    CUOPT_LOG_DEBUG(std::string("[remote_solve] Job ") + job_id + " deleted from server");

    // Deserialize solution from async result response
    return serializer->deserialize_lp_result_response(result_data);
  }
}

//============================================================================
// MIP Remote Solve
//============================================================================

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view,
  const mip_solver_settings_t<i_t, f_t>& settings)
{
  const bool sync_mode = use_sync_mode();
  const bool grpc_mode = use_grpc_transport();

  CUOPT_LOG_INFO("[remote_solve] Solving MIP remotely on %s:%d (%s via %s)",
                 config.host.c_str(),
                 config.port,
                 sync_mode ? "sync" : "async",
                 grpc_mode ? "gRPC" : "socket");

  // Log problem info
  {
    auto n_rows = view.get_constraint_matrix_offsets().size() > 0
                    ? static_cast<i_t>(view.get_constraint_matrix_offsets().size()) - 1
                    : 0;
    auto n_cols = static_cast<i_t>(view.get_objective_coefficients().size());
    auto nnz    = static_cast<i_t>(view.get_constraint_matrix_values().size());
    CUOPT_LOG_INFO(
      "Solving a problem with %d constraints, %d variables, and %d nonzeros (remote MIP)",
      n_rows,
      n_cols,
      nnz);
  }

  auto serializer = get_serializer<i_t, f_t>();

  if (grpc_mode) {
#if CUOPT_ENABLE_GRPC
    const std::string address = config.host + ":" + std::to_string(config.port);

    std::vector<uint8_t> request_data = serializer->serialize_mip_request(view, settings);
    CUOPT_LOG_DEBUG(std::string("[remote_solve] Serialized MIP request (gRPC): ") +
                    std::to_string(request_data.size()) + " bytes");

    std::string job_id;
    std::string err;
    if (!grpc_remote::upload_and_submit(address,
                                        grpc_remote::ProblemType::MIP,
                                        request_data.data(),
                                        request_data.size(),
                                        job_id,
                                        err)) {
      return mip_solution_t<i_t, f_t>(cuopt::logic_error("gRPC UploadAndSubmit failed: " + err,
                                                         cuopt::error_type_t::RuntimeError));
    }

    volatile bool stop_logs = false;
    std::thread log_thread;
    if (settings.log_to_console) {
      log_thread =
        std::thread([&]() { grpc_remote::stream_logs_to_stdout(address, job_id, &stop_logs, ""); });
    }

    std::vector<cuopt::internals::base_solution_callback_t*> callbacks =
      settings.get_mip_callbacks();
    int64_t incumbent_index = 0;
    bool incumbents_done    = callbacks.empty();
    CUOPT_LOG_INFO(std::string("[remote_solve] MIP incumbent callbacks: ") +
                   std::to_string(callbacks.size()));
    if (!callbacks.empty()) {
      size_t n_vars = view.get_objective_coefficients().size();
      for (auto* cb : callbacks) {
        if (cb != nullptr) { cb->setup<f_t>(n_vars); }
      }
    }

    std::string status;
    while (true) {
      std::string st_err;
      if (!grpc_remote::check_status(address, job_id, status, st_err)) {
        stop_logs = true;
        if (log_thread.joinable()) { log_thread.join(); }
        grpc_remote::delete_result(address, job_id);
        return mip_solution_t<i_t, f_t>(cuopt::logic_error("gRPC CheckStatus failed: " + st_err,
                                                           cuopt::error_type_t::RuntimeError));
      }

      if (!incumbents_done) {
        std::vector<grpc_remote::Incumbent> incumbents;
        int64_t next_index = incumbent_index;
        bool job_complete  = false;
        std::string inc_err;
        if (grpc_remote::get_incumbents(address,
                                        job_id,
                                        incumbent_index,
                                        32,
                                        incumbents,
                                        next_index,
                                        job_complete,
                                        inc_err)) {
          if (!incumbents.empty()) {
            CUOPT_LOG_INFO(std::string("[remote_solve] Received ") +
                           std::to_string(incumbents.size()) + " incumbents");
          }
          for (const auto& inc : incumbents) {
            invoke_incumbent_callbacks<f_t>(callbacks, inc.assignment, inc.objective);
          }
          incumbent_index = next_index;
          if (job_complete) { incumbents_done = true; }
        } else if (!inc_err.empty()) {
          CUOPT_LOG_WARN(std::string("[remote_solve] GetIncumbents failed: ") + inc_err);
        }
      }

      if (status == "COMPLETED") { break; }
      if (status == "FAILED" || status == "CANCELLED" || status == "NOT_FOUND") {
        stop_logs = true;
        if (log_thread.joinable()) { log_thread.join(); }
        grpc_remote::delete_result(address, job_id);
        return mip_solution_t<i_t, f_t>(
          cuopt::logic_error("Remote job did not complete successfully (status=" + status + ")",
                             cuopt::error_type_t::RuntimeError));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!incumbents_done) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      // Final drain after completion to catch any last incumbents.
      for (int i = 0; i < 5; ++i) {
        std::vector<grpc_remote::Incumbent> incumbents;
        int64_t next_index = incumbent_index;
        bool job_complete  = false;
        std::string inc_err;
        if (!grpc_remote::get_incumbents(
              address, job_id, incumbent_index, 0, incumbents, next_index, job_complete, inc_err)) {
          break;
        }
        if (incumbents.empty() && next_index == incumbent_index) {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
          continue;
        }
        for (const auto& inc : incumbents) {
          invoke_incumbent_callbacks<f_t>(callbacks, inc.assignment, inc.objective);
        }
        incumbent_index = next_index;
        if (job_complete) { break; }
      }
    }

    std::vector<uint8_t> solution_bytes;
    std::string res_err;
    if (!grpc_remote::stream_result(address, job_id, solution_bytes, res_err)) {
      stop_logs = true;
      if (log_thread.joinable()) { log_thread.join(); }
      grpc_remote::delete_result(address, job_id);
      return mip_solution_t<i_t, f_t>(cuopt::logic_error("gRPC StreamResult failed: " + res_err,
                                                         cuopt::error_type_t::RuntimeError));
    }

    stop_logs = true;
    if (log_thread.joinable()) { log_thread.join(); }

    grpc_remote::delete_result(address, job_id);
    return serializer->deserialize_mip_solution(solution_bytes);
#endif
  }

  if (sync_mode) {
    //=========================================================================
    // SYNC/BLOCKING MODE: Unified architecture
    //
    // Server-side: Job goes through queue, handled by worker process.
    // Client blocks until completion (server uses condition variable).
    // This enables cancellation for "sync" jobs and concurrent solves.
    //=========================================================================

    std::vector<uint8_t> request_data =
      serializer->serialize_async_mip_request(view, settings, true /* blocking */);
    CUOPT_LOG_DEBUG(std::string("[remote_solve] Serialized MIP request (blocking): ") +
                    std::to_string(request_data.size()) + " bytes");

    remote_client_t client(config.host, config.port);
    if (!client.connect()) {
      return mip_solution_t<i_t, f_t>(cuopt::logic_error("Failed to connect to remote server",
                                                         cuopt::error_type_t::RuntimeError));
    }

    if (!client.send_request(request_data)) {
      return mip_solution_t<i_t, f_t>(cuopt::logic_error("Failed to send request to remote server",
                                                         cuopt::error_type_t::RuntimeError));
    }

    // Receive response (server blocks until job completes, then returns result)
    std::vector<uint8_t> response_data;
    if (!client.receive_response(response_data)) {
      return mip_solution_t<i_t, f_t>(cuopt::logic_error(
        "Failed to receive response from remote server", cuopt::error_type_t::RuntimeError));
    }

    CUOPT_LOG_DEBUG(std::string("[remote_solve] Received MIP result (blocking): ") +
                    std::to_string(response_data.size()) + " bytes");

    // Deserialize solution from result response (same format as async GET_RESULT)
    return serializer->deserialize_mip_result_response(response_data);

  } else {
    //=========================================================================
    // ASYNC MODE: Submit → Poll → Get Result → Delete
    //=========================================================================

    std::vector<uint8_t> request_data =
      serializer->serialize_async_mip_request(view, settings, false /* blocking */);
    CUOPT_LOG_DEBUG(std::string("[remote_solve] Serialized MIP request (async): ") +
                    std::to_string(request_data.size()) + " bytes");

    // Submit job
    auto [submit_ok, job_id_or_error] =
      submit_job<i_t, f_t>(config.host, config.port, request_data);
    if (!submit_ok) {
      return mip_solution_t<i_t, f_t>(cuopt::logic_error(
        "Job submission failed: " + job_id_or_error, cuopt::error_type_t::RuntimeError));
    }
    std::string job_id = job_id_or_error;
    CUOPT_LOG_INFO(std::string("[remote_solve] Job submitted, ID: ") + job_id);

    // Poll until complete
    if (!poll_until_complete<i_t, f_t>(config.host, config.port, job_id, true /* verbose */)) {
      delete_job<i_t, f_t>(config.host, config.port, job_id);
      return mip_solution_t<i_t, f_t>(
        cuopt::logic_error("Job failed or not found", cuopt::error_type_t::RuntimeError));
    }

    // Get result
    auto [result_ok, result_data] = get_result<i_t, f_t>(config.host, config.port, job_id);
    if (!result_ok) {
      delete_job<i_t, f_t>(config.host, config.port, job_id);
      return mip_solution_t<i_t, f_t>(
        cuopt::logic_error("Failed to retrieve result", cuopt::error_type_t::RuntimeError));
    }

    // Delete job from server
    delete_job<i_t, f_t>(config.host, config.port, job_id);
    CUOPT_LOG_DEBUG(std::string("[remote_solve] Job ") + job_id + " deleted from server");

    // Deserialize solution from async result response
    return serializer->deserialize_mip_result_response(result_data);
  }
}

//============================================================================
// Cancel Job Remote
//============================================================================

cancel_job_result_t cancel_job_remote(const remote_solve_config_t& config,
                                      const std::string& job_id)
{
  CUOPT_LOG_INFO(std::string("[remote_solve] Cancelling job ") + job_id + " on " + config.host +
                 ":" + std::to_string(config.port));

  // Prefer gRPC cancel when available.
  if (use_grpc_transport()) {
#if CUOPT_ENABLE_GRPC
    const std::string address = config.host + ":" + std::to_string(config.port);
    bool ok                   = false;
    std::string status;
    std::string msg;
    std::string err;
    bool rpc_ok = grpc_remote::cancel_job(address, job_id, ok, status, msg, err);
    cancel_job_result_t result;
    result.success = rpc_ok && ok;
    result.message = rpc_ok ? msg : err;
    if (status == "QUEUED")
      result.job_status = remote_job_status_t::QUEUED;
    else if (status == "PROCESSING")
      result.job_status = remote_job_status_t::PROCESSING;
    else if (status == "COMPLETED")
      result.job_status = remote_job_status_t::COMPLETED;
    else if (status == "FAILED")
      result.job_status = remote_job_status_t::FAILED;
    else if (status == "CANCELLED")
      result.job_status = remote_job_status_t::CANCELLED;
    else
      result.job_status = remote_job_status_t::NOT_FOUND;
    return result;
#endif
  }

  // Fallback: legacy socket cancel.
  auto result = cancel_job_impl<int32_t, double>(config.host, config.port, job_id);

  if (result.success) {
    CUOPT_LOG_INFO(std::string("[remote_solve] Job ") + job_id + " cancelled successfully");
  } else {
    CUOPT_LOG_WARN(std::string("[remote_solve] Failed to cancel job ") + job_id + ": " +
                   result.message);
  }

  return result;
}

// Explicit instantiations
#if CUOPT_INSTANTIATE_FLOAT
template optimization_problem_solution_t<int32_t, float> solve_lp_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<int32_t, float>& view,
  const pdlp_solver_settings_t<int32_t, float>& settings);

template mip_solution_t<int32_t, float> solve_mip_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<int32_t, float>& view,
  const mip_solver_settings_t<int32_t, float>& settings);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template optimization_problem_solution_t<int32_t, double> solve_lp_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<int32_t, double>& view,
  const pdlp_solver_settings_t<int32_t, double>& settings);

template mip_solution_t<int32_t, double> solve_mip_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<int32_t, double>& view,
  const mip_solver_settings_t<int32_t, double>& settings);
#endif

}  // namespace cuopt::linear_programming
