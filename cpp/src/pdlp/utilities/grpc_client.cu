/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/utilities/grpc_client.hpp>

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/utilities/grpc_problem_mapper.hpp>
#include <cuopt/linear_programming/utilities/grpc_service_mapper.hpp>
#include <cuopt/linear_programming/utilities/grpc_settings_mapper.hpp>
#include <cuopt/linear_programming/utilities/grpc_solution_mapper.hpp>
#include <cuopt/linear_programming/utilities/proto_message_stream.hpp>
#include <utilities/logger.hpp>

#include <cuopt_remote_service.grpc.pb.h>
#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>

namespace cuopt::linear_programming {

// =============================================================================
// Constants
// =============================================================================

constexpr int64_t kMinChunkSize = 4 * 1024;  // 4 KiB minimum chunk

// =============================================================================
// Data Integrity - Simple Hash for Transfer Verification
// =============================================================================

/**
 * @brief Compute FNV-1a 64-bit hash for data integrity verification.
 *
 * This is a fast, non-cryptographic hash suitable for detecting data corruption
 * during streaming transfers. The hash is logged by both client and server to
 * allow verification that transferred data matches.
 *
 * @param data Pointer to data buffer
 * @param size Size of data in bytes
 * @return 64-bit hash value (logged as hex string)
 */
inline uint64_t compute_data_hash(const uint8_t* data, size_t size)
{
  // FNV-1a 64-bit hash constants
  constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
  constexpr uint64_t FNV_PRIME        = 1099511628211ULL;

  uint64_t hash = FNV_OFFSET_BASIS;
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint64_t>(data[i]);
    hash *= FNV_PRIME;
  }
  return hash;
}

/**
 * @brief Format hash as hex string for logging.
 */
inline std::string hash_to_hex(uint64_t hash)
{
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << hash;
  return oss.str();
}

// =============================================================================
// Debug Logging Helper
// =============================================================================

// Helper macro to log to debug callback if configured, otherwise to std::cerr
#define GRPC_CLIENT_DEBUG_LOG(config, msg)                                      \
  do {                                                                          \
    std::ostringstream _oss;                                                    \
    _oss << msg;                                                                \
    std::string _msg_str = _oss.str();                                          \
    if ((config).debug_log_callback) { (config).debug_log_callback(_msg_str); } \
    std::cerr << _msg_str << "\n";                                              \
  } while (0)

// Private implementation (PIMPL pattern to hide gRPC types)
struct grpc_client_t::impl_t {
  std::shared_ptr<grpc::Channel> channel;
  // Use StubInterface to support both real stubs and mock stubs for testing
  std::shared_ptr<cuopt::remote::CuOptRemoteService::StubInterface> stub;
  bool mock_mode = false;  // Set to true when using injected mock stub
};

// =============================================================================
// Test Helper Functions (for mock stub injection)
// =============================================================================

void grpc_test_inject_mock_stub(grpc_client_t& client, std::shared_ptr<void> stub)
{
  // Cast from void* to StubInterface* - caller must ensure correct type
  client.impl_->stub =
    std::static_pointer_cast<cuopt::remote::CuOptRemoteService::StubInterface>(stub);
  client.impl_->mock_mode = true;
}

void grpc_test_mark_as_connected(grpc_client_t& client) { client.impl_->mock_mode = true; }

grpc_client_t::grpc_client_t(const grpc_client_config_t& config)
  : impl_(std::make_unique<impl_t>()), config_(config)
{
  // Normalize sentinel value -1 to default timeout of 1 hour
  if (config_.timeout_seconds == -1) { config_.timeout_seconds = 3600; }
}

grpc_client_t::grpc_client_t(const std::string& server_address) : impl_(std::make_unique<impl_t>())
{
  config_.server_address = server_address;
}

grpc_client_t::~grpc_client_t() { stop_log_streaming(); }

bool grpc_client_t::connect()
{
  std::shared_ptr<grpc::ChannelCredentials> creds;

  if (config_.enable_tls) {
    grpc::SslCredentialsOptions ssl_opts;

    // Root CA certificates for verifying the server
    if (!config_.tls_root_certs.empty()) { ssl_opts.pem_root_certs = config_.tls_root_certs; }

    // Client certificate and key for mTLS
    if (!config_.tls_client_cert.empty() && !config_.tls_client_key.empty()) {
      ssl_opts.pem_cert_chain  = config_.tls_client_cert;
      ssl_opts.pem_private_key = config_.tls_client_key;
    }

    creds = grpc::SslCredentials(ssl_opts);
  } else {
    creds = grpc::InsecureChannelCredentials();
  }

  grpc::ChannelArguments channel_args;
  const int channel_limit = (config_.max_message_bytes <= 0)
                              ? -1
                              : static_cast<int>(std::min<int64_t>(
                                  config_.max_message_bytes, std::numeric_limits<int>::max()));
  channel_args.SetMaxReceiveMessageSize(channel_limit);
  channel_args.SetMaxSendMessageSize(channel_limit);

  impl_->channel = grpc::CreateCustomChannel(config_.server_address, creds, channel_args);
  impl_->stub    = cuopt::remote::CuOptRemoteService::NewStub(impl_->channel);

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] Connecting to " << config_.server_address
                                                       << (config_.enable_tls ? " (TLS)" : ""));

  // Try to check connectivity with a short deadline
  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
  if (!impl_->channel->WaitForConnected(deadline)) {
    last_error_ = "Failed to connect to server at " + config_.server_address;
    GRPC_CLIENT_DEBUG_LOG(config_, "[grpc_client] Connection failed: " << last_error_);
    return false;
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] Connected successfully to " << config_.server_address);
  return true;
}

bool grpc_client_t::is_connected() const
{
  // In mock mode, we're always "connected" if a stub is present
  if (impl_->mock_mode) { return impl_->stub != nullptr; }

  if (!impl_->channel) return false;
  auto state = impl_->channel->GetState(false);
  return state == GRPC_CHANNEL_READY || state == GRPC_CHANNEL_IDLE;
}

void grpc_client_t::start_log_streaming(const std::string& job_id)
{
  if (!config_.stream_logs || !config_.log_callback) return;

  stop_logs_.store(false);
  log_thread_ = std::make_unique<std::thread>([this, job_id]() {
    grpc::ClientContext context;
    auto request = build_stream_logs_request(job_id, 0);
    auto reader  = impl_->stub->StreamLogs(&context, request);

    cuopt::remote::LogMessage log_msg;
    while (reader->Read(&log_msg)) {
      if (stop_logs_.load()) {
        context.TryCancel();
        break;
      }

      // Call user's log callback
      if (config_.log_callback) { config_.log_callback(log_msg.line()); }

      if (log_msg.job_complete()) { break; }
    }
    reader->Finish();
  });
}

void grpc_client_t::stop_log_streaming()
{
  stop_logs_.store(true);
  if (log_thread_ && log_thread_->joinable()) { log_thread_->join(); }
  log_thread_.reset();
}

// =============================================================================
// Async Job Management Operations
// =============================================================================

job_status_result_t grpc_client_t::check_status(const std::string& job_id)
{
  job_status_result_t result;

  grpc::ClientContext context;
  auto request = build_status_request(job_id);
  cuopt::remote::StatusResponse response;
  auto status = impl_->stub->CheckStatus(&context, request, &response);

  if (!status.ok()) {
    result.error_message = "CheckStatus failed: " + status.error_message();
    return result;
  }

  result.success           = true;
  result.message           = response.message();
  result.result_size_bytes = response.result_size_bytes();

  // Track server max message size
  if (response.max_message_bytes() > 0) {
    server_max_message_bytes_ = response.max_message_bytes();
  }

  switch (response.job_status()) {
    case cuopt::remote::QUEUED: result.status = job_status_t::QUEUED; break;
    case cuopt::remote::PROCESSING: result.status = job_status_t::PROCESSING; break;
    case cuopt::remote::COMPLETED: result.status = job_status_t::COMPLETED; break;
    case cuopt::remote::FAILED: result.status = job_status_t::FAILED; break;
    case cuopt::remote::CANCELLED: result.status = job_status_t::CANCELLED; break;
    default: result.status = job_status_t::NOT_FOUND; break;
  }

  return result;
}

job_status_result_t grpc_client_t::wait_for_completion(const std::string& job_id)
{
  job_status_result_t result;

  grpc::ClientContext context;
  cuopt::remote::WaitRequest request;
  request.set_job_id(job_id);
  cuopt::remote::WaitResponse response;

  auto status = impl_->stub->WaitForCompletion(&context, request, &response);

  if (!status.ok()) {
    result.error_message = "WaitForCompletion failed: " + status.error_message();
    return result;
  }

  result.success           = true;
  result.message           = response.message();
  result.result_size_bytes = response.result_size_bytes();

  switch (response.job_status()) {
    case cuopt::remote::QUEUED: result.status = job_status_t::QUEUED; break;
    case cuopt::remote::PROCESSING: result.status = job_status_t::PROCESSING; break;
    case cuopt::remote::COMPLETED: result.status = job_status_t::COMPLETED; break;
    case cuopt::remote::FAILED: result.status = job_status_t::FAILED; break;
    case cuopt::remote::CANCELLED: result.status = job_status_t::CANCELLED; break;
    default: result.status = job_status_t::NOT_FOUND; break;
  }

  return result;
}

cancel_result_t grpc_client_t::cancel_job(const std::string& job_id)
{
  cancel_result_t result;

  grpc::ClientContext context;
  auto request = build_cancel_request(job_id);
  cuopt::remote::CancelResponse response;
  auto status = impl_->stub->CancelJob(&context, request, &response);

  if (!status.ok()) {
    result.error_message = "CancelJob failed: " + status.error_message();
    return result;
  }

  result.success = (response.status() == cuopt::remote::SUCCESS);
  result.message = response.message();

  switch (response.job_status()) {
    case cuopt::remote::QUEUED: result.job_status = job_status_t::QUEUED; break;
    case cuopt::remote::PROCESSING: result.job_status = job_status_t::PROCESSING; break;
    case cuopt::remote::COMPLETED: result.job_status = job_status_t::COMPLETED; break;
    case cuopt::remote::FAILED: result.job_status = job_status_t::FAILED; break;
    case cuopt::remote::CANCELLED: result.job_status = job_status_t::CANCELLED; break;
    default: result.job_status = job_status_t::NOT_FOUND; break;
  }

  return result;
}

bool grpc_client_t::delete_job(const std::string& job_id)
{
  grpc::ClientContext context;
  cuopt::remote::DeleteRequest request;
  request.set_job_id(job_id);
  cuopt::remote::DeleteResponse response;
  auto status = impl_->stub->DeleteResult(&context, request, &response);

  if (!status.ok()) {
    last_error_ = "DeleteResult RPC failed: " + status.error_message();
    return false;
  }

  // Check response status - job must exist to be deleted
  if (response.status() == cuopt::remote::ERROR_NOT_FOUND) {
    last_error_ = "Job not found: " + job_id;
    return false;
  }

  if (response.status() != cuopt::remote::SUCCESS) {
    last_error_ = "DeleteResult failed: " + response.message();
    return false;
  }

  return true;
}

incumbents_result_t grpc_client_t::get_incumbents(const std::string& job_id,
                                                  int64_t from_index,
                                                  int32_t max_count)
{
  incumbents_result_t result;

  grpc::ClientContext context;
  cuopt::remote::IncumbentRequest request;
  request.set_job_id(job_id);
  request.set_from_index(from_index);
  request.set_max_count(max_count);

  cuopt::remote::IncumbentResponse response;
  auto status = impl_->stub->GetIncumbents(&context, request, &response);

  if (!status.ok()) {
    result.error_message = "GetIncumbents failed: " + status.error_message();
    return result;
  }

  result.success      = true;
  result.next_index   = response.next_index();
  result.job_complete = response.job_complete();

  for (const auto& inc : response.incumbents()) {
    incumbent_t entry;
    entry.index     = inc.index();
    entry.objective = inc.objective();
    entry.assignment.reserve(inc.assignment_size());
    for (int i = 0; i < inc.assignment_size(); ++i) {
      entry.assignment.push_back(inc.assignment(i));
    }
    result.incumbents.push_back(std::move(entry));
  }

  return result;
}

bool grpc_client_t::stream_logs(
  const std::string& job_id,
  int64_t from_byte,
  std::function<bool(const std::string& line, bool job_complete)> callback)
{
  grpc::ClientContext context;
  cuopt::remote::StreamLogsRequest request;
  request.set_job_id(job_id);
  request.set_from_byte(from_byte);

  auto reader = impl_->stub->StreamLogs(&context, request);

  cuopt::remote::LogMessage log_msg;
  while (reader->Read(&log_msg)) {
    bool should_continue = callback(log_msg.line(), log_msg.job_complete());
    if (!should_continue) {
      context.TryCancel();
      break;
    }
    if (log_msg.job_complete()) { break; }
  }

  auto status = reader->Finish();
  return status.ok() || status.error_code() == grpc::StatusCode::CANCELLED;
}

// =============================================================================
// Submit and Chunk Size Helpers
// =============================================================================

int64_t grpc_client_t::compute_chunk_size(int64_t server_max, int64_t config_max, int64_t preferred)
{
  int64_t effective_max = config_max;
  if (server_max > 0 && (effective_max <= 0 || server_max < effective_max)) {
    effective_max = server_max;
  }
  int64_t chunk_size = preferred;
  if (effective_max > 0 && chunk_size > effective_max / 2) { chunk_size = effective_max / 2; }
  if (chunk_size < kMinChunkSize) { chunk_size = kMinChunkSize; }
  return chunk_size;
}

bool grpc_client_t::submit_unary(const std::vector<uint8_t>& serialized_data,
                                 bool is_lp,
                                 std::string& job_id_out)
{
  job_id_out.clear();

  grpc::ClientContext context;
  cuopt::remote::SubmitJobRequest request;

  bool parse_ok = false;
  if (is_lp) {
    auto* lp_req = request.mutable_lp_request();
    parse_ok =
      lp_req->ParseFromArray(serialized_data.data(), static_cast<int>(serialized_data.size()));
  } else {
    auto* mip_req = request.mutable_mip_request();
    parse_ok =
      mip_req->ParseFromArray(serialized_data.data(), static_cast<int>(serialized_data.size()));
  }

  if (!parse_ok) {
    last_error_ = "Failed to parse serialized request for unary submit";
    return false;
  }

  cuopt::remote::SubmitJobResponse response;
  auto status = impl_->stub->SubmitJob(&context, request, &response);

  if (!status.ok()) {
    last_error_ = "SubmitJob failed: " + status.error_message();
    return false;
  }

  job_id_out = response.job_id();
  if (job_id_out.empty()) {
    last_error_ = "SubmitJob succeeded but no job_id returned";
    return false;
  }

  GRPC_CLIENT_DEBUG_LOG(config_, "[grpc_client] Unary submit succeeded, job_id=" << job_id_out);
  return true;
}

// =============================================================================
// Chunked Array Upload
// =============================================================================

template <typename i_t, typename f_t>
bool grpc_client_t::upload_chunked_arrays(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                          const cuopt::remote::ChunkedProblemHeader& header,
                                          bool is_mip,
                                          std::string& job_id_out)
{
  job_id_out.clear();

  // --- 1. StartChunkedUpload ---
  std::string upload_id;
  {
    grpc::ClientContext context;
    cuopt::remote::StartChunkedUploadRequest request;
    request.set_problem_type(is_mip ? cuopt::remote::MIP : cuopt::remote::LP);
    *request.mutable_problem_header() = header;

    cuopt::remote::StartChunkedUploadResponse response;
    auto status = impl_->stub->StartChunkedUpload(&context, request, &response);

    if (!status.ok()) {
      last_error_ = "StartChunkedUpload failed: " + status.error_message();
      return false;
    }

    upload_id = response.upload_id();
    if (response.max_message_bytes() > 0) {
      server_max_message_bytes_ = response.max_message_bytes();
    }
  }

  GRPC_CLIENT_DEBUG_LOG(config_, "[grpc_client] ChunkedUpload started, upload_id=" << upload_id);

  // --- 2. Build MessageStream and extract array envelopes ---
  auto msg_stream =
    problem_to_messages(problem,
                        static_cast<const pdlp_solver_settings_t<i_t, f_t>*>(nullptr),
                        static_cast<const mip_solver_settings_t<i_t, f_t>*>(nullptr),
                        false);

  // --- 3. Send each array envelope as SendArrayChunk RPCs ---
  int64_t chunk_data_budget = config_.chunk_size_bytes;
  if (chunk_data_budget <= 0) { chunk_data_budget = 1LL * 1024 * 1024; }

  const int64_t proto_overhead = 64;
  if (chunk_data_budget > proto_overhead) { chunk_data_budget -= proto_overhead; }

  int total_chunks = 0;
  int total_arrays = 0;

  for (const auto& env : msg_stream) {
    if (!env.has_array_data()) continue;
    const auto& ad       = env.array_data();
    const auto& raw_data = ad.data();
    if (raw_data.empty()) continue;

    auto field_id = static_cast<cuopt::remote::ArrayFieldId>(ad.field_id());

    // Determine element size based on field type
    int64_t elem_size = sizeof(double);
    switch (field_id) {
      case cuopt::remote::FIELD_A_INDICES:
      case cuopt::remote::FIELD_A_OFFSETS:
      case cuopt::remote::FIELD_Q_INDICES:
      case cuopt::remote::FIELD_Q_OFFSETS: elem_size = sizeof(int32_t); break;
      case cuopt::remote::FIELD_ROW_TYPES:
      case cuopt::remote::FIELD_VARIABLE_TYPES:
      case cuopt::remote::FIELD_VARIABLE_NAMES:
      case cuopt::remote::FIELD_ROW_NAMES: elem_size = 1; break;
      default: break;
    }

    int64_t total_bytes    = static_cast<int64_t>(raw_data.size());
    int64_t total_elements = total_bytes / elem_size;

    int64_t elements_per_chunk = chunk_data_budget / elem_size;
    if (elements_per_chunk <= 0) elements_per_chunk = 1;

    const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(raw_data.data());

    for (int64_t elem_offset = 0; elem_offset < total_elements; elem_offset += elements_per_chunk) {
      int64_t elems_in_chunk = std::min(elements_per_chunk, total_elements - elem_offset);
      int64_t byte_offset    = elem_offset * elem_size;
      int64_t byte_count     = elems_in_chunk * elem_size;

      grpc::ClientContext chunk_context;
      cuopt::remote::SendArrayChunkRequest chunk_request;
      chunk_request.set_upload_id(upload_id);
      auto* ac = chunk_request.mutable_chunk();
      ac->set_field_id(field_id);
      ac->set_element_offset(elem_offset);
      ac->set_total_elements(total_elements);
      ac->set_data(data_ptr + byte_offset, byte_count);

      cuopt::remote::SendArrayChunkResponse chunk_response;
      auto status = impl_->stub->SendArrayChunk(&chunk_context, chunk_request, &chunk_response);

      if (!status.ok()) {
        last_error_ = "SendArrayChunk failed: " + status.error_message();
        return false;
      }

      ++total_chunks;
    }
    ++total_arrays;
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] ChunkedUpload sent " << total_chunks << " array chunks for "
                                                            << total_arrays << " arrays");

  // --- 4. FinishChunkedUpload ---
  {
    grpc::ClientContext context;
    cuopt::remote::FinishChunkedUploadRequest request;
    request.set_upload_id(upload_id);

    cuopt::remote::SubmitJobResponse response;
    auto status = impl_->stub->FinishChunkedUpload(&context, request, &response);

    if (!status.ok()) {
      last_error_ = "FinishChunkedUpload failed: " + status.error_message();
      return false;
    }

    job_id_out = response.job_id();
  }

  GRPC_CLIENT_DEBUG_LOG(
    config_,
    "[grpc_client] ChunkedUpload complete: " << total_chunks << " chunks, job_id=" << job_id_out);
  return true;
}

bool grpc_client_t::get_result_or_download(const std::string& job_id,
                                           std::vector<uint8_t>& result_data_out,
                                           bool& was_chunked)
{
  result_data_out.clear();
  was_chunked = false;

  int64_t result_size_hint = 0;
  {
    grpc::ClientContext context;
    auto request = build_status_request(job_id);
    cuopt::remote::StatusResponse response;
    auto status = impl_->stub->CheckStatus(&context, request, &response);

    if (status.ok()) {
      result_size_hint = response.result_size_bytes();
      if (response.max_message_bytes() > 0) {
        server_max_message_bytes_ = response.max_message_bytes();
      }
    }
  }

  int64_t effective_max = config_.max_message_bytes;
  if (server_max_message_bytes_ > 0 && server_max_message_bytes_ < effective_max) {
    effective_max = server_max_message_bytes_;
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] get_result_or_download: result_size_hint="
                          << result_size_hint << " bytes, client_max=" << config_.max_message_bytes
                          << ", server_max=" << server_max_message_bytes_
                          << ", effective_max=" << effective_max);

  if (result_size_hint > 0 && effective_max > 0 && result_size_hint > effective_max) {
    GRPC_CLIENT_DEBUG_LOG(config_,
                          "[grpc_client] Using chunked download directly (result_size_hint="
                            << result_size_hint << " > effective_max=" << effective_max << ")");
    was_chunked = true;
    return download_chunked_result(job_id, result_data_out);
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] Attempting unary GetResult (result_size_hint="
                          << result_size_hint << " <= effective_max=" << effective_max << ")");

  grpc::ClientContext context;
  auto request = build_get_result_request(job_id);
  cuopt::remote::ResultResponse response;
  auto status = impl_->stub->GetResult(&context, request, &response);

  if (status.ok() && response.status() == cuopt::remote::SUCCESS) {
    if (response.has_lp_solution()) {
      const auto& sol = response.lp_solution();
      result_data_out.resize(sol.ByteSizeLong());
      if (sol.SerializeToArray(result_data_out.data(), result_data_out.size())) {
        GRPC_CLIENT_DEBUG_LOG(config_,
                              "[grpc_client] Unary GetResult succeeded, result_size="
                                << result_data_out.size() << " bytes");
        return true;
      }
      last_error_ = "Failed to serialize LP solution from GetResult";
      return false;
    }
    if (response.has_mip_solution()) {
      const auto& sol = response.mip_solution();
      result_data_out.resize(sol.ByteSizeLong());
      if (sol.SerializeToArray(result_data_out.data(), result_data_out.size())) {
        GRPC_CLIENT_DEBUG_LOG(config_,
                              "[grpc_client] Unary GetResult succeeded, result_size="
                                << result_data_out.size() << " bytes");
        return true;
      }
      last_error_ = "Failed to serialize MIP solution from GetResult";
      return false;
    }
    last_error_ = "GetResult succeeded but no solution in response";
    return false;
  }

  if (status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED) {
    GRPC_CLIENT_DEBUG_LOG(config_,
                          "[grpc_client] GetResult rejected (RESOURCE_EXHAUSTED), "
                          "falling back to chunked download");
    was_chunked = true;
    return download_chunked_result(job_id, result_data_out);
  }

  if (!status.ok()) {
    last_error_ = "GetResult failed: " + status.error_message();
  } else if (response.status() != cuopt::remote::SUCCESS) {
    last_error_ = "GetResult indicates failure: " + response.error_message();
  }
  return false;
}

bool grpc_client_t::download_chunked_result(const std::string& job_id,
                                            std::vector<uint8_t>& result_data_out)
{
  result_data_out.clear();

  GRPC_CLIENT_DEBUG_LOG(config_, "[grpc_client] Starting chunked download for job " << job_id);

  // --- 1. StartChunkedDownload ---
  std::string download_id;
  cuopt::remote::ChunkedResultHeader header;
  {
    grpc::ClientContext context;
    cuopt::remote::StartChunkedDownloadRequest request;
    request.set_job_id(job_id);

    cuopt::remote::StartChunkedDownloadResponse response;
    auto status = impl_->stub->StartChunkedDownload(&context, request, &response);

    if (!status.ok()) {
      last_error_ = "StartChunkedDownload failed: " + status.error_message();
      return false;
    }

    download_id = response.download_id();
    header      = response.header();
    if (response.max_message_bytes() > 0) {
      server_max_message_bytes_ = response.max_message_bytes();
    }
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] ChunkedDownload started, download_id="
                          << download_id << " arrays=" << header.arrays_size()
                          << " is_mip=" << header.is_mip());

  // --- 2. Reconstruct a MessageStream: header envelope + array data envelopes ---
  MessageStream result_stream;

  // First envelope: the result header (scalars)
  cuopt::remote::PipeEnvelope hdr_env;
  *hdr_env.mutable_result_header() = header;
  result_stream.push_back(std::move(hdr_env));

  // --- 3. Fetch each array via GetResultChunk RPCs ---
  int64_t chunk_data_budget = config_.chunk_size_bytes;
  if (chunk_data_budget <= 0) { chunk_data_budget = 1LL * 1024 * 1024; }
  const int64_t proto_overhead = 64;
  if (chunk_data_budget > proto_overhead) { chunk_data_budget -= proto_overhead; }

  int total_chunks = 0;

  for (const auto& arr_desc : header.arrays()) {
    auto field_id       = arr_desc.field_id();
    int64_t total_elems = arr_desc.total_elements();
    int64_t elem_size   = arr_desc.element_size_bytes();
    if (total_elems <= 0) continue;

    int64_t elems_per_chunk = chunk_data_budget / elem_size;
    if (elems_per_chunk <= 0) elems_per_chunk = 1;

    // Accumulate all chunks for this array into a single buffer
    std::vector<uint8_t> array_bytes(static_cast<size_t>(total_elems * elem_size));

    for (int64_t elem_offset = 0; elem_offset < total_elems; elem_offset += elems_per_chunk) {
      int64_t elems_wanted = std::min(elems_per_chunk, total_elems - elem_offset);

      grpc::ClientContext chunk_ctx;
      cuopt::remote::GetResultChunkRequest chunk_req;
      chunk_req.set_download_id(download_id);
      chunk_req.set_field_id(field_id);
      chunk_req.set_element_offset(elem_offset);
      chunk_req.set_max_elements(elems_wanted);

      cuopt::remote::GetResultChunkResponse chunk_resp;
      auto status = impl_->stub->GetResultChunk(&chunk_ctx, chunk_req, &chunk_resp);

      if (!status.ok()) {
        last_error_ = "GetResultChunk failed: " + status.error_message();
        return false;
      }

      int64_t elems_received = chunk_resp.elements_in_chunk();
      const auto& data       = chunk_resp.data();

      if (static_cast<int64_t>(data.size()) != elems_received * elem_size) {
        last_error_ = "GetResultChunk: data size mismatch";
        return false;
      }

      std::memcpy(array_bytes.data() + elem_offset * elem_size, data.data(), data.size());
      ++total_chunks;
    }

    // Create a PipeArrayData envelope for this array
    cuopt::remote::PipeEnvelope arr_env;
    auto* ad = arr_env.mutable_array_data();
    ad->set_field_id(static_cast<int32_t>(field_id));
    ad->set_data(array_bytes.data(), array_bytes.size());
    result_stream.push_back(std::move(arr_env));
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] ChunkedDownload fetched "
                          << total_chunks << " chunks for " << header.arrays_size() << " arrays");

  // --- 4. FinishChunkedDownload ---
  {
    grpc::ClientContext context;
    cuopt::remote::FinishChunkedDownloadRequest request;
    request.set_download_id(download_id);

    cuopt::remote::FinishChunkedDownloadResponse response;
    auto status = impl_->stub->FinishChunkedDownload(&context, request, &response);

    if (!status.ok()) {
      GRPC_CLIENT_DEBUG_LOG(
        config_, "[grpc_client] FinishChunkedDownload warning: " << status.error_message());
    }
  }

  // --- 5. Serialize the MessageStream so get_lp_result/get_mip_result can deserialize ---
  result_data_out = serialize_stream(result_stream);

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] ChunkedDownload complete: "
                          << total_chunks << " chunks, result_size=" << result_data_out.size()
                          << " bytes");

  return true;
}

// =============================================================================
// Submit and Get Result Templates (Async Operations)
// =============================================================================

template <typename i_t, typename f_t>
submit_result_t grpc_client_t::submit_lp(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                         const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  submit_result_t result;

  GRPC_CLIENT_DEBUG_LOG(config_, "[grpc_client] submit_lp: starting submission");

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    GRPC_CLIENT_DEBUG_LOG(config_, "[grpc_client] submit_lp: not connected to server");
    return result;
  }

  // Check if chunked array upload should be used
  bool use_chunked = false;
  if (config_.chunked_array_threshold_bytes >= 0) {
    size_t est  = estimate_problem_proto_size(problem);
    use_chunked = (static_cast<int64_t>(est) > config_.chunked_array_threshold_bytes);
    GRPC_CLIENT_DEBUG_LOG(config_,
                          "[grpc_client] submit_lp: estimated_size="
                            << est << " threshold=" << config_.chunked_array_threshold_bytes
                            << " use_chunked=" << use_chunked);
  }

  if (use_chunked) {
    cuopt::remote::ChunkedProblemHeader header;
    populate_chunked_header_lp(problem, settings, &header);
    if (!upload_chunked_arrays(problem, header, false, result.job_id)) {
      result.error_message = last_error_;
      return result;
    }
  } else {
    auto submit_request = build_lp_submit_request(problem, settings);
    const auto& lp_req  = submit_request.lp_request();
    std::vector<uint8_t> serialized_data(lp_req.ByteSizeLong());
    if (!lp_req.SerializeToArray(serialized_data.data(), serialized_data.size())) {
      result.error_message = "Failed to serialize LP request";
      return result;
    }
    if (!submit_unary(serialized_data, true, result.job_id)) {
      result.error_message = last_error_;
      return result;
    }
  }

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] submit_lp: job submitted, job_id=" << result.job_id);
  result.success = true;
  return result;
}

template <typename i_t, typename f_t>
submit_result_t grpc_client_t::submit_mip(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                          const mip_solver_settings_t<i_t, f_t>& settings,
                                          bool enable_incumbents)
{
  submit_result_t result;

  GRPC_CLIENT_DEBUG_LOG(config_,
                        "[grpc_client] submit_mip: starting submission"
                          << (enable_incumbents ? " (incumbents enabled)" : ""));

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  bool use_chunked = false;
  if (config_.chunked_array_threshold_bytes >= 0) {
    size_t est  = estimate_problem_proto_size(problem);
    use_chunked = (static_cast<int64_t>(est) > config_.chunked_array_threshold_bytes);
    GRPC_CLIENT_DEBUG_LOG(config_,
                          "[grpc_client] submit_mip: estimated_size="
                            << est << " threshold=" << config_.chunked_array_threshold_bytes
                            << " use_chunked=" << use_chunked);
  }

  if (use_chunked) {
    cuopt::remote::ChunkedProblemHeader header;
    populate_chunked_header_mip(problem, settings, enable_incumbents, &header);
    if (!upload_chunked_arrays(problem, header, true, result.job_id)) {
      result.error_message = last_error_;
      return result;
    }
  } else {
    auto submit_request = build_mip_submit_request(problem, settings, enable_incumbents);
    const auto& mip_req = submit_request.mip_request();
    std::vector<uint8_t> serialized_data(mip_req.ByteSizeLong());
    if (!mip_req.SerializeToArray(serialized_data.data(), serialized_data.size())) {
      result.error_message = "Failed to serialize MIP request";
      return result;
    }
    if (!submit_unary(serialized_data, false, result.job_id)) {
      result.error_message = last_error_;
      return result;
    }
  }

  GRPC_CLIENT_DEBUG_LOG(
    config_, "[grpc_client] submit_mip: job submitted successfully, job_id=" << result.job_id);
  result.success = true;
  return result;
}

template <typename i_t, typename f_t>
remote_lp_result_t<i_t, f_t> grpc_client_t::get_lp_result(const std::string& job_id)
{
  remote_lp_result_t<i_t, f_t> result;

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  std::vector<uint8_t> result_data;
  bool was_chunked = false;
  if (!get_result_or_download(job_id, result_data, was_chunked)) {
    result.error_message = last_error_;
    return result;
  }

  if (was_chunked) {
    auto stream = deserialize_stream(result_data);
    result.solution =
      std::make_unique<cpu_lp_solution_t<i_t, f_t>>(messages_to_lp_solution<i_t, f_t>(stream));
  } else {
    cuopt::remote::LPSolution pb_solution;
    if (!pb_solution.ParseFromArray(result_data.data(), result_data.size())) {
      result.error_message = "Failed to parse LP solution from result data";
      return result;
    }
    result.solution = std::make_unique<cpu_lp_solution_t<i_t, f_t>>(
      map_proto_to_lp_solution<i_t, f_t>(pb_solution));
  }
  result.success = true;
  return result;
}

template <typename i_t, typename f_t>
remote_mip_result_t<i_t, f_t> grpc_client_t::get_mip_result(const std::string& job_id)
{
  remote_mip_result_t<i_t, f_t> result;

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  std::vector<uint8_t> result_data;
  bool was_chunked = false;
  if (!get_result_or_download(job_id, result_data, was_chunked)) {
    result.error_message = last_error_;
    return result;
  }

  if (was_chunked) {
    auto stream = deserialize_stream(result_data);
    result.solution =
      std::make_unique<cpu_mip_solution_t<i_t, f_t>>(messages_to_mip_solution<i_t, f_t>(stream));
  } else {
    cuopt::remote::MIPSolution pb_solution;
    if (!pb_solution.ParseFromArray(result_data.data(), result_data.size())) {
      result.error_message = "Failed to parse MIP solution from result data";
      return result;
    }
    result.solution = std::make_unique<cpu_mip_solution_t<i_t, f_t>>(
      map_proto_to_mip_solution<i_t, f_t>(pb_solution));
  }
  result.success = true;
  return result;
}

// =============================================================================
// Blocking Solve Operations
// =============================================================================

// LP solve implementation
template <typename i_t, typename f_t>
remote_lp_result_t<i_t, f_t> grpc_client_t::solve_lp(
  const cpu_optimization_problem_t<i_t, f_t>& problem,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  remote_lp_result_t<i_t, f_t> result;

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  // 1. Submit job (chunked arrays or serialized protobuf based on size)
  std::string job_id;
  bool use_chunked = false;
  if (config_.chunked_array_threshold_bytes >= 0) {
    size_t est  = estimate_problem_proto_size(problem);
    use_chunked = (static_cast<int64_t>(est) > config_.chunked_array_threshold_bytes);
  }

  if (use_chunked) {
    cuopt::remote::ChunkedProblemHeader header;
    populate_chunked_header_lp(problem, settings, &header);
    if (!upload_chunked_arrays(problem, header, false, job_id)) {
      result.error_message = last_error_;
      return result;
    }
  } else {
    auto submit_request = build_lp_submit_request(problem, settings);
    const auto& lp_req  = submit_request.lp_request();
    std::vector<uint8_t> serialized_data(lp_req.ByteSizeLong());
    if (!lp_req.SerializeToArray(serialized_data.data(), serialized_data.size())) {
      result.error_message = "Failed to serialize LP request";
      return result;
    }
    if (!submit_unary(serialized_data, true, job_id)) {
      result.error_message = last_error_;
      return result;
    }
  }

  start_log_streaming(job_id);

  bool completed = false;
  std::string completion_error;

  if (config_.use_wait) {
    CUOPT_LOG_INFO("[grpc_client] Using WaitForCompletion RPC for job %s", job_id.c_str());
    auto wait_result = wait_for_completion(job_id);
    if (!wait_result.success) {
      stop_log_streaming();
      result.error_message = wait_result.error_message;
      return result;
    }
    switch (wait_result.status) {
      case job_status_t::COMPLETED: completed = true; break;
      case job_status_t::FAILED: completion_error = "Job failed: " + wait_result.message; break;
      case job_status_t::CANCELLED: completion_error = "Job was cancelled"; break;
      default:
        completion_error =
          "Unexpected job status: " + std::string(job_status_to_string(wait_result.status));
        break;
    }
  } else {
    CUOPT_LOG_INFO("[grpc_client] Using polling (CheckStatus) for job %s", job_id.c_str());
    int poll_count = 0;
    int max_polls  = (config_.timeout_seconds * 1000) / config_.poll_interval_ms;

    while (!completed && poll_count < max_polls) {
      std::this_thread::sleep_for(std::chrono::milliseconds(config_.poll_interval_ms));

      grpc::ClientContext status_context;
      auto status_request = build_status_request(job_id);
      cuopt::remote::StatusResponse status_response;
      auto status_status =
        impl_->stub->CheckStatus(&status_context, status_request, &status_response);

      if (!status_status.ok()) {
        stop_log_streaming();
        result.error_message = "CheckStatus failed: " + status_status.error_message();
        return result;
      }

      if (status_response.max_message_bytes() > 0) {
        server_max_message_bytes_ = status_response.max_message_bytes();
      }

      switch (status_response.job_status()) {
        case cuopt::remote::COMPLETED: completed = true; break;
        case cuopt::remote::FAILED:
          completion_error = "Job failed: " + status_response.message();
          break;
        case cuopt::remote::CANCELLED: completion_error = "Job was cancelled"; break;
        default: break;
      }

      if (!completion_error.empty()) break;
      poll_count++;
    }

    if (!completed && completion_error.empty()) {
      completion_error = "Timeout waiting for job completion";
    }
  }

  stop_log_streaming();

  if (!completed) {
    result.error_message = completion_error;
    return result;
  }

  std::vector<uint8_t> result_data;
  bool was_chunked = false;
  if (!get_result_or_download(job_id, result_data, was_chunked)) {
    result.error_message = last_error_;
    return result;
  }

  if (was_chunked) {
    auto stream = deserialize_stream(result_data);
    result.solution =
      std::make_unique<cpu_lp_solution_t<i_t, f_t>>(messages_to_lp_solution<i_t, f_t>(stream));
  } else {
    cuopt::remote::LPSolution pb_solution;
    if (!pb_solution.ParseFromArray(result_data.data(), result_data.size())) {
      result.error_message = "Failed to parse LP solution from result data";
      return result;
    }
    result.solution = std::make_unique<cpu_lp_solution_t<i_t, f_t>>(
      map_proto_to_lp_solution<i_t, f_t>(pb_solution));
  }
  result.success = true;

  return result;
}

// MIP solve implementation
template <typename i_t, typename f_t>
remote_mip_result_t<i_t, f_t> grpc_client_t::solve_mip(
  const cpu_optimization_problem_t<i_t, f_t>& problem,
  const mip_solver_settings_t<i_t, f_t>& settings,
  bool enable_incumbents)
{
  remote_mip_result_t<i_t, f_t> result;

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  // Enable incumbents if callback is set
  bool track_incumbents = enable_incumbents || (config_.incumbent_callback != nullptr);

  // 1. Submit job (chunked arrays or serialized protobuf based on size)
  std::string job_id;
  bool use_chunked = false;
  if (config_.chunked_array_threshold_bytes >= 0) {
    size_t est  = estimate_problem_proto_size(problem);
    use_chunked = (static_cast<int64_t>(est) > config_.chunked_array_threshold_bytes);
  }

  if (use_chunked) {
    cuopt::remote::ChunkedProblemHeader header;
    populate_chunked_header_mip(problem, settings, track_incumbents, &header);
    if (!upload_chunked_arrays(problem, header, true, job_id)) {
      result.error_message = last_error_;
      return result;
    }
  } else {
    auto submit_request = build_mip_submit_request(problem, settings, track_incumbents);
    const auto& mip_req = submit_request.mip_request();
    std::vector<uint8_t> serialized_data(mip_req.ByteSizeLong());
    if (!mip_req.SerializeToArray(serialized_data.data(), serialized_data.size())) {
      result.error_message = "Failed to serialize MIP request";
      return result;
    }
    if (!submit_unary(serialized_data, false, job_id)) {
      result.error_message = last_error_;
      return result;
    }
  }

  start_log_streaming(job_id);

  bool cancel_requested = false;
  bool completed        = false;
  std::string completion_error;

  if (config_.use_wait) {
    // Use blocking WaitForCompletion RPC
    // Note: Incumbent callbacks are not supported in wait mode; use polling mode instead
    CUOPT_LOG_INFO("[grpc_client] Using WaitForCompletion RPC for job %s", job_id.c_str());
    auto wait_result = wait_for_completion(job_id);
    if (!wait_result.success) {
      stop_log_streaming();
      result.error_message = wait_result.error_message;
      return result;
    }

    switch (wait_result.status) {
      case job_status_t::COMPLETED: completed = true; break;
      case job_status_t::FAILED: completion_error = "Job failed: " + wait_result.message; break;
      case job_status_t::CANCELLED: completion_error = "Job was cancelled"; break;
      default:
        completion_error =
          "Unexpected job status: " + std::string(job_status_to_string(wait_result.status));
        break;
    }
  } else {
    // Poll for completion
    CUOPT_LOG_INFO("[grpc_client] Using polling (CheckStatus) for job %s", job_id.c_str());
    int poll_count = 0;
    int max_polls  = (config_.timeout_seconds * 1000) / config_.poll_interval_ms;

    // Track next incumbent index for polling
    int64_t incumbent_next_index = 0;
    auto last_incumbent_poll     = std::chrono::steady_clock::now();

    while (!completed && poll_count < max_polls) {
      std::this_thread::sleep_for(std::chrono::milliseconds(config_.poll_interval_ms));

      // Check if incumbent callback requested cancellation
      if (cancel_requested) {
        cancel_job(job_id);
        stop_log_streaming();
        result.error_message = "Cancelled by incumbent callback";
        return result;
      }

      // Poll for incumbents and invoke callbacks on main thread
      if (config_.incumbent_callback) {
        auto now = std::chrono::steady_clock::now();
        auto ms_since_last =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - last_incumbent_poll).count();
        if (ms_since_last >= config_.incumbent_poll_interval_ms) {
          auto inc_result = get_incumbents(job_id, incumbent_next_index, 0);
          if (inc_result.success) {
            for (const auto& inc : inc_result.incumbents) {
              bool should_continue =
                config_.incumbent_callback(inc.index, inc.objective, inc.assignment);
              if (!should_continue) {
                cancel_requested = true;
                break;
              }
            }
            incumbent_next_index = inc_result.next_index;
          }
          last_incumbent_poll = now;
        }
      }

      grpc::ClientContext status_context;
      auto status_request = build_status_request(job_id);
      cuopt::remote::StatusResponse status_response;
      auto status_status =
        impl_->stub->CheckStatus(&status_context, status_request, &status_response);

      if (!status_status.ok()) {
        stop_log_streaming();
        result.error_message = "CheckStatus failed: " + status_status.error_message();
        return result;
      }

      // Track server-reported limits
      if (status_response.max_message_bytes() > 0) {
        server_max_message_bytes_ = status_response.max_message_bytes();
      }

      switch (status_response.job_status()) {
        case cuopt::remote::COMPLETED: completed = true; break;
        case cuopt::remote::FAILED:
          completion_error = "Job failed: " + status_response.message();
          break;
        case cuopt::remote::CANCELLED: completion_error = "Job was cancelled"; break;
        default: break;  // QUEUED or PROCESSING, continue polling
      }

      if (!completion_error.empty()) break;
      poll_count++;
    }

    // Final incumbent poll to catch any remaining incumbents before completion
    if (config_.incumbent_callback && completed) {
      auto inc_result = get_incumbents(job_id, incumbent_next_index, 0);
      if (inc_result.success) {
        for (const auto& inc : inc_result.incumbents) {
          config_.incumbent_callback(inc.index, inc.objective, inc.assignment);
        }
      }
    }

    if (!completed && completion_error.empty()) {
      completion_error = "Timeout waiting for job completion";
    }
  }

  stop_log_streaming();

  if (!completed) {
    result.error_message = completion_error;
    return result;
  }

  // 6. Get result (uses streaming if needed)
  std::vector<uint8_t> result_data;
  bool was_chunked = false;
  if (!get_result_or_download(job_id, result_data, was_chunked)) {
    result.error_message = last_error_;
    return result;
  }

  // 7. Parse solution from serialized data
  if (was_chunked) {
    auto stream = deserialize_stream(result_data);
    result.solution =
      std::make_unique<cpu_mip_solution_t<i_t, f_t>>(messages_to_mip_solution<i_t, f_t>(stream));
  } else {
    cuopt::remote::MIPSolution pb_solution;
    if (!pb_solution.ParseFromArray(result_data.data(), result_data.size())) {
      result.error_message = "Failed to parse MIP solution from result data";
      return result;
    }
    result.solution = std::make_unique<cpu_mip_solution_t<i_t, f_t>>(
      map_proto_to_mip_solution<i_t, f_t>(pb_solution));
  }
  result.success = true;

  return result;
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template remote_lp_result_t<int32_t, float> grpc_client_t::solve_lp(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const pdlp_solver_settings_t<int32_t, float>& settings);
template remote_mip_result_t<int32_t, float> grpc_client_t::solve_mip(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const mip_solver_settings_t<int32_t, float>& settings,
  bool enable_incumbents);
template submit_result_t grpc_client_t::submit_lp(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const pdlp_solver_settings_t<int32_t, float>& settings);
template submit_result_t grpc_client_t::submit_mip(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const mip_solver_settings_t<int32_t, float>& settings,
  bool enable_incumbents);
template remote_lp_result_t<int32_t, float> grpc_client_t::get_lp_result(const std::string& job_id);
template remote_mip_result_t<int32_t, float> grpc_client_t::get_mip_result(
  const std::string& job_id);
template bool grpc_client_t::upload_chunked_arrays(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const cuopt::remote::ChunkedProblemHeader& header,
  bool is_mip,
  std::string& job_id_out);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template remote_lp_result_t<int32_t, double> grpc_client_t::solve_lp(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const pdlp_solver_settings_t<int32_t, double>& settings);
template remote_mip_result_t<int32_t, double> grpc_client_t::solve_mip(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const mip_solver_settings_t<int32_t, double>& settings,
  bool enable_incumbents);
template submit_result_t grpc_client_t::submit_lp(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const pdlp_solver_settings_t<int32_t, double>& settings);
template submit_result_t grpc_client_t::submit_mip(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const mip_solver_settings_t<int32_t, double>& settings,
  bool enable_incumbents);
template remote_lp_result_t<int32_t, double> grpc_client_t::get_lp_result(
  const std::string& job_id);
template remote_mip_result_t<int32_t, double> grpc_client_t::get_mip_result(
  const std::string& job_id);
template bool grpc_client_t::upload_chunked_arrays(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const cuopt::remote::ChunkedProblemHeader& header,
  bool is_mip,
  std::string& job_id_out);
#endif

}  // namespace cuopt::linear_programming
