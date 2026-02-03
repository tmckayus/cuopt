/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/utilities/grpc_client.hpp>

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/utilities/grpc_problem_mapper.hpp>
#include <cuopt/linear_programming/utilities/grpc_service_mapper.hpp>
#include <cuopt/linear_programming/utilities/grpc_settings_mapper.hpp>
#include <cuopt/linear_programming/utilities/grpc_solution_mapper.hpp>

#include <cuopt_remote_service.grpc.pb.h>
#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>

namespace cuopt::linear_programming {

// =============================================================================
// Constants
// =============================================================================

constexpr int64_t kMinChunkSize = 4 * 1024;  // 4 KiB minimum chunk

// Private implementation (PIMPL pattern to hide gRPC types)
struct grpc_client_t::impl_t {
  std::shared_ptr<grpc::Channel> channel;
  std::shared_ptr<cuopt::remote::CuOptRemoteService::Stub> stub;
};

grpc_client_t::grpc_client_t(const grpc_client_config_t& config)
  : impl_(std::make_unique<impl_t>()), config_(config)
{
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

  impl_->channel = grpc::CreateChannel(config_.server_address, creds);
  impl_->stub    = std::shared_ptr<cuopt::remote::CuOptRemoteService::Stub>(
    cuopt::remote::CuOptRemoteService::NewStub(impl_->channel).release());

  // Try to check connectivity with a short deadline
  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
  if (!impl_->channel->WaitForConnected(deadline)) {
    last_error_ = "Failed to connect to server at " + config_.server_address;
    return false;
  }

  return true;
}

bool grpc_client_t::is_connected() const
{
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
    last_error_ = "DeleteResult failed: " + status.error_message();
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
// Streaming Upload/Download Implementation
// =============================================================================

int64_t grpc_client_t::compute_chunk_size(int64_t server_max, int64_t config_max, int64_t preferred)
{
  // Use the most restrictive limit we know about
  int64_t effective_max = config_max;
  if (server_max > 0 && (effective_max <= 0 || server_max < effective_max)) {
    effective_max = server_max;
  }

  // Chunk size should be at most half of max message size to leave room for overhead
  int64_t chunk_size = preferred;
  if (effective_max > 0 && chunk_size > effective_max / 2) { chunk_size = effective_max / 2; }

  // Enforce minimum
  if (chunk_size < kMinChunkSize) { chunk_size = kMinChunkSize; }

  return chunk_size;
}

bool grpc_client_t::submit_or_upload(const std::vector<uint8_t>& serialized_data,
                                     streaming_problem_type_t problem_type,
                                     std::string& job_id_out)
{
  job_id_out.clear();

  const int64_t data_size = static_cast<int64_t>(serialized_data.size());

  // Decide whether to use unary or streaming based on size
  // Use streaming if data exceeds configured max or if we know server has a lower limit
  int64_t effective_max = config_.max_message_bytes;
  if (server_max_message_bytes_ > 0 && server_max_message_bytes_ < effective_max) {
    effective_max = server_max_message_bytes_;
  }

  // Try unary first if data fits
  if (effective_max <= 0 || data_size <= effective_max) {
    // Attempt unary submit
    grpc::ClientContext context;
    cuopt::remote::SubmitJobRequest request;

    bool parse_ok = false;
    if (problem_type == streaming_problem_type_t::LP) {
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

    if (status.ok()) {
      job_id_out = response.job_id();
      if (!job_id_out.empty()) { return true; }
      last_error_ = "SubmitJob succeeded but no job_id returned";
      return false;
    }

    // If RESOURCE_EXHAUSTED, fall through to streaming upload
    if (status.error_code() != grpc::StatusCode::RESOURCE_EXHAUSTED) {
      last_error_ = "SubmitJob failed: " + status.error_message();
      return false;
    }

    // Server rejected due to size - fall through to streaming
    std::cerr << "[grpc_client] Unary submit rejected (RESOURCE_EXHAUSTED), "
              << "falling back to streaming upload\n";
  }

  // Use streaming upload
  int64_t initial_chunk = compute_chunk_size(
    server_max_message_bytes_, config_.max_message_bytes, config_.chunk_size_bytes);

  return upload_and_submit(
    serialized_data.data(), serialized_data.size(), problem_type, job_id_out, initial_chunk);
}

bool grpc_client_t::upload_and_submit(const uint8_t* data,
                                      size_t size,
                                      streaming_problem_type_t problem_type,
                                      std::string& job_id_out,
                                      int64_t initial_chunk_size)
{
  job_id_out.clear();

  // Track the last known server max for retry logic
  int64_t last_server_max = server_max_message_bytes_;

  // Lambda to attempt a single upload
  auto do_upload = [&](int64_t chunk_size) -> bool {
    grpc::ClientContext context;

    // Set deadline for entire upload operation
    auto deadline =
      std::chrono::system_clock::now() + std::chrono::milliseconds(config_.upload_timeout_ms);
    context.set_deadline(deadline);

    // Create bidirectional stream
    auto stream = impl_->stub->UploadAndSubmit(&context);

    // 1. Send UploadStart message
    cuopt::remote::UploadJobRequest start_req;
    auto* start = start_req.mutable_start();
    start->set_problem_type(problem_type == streaming_problem_type_t::LP ? cuopt::remote::LP
                                                                         : cuopt::remote::MIP);
    start->set_resume(false);
    start->set_total_size(static_cast<int64_t>(size));

    if (!stream->Write(start_req)) {
      last_error_ = "UploadAndSubmit: failed to write start message";
      stream->Finish();
      return false;
    }

    // 2. Read ack for start
    cuopt::remote::UploadJobResponse start_resp;
    if (!stream->Read(&start_resp)) {
      last_error_     = "UploadAndSubmit: failed to read response after start";
      grpc::Status st = stream->Finish();
      if (!st.ok()) { last_error_ += " (grpc: " + st.error_message() + ")"; }
      return false;
    }

    if (start_resp.has_error()) {
      last_error_ = "UploadAndSubmit: " + start_resp.error().message();
      if (start_resp.error().max_message_bytes() > 0) {
        last_server_max = start_resp.error().max_message_bytes();
      }
      stream->Finish();
      return false;
    }

    if (!start_resp.has_ack()) {
      last_error_ = "UploadAndSubmit: expected ack after start, got unexpected response";
      stream->Finish();
      return false;
    }

    std::string upload_id  = start_resp.ack().upload_id();
    int64_t committed      = start_resp.ack().committed_size();
    int64_t server_max_msg = start_resp.ack().max_message_bytes();

    // Update our knowledge of server limits
    if (server_max_msg > 0) {
      last_server_max           = server_max_msg;
      server_max_message_bytes_ = server_max_msg;
      // Recompute chunk size if server reports a limit
      chunk_size = compute_chunk_size(server_max_msg, config_.max_message_bytes, chunk_size);
    }

    // 3. Send chunks with acknowledgment-based flow control
    while (static_cast<size_t>(committed) < size) {
      size_t offset = static_cast<size_t>(committed);
      size_t n      = std::min(static_cast<size_t>(chunk_size), size - offset);

      // Build chunk message
      cuopt::remote::UploadJobRequest chunk_req;
      auto* chunk = chunk_req.mutable_chunk();
      chunk->set_upload_id(upload_id);
      chunk->set_offset(committed);
      chunk->set_data(reinterpret_cast<const char*>(data + offset), n);

      if (!stream->Write(chunk_req)) {
        last_error_ = "UploadAndSubmit: failed to write chunk at offset " + std::to_string(offset);
        grpc::Status st = stream->Finish();
        if (!st.ok()) { last_error_ += " (grpc: " + st.error_message() + ")"; }
        return false;
      }

      // Read acknowledgment for this chunk
      cuopt::remote::UploadJobResponse chunk_resp;
      if (!stream->Read(&chunk_resp)) {
        last_error_ =
          "UploadAndSubmit: failed to read ack for chunk at offset " + std::to_string(offset);
        grpc::Status st = stream->Finish();
        if (!st.ok()) { last_error_ += " (grpc: " + st.error_message() + ")"; }
        return false;
      }

      if (chunk_resp.has_error()) {
        last_error_ = "UploadAndSubmit: " + chunk_resp.error().message();
        if (chunk_resp.error().max_message_bytes() > 0) {
          last_server_max = chunk_resp.error().max_message_bytes();
        }
        stream->Finish();
        return false;
      }

      if (!chunk_resp.has_ack()) {
        last_error_ = "UploadAndSubmit: expected ack for chunk, got unexpected response";
        stream->Finish();
        return false;
      }

      committed = chunk_resp.ack().committed_size();

      // Update chunk size if server reports different limits
      if (chunk_resp.ack().max_message_bytes() > 0 &&
          chunk_resp.ack().max_message_bytes() != server_max_msg) {
        server_max_msg            = chunk_resp.ack().max_message_bytes();
        last_server_max           = server_max_msg;
        server_max_message_bytes_ = server_max_msg;
        chunk_size = compute_chunk_size(server_max_msg, config_.max_message_bytes, chunk_size);
      }
    }

    // 4. Send finish message
    cuopt::remote::UploadJobRequest finish_req;
    finish_req.mutable_finish()->set_upload_id(upload_id);
    stream->Write(finish_req);
    stream->WritesDone();

    // 5. Read final response (should contain SubmitJobResponse with job_id)
    cuopt::remote::UploadJobResponse final_resp;
    while (stream->Read(&final_resp)) {
      if (final_resp.has_submit()) {
        job_id_out = final_resp.submit().job_id();
        break;
      }
      if (final_resp.has_error()) {
        last_error_ = "UploadAndSubmit: " + final_resp.error().message();
        break;
      }
    }

    grpc::Status st = stream->Finish();
    if (!st.ok()) {
      if (last_error_.empty()) {
        last_error_ = "UploadAndSubmit: stream finished with error: " + st.error_message();
      }
      return false;
    }

    if (job_id_out.empty()) {
      if (last_error_.empty()) { last_error_ = "UploadAndSubmit: no job_id returned"; }
      return false;
    }

    return true;
  };

  // First attempt
  if (do_upload(initial_chunk_size)) { return true; }

  std::cerr << "[grpc_client] Upload failed: " << last_error_ << "\n";

  // Retry with smaller chunks if we have retries left
  for (int retry = 0; retry < config_.max_upload_retries; ++retry) {
    // Compute a smaller chunk size for retry
    int64_t retry_chunk = initial_chunk_size / 2;
    if (last_server_max > 0) {
      retry_chunk = compute_chunk_size(last_server_max, config_.max_message_bytes, retry_chunk);
    }

    if (retry_chunk < kMinChunkSize) {
      std::cerr << "[grpc_client] Cannot retry: chunk size would be too small\n";
      break;
    }

    if (retry_chunk >= initial_chunk_size) {
      // No point retrying with same or larger chunk
      retry_chunk = initial_chunk_size / 2;
      if (retry_chunk < kMinChunkSize) { break; }
    }

    std::cerr << "[grpc_client] Retrying upload with chunk_size=" << retry_chunk << " (attempt "
              << (retry + 2) << "/" << (config_.max_upload_retries + 1) << ")\n";

    initial_chunk_size = retry_chunk;
    if (do_upload(retry_chunk)) { return true; }

    std::cerr << "[grpc_client] Retry failed: " << last_error_ << "\n";
  }

  return false;
}

bool grpc_client_t::get_result_or_stream(const std::string& job_id,
                                         std::vector<uint8_t>& result_data_out)
{
  result_data_out.clear();

  // First check status to get result size hint
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

  // Decide whether to use streaming based on result size
  int64_t effective_max = config_.max_message_bytes;
  if (server_max_message_bytes_ > 0 && server_max_message_bytes_ < effective_max) {
    effective_max = server_max_message_bytes_;
  }

  // If result is known to be large, go directly to streaming
  if (result_size_hint > 0 && effective_max > 0 && result_size_hint > effective_max) {
    return stream_result(job_id, result_data_out);
  }

  // Try unary GetResult first
  grpc::ClientContext context;
  auto request = build_get_result_request(job_id);
  cuopt::remote::ResultResponse response;
  auto status = impl_->stub->GetResult(&context, request, &response);

  if (status.ok() && response.status() == cuopt::remote::SUCCESS) {
    // Serialize the solution back to bytes for parsing
    if (response.has_lp_solution()) {
      const auto& sol = response.lp_solution();
      result_data_out.resize(sol.ByteSizeLong());
      if (sol.SerializeToArray(result_data_out.data(), result_data_out.size())) { return true; }
      last_error_ = "Failed to serialize LP solution from GetResult";
      return false;
    }
    if (response.has_mip_solution()) {
      const auto& sol = response.mip_solution();
      result_data_out.resize(sol.ByteSizeLong());
      if (sol.SerializeToArray(result_data_out.data(), result_data_out.size())) { return true; }
      last_error_ = "Failed to serialize MIP solution from GetResult";
      return false;
    }
    last_error_ = "GetResult succeeded but no solution in response";
    return false;
  }

  // If RESOURCE_EXHAUSTED, try streaming
  if (status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED) {
    std::cerr << "[grpc_client] GetResult rejected (RESOURCE_EXHAUSTED), "
              << "falling back to streaming\n";
    return stream_result(job_id, result_data_out);
  }

  if (!status.ok()) {
    last_error_ = "GetResult failed: " + status.error_message();
  } else if (response.status() != cuopt::remote::SUCCESS) {
    last_error_ = "GetResult indicates failure: " + response.error_message();
  }
  return false;
}

bool grpc_client_t::stream_result(const std::string& job_id, std::vector<uint8_t>& result_data_out)
{
  result_data_out.clear();

  grpc::ClientContext context;
  cuopt::remote::GetResultRequest request;
  request.set_job_id(job_id);

  auto reader = impl_->stub->StreamResult(&context, request);

  cuopt::remote::ResultChunk chunk;
  while (reader->Read(&chunk)) {
    // Check for error in chunk
    if (!chunk.error_message().empty()) {
      last_error_ = "StreamResult error: " + chunk.error_message();
      reader->Finish();
      return false;
    }

    // Done flag indicates end of stream
    if (chunk.done()) { break; }

    // Append chunk data to result
    // Note: chunks should arrive in order by offset, but we trust the server
    const std::string& data = chunk.data();
    result_data_out.insert(result_data_out.end(), data.begin(), data.end());
  }

  grpc::Status st = reader->Finish();
  if (!st.ok()) {
    last_error_ = "StreamResult failed: " + st.error_message();
    return false;
  }

  if (result_data_out.empty()) {
    last_error_ = "StreamResult returned empty result";
    return false;
  }

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

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  auto submit_request = build_lp_submit_request(problem, settings);
  const auto& lp_req  = submit_request.lp_request();
  std::vector<uint8_t> serialized_data(lp_req.ByteSizeLong());
  if (!lp_req.SerializeToArray(serialized_data.data(), serialized_data.size())) {
    result.error_message = "Failed to serialize LP request";
    return result;
  }

  if (!submit_or_upload(serialized_data, streaming_problem_type_t::LP, result.job_id)) {
    result.error_message = last_error_;
    return result;
  }

  result.success = true;
  return result;
}

template <typename i_t, typename f_t>
submit_result_t grpc_client_t::submit_mip(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                          const mip_solver_settings_t<i_t, f_t>& settings,
                                          bool enable_incumbents)
{
  submit_result_t result;

  if (!is_connected()) {
    result.error_message = "Not connected to server";
    return result;
  }

  auto submit_request = build_mip_submit_request(problem, settings, enable_incumbents);
  const auto& mip_req = submit_request.mip_request();
  std::vector<uint8_t> serialized_data(mip_req.ByteSizeLong());
  if (!mip_req.SerializeToArray(serialized_data.data(), serialized_data.size())) {
    result.error_message = "Failed to serialize MIP request";
    return result;
  }

  if (!submit_or_upload(serialized_data, streaming_problem_type_t::MIP, result.job_id)) {
    result.error_message = last_error_;
    return result;
  }

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
  if (!get_result_or_stream(job_id, result_data)) {
    result.error_message = last_error_;
    return result;
  }

  cuopt::remote::LPSolution pb_solution;
  if (!pb_solution.ParseFromArray(result_data.data(), result_data.size())) {
    result.error_message = "Failed to parse LP solution from result data";
    return result;
  }

  result.solution =
    std::make_unique<cpu_lp_solution_t<i_t, f_t>>(map_proto_to_lp_solution<i_t, f_t>(pb_solution));
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
  if (!get_result_or_stream(job_id, result_data)) {
    result.error_message = last_error_;
    return result;
  }

  cuopt::remote::MIPSolution pb_solution;
  if (!pb_solution.ParseFromArray(result_data.data(), result_data.size())) {
    result.error_message = "Failed to parse MIP solution from result data";
    return result;
  }

  result.solution = std::make_unique<cpu_mip_solution_t<i_t, f_t>>(
    map_proto_to_mip_solution<i_t, f_t>(pb_solution));
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

  // 1. Build and serialize the request
  auto submit_request = build_lp_submit_request(problem, settings);

  // Serialize the LP request portion for potential streaming upload
  const auto& lp_req = submit_request.lp_request();
  std::vector<uint8_t> serialized_data(lp_req.ByteSizeLong());
  if (!lp_req.SerializeToArray(serialized_data.data(), serialized_data.size())) {
    result.error_message = "Failed to serialize LP request";
    return result;
  }

  // 2. Submit job (uses streaming if needed)
  std::string job_id;
  if (!submit_or_upload(serialized_data, streaming_problem_type_t::LP, job_id)) {
    result.error_message = last_error_;
    return result;
  }

  // 3. Start log streaming (if configured)
  start_log_streaming(job_id);

  // 4. Wait for completion (using wait RPC or polling)
  bool completed = false;
  std::string completion_error;

  if (config_.use_wait) {
    // Use blocking WaitForCompletion RPC
    std::cerr << "[grpc_client] Using WaitForCompletion RPC for job " << job_id << "\n";
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
    std::cerr << "[grpc_client] Using polling (CheckStatus) for job " << job_id << "\n";
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

    if (!completed && completion_error.empty()) {
      completion_error = "Timeout waiting for job completion";
    }
  }

  stop_log_streaming();

  if (!completed) {
    result.error_message = completion_error;
    return result;
  }

  // 5. Get result (uses streaming if needed)
  std::vector<uint8_t> result_data;
  if (!get_result_or_stream(job_id, result_data)) {
    result.error_message = last_error_;
    return result;
  }

  // 6. Parse solution from serialized data
  cuopt::remote::LPSolution pb_solution;
  if (!pb_solution.ParseFromArray(result_data.data(), result_data.size())) {
    result.error_message = "Failed to parse LP solution from result data";
    return result;
  }

  result.solution =
    std::make_unique<cpu_lp_solution_t<i_t, f_t>>(map_proto_to_lp_solution<i_t, f_t>(pb_solution));
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

  // 1. Build and serialize the request
  auto submit_request = build_mip_submit_request(problem, settings, track_incumbents);

  // Serialize the MIP request portion for potential streaming upload
  const auto& mip_req = submit_request.mip_request();
  std::vector<uint8_t> serialized_data(mip_req.ByteSizeLong());
  if (!mip_req.SerializeToArray(serialized_data.data(), serialized_data.size())) {
    result.error_message = "Failed to serialize MIP request";
    return result;
  }

  // 2. Submit job (uses streaming if needed)
  std::string job_id;
  if (!submit_or_upload(serialized_data, streaming_problem_type_t::MIP, job_id)) {
    result.error_message = last_error_;
    return result;
  }

  // 3. Start log streaming (if configured)
  start_log_streaming(job_id);

  // 4. Start incumbent polling thread (if callback configured)
  std::atomic<bool> stop_incumbents{false};
  std::atomic<bool> cancel_requested{false};
  std::unique_ptr<std::thread> incumbent_thread;

  if (config_.incumbent_callback) {
    // In wait mode, incumbent thread calls cancel_job directly if callback returns false
    // In poll mode, it sets cancel_requested for main thread to handle
    incumbent_thread =
      std::make_unique<std::thread>([this, &job_id, &stop_incumbents, &cancel_requested]() {
        int64_t next_index = 0;
        while (!stop_incumbents.load()) {
          std::this_thread::sleep_for(
            std::chrono::milliseconds(config_.incumbent_poll_interval_ms));
          if (stop_incumbents.load()) break;

          auto inc_result = get_incumbents(job_id, next_index, 0);
          if (!inc_result.success) continue;

          for (const auto& inc : inc_result.incumbents) {
            bool should_continue =
              config_.incumbent_callback(inc.index, inc.objective, inc.assignment);
            if (!should_continue) {
              cancel_requested.store(true);
              if (config_.use_wait) {
                // Cancel directly so WaitForCompletion returns
                cancel_job(job_id);
              }
              return;
            }
          }
          next_index = inc_result.next_index;

          if (inc_result.job_complete) break;
        }
      });
  }

  // 5. Wait for completion (using wait RPC or polling)
  bool completed = false;
  std::string completion_error;

  if (config_.use_wait) {
    // Use blocking WaitForCompletion RPC
    std::cerr << "[grpc_client] Using WaitForCompletion RPC for job " << job_id << "\n";
    auto wait_result = wait_for_completion(job_id);
    if (!wait_result.success) {
      stop_incumbents.store(true);
      if (incumbent_thread && incumbent_thread->joinable()) incumbent_thread->join();
      stop_log_streaming();
      result.error_message = wait_result.error_message;
      return result;
    }

    // Check if cancelled by incumbent callback
    if (cancel_requested.load() && wait_result.status == job_status_t::CANCELLED) {
      stop_incumbents.store(true);
      if (incumbent_thread && incumbent_thread->joinable()) incumbent_thread->join();
      stop_log_streaming();
      result.error_message = "Cancelled by incumbent callback";
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
    std::cerr << "[grpc_client] Using polling (CheckStatus) for job " << job_id << "\n";
    int poll_count = 0;
    int max_polls  = (config_.timeout_seconds * 1000) / config_.poll_interval_ms;

    while (!completed && poll_count < max_polls) {
      std::this_thread::sleep_for(std::chrono::milliseconds(config_.poll_interval_ms));

      // Check if incumbent callback requested cancellation
      if (cancel_requested.load()) {
        cancel_job(job_id);
        stop_logs_.store(true);
        stop_incumbents.store(true);
        if (incumbent_thread && incumbent_thread->joinable()) incumbent_thread->join();
        stop_log_streaming();
        result.error_message = "Cancelled by incumbent callback";
        return result;
      }

      grpc::ClientContext status_context;
      auto status_request = build_status_request(job_id);
      cuopt::remote::StatusResponse status_response;
      auto status_status =
        impl_->stub->CheckStatus(&status_context, status_request, &status_response);

      if (!status_status.ok()) {
        stop_incumbents.store(true);
        if (incumbent_thread && incumbent_thread->joinable()) incumbent_thread->join();
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

    if (!completed && completion_error.empty()) {
      completion_error = "Timeout waiting for job completion";
    }
  }

  // Stop background threads
  stop_incumbents.store(true);
  if (incumbent_thread && incumbent_thread->joinable()) incumbent_thread->join();
  stop_log_streaming();

  if (!completed) {
    result.error_message = completion_error;
    return result;
  }

  // 6. Get result (uses streaming if needed)
  std::vector<uint8_t> result_data;
  if (!get_result_or_stream(job_id, result_data)) {
    result.error_message = last_error_;
    return result;
  }

  // 7. Parse solution from serialized data
  cuopt::remote::MIPSolution pb_solution;
  if (!pb_solution.ParseFromArray(result_data.data(), result_data.size())) {
    result.error_message = "Failed to parse MIP solution from result data";
    return result;
  }

  result.solution = std::make_unique<cpu_mip_solution_t<i_t, f_t>>(
    map_proto_to_mip_solution<i_t, f_t>(pb_solution));
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
#endif

}  // namespace cuopt::linear_programming
