/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "remote_solve_grpc.hpp"

#include <cuopt_remote_service.grpc.pb.h>

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>

namespace cuopt::linear_programming::grpc_remote {
namespace {

constexpr int64_t kMiB = 1024LL * 1024;

int64_t get_submit_max_bytes();

void append_stream_closed_hint(std::string& message, const grpc::Status& status)
{
  if (status.ok()) { return; }
  if (message.find("max message") != std::string::npos) { return; }
  if (status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED ||
      status.error_code() == grpc::StatusCode::CANCELLED) {
    message +=
      " (stream closed; check server --max-message-mb and client "
      "CUOPT_GRPC_MAX_MESSAGE_MB)";
  }
}

std::unique_ptr<cuopt::remote::CuOptRemoteService::Stub> make_stub(const std::string& address)
{
  grpc::ChannelArguments args;
  // Align channel max sizes with client max message configuration.
  const int64_t max_bytes = get_submit_max_bytes();
  const int channel_limit =
    (max_bytes <= 0)
      ? -1
      : static_cast<int>(std::min<int64_t>(max_bytes, std::numeric_limits<int>::max()));
  args.SetMaxReceiveMessageSize(channel_limit);
  args.SetMaxSendMessageSize(channel_limit);

  auto channel = grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args);
  return cuopt::remote::CuOptRemoteService::NewStub(channel);
}

int64_t get_submit_max_bytes()
{
  constexpr int64_t kDefaultMax = 256LL * kMiB;

  const char* val = std::getenv("CUOPT_GRPC_MAX_MESSAGE_MB");
  if (!val || val[0] == '\0') { return kDefaultMax; }
  try {
    int64_t mb = std::stoll(val);
    if (mb <= 0) { return std::numeric_limits<int64_t>::max(); }
    return mb * kMiB;
  } catch (...) {
    return kDefaultMax;
  }
}

bool submit_job(const std::string& address,
                ProblemType problem_type,
                const uint8_t* data,
                size_t size,
                std::string& job_id,
                std::string& error_message,
                grpc::StatusCode& status_code)
{
  job_id.clear();
  error_message.clear();
  status_code = grpc::StatusCode::OK;

  auto stub = make_stub(address);
  grpc::ClientContext ctx;
  cuopt::remote::SubmitJobRequest req;

  if (problem_type == ProblemType::LP) {
    auto* lp_req = req.mutable_lp_request();
    if (!lp_req->ParseFromArray(data, static_cast<int>(size))) {
      error_message = "SubmitJob: failed to parse LP request";
      status_code   = grpc::StatusCode::INVALID_ARGUMENT;
      return false;
    }
  } else {
    auto* mip_req = req.mutable_mip_request();
    if (!mip_req->ParseFromArray(data, static_cast<int>(size))) {
      error_message = "SubmitJob: failed to parse MIP request";
      status_code   = grpc::StatusCode::INVALID_ARGUMENT;
      return false;
    }
  }

  cuopt::remote::SubmitJobResponse resp;
  grpc::Status st = stub->SubmitJob(&ctx, req, &resp);
  if (!st.ok()) {
    status_code   = st.error_code();
    error_message = "SubmitJob: " + st.error_message();
    return false;
  }

  job_id = resp.job_id();
  if (job_id.empty()) {
    error_message = "SubmitJob: no job_id returned";
    status_code   = grpc::StatusCode::INTERNAL;
    return false;
  }

  return true;
}

bool read_upload_start_ack(grpc::ClientReaderWriter<cuopt::remote::UploadJobRequest,
                                                    cuopt::remote::UploadJobResponse>* stream,
                           std::string& upload_id,
                           int64_t& committed,
                           int64_t* max_message_bytes_out,
                           std::string& error_message)
{
  cuopt::remote::UploadJobResponse resp;
  if (!stream->Read(&resp)) {
    error_message = "UploadAndSubmit: failed to read response after start";
    return false;
  }
  if (resp.has_error()) {
    error_message = "UploadAndSubmit: " + resp.error().message();
    if (max_message_bytes_out) { *max_message_bytes_out = resp.error().max_message_bytes(); }
    return false;
  }
  if (!resp.has_ack()) {
    error_message = "UploadAndSubmit: expected ack after start";
    return false;
  }
  upload_id = resp.ack().upload_id();
  committed = resp.ack().committed_size();
  if (max_message_bytes_out) { *max_message_bytes_out = resp.ack().max_message_bytes(); }
  return true;
}

bool write_chunk_and_read_ack(grpc::ClientReaderWriter<cuopt::remote::UploadJobRequest,
                                                       cuopt::remote::UploadJobResponse>* stream,
                              const std::string& upload_id,
                              int64_t offset,
                              const uint8_t* data,
                              size_t n,
                              int64_t& committed_out,
                              int64_t* max_message_bytes_out,
                              std::string& error_message)
{
  cuopt::remote::UploadJobRequest req;
  auto* chunk = req.mutable_chunk();
  chunk->set_upload_id(upload_id);
  chunk->set_offset(offset);
  chunk->set_data(reinterpret_cast<const char*>(data), n);

  if (!stream->Write(req)) {
    error_message = "UploadAndSubmit: failed to write chunk";
    return false;
  }

  cuopt::remote::UploadJobResponse resp;
  if (!stream->Read(&resp)) {
    error_message = "UploadAndSubmit: failed to read response after chunk";
    return false;
  }
  if (resp.has_error()) {
    error_message = "UploadAndSubmit: " + resp.error().message();
    committed_out = resp.error().committed_size();
    if (max_message_bytes_out) { *max_message_bytes_out = resp.error().max_message_bytes(); }
    return false;
  }
  if (!resp.has_ack()) {
    error_message = "UploadAndSubmit: expected ack after chunk";
    return false;
  }

  committed_out = resp.ack().committed_size();
  if (max_message_bytes_out) { *max_message_bytes_out = resp.ack().max_message_bytes(); }
  return true;
}

}  // namespace

bool submit_or_upload(const std::string& address,
                      ProblemType problem_type,
                      const uint8_t* data,
                      size_t size,
                      std::string& job_id,
                      std::string& error_message,
                      bool* used_upload_out,
                      int64_t* max_bytes_out)
{
  try {
    const int64_t max_bytes = get_submit_max_bytes();
    if (max_bytes_out) { *max_bytes_out = max_bytes; }
    if (max_bytes >= 0 && static_cast<int64_t>(size) <= max_bytes) {
      grpc::StatusCode status_code = grpc::StatusCode::OK;
      if (submit_job(address, problem_type, data, size, job_id, error_message, status_code)) {
        if (used_upload_out) { *used_upload_out = false; }
        return true;
      }
      if (status_code != grpc::StatusCode::RESOURCE_EXHAUSTED) { return false; }
    }

    if (used_upload_out) { *used_upload_out = true; }
    return upload_and_submit(address, problem_type, data, size, job_id, error_message);
  } catch (const std::exception& ex) {
    error_message = std::string("SubmitOrUpload: exception: ") + ex.what();
    std::cerr << "[remote_solve] SubmitOrUpload exception: " << ex.what() << "\n";
    std::cerr.flush();
    return false;
  } catch (...) {
    error_message = "SubmitOrUpload: unknown exception";
    std::cerr << "[remote_solve] SubmitOrUpload unknown exception\n";
    std::cerr.flush();
    return false;
  }
}

bool upload_and_submit(const std::string& address,
                       ProblemType problem_type,
                       const uint8_t* data,
                       size_t size,
                       std::string& job_id,
                       std::string& error_message)
{
  try {
    constexpr size_t kMinChunkSize = 4 * 1024;
    size_t default_chunk_size      = 1 << 20;  // 1 MiB
    if (const char* chunk_kb = std::getenv("CUOPT_GRPC_UPLOAD_CHUNK_KB")) {
      try {
        auto kb = std::stoll(chunk_kb);
        if (kb > 0) { default_chunk_size = static_cast<size_t>(kb) * 1024; }
      } catch (...) {
      }
    }

    auto compute_chunk_size = [&](int64_t max_bytes, size_t fallback) -> size_t {
      size_t chunk_size = fallback;
      if (max_bytes > 0 && max_bytes < static_cast<int64_t>(chunk_size)) {
        chunk_size = static_cast<size_t>(max_bytes / 2);
        if (chunk_size < kMinChunkSize) { chunk_size = kMinChunkSize; }
      }
      return chunk_size;
    };

    int64_t last_max_message_bytes = 0;
    auto parse_max_mb_from_error   = [](const std::string& msg) -> int64_t {
      const char* key = "max_message_mb=";
      auto pos        = msg.find(key);
      if (pos == std::string::npos) { return 0; }
      pos += std::strlen(key);
      const char* start = msg.c_str() + pos;
      char* end         = nullptr;
      long long mb      = std::strtoll(start, &end, 10);
      if (end == start || mb <= 0) { return 0; }
      return mb * kMiB;
    };

    auto do_upload = [&](size_t initial_chunk_size, std::string& err_out) -> bool {
      job_id.clear();
      err_out.clear();

      auto stub = make_stub(address);
      grpc::ClientContext ctx;
      int64_t timeout_ms = 30000;
      if (const char* timeout_env = std::getenv("CUOPT_GRPC_UPLOAD_TIMEOUT_MS")) {
        try {
          auto parsed = std::stoll(timeout_env);
          if (parsed > 0) { timeout_ms = parsed; }
        } catch (...) {
        }
      }
      ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms));
      auto stream = stub->UploadAndSubmit(&ctx);

      cuopt::remote::UploadJobRequest start_req;
      auto* start = start_req.mutable_start();
      start->set_problem_type(problem_type == ProblemType::LP ? cuopt::remote::LP
                                                              : cuopt::remote::MIP);
      start->set_resume(false);
      start->set_total_size(static_cast<int64_t>(size));

      if (!stream->Write(start_req)) {
        err_out = "UploadAndSubmit: failed to write start";
        return false;
      }

      std::string upload_id;
      int64_t committed         = 0;
      int64_t max_message_bytes = 0;
      if (!read_upload_start_ack(stream.get(), upload_id, committed, &max_message_bytes, err_out)) {
        if (max_message_bytes != 0) { last_max_message_bytes = max_message_bytes; }
        grpc::Status st = stream->Finish();
        if (!st.ok()) {
          err_out +=
            " (grpc_status=" + std::to_string(st.error_code()) + " " + st.error_message() + ")";
          append_stream_closed_hint(err_out, st);
        }
        if (last_max_message_bytes == 0) {
          last_max_message_bytes = parse_max_mb_from_error(err_out);
        }
        return false;
      }

      int64_t active_max_bytes = get_submit_max_bytes();
      if (max_message_bytes != 0) {
        active_max_bytes       = max_message_bytes;
        last_max_message_bytes = max_message_bytes;
      }
      size_t chunk_size = compute_chunk_size(active_max_bytes, initial_chunk_size);

      while (static_cast<size_t>(committed) < size) {
        size_t off = static_cast<size_t>(committed);
        size_t n   = std::min(chunk_size, size - off);

        int64_t committed2    = committed;
        int64_t ack_max_bytes = 0;
        if (!write_chunk_and_read_ack(stream.get(),
                                      upload_id,
                                      committed,
                                      data + off,
                                      n,
                                      committed2,
                                      &ack_max_bytes,
                                      err_out)) {
          if (ack_max_bytes != 0) { last_max_message_bytes = ack_max_bytes; }
          grpc::Status st = stream->Finish();
          if (!st.ok()) {
            err_out +=
              " (grpc_status=" + std::to_string(st.error_code()) + " " + st.error_message() + ")";
            append_stream_closed_hint(err_out, st);
          }
          if (last_max_message_bytes == 0) {
            last_max_message_bytes = parse_max_mb_from_error(err_out);
          }
          return false;
        }
        committed = committed2;
        if (ack_max_bytes != 0) {
          active_max_bytes       = ack_max_bytes;
          last_max_message_bytes = ack_max_bytes;
          chunk_size             = compute_chunk_size(active_max_bytes, chunk_size);
        }
      }

      cuopt::remote::UploadJobRequest finish_req;
      finish_req.mutable_finish()->set_upload_id(upload_id);
      stream->Write(finish_req);
      stream->WritesDone();

      cuopt::remote::UploadJobResponse resp;
      while (stream->Read(&resp)) {
        if (resp.has_submit()) {
          job_id = resp.submit().job_id();
          break;
        }
        if (resp.has_error()) {
          err_out = "UploadAndSubmit: " + resp.error().message();
          break;
        }
      }

      grpc::Status st = stream->Finish();
      if (!st.ok()) {
        if (err_out.empty()) {
          err_out = "UploadAndSubmit: grpc_status=" + std::to_string(st.error_code()) + " " +
                    st.error_message();
        }
        append_stream_closed_hint(err_out, st);
        if (last_max_message_bytes == 0) {
          last_max_message_bytes = parse_max_mb_from_error(err_out);
        }
        return false;
      }
      if (job_id.empty()) {
        if (err_out.empty()) { err_out = "UploadAndSubmit: no job_id returned"; }
        return false;
      }

      return true;
    };

    size_t first_chunk = default_chunk_size;
    if (do_upload(first_chunk, error_message)) { return true; }

    std::cout << "[remote_solve] UploadAndSubmit failed: " << error_message << "\n";
    if (last_max_message_bytes > 0) {
      std::cout << "[remote_solve] Server max message MiB: " << (last_max_message_bytes / kMiB)
                << "\n";
    }
    std::cout.flush();

    size_t retry_chunk = first_chunk / 2;
    if (last_max_message_bytes > 0) {
      retry_chunk = compute_chunk_size(last_max_message_bytes, first_chunk);
      if (retry_chunk >= first_chunk) { retry_chunk = first_chunk / 2; }
    }
    if (retry_chunk < kMinChunkSize) { return false; }
    std::cout << "[remote_solve] UploadAndSubmit retry with chunk_size=" << retry_chunk << "\n";
    std::cout.flush();
    return do_upload(retry_chunk, error_message);
  } catch (const std::exception& ex) {
    error_message = std::string("UploadAndSubmit: exception: ") + ex.what();
    std::cerr << "[remote_solve] UploadAndSubmit exception: " << ex.what() << "\n";
    std::cerr.flush();
    return false;
  } catch (...) {
    error_message = "UploadAndSubmit: unknown exception";
    std::cerr << "[remote_solve] UploadAndSubmit unknown exception\n";
    std::cerr.flush();
    return false;
  }
}

bool check_status(const std::string& address,
                  const std::string& job_id,
                  std::string& status_out,
                  std::string& error_message,
                  int64_t* result_size_bytes_out,
                  int64_t* max_message_bytes_out)
{
  status_out.clear();
  error_message.clear();
  if (result_size_bytes_out) { *result_size_bytes_out = 0; }
  if (max_message_bytes_out) { *max_message_bytes_out = 0; }

  auto stub = make_stub(address);
  grpc::ClientContext ctx;
  cuopt::remote::StatusRequest req;
  req.set_job_id(job_id);
  cuopt::remote::StatusResponse resp;

  grpc::Status st = stub->CheckStatus(&ctx, req, &resp);
  if (!st.ok()) {
    error_message = st.error_message();
    return false;
  }

  if (result_size_bytes_out) { *result_size_bytes_out = resp.result_size_bytes(); }
  if (max_message_bytes_out) { *max_message_bytes_out = resp.max_message_bytes(); }

  switch (resp.job_status()) {
    case cuopt::remote::QUEUED: status_out = "QUEUED"; break;
    case cuopt::remote::PROCESSING: status_out = "PROCESSING"; break;
    case cuopt::remote::COMPLETED: status_out = "COMPLETED"; break;
    case cuopt::remote::FAILED: status_out = "FAILED"; break;
    case cuopt::remote::CANCELLED: status_out = "CANCELLED"; break;
    case cuopt::remote::NOT_FOUND: status_out = "NOT_FOUND"; break;
    default: status_out = "UNKNOWN"; break;
  }

  return true;
}

bool stream_result(const std::string& address,
                   const std::string& job_id,
                   std::vector<uint8_t>& out,
                   std::string& error_message)
{
  out.clear();
  error_message.clear();

  auto stub = make_stub(address);
  grpc::ClientContext ctx;
  cuopt::remote::GetResultRequest req;
  req.set_job_id(job_id);

  std::unique_ptr<grpc::ClientReader<cuopt::remote::ResultChunk>> reader =
    stub->StreamResult(&ctx, req);

  cuopt::remote::ResultChunk chunk;
  while (reader->Read(&chunk)) {
    if (!chunk.error_message().empty()) {
      error_message = chunk.error_message();
      break;
    }
    if (chunk.done()) { break; }
    const std::string& data = chunk.data();
    out.insert(out.end(), data.begin(), data.end());
  }

  grpc::Status st = reader->Finish();
  if (!st.ok()) {
    if (error_message.empty()) { error_message = st.error_message(); }
    return false;
  }
  if (!error_message.empty()) { return false; }
  return true;
}

bool get_result(const std::string& address,
                const std::string& job_id,
                std::vector<uint8_t>& out,
                std::string& error_message)
{
  out.clear();
  error_message.clear();

  auto stub = make_stub(address);
  grpc::ClientContext ctx;
  cuopt::remote::GetResultRequest req;
  req.set_job_id(job_id);

  cuopt::remote::ResultResponse resp;
  grpc::Status st = stub->GetResult(&ctx, req, &resp);
  if (!st.ok()) {
    error_message = st.error_message();
    return false;
  }
  if (resp.status() != cuopt::remote::SUCCESS) {
    error_message = resp.error_message().empty() ? "GetResult failed" : resp.error_message();
    return false;
  }

  if (resp.has_lp_solution()) {
    const auto& lp = resp.lp_solution();
    out.resize(lp.ByteSizeLong());
    if (!lp.SerializeToArray(out.data(), out.size())) {
      error_message = "GetResult: failed to serialize LP solution";
      return false;
    }
    return true;
  }
  if (resp.has_mip_solution()) {
    const auto& mip = resp.mip_solution();
    out.resize(mip.ByteSizeLong());
    if (!mip.SerializeToArray(out.data(), out.size())) {
      error_message = "GetResult: failed to serialize MIP solution";
      return false;
    }
    return true;
  }

  error_message = "GetResult: missing solution payload";
  return false;
}

void delete_result(const std::string& address, const std::string& job_id)
{
  auto stub = make_stub(address);
  grpc::ClientContext ctx;
  cuopt::remote::DeleteRequest req;
  req.set_job_id(job_id);
  cuopt::remote::DeleteResponse resp;
  (void)stub->DeleteResult(&ctx, req, &resp);
}

bool cancel_job(const std::string& address,
                const std::string& job_id,
                bool& success_out,
                std::string& status_out,
                std::string& message_out,
                std::string& error_message)
{
  success_out = false;
  status_out.clear();
  message_out.clear();
  error_message.clear();

  auto stub = make_stub(address);
  grpc::ClientContext ctx;
  cuopt::remote::CancelRequest req;
  req.set_job_id(job_id);
  cuopt::remote::CancelResponse resp;

  grpc::Status st = stub->CancelJob(&ctx, req, &resp);
  if (!st.ok()) {
    error_message = st.error_message();
    return false;
  }

  success_out = (resp.status() == cuopt::remote::SUCCESS);
  message_out = resp.message();
  switch (resp.job_status()) {
    case cuopt::remote::QUEUED: status_out = "QUEUED"; break;
    case cuopt::remote::PROCESSING: status_out = "PROCESSING"; break;
    case cuopt::remote::COMPLETED: status_out = "COMPLETED"; break;
    case cuopt::remote::FAILED: status_out = "FAILED"; break;
    case cuopt::remote::CANCELLED: status_out = "CANCELLED"; break;
    case cuopt::remote::NOT_FOUND: status_out = "NOT_FOUND"; break;
    default: status_out = "UNKNOWN"; break;
  }

  return true;
}

void stream_logs_to_stdout(const std::string& address,
                           const std::string& job_id,
                           volatile bool* stop_flag,
                           const std::string& print_prefix)
{
  auto stub = make_stub(address);
  grpc::ClientContext ctx;
  cuopt::remote::StreamLogsRequest req;
  req.set_job_id(job_id);
  req.set_from_byte(0);

  std::unique_ptr<grpc::ClientReader<cuopt::remote::LogMessage>> reader =
    stub->StreamLogs(&ctx, req);

  cuopt::remote::LogMessage msg;
  while (reader->Read(&msg)) {
    if (stop_flag != nullptr && *stop_flag) { ctx.TryCancel(); }
    if (!msg.line().empty()) {
      if (!print_prefix.empty()) { std::cout << print_prefix; }
      std::cout << msg.line() << "\n";
      std::cout.flush();
    }
    if (msg.job_complete()) { break; }
  }
  reader->Finish();
}

bool get_incumbents(const std::string& address,
                    const std::string& job_id,
                    int64_t from_index,
                    int32_t max_count,
                    std::vector<Incumbent>& incumbents_out,
                    int64_t& next_index_out,
                    bool& job_complete_out,
                    std::string& error_message)
{
  incumbents_out.clear();
  next_index_out   = from_index;
  job_complete_out = false;
  error_message.clear();

  auto stub = make_stub(address);
  grpc::ClientContext ctx;
  cuopt::remote::IncumbentRequest req;
  req.set_job_id(job_id);
  req.set_from_index(from_index);
  req.set_max_count(max_count);

  cuopt::remote::IncumbentResponse resp;
  grpc::Status st = stub->GetIncumbents(&ctx, req, &resp);
  if (!st.ok()) {
    error_message = st.error_message();
    return false;
  }

  incumbents_out.reserve(resp.incumbents_size());
  for (const auto& inc : resp.incumbents()) {
    Incumbent entry;
    entry.index     = inc.index();
    entry.objective = inc.objective();
    entry.assignment.reserve(inc.assignment_size());
    for (int i = 0; i < inc.assignment_size(); ++i) {
      entry.assignment.push_back(inc.assignment(i));
    }
    incumbents_out.push_back(std::move(entry));
  }

  next_index_out   = resp.next_index();
  job_complete_out = resp.job_complete();
  return true;
}

}  // namespace cuopt::linear_programming::grpc_remote
