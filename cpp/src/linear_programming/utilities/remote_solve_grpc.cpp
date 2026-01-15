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
#include <cstring>
#include <iostream>
#include <thread>

namespace cuopt::linear_programming::grpc_remote {
namespace {

std::unique_ptr<cuopt::remote::CuOptRemoteService::Stub> make_stub(const std::string& address)
{
  grpc::ChannelArguments args;
  // Keep per-message limits reasonably high; total transfer is handled via streaming.
  // Users can still override process-wide gRPC defaults if needed.
  args.SetMaxReceiveMessageSize(64 * 1024 * 1024);
  args.SetMaxSendMessageSize(64 * 1024 * 1024);

  auto channel = grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args);
  return cuopt::remote::CuOptRemoteService::NewStub(channel);
}

bool read_upload_start_ack(grpc::ClientReaderWriter<cuopt::remote::UploadJobRequest,
                                                    cuopt::remote::UploadJobResponse>* stream,
                           std::string& upload_id,
                           int64_t& committed,
                           std::string& error_message)
{
  cuopt::remote::UploadJobResponse resp;
  if (!stream->Read(&resp)) {
    error_message = "UploadAndSubmit: failed to read response after start";
    return false;
  }
  if (resp.has_error()) {
    error_message = "UploadAndSubmit: " + resp.error().message();
    return false;
  }
  if (!resp.has_ack()) {
    error_message = "UploadAndSubmit: expected ack after start";
    return false;
  }
  upload_id = resp.ack().upload_id();
  committed = resp.ack().committed_size();
  return true;
}

bool write_chunk_and_read_ack(grpc::ClientReaderWriter<cuopt::remote::UploadJobRequest,
                                                       cuopt::remote::UploadJobResponse>* stream,
                              const std::string& upload_id,
                              int64_t offset,
                              const uint8_t* data,
                              size_t n,
                              int64_t& committed_out,
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
    return false;
  }
  if (!resp.has_ack()) {
    error_message = "UploadAndSubmit: expected ack after chunk";
    return false;
  }

  committed_out = resp.ack().committed_size();
  return true;
}

}  // namespace

bool upload_and_submit(const std::string& address,
                       ProblemType problem_type,
                       const uint8_t* data,
                       size_t size,
                       std::string& job_id,
                       std::string& error_message)
{
  job_id.clear();
  error_message.clear();

  auto stub = make_stub(address);
  grpc::ClientContext ctx;
  auto stream = stub->UploadAndSubmit(&ctx);

  cuopt::remote::UploadJobRequest start_req;
  auto* start = start_req.mutable_start();
  start->set_problem_type(problem_type == ProblemType::LP ? cuopt::remote::LP : cuopt::remote::MIP);
  start->set_resume(false);
  start->set_total_size(static_cast<int64_t>(size));

  if (!stream->Write(start_req)) {
    error_message = "UploadAndSubmit: failed to write start";
    return false;
  }

  std::string upload_id;
  int64_t committed = 0;
  if (!read_upload_start_ack(stream.get(), upload_id, committed, error_message)) { return false; }

  const size_t chunk_size = 1 << 20;  // 1 MiB
  while (static_cast<size_t>(committed) < size) {
    size_t off = static_cast<size_t>(committed);
    size_t n   = std::min(chunk_size, size - off);

    int64_t committed2 = committed;
    if (!write_chunk_and_read_ack(
          stream.get(), upload_id, committed, data + off, n, committed2, error_message)) {
      return false;
    }
    committed = committed2;
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
      error_message = "UploadAndSubmit: " + resp.error().message();
      break;
    }
  }

  grpc::Status st = stream->Finish();
  if (!st.ok()) {
    if (error_message.empty()) { error_message = st.error_message(); }
    return false;
  }
  if (job_id.empty()) {
    if (error_message.empty()) { error_message = "UploadAndSubmit: no job_id returned"; }
    return false;
  }

  return true;
}

bool check_status(const std::string& address,
                  const std::string& job_id,
                  std::string& status_out,
                  std::string& error_message)
{
  status_out.clear();
  error_message.clear();

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

}  // namespace cuopt::linear_programming::grpc_remote
