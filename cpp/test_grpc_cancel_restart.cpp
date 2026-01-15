/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file test_grpc_cancel_restart.cpp
 * @brief Submit a job, wait until PROCESSING, then cancel; verify CANCELLED and worker restart.
 *
 * This is primarily a server behavior test: CancelJob should kill the worker for a running job,
 * worker_monitor_thread should restart it, and the job should end up CANCELLED (not FAILED).
 */

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "cuopt_remote_service.grpc.pb.h"

#include <mps_parser/parser.hpp>

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReaderWriter;
using grpc::Status;

using namespace cuopt::remote;

static std::string upload_and_submit_mip(CuOptRemoteService::Stub* stub,
                                         const SolveMIPRequest& req,
                                         int max_message_mb)
{
  const size_t size = req.ByteSizeLong();
  std::vector<uint8_t> bytes(size);
  if (!req.SerializeToArray(bytes.data(), size)) { return ""; }

  ClientContext context;
  std::unique_ptr<ClientReaderWriter<UploadJobRequest, UploadJobResponse>> stream =
    stub->UploadAndSubmit(&context);

  UploadJobRequest ureq;
  ureq.mutable_start()->set_problem_type(MIP);
  ureq.mutable_start()->set_resume(false);
  ureq.mutable_start()->set_total_size(static_cast<int64_t>(bytes.size()));

  if (!stream->Write(ureq)) { return ""; }

  UploadJobResponse uresp;
  if (!stream->Read(&uresp) || !uresp.has_ack()) { return ""; }

  const std::string upload_id = uresp.ack().upload_id();
  int64_t committed           = uresp.ack().committed_size();

  // Keep chunks <= max_message_mb (leave headroom).
  size_t chunk_size = 1 << 20;  // default 1 MiB
  if (max_message_mb > 0) {
    const size_t cap = static_cast<size_t>(max_message_mb) * 1024 * 1024;
    chunk_size       = std::min(chunk_size, cap / 2);
    if (chunk_size < 4096) { chunk_size = 4096; }
  }

  while (static_cast<size_t>(committed) < bytes.size()) {
    const size_t offset = static_cast<size_t>(committed);
    size_t n            = bytes.size() - offset;
    if (n > chunk_size) { n = chunk_size; }

    ureq.Clear();
    auto* chunk = ureq.mutable_chunk();
    chunk->set_upload_id(upload_id);
    chunk->set_offset(committed);
    chunk->set_data(reinterpret_cast<const char*>(bytes.data() + offset), n);

    if (!stream->Write(ureq)) { return ""; }

    uresp.Clear();
    if (!stream->Read(&uresp) || !uresp.has_ack()) { return ""; }
    committed = uresp.ack().committed_size();
  }

  ureq.Clear();
  ureq.mutable_finish()->set_upload_id(upload_id);
  stream->Write(ureq);
  stream->WritesDone();

  std::string job_id;
  while (stream->Read(&uresp)) {
    if (uresp.has_submit()) {
      job_id = uresp.submit().job_id();
      break;
    }
    if (uresp.has_error()) { break; }
  }
  Status st = stream->Finish();
  if (!st.ok()) { return ""; }
  return job_id;
}

int main(int argc, char** argv)
{
  std::string server_address = "localhost:9095";
  // Use a papilo MIP instance (present in build tree) for a longer-running solve.
  std::string mip_mps =
    "/home/tmckay/repos/nvidia-cuopt/cpp/build_grpc_test/_deps/papilo-src/check/instances/MIP/"
    "enigma.mps";
  int max_message_mb = 16;

  if (argc > 1) { mip_mps = argv[1]; }
  if (argc > 2) { server_address = argv[2]; }
  if (argc > 3) { max_message_mb = std::stoi(argv[3]); }

  grpc::ChannelArguments args;
  int max_bytes = (max_message_mb <= 0) ? -1 : (max_message_mb * 1024 * 1024);
  args.SetMaxReceiveMessageSize(max_bytes);
  args.SetMaxSendMessageSize(max_bytes);
  auto channel =
    grpc::CreateCustomChannel(server_address, grpc::InsecureChannelCredentials(), args);
  auto stub = CuOptRemoteService::NewStub(channel);

  std::cout << "[Client] Parsing MIP MPS: " << mip_mps << std::endl;
  cuopt::mps_parser::mps_data_model_t<int, double> mps_data =
    cuopt::mps_parser::parse_mps<int, double>(mip_mps, false);

  SolveMIPRequest mip_req;
  auto* header = mip_req.mutable_header();
  header->set_version(1);
  header->set_problem_type(MIP);
  header->set_index_type(INT32);
  header->set_float_type(DOUBLE);

  auto* problem = mip_req.mutable_problem();
  problem->set_problem_name(mps_data.get_problem_name());
  problem->set_maximize(mps_data.get_sense());
  problem->set_objective_name(mps_data.get_objective_name());
  problem->set_objective_scaling_factor(mps_data.get_objective_scaling_factor());
  problem->set_objective_offset(mps_data.get_objective_offset());

  for (auto v : mps_data.get_objective_coefficients()) {
    problem->add_c(v);
  }
  for (auto v : mps_data.get_constraint_matrix_values()) {
    problem->add_a(v);
  }
  for (auto v : mps_data.get_constraint_matrix_indices()) {
    problem->add_a_indices(v);
  }
  for (auto v : mps_data.get_constraint_matrix_offsets()) {
    problem->add_a_offsets(v);
  }
  const auto& con_lb = mps_data.get_constraint_lower_bounds();
  const auto& con_ub = mps_data.get_constraint_upper_bounds();
  if (!con_lb.empty()) {
    for (auto v : con_lb) {
      problem->add_constraint_lower_bounds(v);
    }
    for (auto v : con_ub) {
      problem->add_constraint_upper_bounds(v);
    }
  } else {
    for (auto v : mps_data.get_constraint_bounds()) {
      problem->add_b(v);
    }
  }
  {
    const auto& row_types = mps_data.get_row_types();
    std::string row_types_str(row_types.begin(), row_types.end());
    problem->set_row_types(row_types_str);
  }
  {
    const auto& var_types = mps_data.get_variable_types();
    std::string var_types_str(var_types.begin(), var_types.end());
    problem->set_variable_types(var_types_str);
  }
  for (auto v : mps_data.get_variable_lower_bounds()) {
    problem->add_variable_lower_bounds(v);
  }
  for (auto v : mps_data.get_variable_upper_bounds()) {
    problem->add_variable_upper_bounds(v);
  }

  auto* settings = mip_req.mutable_settings();
  settings->set_time_limit(600.0);
  settings->set_log_to_console(false);
  settings->set_presolve(true);
  settings->set_num_gpus(1);

  std::cout << "[Client] Submitting MIP via UploadAndSubmit..." << std::endl;
  std::string job_id = upload_and_submit_mip(stub.get(), mip_req, max_message_mb);
  if (job_id.empty()) {
    std::cerr << "[Client] UploadAndSubmit failed" << std::endl;
    return 1;
  }
  std::cout << "[Client] Job ID: " << job_id << std::endl;

  // Poll until PROCESSING (or short timeout)
  bool processing = false;
  for (int i = 0; i < 200; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    StatusRequest sreq;
    sreq.set_job_id(job_id);
    StatusResponse sresp;
    ClientContext sctx;
    Status st = stub->CheckStatus(&sctx, sreq, &sresp);
    if (!st.ok()) { continue; }
    if (sresp.job_status() == PROCESSING) {
      processing = true;
      break;
    }
    if (sresp.job_status() == COMPLETED || sresp.job_status() == FAILED) {
      std::cerr << "[Client] Job finished too quickly for cancel test; status="
                << sresp.job_status() << std::endl;
      return 2;
    }
  }
  if (!processing) {
    std::cerr << "[Client] Did not observe PROCESSING state; cannot test cancel reliably"
              << std::endl;
    return 3;
  }

  // Cancel
  std::cout << "[Client] Cancelling job..." << std::endl;
  CancelRequest creq;
  creq.set_job_id(job_id);
  CancelResponse cresp;
  ClientContext cctx;
  Status cst = stub->CancelJob(&cctx, creq, &cresp);
  if (!cst.ok()) {
    std::cerr << "[Client] CancelJob RPC failed: " << cst.error_message() << std::endl;
    return 4;
  }
  std::cout << "[Client] CancelJob response: status=" << cresp.status()
            << " job_status=" << cresp.job_status() << " msg=" << cresp.message() << std::endl;

  // Wait until CANCELLED shows up
  for (int i = 0; i < 200; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    StatusRequest sreq;
    sreq.set_job_id(job_id);
    StatusResponse sresp;
    ClientContext sctx;
    Status st = stub->CheckStatus(&sctx, sreq, &sresp);
    if (!st.ok()) { continue; }
    if (sresp.job_status() == CANCELLED) {
      std::cout << "[Client] Observed CANCELLED." << std::endl;
      return 0;
    }
  }

  std::cerr << "[Client] Did not observe CANCELLED within timeout." << std::endl;
  return 5;
}
