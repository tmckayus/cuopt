/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file test_grpc_mip_sample.cpp
 * @brief Test gRPC remote solve server with a MIP problem (mip_sample.mps)
 *
 * This exercises the server's true MIP path (SubmitJob.mip_request -> solve_mip).
 */

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>
#include "cuopt_remote_service.grpc.pb.h"

#include <mps_parser/parser.hpp>

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using namespace cuopt::remote;

class CuOptGrpcMipClient {
 public:
  explicit CuOptGrpcMipClient(std::shared_ptr<Channel> channel)
    : stub_(CuOptRemoteService::NewStub(std::move(channel)))
  {
  }

  std::string SubmitMpsFileAsMip(const std::string& mps_file_path, double time_limit_s)
  {
    std::cout << "[Client] Parsing MPS file: " << mps_file_path << std::endl;
    cuopt::mps_parser::mps_data_model_t<int, double> mps_data;
    try {
      mps_data = cuopt::mps_parser::parse_mps<int, double>(mps_file_path, false);
    } catch (const std::exception& e) {
      std::cerr << "[Client] Failed to parse MPS file: " << e.what() << std::endl;
      return "";
    }

    std::cout << "[Client] MPS parsed successfully: constraints=" << mps_data.get_n_constraints()
              << " variables=" << mps_data.get_n_variables() << std::endl;

    SubmitJobRequest request;
    auto* mip_request = request.mutable_mip_request();

    // Header
    auto* header = mip_request->mutable_header();
    header->set_version(1);
    header->set_problem_type(MIP);
    header->set_index_type(INT32);
    header->set_float_type(DOUBLE);

    // Problem
    auto* problem = mip_request->mutable_problem();
    problem->set_problem_name(mps_data.get_problem_name());
    problem->set_maximize(mps_data.get_sense());
    problem->set_objective_name(mps_data.get_objective_name());
    problem->set_objective_scaling_factor(mps_data.get_objective_scaling_factor());
    problem->set_objective_offset(mps_data.get_objective_offset());

    // Objective coefficients
    const auto& obj_coeffs = mps_data.get_objective_coefficients();
    for (auto v : obj_coeffs) {
      problem->add_c(v);
    }

    // Constraint matrix (CSR)
    const auto& A_vals = mps_data.get_constraint_matrix_values();
    for (auto v : A_vals) {
      problem->add_a(v);
    }
    const auto& A_idx = mps_data.get_constraint_matrix_indices();
    for (auto v : A_idx) {
      problem->add_a_indices(v);
    }
    const auto& A_off = mps_data.get_constraint_matrix_offsets();
    for (auto v : A_off) {
      problem->add_a_offsets(v);
    }

    // Constraint bounds (prefer lower/upper)
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
      const auto& b = mps_data.get_constraint_bounds();
      for (auto v : b) {
        problem->add_b(v);
      }
    }

    // Row / var types
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

    // Bounds
    const auto& var_lb = mps_data.get_variable_lower_bounds();
    for (auto v : var_lb) {
      problem->add_variable_lower_bounds(v);
    }
    const auto& var_ub = mps_data.get_variable_upper_bounds();
    for (auto v : var_ub) {
      problem->add_variable_upper_bounds(v);
    }

    // MIP settings (keep minimal; rely on server/library defaults)
    auto* settings = mip_request->mutable_settings();
    settings->set_time_limit(time_limit_s);
    settings->set_log_to_console(false);
    settings->set_presolve(true);
    settings->set_num_gpus(1);

    std::cout << "[Client] Prepared SolveMIPRequest: bytes=" << mip_request->ByteSizeLong()
              << " time_limit=" << settings->time_limit() << std::endl;

    SubmitJobResponse response;
    ClientContext context;
    Status status = stub_->SubmitJob(&context, request, &response);
    if (!status.ok()) {
      std::cerr << "[Client] SubmitJob RPC failed: " << status.error_message() << std::endl;
      return "";
    }

    std::cout << "[Client] Job submitted. Job ID: " << response.job_id() << std::endl;
    return response.job_id();
  }

  bool CheckCompleted(const std::string& job_id)
  {
    StatusRequest request;
    request.set_job_id(job_id);
    StatusResponse response;
    ClientContext context;
    Status status = stub_->CheckStatus(&context, request, &response);
    if (!status.ok()) {
      std::cerr << "[Client] CheckStatus RPC failed: " << status.error_message() << std::endl;
      return false;
    }
    return response.job_status() == COMPLETED;
  }

  bool GetResult(const std::string& job_id)
  {
    GetResultRequest request;
    request.set_job_id(job_id);
    ResultResponse response;
    ClientContext context;
    Status status = stub_->GetResult(&context, request, &response);
    if (!status.ok()) {
      std::cerr << "[Client] GetResult RPC failed: " << status.error_message() << std::endl;
      return false;
    }

    if (!response.error_message().empty()) {
      std::cout << "[Client] Response error: " << response.error_message() << std::endl;
    }

    if (!response.has_mip_solution()) {
      std::cerr << "[Client] Expected mip_solution but did not receive one." << std::endl;
      return false;
    }

    const auto& sol = response.mip_solution();
    std::cout << "\n[Client] ============ REMOTE MIP RESULT ============\n";
    std::cout << "[Client] Termination Status: " << sol.termination_status() << "\n";
    std::cout << "[Client] Objective: " << sol.objective() << "\n";
    std::cout << "[Client] Nodes: " << sol.nodes() << "\n";
    std::cout << "[Client] Total solve time: " << sol.total_solve_time() << " seconds\n";
    if (!sol.error_message().empty()) {
      std::cout << "[Client] Solver error message: " << sol.error_message() << "\n";
    }
    std::cout << "[Client] Solution size: " << sol.solution_size() << "\n";
    std::cout << "[Client] ===========================================\n" << std::endl;

    return true;
  }

 private:
  std::unique_ptr<CuOptRemoteService::Stub> stub_;
};

int main(int argc, char** argv)
{
  std::string server_address = "localhost:9091";
  std::string mps_file =
    "/home/tmckay/repos/nvidia-cuopt/docs/cuopt/source/cuopt-c/lp-qp-milp/examples/mip_sample.mps";
  int max_message_mb  = 256;
  bool submit_only    = false;
  double time_limit_s = 30.0;

  if (argc > 1) { mps_file = argv[1]; }
  if (argc > 2) { server_address = argv[2]; }
  if (argc > 3) { max_message_mb = std::stoi(argv[3]); }
  if (argc > 4) {
    std::string arg = argv[4];
    if (arg == "--submit-only") { submit_only = true; }
  }

  grpc::ChannelArguments args;
  int max_bytes = (max_message_mb <= 0) ? -1 : (max_message_mb * 1024 * 1024);
  args.SetMaxReceiveMessageSize(max_bytes);
  args.SetMaxSendMessageSize(max_bytes);

  std::cout << "[Client] Connecting to " << server_address << std::endl;
  CuOptGrpcMipClient client(
    grpc::CreateCustomChannel(server_address, grpc::InsecureChannelCredentials(), args));

  std::string job_id = client.SubmitMpsFileAsMip(mps_file, time_limit_s);
  if (job_id.empty()) { return 1; }
  if (submit_only) {
    std::cout << "[Client] Submit-only mode; exiting after SubmitJob." << std::endl;
    return 0;
  }

  std::cout << "[Client] Waiting for result..." << std::endl;
  bool completed = false;
  for (int i = 0; i < 120; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (client.CheckCompleted(job_id)) {
      completed = true;
      break;
    }
  }
  if (!completed) {
    std::cerr << "[Client] Job did not complete in time" << std::endl;
    return 1;
  }

  if (!client.GetResult(job_id)) { return 1; }
  std::cout << "[Client] MIP test completed successfully" << std::endl;
  return 0;
}
