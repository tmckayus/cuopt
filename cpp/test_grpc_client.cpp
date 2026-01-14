/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file test_grpc_client.cpp
 * @brief Simple test client for gRPC remote solve server
 */

#ifdef CUOPT_ENABLE_GRPC

#include <grpcpp/grpcpp.h>
#include "cuopt_remote.pb.h"
#include "cuopt_remote_service.grpc.pb.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using namespace cuopt::remote;

class CuOptGrpcTestClient {
 public:
  CuOptGrpcTestClient(std::shared_ptr<Channel> channel)
    : stub_(CuOptRemoteService::NewStub(channel))
  {
  }

  std::string SubmitJob()
  {
    // Create a simple LP request
    SubmitJobRequest request;
    auto* lp_request = request.mutable_lp_request();

    // Set header
    auto* header = lp_request->mutable_header();
    header->set_version(1);
    header->set_problem_type(LP);
    header->set_index_type(INT32);
    header->set_float_type(DOUBLE);

    // Create a minimal problem (LP: minimize c^T x  s.t. Ax = b, x >= 0)
    // Example: minimize -x1 - 2*x2
    //          s.t. x1 + x2 <= 4
    //               x1, x2 >= 0
    auto* problem = lp_request->mutable_problem();
    problem->set_problem_name("test_lp");
    problem->set_maximize(false);

    // Objective: -x1 - 2*x2
    problem->add_c(-1.0);
    problem->add_c(-2.0);

    // Constraint matrix A (1 row, 2 columns): [1.0, 1.0]
    problem->add_a(1.0);
    problem->add_a(1.0);
    problem->add_a_indices(0);  // column 0
    problem->add_a_indices(1);  // column 1
    problem->add_a_offsets(0);  // row 0 starts at index 0
    problem->add_a_offsets(2);  // row 0 ends at index 2

    // RHS: b = [4.0]
    problem->add_b(4.0);

    // Row types: 'L' for <=
    problem->set_row_types("L");

    // Variable bounds: x1, x2 >= 0
    problem->add_variable_lower_bounds(0.0);
    problem->add_variable_lower_bounds(0.0);
    problem->add_variable_upper_bounds(1e30);  // effectively unbounded
    problem->add_variable_upper_bounds(1e30);

    // Variable types: continuous
    problem->set_variable_types("CC");

    // Default settings
    auto* settings = lp_request->mutable_settings();
    settings->set_time_limit(60.0);
    settings->set_log_to_console(false);

    // Make RPC call
    SubmitJobResponse response;
    ClientContext context;

    Status status = stub_->SubmitJob(&context, request, &response);

    if (status.ok()) {
      std::cout << "[Client] Job submitted successfully" << std::endl;
      std::cout << "[Client] Job ID: " << response.job_id() << std::endl;
      std::cout << "[Client] Message: " << response.message() << std::endl;
      return response.job_id();
    } else {
      std::cerr << "[Client] SubmitJob failed: " << status.error_message() << std::endl;
      return "";
    }
  }

  bool CheckStatus(const std::string& job_id)
  {
    StatusRequest request;
    request.set_job_id(job_id);

    StatusResponse response;
    ClientContext context;

    Status status = stub_->CheckStatus(&context, request, &response);

    if (status.ok()) {
      std::cout << "[Client] Job status: ";
      switch (response.job_status()) {
        case QUEUED: std::cout << "QUEUED"; break;
        case PROCESSING: std::cout << "PROCESSING"; break;
        case COMPLETED: std::cout << "COMPLETED"; break;
        case FAILED: std::cout << "FAILED"; break;
        case NOT_FOUND: std::cout << "NOT_FOUND"; break;
        case CANCELLED: std::cout << "CANCELLED"; break;
      }
      std::cout << " - " << response.message() << std::endl;
      return response.job_status() == COMPLETED;
    } else {
      std::cerr << "[Client] CheckStatus failed: " << status.error_message() << std::endl;
      return false;
    }
  }

  bool GetResult(const std::string& job_id)
  {
    GetResultRequest request;
    request.set_job_id(job_id);

    ResultResponse response;
    ClientContext context;

    Status status = stub_->GetResult(&context, request, &response);

    if (status.ok()) {
      std::cout << "[Client] Result retrieved successfully" << std::endl;
      if (response.has_lp_solution()) {
        auto& lp_sol = response.lp_solution();
        std::cout << "[Client] LP Solution:" << std::endl;
        std::cout << "[Client]   Status: " << lp_sol.termination_status() << std::endl;
        std::cout << "[Client]   Objective: " << lp_sol.primal_objective() << std::endl;
        std::cout << "[Client]   Primal solution size: " << lp_sol.primal_solution_size()
                  << std::endl;
      } else if (response.has_mip_solution()) {
        std::cout << "[Client] MIP solution received" << std::endl;
      }
      return true;
    } else {
      std::cerr << "[Client] GetResult failed: " << status.error_code() << " - "
                << status.error_message() << std::endl;
      return false;
    }
  }

 private:
  std::unique_ptr<CuOptRemoteService::Stub> stub_;
};

int main(int argc, char** argv)
{
  std::string server_address = "localhost:8765";

  if (argc > 1) { server_address = argv[1]; }

  std::cout << "cuOpt gRPC Test Client\n"
            << "======================\n"
            << "Connecting to " << server_address << "\n"
            << std::endl;

  // Create client
  CuOptGrpcTestClient client(
    grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials()));

  // Submit job
  std::string job_id = client.SubmitJob();
  if (job_id.empty()) {
    std::cerr << "Failed to submit job" << std::endl;
    return 1;
  }

  // Poll for completion
  std::cout << "\n[Client] Polling for job completion..." << std::endl;
  bool completed = false;
  for (int i = 0; i < 10; i++) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    completed = client.CheckStatus(job_id);
    if (completed) { break; }
  }

  if (!completed) {
    std::cerr << "\n[Client] Job did not complete in time" << std::endl;
    return 1;
  }

  // Get result
  std::cout << "\n[Client] Retrieving result..." << std::endl;
  if (!client.GetResult(job_id)) {
    std::cerr << "[Client] Failed to get result" << std::endl;
    return 1;
  }

  std::cout << "\n[Client] Test completed successfully!" << std::endl;
  return 0;
}

#else  // !CUOPT_ENABLE_GRPC

#include <iostream>

int main()
{
  std::cerr << "Error: test_grpc_client requires gRPC support.\n"
            << "Rebuild with gRPC enabled (CUOPT_ENABLE_GRPC=ON)" << std::endl;
  return 1;
}

#endif  // CUOPT_ENABLE_GRPC
