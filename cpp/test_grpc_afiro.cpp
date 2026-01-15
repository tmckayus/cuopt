/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Test gRPC remote solve with afiro.mps
 */

#include <chrono>
#include <fstream>
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

class CuOptGrpcClient {
 public:
  CuOptGrpcClient(std::shared_ptr<Channel> channel) : stub_(CuOptRemoteService::NewStub(channel)) {}

  std::string SubmitMpsFile(const std::string& mps_file_path)
  {
    // Parse MPS file
    std::cout << "[Client] Parsing MPS file: " << mps_file_path << std::endl;

    cuopt::mps_parser::mps_data_model_t<int, double> mps_data;
    try {
      mps_data = cuopt::mps_parser::parse_mps<int, double>(mps_file_path, false);
    } catch (const std::exception& e) {
      std::cerr << "[Client] Failed to parse MPS file: " << e.what() << std::endl;
      return "";
    }

    std::cout << "[Client] MPS parsed successfully" << std::endl;
    std::cout << "[Client] Problem: " << mps_data.get_n_constraints() << " constraints, "
              << mps_data.get_n_variables() << " variables" << std::endl;
    {
      const auto& obj_coeffs_dbg = mps_data.get_objective_coefficients();
      double min_c = 0.0, max_c = 0.0;
      if (!obj_coeffs_dbg.empty()) {
        min_c = max_c = obj_coeffs_dbg[0];
        for (auto v : obj_coeffs_dbg) {
          min_c = std::min(min_c, v);
          max_c = std::max(max_c, v);
        }
      }
      std::cout << "[Client] MPS objective: maximize=" << mps_data.get_sense()
                << " scaling_factor=" << mps_data.get_objective_scaling_factor()
                << " offset=" << mps_data.get_objective_offset()
                << " c_size=" << obj_coeffs_dbg.size() << " c_range=[" << min_c << "," << max_c
                << "]" << std::endl;
    }

    // Build gRPC request
    SubmitJobRequest request;
    auto* lp_request = request.mutable_lp_request();

    // Set header
    auto* header = lp_request->mutable_header();
    header->set_version(1);
    header->set_problem_type(LP);
    header->set_index_type(INT32);
    header->set_float_type(DOUBLE);

    // Fill problem from MPS data
    auto* problem = lp_request->mutable_problem();
    problem->set_problem_name(mps_data.get_problem_name());
    problem->set_maximize(mps_data.get_sense());
    problem->set_objective_name(mps_data.get_objective_name());
    problem->set_objective_scaling_factor(mps_data.get_objective_scaling_factor());
    problem->set_objective_offset(mps_data.get_objective_offset());

    // Objective coefficients
    const auto& obj_coeffs = mps_data.get_objective_coefficients();
    for (size_t i = 0; i < obj_coeffs.size(); ++i) {
      problem->add_c(obj_coeffs[i]);
    }

    // Constraint matrix (CSR format)
    const auto& matrix_values = mps_data.get_constraint_matrix_values();
    for (size_t i = 0; i < matrix_values.size(); ++i) {
      problem->add_a(matrix_values[i]);
    }
    const auto& matrix_indices = mps_data.get_constraint_matrix_indices();
    for (size_t i = 0; i < matrix_indices.size(); ++i) {
      problem->add_a_indices(matrix_indices[i]);
    }
    const auto& matrix_offsets = mps_data.get_constraint_matrix_offsets();
    for (size_t i = 0; i < matrix_offsets.size(); ++i) {
      problem->add_a_offsets(matrix_offsets[i]);
    }

    // Constraint bounds
    //
    // IMPORTANT: cuOpt accepts either:
    // - (constraint_lower_bounds + constraint_upper_bounds) OR
    // - (b + row_types)
    //
    // The MPS parser may populate lower/upper bounds without populating constraint_bounds(),
    // so prefer lower/upper here to avoid sending an empty b-vector.
    const auto& con_lb = mps_data.get_constraint_lower_bounds();
    const auto& con_ub = mps_data.get_constraint_upper_bounds();
    if (!con_lb.empty()) {
      for (size_t i = 0; i < con_lb.size(); ++i) {
        problem->add_constraint_lower_bounds(con_lb[i]);
      }
      for (size_t i = 0; i < con_ub.size(); ++i) {
        problem->add_constraint_upper_bounds(con_ub[i]);
      }
    } else {
      const auto& constraint_bounds = mps_data.get_constraint_bounds();
      for (size_t i = 0; i < constraint_bounds.size(); ++i) {
        problem->add_b(constraint_bounds[i]);
      }
    }

    // Row types
    const auto& row_types = mps_data.get_row_types();
    std::string row_types_str(row_types.begin(), row_types.end());
    problem->set_row_types(row_types_str);

    // Variable bounds
    const auto& var_lb = mps_data.get_variable_lower_bounds();
    for (size_t i = 0; i < var_lb.size(); ++i) {
      problem->add_variable_lower_bounds(var_lb[i]);
    }
    const auto& var_ub = mps_data.get_variable_upper_bounds();
    for (size_t i = 0; i < var_ub.size(); ++i) {
      problem->add_variable_upper_bounds(var_ub[i]);
    }

    // Variable types
    const auto& var_types = mps_data.get_variable_types();
    std::string var_types_str(var_types.begin(), var_types.end());
    problem->set_variable_types(var_types_str);

    // Settings
    auto* settings = lp_request->mutable_settings();
    settings->set_time_limit(60.0);
    settings->set_log_to_console(false);
    // IMPORTANT: proto3 defaults numeric fields to 0. If we don't set this,
    // cuOpt may interpret iteration_limit=0 as "do zero iterations" and return
    // PDLP_ITERATION_LIMIT immediately with a trivial objective.
    // Use -1 sentinel for "unset" so server/library defaults apply.
    settings->set_iteration_limit(-1);

    // Sanity-check what we're about to send (proto3 only serializes non-default fields)
    std::cout << "[Client] Prepared SolveLPRequest: bytes=" << lp_request->ByteSizeLong()
              << " objective_scaling_factor=" << problem->objective_scaling_factor()
              << " iteration_limit=" << settings->iteration_limit() << std::endl;

    // Make RPC call
    std::cout << "[Client] Submitting job..." << std::endl;
    SubmitJobResponse response;
    ClientContext context;

    Status status = stub_->SubmitJob(&context, request, &response);

    if (status.ok()) {
      std::cout << "[Client] Job submitted successfully" << std::endl;
      std::cout << "[Client] Job ID: " << response.job_id() << std::endl;
      return response.job_id();
    } else {
      std::cerr << "[Client] RPC failed: " << status.error_message() << std::endl;
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
      std::cout << "[Client] Status: " << response.job_status() << std::endl;
      return response.job_status() == COMPLETED;
    } else {
      std::cerr << "[Client] CheckStatus RPC failed: " << status.error_message() << std::endl;
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
      std::cout << "\n[Client] ============ REMOTE RESULT ============" << std::endl;
      std::cout << "[Client] Response Status: " << response.status() << std::endl;

      if (response.has_lp_solution()) {
        const auto& lp_sol = response.lp_solution();
        std::cout << "[Client] Termination Status: " << lp_sol.termination_status() << std::endl;
        std::cout << "[Client] Objective: " << lp_sol.primal_objective() << std::endl;
        std::cout << "[Client] Primal solution size: " << lp_sol.primal_solution_size()
                  << std::endl;
        std::cout << "[Client] Dual solution size: " << lp_sol.dual_solution_size() << std::endl;
        std::cout << "[Client] Iterations: " << lp_sol.nb_iterations() << std::endl;
        std::cout << "[Client] Solve time: " << lp_sol.solve_time() << " seconds" << std::endl;

        if (!lp_sol.error_message().empty()) {
          std::cout << "[Client] Error: " << lp_sol.error_message() << std::endl;
        }
      }

      if (!response.error_message().empty()) {
        std::cout << "[Client] Response Error: " << response.error_message() << std::endl;
      }

      std::cout << "[Client] ========================================\n" << std::endl;
      return true;
    } else {
      std::cerr << "[Client] GetResult RPC failed: " << status.error_message() << std::endl;
      return false;
    }
  }

 private:
  std::unique_ptr<CuOptRemoteService::Stub> stub_;
};

int main(int argc, char** argv)
{
  std::string server_address = "localhost:9091";
  std::string mps_file       = "../datasets/afiro.mps";
  int max_message_mb         = 256;
  bool submit_only           = false;

  if (argc > 1) { mps_file = argv[1]; }
  if (argc > 2) { server_address = argv[2]; }
  if (argc > 3) { max_message_mb = std::stoi(argv[3]); }
  if (argc > 4) {
    std::string arg = argv[4];
    if (arg == "--submit-only") { submit_only = true; }
  }

  std::cout << "[Client] Connecting to " << server_address << std::endl;

  grpc::ChannelArguments args;
  int max_bytes = (max_message_mb <= 0) ? -1 : (max_message_mb * 1024 * 1024);
  args.SetMaxReceiveMessageSize(max_bytes);
  args.SetMaxSendMessageSize(max_bytes);
  CuOptGrpcClient client(
    grpc::CreateCustomChannel(server_address, grpc::InsecureChannelCredentials(), args));

  // Submit job
  std::string job_id = client.SubmitMpsFile(mps_file);
  if (job_id.empty()) { return 1; }
  if (submit_only) {
    std::cout << "[Client] Submit-only mode; exiting after SubmitJob." << std::endl;
    return 0;
  }

  // Poll for completion
  std::cout << "[Client] Waiting for result..." << std::endl;
  bool completed = false;
  for (int i = 0; i < 60; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (client.CheckStatus(job_id)) {
      completed = true;
      break;
    }
  }

  if (!completed) {
    std::cerr << "[Client] Job did not complete in time" << std::endl;
    return 1;
  }

  // Get result
  if (!client.GetResult(job_id)) { return 1; }

  std::cout << "[Client] Test completed successfully" << std::endl;
  return 0;
}
