/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Test gRPC remote solve with afiro.mps
 */

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
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
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::Status;

using namespace cuopt::remote;

class CuOptGrpcClient {
 public:
  CuOptGrpcClient(std::shared_ptr<Channel> channel) : stub_(CuOptRemoteService::NewStub(channel)) {}

  void StreamLogsToStdout(const std::string& job_id, std::atomic<bool>& stop_flag)
  {
    StreamLogsRequest req;
    req.set_job_id(job_id);
    req.set_from_byte(0);

    ClientContext ctx;
    std::unique_ptr<ClientReader<LogMessage>> reader = stub_->StreamLogs(&ctx, req);

    LogMessage msg;
    while (!stop_flag.load() && reader->Read(&msg)) {
      if (!msg.line().empty()) { std::cout << "[ServerLog] " << msg.line() << std::endl; }
      if (msg.job_complete()) { break; }
    }

    // If the caller wants to stop early, cancel the stream.
    if (stop_flag.load()) { ctx.TryCancel(); }
    reader->Finish();
  }

  std::string UploadAndSubmitLPFromFD(int fd, int64_t total_size, cuopt::remote::LPMethod method)
  {
    // We keep fd open and can retry the streaming RPC (in-process) using resume semantics.
    // The server enforces sequential offsets and reports committed_size acks.
    std::string upload_id;
    bool resume = false;

    for (int attempt = 0; attempt < 5; ++attempt) {
      ClientContext context;
      std::unique_ptr<ClientReaderWriter<UploadJobRequest, UploadJobResponse>> stream =
        stub_->UploadAndSubmit(&context);

      UploadJobRequest req;
      auto* start = req.mutable_start();
      start->set_problem_type(LP);
      start->set_resume(resume);
      if (!upload_id.empty()) { start->set_upload_id(upload_id); }
      start->set_total_size(total_size);

      if (!stream->Write(req)) {
        // immediate transport failure, retry
        resume = true;
        continue;
      }

      UploadJobResponse resp;
      if (!stream->Read(&resp) || !resp.has_ack()) {
        // failed to read ack, retry
        resume = true;
        continue;
      }

      upload_id         = resp.ack().upload_id();
      int64_t committed = resp.ack().committed_size();

      // Resume reading from committed offset
      if (lseek(fd, committed, SEEK_SET) < 0) {
        std::cerr << "[Client] lseek failed for resume offset " << committed << std::endl;
        return "";
      }

      const size_t chunk_size = 1 << 20;  // 1 MiB
      std::vector<uint8_t> buf(chunk_size);

      while (committed < total_size) {
        ssize_t nread =
          ::read(fd,
                 buf.data(),
                 static_cast<size_t>(std::min<int64_t>(chunk_size, total_size - committed)));
        if (nread < 0) {
          if (errno == EINTR) continue;
          std::cerr << "[Client] read() failed: " << strerror(errno) << std::endl;
          break;
        }
        if (nread == 0) {
          std::cerr << "[Client] Unexpected EOF reading serialized protobuf" << std::endl;
          break;
        }

        req.Clear();
        auto* chunk = req.mutable_chunk();
        chunk->set_upload_id(upload_id);
        chunk->set_offset(committed);
        chunk->set_data(buf.data(), static_cast<size_t>(nread));

        if (!stream->Write(req)) { break; }

        resp.Clear();
        if (!stream->Read(&resp) || !resp.has_ack()) { break; }
        committed = resp.ack().committed_size();
      }

      // Finish upload
      req.Clear();
      req.mutable_finish()->set_upload_id(upload_id);
      stream->Write(req);
      stream->WritesDone();

      std::string job_id;
      while (stream->Read(&resp)) {
        if (resp.has_submit()) {
          job_id = resp.submit().job_id();
          break;
        }
        if (resp.has_error()) {
          std::cerr << "[Client] UploadAndSubmit error: " << resp.error().message()
                    << " committed=" << resp.error().committed_size() << std::endl;
          break;
        }
      }

      Status st = stream->Finish();
      if (st.ok() && !job_id.empty()) { return job_id; }

      // Retry
      resume = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    return "";
  }

  std::string UploadAndSubmitLP(const SolveLPRequest& lp_request)
  {
    // Serialize to a private temp file (0600) and unlink immediately (unlink-on-open).
    // This avoids allocating a multi-GB contiguous buffer for serialized protobuf bytes.
    char tmp[] = "/tmp/cuopt_req_XXXXXX";
    int fd     = mkstemp(tmp);
    if (fd < 0) {
      std::cerr << "[Client] mkstemp failed: " << strerror(errno) << std::endl;
      return "";
    }
    // Ensure permissions are owner-only and unlink so it auto-cleans.
    fchmod(fd, 0600);
    unlink(tmp);

    if (!lp_request.SerializeToFileDescriptor(fd)) {
      std::cerr << "[Client] Failed to serialize SolveLPRequest to temp file" << std::endl;
      close(fd);
      return "";
    }
    int64_t total_size = lseek(fd, 0, SEEK_END);
    if (total_size < 0) {
      std::cerr << "[Client] lseek(SEEK_END) failed" << std::endl;
      close(fd);
      return "";
    }
    lseek(fd, 0, SEEK_SET);

    std::string job_id = UploadAndSubmitLPFromFD(fd, total_size, lp_request.settings().method());
    close(fd);
    return job_id;
  }

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
    // Give large instances more time by default.
    settings->set_time_limit(60.0);
    settings->set_log_to_console(true);
    // IMPORTANT: proto3 defaults numeric fields to 0. If we don't set this,
    // cuOpt may interpret iteration_limit=0 as "do zero iterations" and return
    // PDLP_ITERATION_LIMIT immediately with a trivial objective.
    // Use -1 sentinel for "unset" so server/library defaults apply.
    settings->set_iteration_limit(-1);
    // Allow overriding LP method for known-problem cases (e.g. avoid Concurrent/Barrier issues).
    if (mps_file_path.find("L2CTA3D.mps") != std::string::npos) {
      settings->set_method(PDLP);
      settings->set_time_limit(1800.0);
    }

    // Sanity-check what we're about to send (proto3 only serializes non-default fields)
    std::cout << "[Client] Prepared SolveLPRequest: bytes=" << lp_request->ByteSizeLong()
              << " objective_scaling_factor=" << problem->objective_scaling_factor()
              << " iteration_limit=" << settings->iteration_limit() << std::endl;

    // Streaming Upload+Submit (chunked)
    std::cout << "[Client] Uploading + submitting job (streaming)..." << std::endl;
    std::string job_id = UploadAndSubmitLP(*lp_request);
    if (job_id.empty()) {
      std::cerr << "[Client] UploadAndSubmit failed" << std::endl;
      return "";
    }
    std::cout << "[Client] Job submitted successfully" << std::endl;
    std::cout << "[Client] Job ID: " << job_id << std::endl;
    return job_id;
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
    // Prefer streaming result to avoid any total result size limit.
    // (Each streamed chunk must still fit within gRPC per-message limits.)
    {
      GetResultRequest request;
      request.set_job_id(job_id);
      ClientContext context;
      std::unique_ptr<ClientReader<ResultChunk>> reader = stub_->StreamResult(&context, request);

      // Spool streamed bytes to a private temp file (0600) and unlink immediately so it
      // auto-cleans.
      char tmp[] = "/tmp/cuopt_result_XXXXXX";
      int fd     = mkstemp(tmp);
      if (fd < 0) {
        std::cerr << "[Client] mkstemp failed for result spool: " << strerror(errno) << std::endl;
        return false;
      }
      fchmod(fd, 0600);
      unlink(tmp);

      ResultChunk chunk;
      bool saw_done = false;
      while (reader->Read(&chunk)) {
        if (!chunk.error_message().empty()) {
          std::cerr << "[Client] StreamResult error: " << chunk.error_message() << std::endl;
          break;
        }
        if (chunk.done()) {
          saw_done = true;
          break;
        }
        const std::string& data = chunk.data();
        const char* p           = data.data();
        size_t remaining        = data.size();
        while (remaining > 0) {
          ssize_t n = ::write(fd, p, remaining);
          if (n < 0) {
            if (errno == EINTR) continue;
            std::cerr << "[Client] write() failed spooling streamed result: " << strerror(errno)
                      << std::endl;
            remaining = 0;
            break;
          }
          p += n;
          remaining -= static_cast<size_t>(n);
        }
      }

      Status status = reader->Finish();
      if (!status.ok()) {
        std::cerr << "[Client] StreamResult RPC failed: " << status.error_message() << std::endl;
        close(fd);
        // fall through to unary GetResult as a fallback
      } else if (saw_done) {
        // Parse as LP solution (this test client is LP-only)
        cuopt::remote::LPSolution lp_solution;
        lseek(fd, 0, SEEK_SET);
        if (!lp_solution.ParseFromFileDescriptor(fd)) {
          std::cerr << "[Client] Failed to parse streamed LP solution" << std::endl;
          close(fd);
          return false;
        }
        close(fd);

        std::cout << "\n[Client] ============ REMOTE RESULT (STREAM) ============" << std::endl;
        std::cout << "[Client] Termination Status: " << lp_solution.termination_status()
                  << std::endl;
        std::cout << "[Client] Objective: " << lp_solution.primal_objective() << std::endl;
        std::cout << "[Client] Primal solution size: " << lp_solution.primal_solution_size()
                  << std::endl;
        std::cout << "[Client] Dual solution size: " << lp_solution.dual_solution_size()
                  << std::endl;
        std::cout << "[Client] Iterations: " << lp_solution.nb_iterations() << std::endl;
        std::cout << "[Client] Solve time: " << lp_solution.solve_time() << " seconds" << std::endl;
        if (!lp_solution.error_message().empty()) {
          std::cout << "[Client] Error: " << lp_solution.error_message() << std::endl;
        }
        std::cout << "[Client] ================================================\n" << std::endl;
        return true;
      } else {
        close(fd);
      }
    }

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
  int max_wait_sec           = 30;
  bool stream_logs           = false;

  if (argc > 1) { mps_file = argv[1]; }
  if (argc > 2) { server_address = argv[2]; }
  if (argc > 3) { max_message_mb = std::stoi(argv[3]); }
  // Parse optional flags
  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--submit-only") {
      submit_only = true;
    } else if (arg == "--max-wait-sec" && i + 1 < argc) {
      max_wait_sec = std::stoi(argv[++i]);
    } else if (arg == "--stream-logs") {
      stream_logs = true;
    }
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

  std::atomic<bool> stop_logs{false};
  std::thread log_thread;
  if (stream_logs) {
    std::cout << "[Client] Streaming server logs..." << std::endl;
    log_thread = std::thread(
      [&client, &job_id, &stop_logs]() { client.StreamLogsToStdout(job_id, stop_logs); });
  }

  // Poll for completion
  std::cout << "[Client] Waiting for result..." << std::endl;
  bool completed = false;
  auto start     = std::chrono::steady_clock::now();
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (client.CheckStatus(job_id)) {
      completed = true;
      break;
    }
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= max_wait_sec) {
      break;
    }
  }

  if (!completed) {
    std::cerr << "[Client] Job did not complete in time" << std::endl;
    return 1;
  }

  // Get result
  if (!client.GetResult(job_id)) { return 1; }

  if (stream_logs) {
    stop_logs = true;
    if (log_thread.joinable()) { log_thread.join(); }
  }

  std::cout << "[Client] Test completed successfully" << std::endl;
  return 0;
}
