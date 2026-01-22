/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cuopt::linear_programming::grpc_remote {

enum class ProblemType { LP = 0, MIP = 1 };

struct Incumbent {
  int64_t index    = 0;
  double objective = 0.0;
  std::vector<double> assignment;
};

// Upload serialized SolveLPRequest / SolveMIPRequest bytes and enqueue a job.
bool upload_and_submit(const std::string& address,
                       ProblemType problem_type,
                       const uint8_t* data,
                       size_t size,
                       std::string& job_id,
                       std::string& error_message);

// Submit as unary if payload fits, otherwise fall back to UploadAndSubmit.
bool submit_or_upload(const std::string& address,
                      ProblemType problem_type,
                      const uint8_t* data,
                      size_t size,
                      std::string& job_id,
                      std::string& error_message,
                      bool* used_upload_out  = nullptr,
                      int64_t* max_bytes_out = nullptr);

// Return one of: "QUEUED", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED", "NOT_FOUND"
bool check_status(const std::string& address,
                  const std::string& job_id,
                  std::string& status_out,
                  std::string& error_message,
                  int64_t* result_size_bytes_out = nullptr,
                  int64_t* max_message_bytes_out = nullptr);

// Stream raw serialized solution bytes (LPSolution or MIPSolution) into out.
bool stream_result(const std::string& address,
                   const std::string& job_id,
                   std::vector<uint8_t>& out,
                   std::string& error_message);

// Unary GetResult (returns full serialized solution bytes).
bool get_result(const std::string& address,
                const std::string& job_id,
                std::vector<uint8_t>& out,
                std::string& error_message);

// Best-effort delete of server-side stored result for a job.
void delete_result(const std::string& address, const std::string& job_id);

// Best-effort cancel.
bool cancel_job(const std::string& address,
                const std::string& job_id,
                bool& success_out,
                std::string& status_out,
                std::string& message_out,
                std::string& error_message);

// Stream logs to stdout until stop_flag is true or server indicates job_complete.
// If print_prefix is non-empty, it will be printed before each log line.
void stream_logs_to_stdout(const std::string& address,
                           const std::string& job_id,
                           volatile bool* stop_flag,
                           const std::string& print_prefix);

// Fetch incumbent solutions for a job starting at from_index.
bool get_incumbents(const std::string& address,
                    const std::string& job_id,
                    int64_t from_index,
                    int32_t max_count,
                    std::vector<Incumbent>& incumbents_out,
                    int64_t& next_index_out,
                    bool& job_complete_out,
                    std::string& error_message);

}  // namespace cuopt::linear_programming::grpc_remote
