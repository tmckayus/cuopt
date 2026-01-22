/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <cuopt/linear_programming/utilities/remote_serialization.hpp>
#include <cuopt/linear_programming/utilities/remote_solve.hpp>
#include <utilities/logger.hpp>

#if CUOPT_ENABLE_GRPC
#include "remote_solve_grpc.hpp"
#endif

#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace cuopt::linear_programming {

namespace {

// gRPC is the only supported remote transport.
template <typename f_t>
bool copy_incumbent_to_device(const std::vector<double>& host_assignment,
                              double host_objective,
                              f_t** d_assignment_out,
                              f_t** d_objective_out)
{
  *d_assignment_out = nullptr;
  *d_objective_out  = nullptr;
  if (host_assignment.empty()) { return false; }

  int device_count       = 0;
  cudaError_t device_err = cudaGetDeviceCount(&device_count);
  if (device_err != cudaSuccess || device_count == 0) {
    static bool logged_no_device = false;
    if (!logged_no_device) {
      CUOPT_LOG_INFO("[remote_solve] No CUDA device available; using host incumbents");
      logged_no_device = true;
    }
    return false;
  }

  size_t n = host_assignment.size();
  std::vector<f_t> assignment(n);
  for (size_t i = 0; i < n; ++i) {
    assignment[i] = static_cast<f_t>(host_assignment[i]);
  }
  f_t objective = static_cast<f_t>(host_objective);

  if (cudaMalloc(reinterpret_cast<void**>(d_assignment_out), n * sizeof(f_t)) != cudaSuccess) {
    CUOPT_LOG_WARN("[remote_solve] Failed to cudaMalloc for incumbent assignment");
    return false;
  }
  if (cudaMalloc(reinterpret_cast<void**>(d_objective_out), sizeof(f_t)) != cudaSuccess) {
    CUOPT_LOG_WARN("[remote_solve] Failed to cudaMalloc for incumbent objective");
    cudaFree(*d_assignment_out);
    *d_assignment_out = nullptr;
    return false;
  }

  if (cudaMemcpy(*d_assignment_out, assignment.data(), n * sizeof(f_t), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    CUOPT_LOG_WARN("[remote_solve] Failed to cudaMemcpy incumbent assignment");
    cudaFree(*d_assignment_out);
    cudaFree(*d_objective_out);
    *d_assignment_out = nullptr;
    *d_objective_out  = nullptr;
    return false;
  }
  if (cudaMemcpy(*d_objective_out, &objective, sizeof(f_t), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    CUOPT_LOG_WARN("[remote_solve] Failed to cudaMemcpy incumbent objective");
    cudaFree(*d_assignment_out);
    cudaFree(*d_objective_out);
    *d_assignment_out = nullptr;
    *d_objective_out  = nullptr;
    return false;
  }

  return true;
}

template <typename f_t>
void invoke_incumbent_callbacks(
  const std::vector<cuopt::internals::base_solution_callback_t*>& callbacks,
  const std::vector<double>& assignment,
  double objective)
{
  f_t* d_assignment = nullptr;
  f_t* d_objective  = nullptr;
  bool on_device =
    copy_incumbent_to_device<f_t>(assignment, objective, &d_assignment, &d_objective);
  std::vector<f_t> h_assignment;
  f_t h_objective     = static_cast<f_t>(objective);
  f_t* assignment_ptr = nullptr;
  f_t* objective_ptr  = nullptr;
  if (on_device) {
    assignment_ptr = d_assignment;
    objective_ptr  = d_objective;
  } else {
    if (assignment.empty()) { return; }
    h_assignment.resize(assignment.size());
    for (size_t i = 0; i < assignment.size(); ++i) {
      h_assignment[i] = static_cast<f_t>(assignment[i]);
    }
    assignment_ptr = h_assignment.data();
    objective_ptr  = &h_objective;
  }

  for (auto* cb : callbacks) {
    if (cb == nullptr) { continue; }
    if (cb->get_type() != cuopt::internals::base_solution_callback_type::GET_SOLUTION) { continue; }
    cb->set_memory_location(on_device ? cuopt::internals::callback_memory_location::DEVICE
                                      : cuopt::internals::callback_memory_location::HOST);
    auto* get_cb = static_cast<cuopt::internals::get_solution_callback_t*>(cb);
    get_cb->get_solution(assignment_ptr, objective_ptr);
  }

  if (on_device) {
    cudaDeviceSynchronize();
    cudaFree(d_assignment);
    cudaFree(d_objective);
  }
}

// Socket transport removed. gRPC is the only supported remote transport.

}  // namespace

//============================================================================
// LP Remote Solve
//============================================================================

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  CUOPT_LOG_INFO(
    "[remote_solve] Solving LP remotely on %s:%d (gRPC)", config.host.c_str(), config.port);

  // Log problem info (similar to local solve)
  if (settings.log_to_console) {
    auto n_rows = view.get_constraint_matrix_offsets().size() > 0
                    ? static_cast<i_t>(view.get_constraint_matrix_offsets().size()) - 1
                    : 0;
    auto n_cols = static_cast<i_t>(view.get_objective_coefficients().size());
    auto nnz    = static_cast<i_t>(view.get_constraint_matrix_values().size());
    CUOPT_LOG_INFO("Solving a problem with %d constraints, %d variables, and %d nonzeros (remote)",
                   n_rows,
                   n_cols,
                   nnz);
  }

  auto serializer = get_serializer<i_t, f_t>();

#if CUOPT_ENABLE_GRPC
  const std::string address = config.host + ":" + std::to_string(config.port);

  // Serialize as SolveLPRequest (server expects this protobuf, not AsyncRequest)
  std::vector<uint8_t> request_data = serializer->serialize_lp_request(view, settings);
  CUOPT_LOG_DEBUG(std::string("[remote_solve] Serialized LP request (gRPC): ") +
                  std::to_string(request_data.size()) + " bytes");

  std::string job_id;
  std::string err;
  bool used_upload  = false;
  int64_t max_bytes = -1;
  if (!grpc_remote::submit_or_upload(address,
                                     grpc_remote::ProblemType::LP,
                                     request_data.data(),
                                     request_data.size(),
                                     job_id,
                                     err,
                                     &used_upload,
                                     &max_bytes)) {
    std::cerr << "[remote_solve] UploadAndSubmit failed: " << err << "\n";
    std::cerr.flush();
    return optimization_problem_solution_t<i_t, f_t>(
      cuopt::logic_error("gRPC UploadAndSubmit failed: " + err, cuopt::error_type_t::RuntimeError));
  }
  if (settings.log_to_console) {
    CUOPT_LOG_INFO("gRPC submit path for LP (%zu bytes, max=%ld): %s",
                   request_data.size(),
                   static_cast<long>(max_bytes),
                   used_upload ? "UploadAndSubmit" : "SubmitJob");
  }

  // Optional realtime logs on client side
  volatile bool stop_logs = false;
  std::thread log_thread;
  if (settings.log_to_console) {
    log_thread =
      std::thread([&]() { grpc_remote::stream_logs_to_stdout(address, job_id, &stop_logs, ""); });
  }

  // Poll status until terminal, allowing log streaming and cancellation in other threads.
  std::string status;
  int64_t result_size_bytes = 0;
  int64_t max_message_bytes = 0;
  while (true) {
    std::string st_err;
    if (!grpc_remote::check_status(
          address, job_id, status, st_err, &result_size_bytes, &max_message_bytes)) {
      stop_logs = true;
      if (log_thread.joinable()) { log_thread.join(); }
      grpc_remote::delete_result(address, job_id);
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "gRPC CheckStatus failed: " + st_err, cuopt::error_type_t::RuntimeError));
    }

    if (status == "COMPLETED") { break; }
    if (status == "FAILED" || status == "CANCELLED" || status == "NOT_FOUND") {
      stop_logs = true;
      if (log_thread.joinable()) { log_thread.join(); }
      grpc_remote::delete_result(address, job_id);
      return optimization_problem_solution_t<i_t, f_t>(
        cuopt::logic_error("Remote job did not complete successfully (status=" + status + ")",
                           cuopt::error_type_t::RuntimeError));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Retrieve result bytes (unary if it fits, otherwise streaming)
  std::vector<uint8_t> solution_bytes;
  std::string res_err;
  bool use_get_result = false;
  if (max_message_bytes < 0) {
    use_get_result = true;
  } else if (result_size_bytes > 0 && result_size_bytes <= max_message_bytes) {
    use_get_result = true;
  }
  if (use_get_result) {
    if (!grpc_remote::get_result(address, job_id, solution_bytes, res_err)) {
      use_get_result = false;
    }
  }
  if (!use_get_result) {
    if (!grpc_remote::stream_result(address, job_id, solution_bytes, res_err)) {
      stop_logs = true;
      if (log_thread.joinable()) { log_thread.join(); }
      grpc_remote::delete_result(address, job_id);
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        "gRPC StreamResult failed: " + res_err, cuopt::error_type_t::RuntimeError));
    }
  }
  if (settings.log_to_console) {
    CUOPT_LOG_INFO("gRPC result fetch: %s (size=%ld max=%ld)",
                   use_get_result ? "GetResult" : "StreamResult",
                   static_cast<long>(result_size_bytes),
                   static_cast<long>(max_message_bytes));
  }
  if (solution_bytes.empty()) {
    stop_logs = true;
    if (log_thread.joinable()) { log_thread.join(); }
    grpc_remote::delete_result(address, job_id);
    return optimization_problem_solution_t<i_t, f_t>(
      cuopt::logic_error("gRPC result payload empty", cuopt::error_type_t::RuntimeError));
  }

  stop_logs = true;
  if (log_thread.joinable()) { log_thread.join(); }

  grpc_remote::delete_result(address, job_id);
  return serializer->deserialize_lp_solution(solution_bytes);
#else
  (void)serializer;
  return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
    "gRPC support is disabled; remote solve is unavailable", cuopt::error_type_t::RuntimeError));
#endif
}

//============================================================================
// MIP Remote Solve
//============================================================================

template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<i_t, f_t>& view,
  const mip_solver_settings_t<i_t, f_t>& settings)
{
  CUOPT_LOG_INFO(
    "[remote_solve] Solving MIP remotely on %s:%d (gRPC)", config.host.c_str(), config.port);

  // Log problem info
  {
    auto n_rows = view.get_constraint_matrix_offsets().size() > 0
                    ? static_cast<i_t>(view.get_constraint_matrix_offsets().size()) - 1
                    : 0;
    auto n_cols = static_cast<i_t>(view.get_objective_coefficients().size());
    auto nnz    = static_cast<i_t>(view.get_constraint_matrix_values().size());
    CUOPT_LOG_INFO(
      "Solving a problem with %d constraints, %d variables, and %d nonzeros (remote MIP)",
      n_rows,
      n_cols,
      nnz);
  }

  auto serializer = get_serializer<i_t, f_t>();

#if CUOPT_ENABLE_GRPC
  const std::string address = config.host + ":" + std::to_string(config.port);

  std::vector<uint8_t> request_data = serializer->serialize_mip_request(view, settings);
  CUOPT_LOG_DEBUG(std::string("[remote_solve] Serialized MIP request (gRPC): ") +
                  std::to_string(request_data.size()) + " bytes");

  std::string job_id;
  std::string err;
  bool used_upload  = false;
  int64_t max_bytes = -1;
  if (!grpc_remote::submit_or_upload(address,
                                     grpc_remote::ProblemType::MIP,
                                     request_data.data(),
                                     request_data.size(),
                                     job_id,
                                     err,
                                     &used_upload,
                                     &max_bytes)) {
    std::cerr << "[remote_solve] UploadAndSubmit failed: " << err << "\n";
    std::cerr.flush();
    return mip_solution_t<i_t, f_t>(
      cuopt::logic_error("gRPC UploadAndSubmit failed: " + err, cuopt::error_type_t::RuntimeError));
  }
  if (settings.log_to_console) {
    CUOPT_LOG_INFO("gRPC submit path for MIP (%zu bytes, max=%ld): %s",
                   request_data.size(),
                   static_cast<long>(max_bytes),
                   used_upload ? "UploadAndSubmit" : "SubmitJob");
  }

  volatile bool stop_logs = false;
  std::thread log_thread;
  if (settings.log_to_console) {
    log_thread =
      std::thread([&]() { grpc_remote::stream_logs_to_stdout(address, job_id, &stop_logs, ""); });
  }

  std::vector<cuopt::internals::base_solution_callback_t*> callbacks = settings.get_mip_callbacks();
  int64_t incumbent_index                                            = 0;
  bool incumbents_done                                               = callbacks.empty();
  CUOPT_LOG_INFO(std::string("[remote_solve] MIP incumbent callbacks: ") +
                 std::to_string(callbacks.size()));
  if (!callbacks.empty()) {
    size_t n_vars = view.get_objective_coefficients().size();
    for (auto* cb : callbacks) {
      if (cb != nullptr) { cb->setup<f_t>(n_vars); }
    }
  }

  std::string status;
  int64_t result_size_bytes = 0;
  int64_t max_message_bytes = 0;
  while (true) {
    std::string st_err;
    if (!grpc_remote::check_status(
          address, job_id, status, st_err, &result_size_bytes, &max_message_bytes)) {
      stop_logs = true;
      if (log_thread.joinable()) { log_thread.join(); }
      grpc_remote::delete_result(address, job_id);
      return mip_solution_t<i_t, f_t>(cuopt::logic_error("gRPC CheckStatus failed: " + st_err,
                                                         cuopt::error_type_t::RuntimeError));
    }

    if (!incumbents_done) {
      std::vector<grpc_remote::Incumbent> incumbents;
      int64_t next_index = incumbent_index;
      bool job_complete  = false;
      std::string inc_err;
      if (grpc_remote::get_incumbents(
            address, job_id, incumbent_index, 32, incumbents, next_index, job_complete, inc_err)) {
        if (!incumbents.empty()) {
          CUOPT_LOG_INFO(std::string("[remote_solve] Received ") +
                         std::to_string(incumbents.size()) + " incumbents");
        } else if (next_index != incumbent_index || job_complete) {
          CUOPT_LOG_INFO(std::string("[remote_solve] GetIncumbents returned 0 incumbents (from=") +
                         std::to_string(incumbent_index) + " next=" + std::to_string(next_index) +
                         " done=" + std::to_string(static_cast<int>(job_complete)) + ")");
        }
        for (const auto& inc : incumbents) {
          CUOPT_LOG_INFO(std::string("[remote_solve] Incumbent idx=") + std::to_string(inc.index) +
                         " obj=" + std::to_string(inc.objective) +
                         " vars=" + std::to_string(inc.assignment.size()));
          invoke_incumbent_callbacks<f_t>(callbacks, inc.assignment, inc.objective);
        }
        incumbent_index = next_index;
        if (job_complete) { incumbents_done = true; }
      } else if (!inc_err.empty()) {
        CUOPT_LOG_WARN(std::string("[remote_solve] GetIncumbents failed: ") + inc_err);
      }
    }

    if (status == "COMPLETED") { break; }
    if (status == "FAILED" || status == "CANCELLED" || status == "NOT_FOUND") {
      stop_logs = true;
      if (log_thread.joinable()) { log_thread.join(); }
      grpc_remote::delete_result(address, job_id);
      return mip_solution_t<i_t, f_t>(
        cuopt::logic_error("Remote job did not complete successfully (status=" + status + ")",
                           cuopt::error_type_t::RuntimeError));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  if (!incumbents_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // Final drain after completion to catch any last incumbents.
    for (int i = 0; i < 5; ++i) {
      std::vector<grpc_remote::Incumbent> incumbents;
      int64_t next_index = incumbent_index;
      bool job_complete  = false;
      std::string inc_err;
      if (!grpc_remote::get_incumbents(
            address, job_id, incumbent_index, 0, incumbents, next_index, job_complete, inc_err)) {
        break;
      }
      if (incumbents.empty() && next_index == incumbent_index) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
      for (const auto& inc : incumbents) {
        CUOPT_LOG_INFO(std::string("[remote_solve] Final drain incumbent idx=") +
                       std::to_string(inc.index) + " obj=" + std::to_string(inc.objective) +
                       " vars=" + std::to_string(inc.assignment.size()));
        invoke_incumbent_callbacks<f_t>(callbacks, inc.assignment, inc.objective);
      }
      incumbent_index = next_index;
      if (job_complete) { break; }
    }
  }

  std::vector<uint8_t> solution_bytes;
  std::string res_err;
  bool use_get_result = false;
  if (max_message_bytes < 0) {
    use_get_result = true;
  } else if (result_size_bytes > 0 && result_size_bytes <= max_message_bytes) {
    use_get_result = true;
  }
  if (use_get_result) {
    if (!grpc_remote::get_result(address, job_id, solution_bytes, res_err)) {
      use_get_result = false;
    }
  }
  if (!use_get_result) {
    if (!grpc_remote::stream_result(address, job_id, solution_bytes, res_err)) {
      stop_logs = true;
      if (log_thread.joinable()) { log_thread.join(); }
      grpc_remote::delete_result(address, job_id);
      return mip_solution_t<i_t, f_t>(cuopt::logic_error("gRPC StreamResult failed: " + res_err,
                                                         cuopt::error_type_t::RuntimeError));
    }
  }
  if (settings.log_to_console) {
    CUOPT_LOG_INFO("gRPC result fetch: %s (size=%ld max=%ld)",
                   use_get_result ? "GetResult" : "StreamResult",
                   static_cast<long>(result_size_bytes),
                   static_cast<long>(max_message_bytes));
  }
  if (solution_bytes.empty()) {
    stop_logs = true;
    if (log_thread.joinable()) { log_thread.join(); }
    grpc_remote::delete_result(address, job_id);
    return mip_solution_t<i_t, f_t>(
      cuopt::logic_error("gRPC result payload empty", cuopt::error_type_t::RuntimeError));
  }

  stop_logs = true;
  if (log_thread.joinable()) { log_thread.join(); }

  grpc_remote::delete_result(address, job_id);
  return serializer->deserialize_mip_solution(solution_bytes);
#else
  (void)serializer;
  return mip_solution_t<i_t, f_t>(cuopt::logic_error(
    "gRPC support is disabled; remote solve is unavailable", cuopt::error_type_t::RuntimeError));
#endif
}

//============================================================================
// Cancel Job Remote
//============================================================================

cancel_job_result_t cancel_job_remote(const remote_solve_config_t& config,
                                      const std::string& job_id)
{
  CUOPT_LOG_INFO(std::string("[remote_solve] Cancelling job ") + job_id + " on " + config.host +
                 ":" + std::to_string(config.port));

#if CUOPT_ENABLE_GRPC
  const std::string address = config.host + ":" + std::to_string(config.port);
  bool ok                   = false;
  std::string status;
  std::string msg;
  std::string err;
  bool rpc_ok = grpc_remote::cancel_job(address, job_id, ok, status, msg, err);
  cancel_job_result_t result;
  result.success = rpc_ok && ok;
  result.message = rpc_ok ? msg : err;
  if (status == "QUEUED")
    result.job_status = remote_job_status_t::QUEUED;
  else if (status == "PROCESSING")
    result.job_status = remote_job_status_t::PROCESSING;
  else if (status == "COMPLETED")
    result.job_status = remote_job_status_t::COMPLETED;
  else if (status == "FAILED")
    result.job_status = remote_job_status_t::FAILED;
  else if (status == "CANCELLED")
    result.job_status = remote_job_status_t::CANCELLED;
  else
    result.job_status = remote_job_status_t::NOT_FOUND;
  return result;
#else
  return cancel_job_result_t{false,
                             "gRPC support is disabled; remote cancel is unavailable",
                             remote_job_status_t::NOT_FOUND};
#endif
}

// Explicit instantiations
#if CUOPT_INSTANTIATE_FLOAT
template optimization_problem_solution_t<int32_t, float> solve_lp_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<int32_t, float>& view,
  const pdlp_solver_settings_t<int32_t, float>& settings);

template mip_solution_t<int32_t, float> solve_mip_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<int32_t, float>& view,
  const mip_solver_settings_t<int32_t, float>& settings);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template optimization_problem_solution_t<int32_t, double> solve_lp_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<int32_t, double>& view,
  const pdlp_solver_settings_t<int32_t, double>& settings);

template mip_solution_t<int32_t, double> solve_mip_remote(
  const remote_solve_config_t& config,
  const cuopt::mps_parser::data_model_view_t<int32_t, double>& view,
  const mip_solver_settings_t<int32_t, double>& settings);
#endif

}  // namespace cuopt::linear_programming
