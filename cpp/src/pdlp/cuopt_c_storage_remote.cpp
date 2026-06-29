/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/cuopt_c.h>
#include <cuopt/mathematical_optimization/backend_selection.hpp>
#include <pdlp/cuopt_c_internal.hpp>

#include <cstdio>
#include <cstring>

using namespace cuopt::mathematical_optimization;

namespace {

problem_storage_default_t to_cpp_storage(cuOptProblemStorage_t storage)
{
  switch (storage) {
    case CUOPT_PROBLEM_STORAGE_AUTO: return problem_storage_default_t::AUTO;
    case CUOPT_PROBLEM_STORAGE_HOST: return problem_storage_default_t::HOST;
    case CUOPT_PROBLEM_STORAGE_DEVICE: return problem_storage_default_t::DEVICE;
    default: return problem_storage_default_t::AUTO;
  }
}

cuOptProblemStorage_t to_c_storage(problem_storage_default_t storage)
{
  switch (storage) {
    case problem_storage_default_t::AUTO: return CUOPT_PROBLEM_STORAGE_AUTO;
    case problem_storage_default_t::HOST: return CUOPT_PROBLEM_STORAGE_HOST;
    case problem_storage_default_t::DEVICE: return CUOPT_PROBLEM_STORAGE_DEVICE;
  }
  return CUOPT_PROBLEM_STORAGE_AUTO;
}

cuOptProblemStorage_t backend_to_c_storage(memory_backend_t backend)
{
  return backend == memory_backend_t::CPU ? CUOPT_PROBLEM_STORAGE_HOST
                                          : CUOPT_PROBLEM_STORAGE_DEVICE;
}

problem_and_stream_view_t* as_problem(cuOptOptimizationProblem problem)
{
  return static_cast<problem_and_stream_view_t*>(problem);
}

void copy_string_to_buffer(const std::string& value, char* buffer, cuopt_int_t buffer_size)
{
  if (buffer == nullptr || buffer_size <= 0) { return; }
  std::snprintf(buffer, static_cast<std::size_t>(buffer_size), "%s", value.c_str());
}

}  // namespace

cuopt_int_t cuOptSetDefaultProblemStorage(cuOptProblemStorage_t storage)
{
  if (storage != CUOPT_PROBLEM_STORAGE_AUTO && storage != CUOPT_PROBLEM_STORAGE_HOST &&
      storage != CUOPT_PROBLEM_STORAGE_DEVICE) {
    return CUOPT_INVALID_ARGUMENT;
  }
  set_default_problem_storage(to_cpp_storage(storage));
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetDefaultProblemStorage(cuOptProblemStorage_t* storage_out)
{
  if (storage_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  *storage_out = to_c_storage(get_default_problem_storage());
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetProblemStorage(cuOptOptimizationProblem problem,
                                   cuOptProblemStorage_t* storage_out)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (storage_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  *storage_out = backend_to_c_storage(as_problem(problem)->memory_backend);
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetProblemRemoteServer(cuOptOptimizationProblem problem,
                                        const char* host,
                                        const char* port)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (host == nullptr || port == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (host[0] == '\0' || port[0] == '\0') { return CUOPT_INVALID_ARGUMENT; }

  auto* problem_view = as_problem(problem);
  if (problem_view->memory_backend == memory_backend_t::GPU) { return CUOPT_INVALID_ARGUMENT; }

  problem_view->set_remote_server(host, port);
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptClearProblemRemoteServer(cuOptOptimizationProblem problem)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  as_problem(problem)->clear_remote_server();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetProblemRemoteServer(cuOptOptimizationProblem problem,
                                        char* host,
                                        cuopt_int_t host_buffer_size,
                                        char* port,
                                        cuopt_int_t port_buffer_size,
                                        cuopt_int_t* is_remote_ptr)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (is_remote_ptr == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  auto* problem_view = as_problem(problem);
  *is_remote_ptr     = problem_view->has_remote_server() ? 1 : 0;
  if (*is_remote_ptr) {
    copy_string_to_buffer(problem_view->remote_host(), host, host_buffer_size);
    copy_string_to_buffer(problem_view->remote_port(), port, port_buffer_size);
  }
  return CUOPT_SUCCESS;
}
