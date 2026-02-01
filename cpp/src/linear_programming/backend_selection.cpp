/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <utilities/logger.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace cuopt::linear_programming {

bool is_remote_execution_enabled()
{
  const char* remote_host = std::getenv("CUOPT_REMOTE_HOST");
  const char* remote_port = std::getenv("CUOPT_REMOTE_PORT");
  return (remote_host != nullptr && remote_port != nullptr);
}

bool force_gpu_memory()
{
  const char* use_gpu_mem = std::getenv("CUOPT_USE_GPU_MEM");
  if (use_gpu_mem != nullptr) {
    std::string value(use_gpu_mem);
    // Convert to lowercase for case-insensitive comparison
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return (value == "true" || value == "1");
  }
  return false;
}

problem_backend_t get_backend_type()
{
  if (force_gpu_memory()) {
    fprintf(stderr, "[cuOpt] CUOPT_USE_GPU_MEM=true - forcing GPU backend\n");
    return problem_backend_t::GPU;
  }
  if (is_remote_execution_enabled()) {
    fprintf(stderr, "[cuOpt] Remote execution enabled - using CPU backend\n");
    return problem_backend_t::CPU;
  }
  fprintf(stderr, "[cuOpt] Using GPU backend\n");
  return problem_backend_t::GPU;
}

}  // namespace cuopt::linear_programming
