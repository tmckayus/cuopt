/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/backend_selection.hpp>
#include <utilities/logger.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace cuopt::mathematical_optimization {

namespace {

problem_storage_default_t g_default_problem_storage = problem_storage_default_t::AUTO;

problem_storage_default_t parse_problem_storage_env()
{
  const char* value = std::getenv("CUOPT_PROBLEM_STORAGE");
  if (value == nullptr) { return problem_storage_default_t::AUTO; }

  std::string storage(value);
  std::transform(storage.begin(), storage.end(), storage.begin(), ::tolower);
  if (storage == "host") { return problem_storage_default_t::HOST; }
  if (storage == "device") { return problem_storage_default_t::DEVICE; }
  return problem_storage_default_t::AUTO;
}

}  // namespace

bool is_remote_execution_enabled()
{
  const char* remote_host = std::getenv("CUOPT_REMOTE_HOST");
  const char* remote_port = std::getenv("CUOPT_REMOTE_PORT");
  return (remote_host != nullptr && remote_port != nullptr);
}

execution_mode_t get_execution_mode()
{
  return is_remote_execution_enabled() ? execution_mode_t::REMOTE : execution_mode_t::LOCAL;
}

bool use_cpu_memory_for_local()
{
  const char* use_cpu_mem = std::getenv("CUOPT_USE_CPU_MEM_FOR_LOCAL");
  if (use_cpu_mem != nullptr) {
    std::string value(use_cpu_mem);
    // Convert to lowercase for case-insensitive comparison
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return (value == "true" || value == "1");
  }
  return false;
}

memory_backend_t get_memory_backend_type()
{
  if (get_execution_mode() == execution_mode_t::REMOTE) { return memory_backend_t::CPU; }
  // Local execution: GPU memory by default, CPU if CUOPT_USE_CPU_MEM_FOR_LOCAL is set
  return use_cpu_memory_for_local() ? memory_backend_t::CPU : memory_backend_t::GPU;
}

void set_default_problem_storage(problem_storage_default_t storage)
{
  g_default_problem_storage = storage;
}

problem_storage_default_t get_default_problem_storage() { return g_default_problem_storage; }

memory_backend_t resolve_memory_backend_for_new_problem()
{
  auto storage = g_default_problem_storage;
  if (storage == problem_storage_default_t::AUTO) { storage = parse_problem_storage_env(); }
  if (storage == problem_storage_default_t::HOST) { return memory_backend_t::CPU; }
  if (storage == problem_storage_default_t::DEVICE) { return memory_backend_t::GPU; }
  return get_memory_backend_type();
}

}  // namespace cuopt::mathematical_optimization
