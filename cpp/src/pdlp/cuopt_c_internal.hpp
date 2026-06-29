/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/cuopt_c.h>
#include <cuopt/mathematical_optimization/backend_selection.hpp>
#include <cuopt/mathematical_optimization/cpu_optimization_problem.hpp>
#include <cuopt/mathematical_optimization/mip/solver_solution.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_solution_interface.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization {

struct problem_and_stream_view_t {
  problem_and_stream_view_t(memory_backend_t mem_backend)
    : memory_backend(mem_backend), stream_view_ptr(nullptr), handle_ptr(nullptr)
  {
    if (mem_backend == memory_backend_t::GPU) {
      // Use RAII locals so partial allocations are cleaned up if a later new throws
      std::unique_ptr<rmm::cuda_stream_view> sv(
        new rmm::cuda_stream_view(rmm::cuda_stream_per_thread));
      std::unique_ptr<raft::handle_t> h(new raft::handle_t(*sv));
      std::unique_ptr<optimization_problem_t<cuopt_int_t, cuopt_float_t>> gp(
        new optimization_problem_t<cuopt_int_t, cuopt_float_t>(h.get()));
      stream_view_ptr = sv.release();
      handle_ptr      = h.release();
      gpu_problem     = gp.release();
      cpu_problem     = nullptr;
    } else {
      cpu_problem = new cpu_optimization_problem_t<cuopt_int_t, cuopt_float_t>();
      gpu_problem = nullptr;
    }
  }

  // Non-copyable
  problem_and_stream_view_t(const problem_and_stream_view_t&)            = delete;
  problem_and_stream_view_t& operator=(const problem_and_stream_view_t&) = delete;

  // Movable
  problem_and_stream_view_t(problem_and_stream_view_t&& other) noexcept
    : float_array_cache_(std::move(other.float_array_cache_)),
      int_array_cache_(std::move(other.int_array_cache_)),
      char_array_cache_(std::move(other.char_array_cache_)),
      string_array_view_cache_(std::move(other.string_array_view_cache_)),
      scalar_string_cache_(std::move(other.scalar_string_cache_)),
      memory_backend(other.memory_backend),
      gpu_problem(other.gpu_problem),
      cpu_problem(other.cpu_problem),
      stream_view_ptr(other.stream_view_ptr),
      handle_ptr(other.handle_ptr),
      remote_configured_(other.remote_configured_),
      remote_host_(std::move(other.remote_host_)),
      remote_port_(std::move(other.remote_port_))
  {
    other.gpu_problem        = nullptr;
    other.cpu_problem        = nullptr;
    other.stream_view_ptr    = nullptr;
    other.handle_ptr         = nullptr;
    other.remote_configured_ = false;
  }

  problem_and_stream_view_t& operator=(problem_and_stream_view_t&& other) noexcept
  {
    if (this != &other) {
      if (gpu_problem) delete gpu_problem;
      if (cpu_problem) delete cpu_problem;
      if (handle_ptr) delete handle_ptr;
      if (stream_view_ptr) delete stream_view_ptr;

      float_array_cache_       = std::move(other.float_array_cache_);
      int_array_cache_         = std::move(other.int_array_cache_);
      char_array_cache_        = std::move(other.char_array_cache_);
      string_array_view_cache_ = std::move(other.string_array_view_cache_);
      scalar_string_cache_     = std::move(other.scalar_string_cache_);
      memory_backend           = other.memory_backend;
      gpu_problem              = other.gpu_problem;
      cpu_problem              = other.cpu_problem;
      stream_view_ptr          = other.stream_view_ptr;
      handle_ptr               = other.handle_ptr;
      remote_configured_       = other.remote_configured_;
      remote_host_             = std::move(other.remote_host_);
      remote_port_             = std::move(other.remote_port_);

      other.gpu_problem        = nullptr;
      other.cpu_problem        = nullptr;
      other.stream_view_ptr    = nullptr;
      other.handle_ptr         = nullptr;
      other.remote_configured_ = false;
    }
    return *this;
  }

  ~problem_and_stream_view_t()
  {
    if (gpu_problem) delete gpu_problem;
    if (cpu_problem) delete cpu_problem;
    if (handle_ptr) delete handle_ptr;
    if (stream_view_ptr) delete stream_view_ptr;
  }

  raft::handle_t* get_handle_ptr() { return handle_ptr; }

  optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>* get_problem()
  {
    return memory_backend == memory_backend_t::GPU
             ? static_cast<optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>*>(
                 gpu_problem)
             : static_cast<optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>*>(
                 cpu_problem);
  }

  optimization_problem_t<cuopt_int_t, cuopt_float_t>* get_gpu_problem()
  {
    if (memory_backend == memory_backend_t::GPU) {
      return gpu_problem;
    } else {
      return nullptr;
    }
  }

  bool has_remote_server() const { return remote_configured_; }

  const std::string& remote_host() const { return remote_host_; }

  const std::string& remote_port() const { return remote_port_; }

  std::string remote_server_address() const
  {
    if (!remote_configured_) { return {}; }
    return remote_host_ + ":" + remote_port_;
  }

  void set_remote_server(const std::string& host, const std::string& port)
  {
    remote_host_       = host;
    remote_port_       = port;
    remote_configured_ = true;
    invalidate_c_api_caches();
  }

  void clear_remote_server()
  {
    remote_host_.clear();
    remote_port_.clear();
    remote_configured_ = false;
  }

  void invalidate_c_api_caches()
  {
    float_array_cache_.clear();
    int_array_cache_.clear();
    char_array_cache_.clear();
    string_array_view_cache_.clear();
    scalar_string_cache_.clear();
  }

  void refresh_string_array_view(const std::vector<std::string>& names)
  {
    string_array_view_cache_.clear();
    string_array_view_cache_.reserve(names.size());
    for (const auto& name : names) {
      string_array_view_cache_.push_back(name.c_str());
    }
  }

  std::vector<cuopt_float_t> float_array_cache_;
  std::vector<cuopt_int_t> int_array_cache_;
  std::vector<char> char_array_cache_;
  std::vector<const char*> string_array_view_cache_;
  std::string scalar_string_cache_;

  memory_backend_t memory_backend;
  optimization_problem_t<cuopt_int_t, cuopt_float_t>* gpu_problem;
  cpu_optimization_problem_t<cuopt_int_t, cuopt_float_t>* cpu_problem;
  rmm::cuda_stream_view*
    stream_view_ptr;           // nullptr for CPU memory backend to avoid CUDA initialization
  raft::handle_t* handle_ptr;  // nullptr for CPU memory backend to avoid CUDA initialization

 private:
  bool remote_configured_ = false;
  std::string remote_host_;
  std::string remote_port_;
};

struct solution_and_stream_view_t {
  solution_and_stream_view_t(bool solution_for_mip, memory_backend_t mem_backend)
    : is_mip(solution_for_mip),
      mip_solution_interface_ptr(nullptr),
      lp_solution_interface_ptr(nullptr),
      memory_backend(mem_backend)
  {
  }

  // Non-copyable
  solution_and_stream_view_t(const solution_and_stream_view_t&)            = delete;
  solution_and_stream_view_t& operator=(const solution_and_stream_view_t&) = delete;

  // Movable
  solution_and_stream_view_t(solution_and_stream_view_t&& other) noexcept
    : is_mip(other.is_mip),
      mip_solution_interface_ptr(other.mip_solution_interface_ptr),
      lp_solution_interface_ptr(other.lp_solution_interface_ptr),
      memory_backend(other.memory_backend)
  {
    other.mip_solution_interface_ptr = nullptr;
    other.lp_solution_interface_ptr  = nullptr;
  }

  solution_and_stream_view_t& operator=(solution_and_stream_view_t&& other) noexcept
  {
    if (this != &other) {
      if (mip_solution_interface_ptr) delete mip_solution_interface_ptr;
      if (lp_solution_interface_ptr) delete lp_solution_interface_ptr;

      is_mip                     = other.is_mip;
      mip_solution_interface_ptr = other.mip_solution_interface_ptr;
      lp_solution_interface_ptr  = other.lp_solution_interface_ptr;
      memory_backend             = other.memory_backend;

      other.mip_solution_interface_ptr = nullptr;
      other.lp_solution_interface_ptr  = nullptr;
    }
    return *this;
  }

  ~solution_and_stream_view_t()
  {
    if (mip_solution_interface_ptr) delete mip_solution_interface_ptr;
    if (lp_solution_interface_ptr) delete lp_solution_interface_ptr;
  }

  /**
   * @brief Get the solution as base interface pointer
   * @return Base interface pointer for polymorphic access to common methods
   * @note Allows uniform access to get_solution_host(), get_error_status(), get_solve_time()
   */
  optimization_problem_solution_interface_t<cuopt_int_t, cuopt_float_t>* get_solution()
  {
    return is_mip
             ? static_cast<optimization_problem_solution_interface_t<cuopt_int_t, cuopt_float_t>*>(
                 mip_solution_interface_ptr)
             : static_cast<optimization_problem_solution_interface_t<cuopt_int_t, cuopt_float_t>*>(
                 lp_solution_interface_ptr);
  }

  bool is_mip;
  mip_solution_interface_t<cuopt_int_t, cuopt_float_t>* mip_solution_interface_ptr;
  lp_solution_interface_t<cuopt_int_t, cuopt_float_t>* lp_solution_interface_ptr;
  memory_backend_t memory_backend;  // Track if GPU or CPU memory for data access
};

}  // namespace cuopt::mathematical_optimization
