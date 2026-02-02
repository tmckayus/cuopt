/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/cuopt_c.h>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/optimization_problem_solution_interface.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cuopt::linear_programming {

struct problem_and_stream_view_t {
  problem_and_stream_view_t(problem_backend_t backend)
    : backend_type(backend), stream_view_ptr(nullptr), handle_ptr(nullptr)
  {
    if (backend == problem_backend_t::GPU) {
      // GPU backend: Allocate CUDA resources
      stream_view_ptr = new rmm::cuda_stream_view(rmm::cuda_stream_per_thread);
      handle_ptr      = new raft::handle_t(*stream_view_ptr);
      gpu_problem     = new gpu_optimization_problem_t<cuopt_int_t, cuopt_float_t>(handle_ptr);
      cpu_problem     = nullptr;
    } else {
      // CPU backend: No CUDA resources allocated (for remote execution on CPU-only hosts)
      cpu_problem = new cpu_optimization_problem_t<cuopt_int_t, cuopt_float_t>(nullptr);
      gpu_problem = nullptr;
    }
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
    return backend_type == problem_backend_t::GPU
             ? static_cast<optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>*>(
                 gpu_problem)
             : static_cast<optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>*>(
                 cpu_problem);
  }

  optimization_problem_t<cuopt_int_t, cuopt_float_t> to_optimization_problem()
  {
    if (backend_type == problem_backend_t::GPU) {
      return gpu_problem->to_optimization_problem();
    } else {
      return cpu_problem->to_optimization_problem();
    }
  }

  problem_backend_t backend_type;
  gpu_optimization_problem_t<cuopt_int_t, cuopt_float_t>* gpu_problem;
  cpu_optimization_problem_t<cuopt_int_t, cuopt_float_t>* cpu_problem;
  rmm::cuda_stream_view* stream_view_ptr;  // nullptr for CPU backend to avoid CUDA initialization
  raft::handle_t* handle_ptr;              // nullptr for CPU backend to avoid CUDA initialization
};

struct solution_and_stream_view_t {
  solution_and_stream_view_t(bool solution_for_mip, problem_backend_t backend)
    : is_mip(solution_for_mip),
      mip_solution_interface_ptr(nullptr),
      lp_solution_interface_ptr(nullptr),
      backend_type(backend)
  {
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
  problem_backend_t backend_type;  // Track if GPU or CPU backend for data access
};

}  // namespace cuopt::linear_programming
