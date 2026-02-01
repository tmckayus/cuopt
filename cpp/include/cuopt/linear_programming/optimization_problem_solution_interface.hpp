/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/constants.h>
#include <cuopt/error.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>  // For mip_termination_status_t
#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>  // For pdlp_termination_status_t

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <string>
#include <vector>

namespace cuopt::linear_programming {

// Forward declarations
template <typename i_t, typename f_t>
class optimization_problem_solution_t;
template <typename i_t, typename f_t>
class mip_solution_t;

/**
 * @brief Abstract interface for optimization problem solutions (LP and MIP)
 *
 * This interface allows for CPU or GPU-backed solution storage.
 * - gpu_optimization_problem_solution_t: Uses rmm::device_uvector (GPU memory)
 * - cpu_optimization_problem_solution_t: Uses std::vector (CPU/host memory)
 *
 * @tparam i_t Integer type for indices
 * @tparam f_t Floating point type for values
 */
template <typename i_t, typename f_t>
class optimization_problem_solution_interface_t {
 public:
  virtual ~optimization_problem_solution_interface_t() = default;

  /**
   * @brief Check if this is a MIP solution or LP solution
   * @return true if MIP, false if LP
   */
  virtual bool is_mip() const = 0;

  /**
   * @brief Get the error status
   * @return The error status
   */
  virtual cuopt::logic_error get_error_status() const = 0;

  /**
   * @brief Get the solve time in seconds
   * @return Time in seconds
   */
  virtual double get_solve_time() const = 0;
};

/**
 * @brief Interface for LP/PDLP solutions
 */
template <typename i_t, typename f_t>
class lp_solution_interface_t : public optimization_problem_solution_interface_t<i_t, f_t> {
 public:
  bool is_mip() const override { return false; }

  /**
   * @brief Get the primal solution size
   * @return Number of variables
   */
  virtual i_t get_primal_solution_size() const = 0;

  /**
   * @brief Get the dual solution size
   * @return Number of constraints
   */
  virtual i_t get_dual_solution_size() const = 0;

  /**
   * @brief Get the reduced cost size
   * @return Number of variables
   */
  virtual i_t get_reduced_cost_size() const = 0;

  /**
   * @brief Get primal solution as host vector
   * @return Host vector of primal solution
   */
  virtual const std::vector<f_t>& get_primal_solution_host() const = 0;

  /**
   * @brief Get dual solution as host vector
   * @return Host vector of dual solution
   */
  virtual const std::vector<f_t>& get_dual_solution_host() const = 0;

  /**
   * @brief Get reduced cost as host vector
   * @return Host vector of reduced costs
   */
  virtual const std::vector<f_t>& get_reduced_cost_host() const = 0;

  /**
   * @brief Get solve time
   * @return Total solve time in seconds
   */
  virtual f_t get_solve_time() const = 0;

  /**
   * @brief Get primal objective value
   * @return Primal objective value
   */
  virtual f_t get_objective_value(i_t id = 0) const = 0;

  /**
   * @brief Get dual objective value
   * @return Dual objective value
   */
  virtual f_t get_dual_objective_value(i_t id = 0) const = 0;

  /**
   * @brief Get termination status
   * @return Termination status
   */
  virtual pdlp_termination_status_t get_termination_status(i_t id = 0) const = 0;

  /**
   * @brief Get L2 primal residual
   * @return L2 primal residual
   */
  virtual f_t get_l2_primal_residual(i_t id = 0) const = 0;

  /**
   * @brief Get L2 dual residual
   * @return L2 dual residual
   */
  virtual f_t get_l2_dual_residual(i_t id = 0) const = 0;

  /**
   * @brief Get gap
   * @return Gap value
   */
  virtual f_t get_gap(i_t id = 0) const = 0;

  /**
   * @brief Get number of iterations
   * @return Number of iterations
   */
  virtual i_t get_num_iterations(i_t id = 0) const = 0;

  /**
   * @brief Check if solved by PDLP
   * @return true if solved by PDLP
   */
  virtual bool is_solved_by_pdlp(i_t id = 0) const = 0;

  /**
   * @brief Get PDLP warm start data (GPU solutions only)
   * @return Reference to warm start data
   * @note GPU solutions only - throws for CPU solutions
   */
  virtual const pdlp_warm_start_data_t<i_t, f_t>& get_pdlp_warm_start_data() const = 0;

  /**
   * @brief Check if warm start data is available
   * @return true if warm start data is available, false otherwise
   */
  virtual bool has_warm_start_data() const = 0;

  // Individual warm start data accessors (work for both GPU and CPU)
  // Return empty vectors if no warm start data available
  virtual std::vector<f_t> get_current_primal_solution_host() const                  = 0;
  virtual std::vector<f_t> get_current_dual_solution_host() const                    = 0;
  virtual std::vector<f_t> get_initial_primal_average_host() const                   = 0;
  virtual std::vector<f_t> get_initial_dual_average_host() const                     = 0;
  virtual std::vector<f_t> get_current_ATY_host() const                              = 0;
  virtual std::vector<f_t> get_sum_primal_solutions_host() const                     = 0;
  virtual std::vector<f_t> get_sum_dual_solutions_host() const                       = 0;
  virtual std::vector<f_t> get_last_restart_duality_gap_primal_solution_host() const = 0;
  virtual std::vector<f_t> get_last_restart_duality_gap_dual_solution_host() const   = 0;
  virtual f_t get_initial_primal_weight() const                                      = 0;
  virtual f_t get_initial_step_size() const                                          = 0;
  virtual i_t get_total_pdlp_iterations() const                                      = 0;
  virtual i_t get_total_pdhg_iterations() const                                      = 0;
  virtual f_t get_last_candidate_kkt_score() const                                   = 0;
  virtual f_t get_last_restart_kkt_score() const                                     = 0;
  virtual f_t get_sum_solution_weight() const                                        = 0;
  virtual i_t get_iterations_since_last_restart() const                              = 0;

  /**
   * @brief Convert to optimization_problem_solution_t (GPU-backed)
   * This is used for remote execution: CPU solution -> GPU solution for return
   * @param stream_view CUDA stream for device allocations
   * @return GPU-backed solution
   */
  virtual optimization_problem_solution_t<i_t, f_t> to_gpu_solution(
    rmm::cuda_stream_view stream_view) = 0;
};

/**
 * @brief Interface for MIP solutions
 */
template <typename i_t, typename f_t>
class mip_solution_interface_t : public optimization_problem_solution_interface_t<i_t, f_t> {
 public:
  bool is_mip() const override { return true; }

  /**
   * @brief Get the solution size
   * @return Number of variables
   */
  virtual i_t get_solution_size() const = 0;

  /**
   * @brief Get solution as host vector
   * @return Host vector of solution
   */
  virtual const std::vector<f_t>& get_solution_host() const = 0;

  /**
   * @brief Get objective value
   * @return Objective value
   */
  virtual f_t get_objective_value() const = 0;

  /**
   * @brief Get solve time
   * @return Total solve time in seconds
   */
  virtual f_t get_solve_time() const = 0;

  /**
   * @brief Get MIP gap
   * @return MIP gap
   */
  virtual f_t get_mip_gap() const = 0;

  /**
   * @brief Get solution bound
   * @return Solution bound
   */
  virtual f_t get_solution_bound() const = 0;

  /**
   * @brief Get termination status
   * @return Termination status
   */
  virtual mip_termination_status_t get_termination_status() const = 0;

  /**
   * @brief Get presolve time
   * @return Presolve time in seconds
   */
  virtual f_t get_presolve_time() const = 0;

  /**
   * @brief Get max constraint violation
   * @return Maximum constraint violation
   */
  virtual f_t get_max_constraint_violation() const = 0;

  /**
   * @brief Get max integer violation
   * @return Maximum integer violation
   */
  virtual f_t get_max_int_violation() const = 0;

  /**
   * @brief Get max variable bound violation
   * @return Maximum variable bound violation
   */
  virtual f_t get_max_variable_bound_violation() const = 0;

  /**
   * @brief Get number of nodes
   * @return Number of nodes explored
   */
  virtual i_t get_num_nodes() const = 0;

  /**
   * @brief Get number of simplex iterations
   * @return Number of simplex iterations
   */
  virtual i_t get_num_simplex_iterations() const = 0;

  /**
   * @brief Convert to mip_solution_t (GPU-backed)
   * This is used for remote execution: CPU solution -> GPU solution for return
   * @param stream_view CUDA stream for device allocations
   * @return GPU-backed solution
   */
  virtual mip_solution_t<i_t, f_t> to_gpu_solution(rmm::cuda_stream_view stream_view) = 0;
};

// Forward declarations of concrete implementations
template <typename i_t, typename f_t>
class gpu_lp_solution_t;
template <typename i_t, typename f_t>
class cpu_lp_solution_t;
template <typename i_t, typename f_t>
class gpu_mip_solution_t;
template <typename i_t, typename f_t>
class cpu_mip_solution_t;

}  // namespace cuopt::linear_programming
