/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/utilities/grpc_solution_mapper.hpp>

#include <cuopt/linear_programming/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>
#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/utilities/proto_message_stream.hpp>

#include <cstring>
#include <map>

namespace cuopt::linear_programming {

// Convert cuOpt termination status to protobuf enum
cuopt::remote::PDLPTerminationStatus to_proto_pdlp_status(pdlp_termination_status_t status)
{
  switch (status) {
    case pdlp_termination_status_t::NoTermination: return cuopt::remote::PDLP_NO_TERMINATION;
    case pdlp_termination_status_t::NumericalError: return cuopt::remote::PDLP_NUMERICAL_ERROR;
    case pdlp_termination_status_t::Optimal: return cuopt::remote::PDLP_OPTIMAL;
    case pdlp_termination_status_t::PrimalInfeasible: return cuopt::remote::PDLP_PRIMAL_INFEASIBLE;
    case pdlp_termination_status_t::DualInfeasible: return cuopt::remote::PDLP_DUAL_INFEASIBLE;
    case pdlp_termination_status_t::IterationLimit: return cuopt::remote::PDLP_ITERATION_LIMIT;
    case pdlp_termination_status_t::TimeLimit: return cuopt::remote::PDLP_TIME_LIMIT;
    case pdlp_termination_status_t::ConcurrentLimit: return cuopt::remote::PDLP_CONCURRENT_LIMIT;
    case pdlp_termination_status_t::PrimalFeasible: return cuopt::remote::PDLP_PRIMAL_FEASIBLE;
    default: return cuopt::remote::PDLP_NO_TERMINATION;
  }
}

// Convert protobuf enum to cuOpt termination status
pdlp_termination_status_t from_proto_pdlp_status(cuopt::remote::PDLPTerminationStatus status)
{
  switch (status) {
    case cuopt::remote::PDLP_NO_TERMINATION: return pdlp_termination_status_t::NoTermination;
    case cuopt::remote::PDLP_NUMERICAL_ERROR: return pdlp_termination_status_t::NumericalError;
    case cuopt::remote::PDLP_OPTIMAL: return pdlp_termination_status_t::Optimal;
    case cuopt::remote::PDLP_PRIMAL_INFEASIBLE: return pdlp_termination_status_t::PrimalInfeasible;
    case cuopt::remote::PDLP_DUAL_INFEASIBLE: return pdlp_termination_status_t::DualInfeasible;
    case cuopt::remote::PDLP_ITERATION_LIMIT: return pdlp_termination_status_t::IterationLimit;
    case cuopt::remote::PDLP_TIME_LIMIT: return pdlp_termination_status_t::TimeLimit;
    case cuopt::remote::PDLP_CONCURRENT_LIMIT: return pdlp_termination_status_t::ConcurrentLimit;
    case cuopt::remote::PDLP_PRIMAL_FEASIBLE: return pdlp_termination_status_t::PrimalFeasible;
    default: return pdlp_termination_status_t::NoTermination;
  }
}

// Convert MIP termination status
cuopt::remote::MIPTerminationStatus to_proto_mip_status(mip_termination_status_t status)
{
  switch (status) {
    case mip_termination_status_t::NoTermination: return cuopt::remote::MIP_NO_TERMINATION;
    case mip_termination_status_t::Optimal: return cuopt::remote::MIP_OPTIMAL;
    case mip_termination_status_t::FeasibleFound: return cuopt::remote::MIP_FEASIBLE_FOUND;
    case mip_termination_status_t::Infeasible: return cuopt::remote::MIP_INFEASIBLE;
    case mip_termination_status_t::Unbounded: return cuopt::remote::MIP_UNBOUNDED;
    case mip_termination_status_t::TimeLimit: return cuopt::remote::MIP_TIME_LIMIT;
    default: return cuopt::remote::MIP_NO_TERMINATION;
  }
}

mip_termination_status_t from_proto_mip_status(cuopt::remote::MIPTerminationStatus status)
{
  switch (status) {
    case cuopt::remote::MIP_NO_TERMINATION: return mip_termination_status_t::NoTermination;
    case cuopt::remote::MIP_OPTIMAL: return mip_termination_status_t::Optimal;
    case cuopt::remote::MIP_FEASIBLE_FOUND: return mip_termination_status_t::FeasibleFound;
    case cuopt::remote::MIP_INFEASIBLE: return mip_termination_status_t::Infeasible;
    case cuopt::remote::MIP_UNBOUNDED: return mip_termination_status_t::Unbounded;
    case cuopt::remote::MIP_TIME_LIMIT: return mip_termination_status_t::TimeLimit;
    default: return mip_termination_status_t::NoTermination;
  }
}

template <typename i_t, typename f_t>
void map_lp_solution_to_proto(const cpu_lp_solution_t<i_t, f_t>& solution,
                              cuopt::remote::LPSolution* pb_solution)
{
  pb_solution->set_termination_status(to_proto_pdlp_status(solution.get_termination_status()));
  pb_solution->set_error_message(solution.get_error_status().what());

  // Solution vectors - CPU solution already has data in host memory
  const auto& primal       = solution.get_primal_solution_host();
  const auto& dual         = solution.get_dual_solution_host();
  const auto& reduced_cost = solution.get_reduced_cost_host();

  for (const auto& v : primal) {
    pb_solution->add_primal_solution(static_cast<double>(v));
  }
  for (const auto& v : dual) {
    pb_solution->add_dual_solution(static_cast<double>(v));
  }
  for (const auto& v : reduced_cost) {
    pb_solution->add_reduced_cost(static_cast<double>(v));
  }

  // Statistics
  pb_solution->set_l2_primal_residual(solution.get_l2_primal_residual());
  pb_solution->set_l2_dual_residual(solution.get_l2_dual_residual());
  pb_solution->set_primal_objective(solution.get_objective_value());
  pb_solution->set_dual_objective(solution.get_dual_objective_value());
  pb_solution->set_gap(solution.get_gap());
  pb_solution->set_nb_iterations(solution.get_num_iterations());
  pb_solution->set_solve_time(solution.get_solve_time());
  pb_solution->set_solved_by_pdlp(solution.is_solved_by_pdlp());

  if (solution.has_warm_start_data()) {
    auto* pb_ws    = pb_solution->mutable_warm_start_data();
    const auto& ws = solution.get_cpu_pdlp_warm_start_data();

    for (const auto& v : ws.current_primal_solution_)
      pb_ws->add_current_primal_solution(static_cast<double>(v));
    for (const auto& v : ws.current_dual_solution_)
      pb_ws->add_current_dual_solution(static_cast<double>(v));
    for (const auto& v : ws.initial_primal_average_)
      pb_ws->add_initial_primal_average(static_cast<double>(v));
    for (const auto& v : ws.initial_dual_average_)
      pb_ws->add_initial_dual_average(static_cast<double>(v));
    for (const auto& v : ws.current_ATY_)
      pb_ws->add_current_aty(static_cast<double>(v));
    for (const auto& v : ws.sum_primal_solutions_)
      pb_ws->add_sum_primal_solutions(static_cast<double>(v));
    for (const auto& v : ws.sum_dual_solutions_)
      pb_ws->add_sum_dual_solutions(static_cast<double>(v));
    for (const auto& v : ws.last_restart_duality_gap_primal_solution_)
      pb_ws->add_last_restart_duality_gap_primal_solution(static_cast<double>(v));
    for (const auto& v : ws.last_restart_duality_gap_dual_solution_)
      pb_ws->add_last_restart_duality_gap_dual_solution(static_cast<double>(v));

    pb_ws->set_initial_primal_weight(static_cast<double>(ws.initial_primal_weight_));
    pb_ws->set_initial_step_size(static_cast<double>(ws.initial_step_size_));
    pb_ws->set_total_pdlp_iterations(static_cast<int32_t>(ws.total_pdlp_iterations_));
    pb_ws->set_total_pdhg_iterations(static_cast<int32_t>(ws.total_pdhg_iterations_));
    pb_ws->set_last_candidate_kkt_score(static_cast<double>(ws.last_candidate_kkt_score_));
    pb_ws->set_last_restart_kkt_score(static_cast<double>(ws.last_restart_kkt_score_));
    pb_ws->set_sum_solution_weight(static_cast<double>(ws.sum_solution_weight_));
    pb_ws->set_iterations_since_last_restart(
      static_cast<int32_t>(ws.iterations_since_last_restart_));
  }
}

template <typename i_t, typename f_t>
cpu_lp_solution_t<i_t, f_t> map_proto_to_lp_solution(const cuopt::remote::LPSolution& pb_solution)
{
  // Convert solution vectors
  std::vector<f_t> primal(pb_solution.primal_solution().begin(),
                          pb_solution.primal_solution().end());
  std::vector<f_t> dual(pb_solution.dual_solution().begin(), pb_solution.dual_solution().end());
  std::vector<f_t> reduced_cost(pb_solution.reduced_cost().begin(),
                                pb_solution.reduced_cost().end());

  auto status   = from_proto_pdlp_status(pb_solution.termination_status());
  auto obj      = static_cast<f_t>(pb_solution.primal_objective());
  auto dual_obj = static_cast<f_t>(pb_solution.dual_objective());
  auto solve_t  = pb_solution.solve_time();
  auto l2_pr    = static_cast<f_t>(pb_solution.l2_primal_residual());
  auto l2_dr    = static_cast<f_t>(pb_solution.l2_dual_residual());
  auto g        = static_cast<f_t>(pb_solution.gap());
  auto iters    = static_cast<i_t>(pb_solution.nb_iterations());
  auto by_pdlp  = pb_solution.solved_by_pdlp();

  if (pb_solution.has_warm_start_data()) {
    const auto& pb_ws = pb_solution.warm_start_data();
    cpu_pdlp_warm_start_data_t<i_t, f_t> ws;

    ws.current_primal_solution_.assign(pb_ws.current_primal_solution().begin(),
                                       pb_ws.current_primal_solution().end());
    ws.current_dual_solution_.assign(pb_ws.current_dual_solution().begin(),
                                     pb_ws.current_dual_solution().end());
    ws.initial_primal_average_.assign(pb_ws.initial_primal_average().begin(),
                                      pb_ws.initial_primal_average().end());
    ws.initial_dual_average_.assign(pb_ws.initial_dual_average().begin(),
                                    pb_ws.initial_dual_average().end());
    ws.current_ATY_.assign(pb_ws.current_aty().begin(), pb_ws.current_aty().end());
    ws.sum_primal_solutions_.assign(pb_ws.sum_primal_solutions().begin(),
                                    pb_ws.sum_primal_solutions().end());
    ws.sum_dual_solutions_.assign(pb_ws.sum_dual_solutions().begin(),
                                  pb_ws.sum_dual_solutions().end());
    ws.last_restart_duality_gap_primal_solution_.assign(
      pb_ws.last_restart_duality_gap_primal_solution().begin(),
      pb_ws.last_restart_duality_gap_primal_solution().end());
    ws.last_restart_duality_gap_dual_solution_.assign(
      pb_ws.last_restart_duality_gap_dual_solution().begin(),
      pb_ws.last_restart_duality_gap_dual_solution().end());

    ws.initial_primal_weight_         = static_cast<f_t>(pb_ws.initial_primal_weight());
    ws.initial_step_size_             = static_cast<f_t>(pb_ws.initial_step_size());
    ws.total_pdlp_iterations_         = static_cast<i_t>(pb_ws.total_pdlp_iterations());
    ws.total_pdhg_iterations_         = static_cast<i_t>(pb_ws.total_pdhg_iterations());
    ws.last_candidate_kkt_score_      = static_cast<f_t>(pb_ws.last_candidate_kkt_score());
    ws.last_restart_kkt_score_        = static_cast<f_t>(pb_ws.last_restart_kkt_score());
    ws.sum_solution_weight_           = static_cast<f_t>(pb_ws.sum_solution_weight());
    ws.iterations_since_last_restart_ = static_cast<i_t>(pb_ws.iterations_since_last_restart());

    return cpu_lp_solution_t<i_t, f_t>(std::move(primal),
                                       std::move(dual),
                                       std::move(reduced_cost),
                                       status,
                                       obj,
                                       dual_obj,
                                       solve_t,
                                       l2_pr,
                                       l2_dr,
                                       g,
                                       iters,
                                       by_pdlp,
                                       std::move(ws));
  }

  return cpu_lp_solution_t<i_t, f_t>(std::move(primal),
                                     std::move(dual),
                                     std::move(reduced_cost),
                                     status,
                                     obj,
                                     dual_obj,
                                     solve_t,
                                     l2_pr,
                                     l2_dr,
                                     g,
                                     iters,
                                     by_pdlp);
}

template <typename i_t, typename f_t>
void map_mip_solution_to_proto(const cpu_mip_solution_t<i_t, f_t>& solution,
                               cuopt::remote::MIPSolution* pb_solution)
{
  pb_solution->set_termination_status(to_proto_mip_status(solution.get_termination_status()));
  pb_solution->set_error_message(solution.get_error_status().what());

  // Solution vector - CPU solution already has data in host memory
  const auto& sol_vec = solution.get_solution_host();
  for (const auto& v : sol_vec) {
    pb_solution->add_solution(static_cast<double>(v));
  }

  // Solution statistics
  pb_solution->set_objective(solution.get_objective_value());
  pb_solution->set_mip_gap(solution.get_mip_gap());
  pb_solution->set_solution_bound(solution.get_solution_bound());
  pb_solution->set_total_solve_time(solution.get_solve_time());
  pb_solution->set_presolve_time(solution.get_presolve_time());
  pb_solution->set_max_constraint_violation(solution.get_max_constraint_violation());
  pb_solution->set_max_int_violation(solution.get_max_int_violation());
  pb_solution->set_max_variable_bound_violation(solution.get_max_variable_bound_violation());
  pb_solution->set_nodes(solution.get_num_nodes());
  pb_solution->set_simplex_iterations(solution.get_num_simplex_iterations());
}

template <typename i_t, typename f_t>
cpu_mip_solution_t<i_t, f_t> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution)
{
  // Convert solution vector
  std::vector<f_t> solution_vec(pb_solution.solution().begin(), pb_solution.solution().end());

  // Create CPU MIP solution with data
  return cpu_mip_solution_t<i_t, f_t>(std::move(solution_vec),
                                      from_proto_mip_status(pb_solution.termination_status()),
                                      static_cast<f_t>(pb_solution.objective()),
                                      static_cast<f_t>(pb_solution.mip_gap()),
                                      static_cast<f_t>(pb_solution.solution_bound()),
                                      pb_solution.total_solve_time(),
                                      pb_solution.presolve_time(),
                                      static_cast<f_t>(pb_solution.max_constraint_violation()),
                                      static_cast<f_t>(pb_solution.max_int_violation()),
                                      static_cast<f_t>(pb_solution.max_variable_bound_violation()),
                                      static_cast<i_t>(pb_solution.nodes()),
                                      static_cast<i_t>(pb_solution.simplex_iterations()));
}

// ============================================================================
// Size estimation
// ============================================================================

template <typename i_t, typename f_t>
size_t estimate_lp_solution_proto_size(const cpu_lp_solution_t<i_t, f_t>& solution)
{
  size_t est = 0;
  est += static_cast<size_t>(solution.get_primal_solution_size()) * sizeof(double);
  est += static_cast<size_t>(solution.get_dual_solution_size()) * sizeof(double);
  est += static_cast<size_t>(solution.get_reduced_cost_size()) * sizeof(double);
  if (solution.has_warm_start_data()) {
    const auto& ws = solution.get_cpu_pdlp_warm_start_data();
    est += ws.current_primal_solution_.size() * sizeof(double);
    est += ws.current_dual_solution_.size() * sizeof(double);
    est += ws.initial_primal_average_.size() * sizeof(double);
    est += ws.initial_dual_average_.size() * sizeof(double);
    est += ws.current_ATY_.size() * sizeof(double);
    est += ws.sum_primal_solutions_.size() * sizeof(double);
    est += ws.sum_dual_solutions_.size() * sizeof(double);
    est += ws.last_restart_duality_gap_primal_solution_.size() * sizeof(double);
    est += ws.last_restart_duality_gap_dual_solution_.size() * sizeof(double);
  }
  est += 512;  // scalars + tags overhead
  return est;
}

template <typename i_t, typename f_t>
size_t estimate_mip_solution_proto_size(const cpu_mip_solution_t<i_t, f_t>& solution)
{
  size_t est = 0;
  est += static_cast<size_t>(solution.get_solution_size()) * sizeof(double);
  est += 256;  // scalars + tags overhead
  return est;
}

// ============================================================================
// Chunked result header population
// ============================================================================

namespace {
void add_result_array_descriptor(cuopt::remote::ChunkedResultHeader* header,
                                 cuopt::remote::ResultFieldId fid,
                                 int64_t count,
                                 int64_t elem_size)
{
  if (count <= 0) return;
  auto* desc = header->add_arrays();
  desc->set_field_id(fid);
  desc->set_total_elements(count);
  desc->set_element_size_bytes(elem_size);
}

template <typename f_t>
std::vector<uint8_t> doubles_to_bytes(const std::vector<f_t>& vec)
{
  std::vector<double> tmp(vec.begin(), vec.end());
  std::vector<uint8_t> bytes(tmp.size() * sizeof(double));
  std::memcpy(bytes.data(), tmp.data(), bytes.size());
  return bytes;
}
}  // namespace

template <typename i_t, typename f_t>
void populate_chunked_result_header_lp(const cpu_lp_solution_t<i_t, f_t>& solution,
                                       cuopt::remote::ChunkedResultHeader* header)
{
  header->set_is_mip(false);
  header->set_lp_termination_status(to_proto_pdlp_status(solution.get_termination_status()));
  header->set_error_message(solution.get_error_status().what());
  header->set_l2_primal_residual(solution.get_l2_primal_residual());
  header->set_l2_dual_residual(solution.get_l2_dual_residual());
  header->set_primal_objective(solution.get_objective_value());
  header->set_dual_objective(solution.get_dual_objective_value());
  header->set_gap(solution.get_gap());
  header->set_nb_iterations(solution.get_num_iterations());
  header->set_solve_time(solution.get_solve_time());
  header->set_solved_by_pdlp(solution.is_solved_by_pdlp());

  const auto& primal       = solution.get_primal_solution_host();
  const auto& dual         = solution.get_dual_solution_host();
  const auto& reduced_cost = solution.get_reduced_cost_host();

  add_result_array_descriptor(
    header, cuopt::remote::RESULT_PRIMAL_SOLUTION, primal.size(), sizeof(double));
  add_result_array_descriptor(
    header, cuopt::remote::RESULT_DUAL_SOLUTION, dual.size(), sizeof(double));
  add_result_array_descriptor(
    header, cuopt::remote::RESULT_REDUCED_COST, reduced_cost.size(), sizeof(double));

  if (solution.has_warm_start_data()) {
    const auto& ws = solution.get_cpu_pdlp_warm_start_data();
    header->set_ws_initial_primal_weight(static_cast<double>(ws.initial_primal_weight_));
    header->set_ws_initial_step_size(static_cast<double>(ws.initial_step_size_));
    header->set_ws_total_pdlp_iterations(static_cast<int32_t>(ws.total_pdlp_iterations_));
    header->set_ws_total_pdhg_iterations(static_cast<int32_t>(ws.total_pdhg_iterations_));
    header->set_ws_last_candidate_kkt_score(static_cast<double>(ws.last_candidate_kkt_score_));
    header->set_ws_last_restart_kkt_score(static_cast<double>(ws.last_restart_kkt_score_));
    header->set_ws_sum_solution_weight(static_cast<double>(ws.sum_solution_weight_));
    header->set_ws_iterations_since_last_restart(
      static_cast<int32_t>(ws.iterations_since_last_restart_));

    add_result_array_descriptor(header,
                                cuopt::remote::RESULT_WS_CURRENT_PRIMAL,
                                ws.current_primal_solution_.size(),
                                sizeof(double));
    add_result_array_descriptor(header,
                                cuopt::remote::RESULT_WS_CURRENT_DUAL,
                                ws.current_dual_solution_.size(),
                                sizeof(double));
    add_result_array_descriptor(header,
                                cuopt::remote::RESULT_WS_INITIAL_PRIMAL_AVG,
                                ws.initial_primal_average_.size(),
                                sizeof(double));
    add_result_array_descriptor(header,
                                cuopt::remote::RESULT_WS_INITIAL_DUAL_AVG,
                                ws.initial_dual_average_.size(),
                                sizeof(double));
    add_result_array_descriptor(
      header, cuopt::remote::RESULT_WS_CURRENT_ATY, ws.current_ATY_.size(), sizeof(double));
    add_result_array_descriptor(
      header, cuopt::remote::RESULT_WS_SUM_PRIMAL, ws.sum_primal_solutions_.size(), sizeof(double));
    add_result_array_descriptor(
      header, cuopt::remote::RESULT_WS_SUM_DUAL, ws.sum_dual_solutions_.size(), sizeof(double));
    add_result_array_descriptor(header,
                                cuopt::remote::RESULT_WS_LAST_RESTART_GAP_PRIMAL,
                                ws.last_restart_duality_gap_primal_solution_.size(),
                                sizeof(double));
    add_result_array_descriptor(header,
                                cuopt::remote::RESULT_WS_LAST_RESTART_GAP_DUAL,
                                ws.last_restart_duality_gap_dual_solution_.size(),
                                sizeof(double));
  }
}

template <typename i_t, typename f_t>
void populate_chunked_result_header_mip(const cpu_mip_solution_t<i_t, f_t>& solution,
                                        cuopt::remote::ChunkedResultHeader* header)
{
  header->set_is_mip(true);
  header->set_mip_termination_status(to_proto_mip_status(solution.get_termination_status()));
  header->set_mip_error_message(solution.get_error_status().what());
  header->set_mip_objective(solution.get_objective_value());
  header->set_mip_gap(solution.get_mip_gap());
  header->set_solution_bound(solution.get_solution_bound());
  header->set_total_solve_time(solution.get_solve_time());
  header->set_presolve_time(solution.get_presolve_time());
  header->set_max_constraint_violation(solution.get_max_constraint_violation());
  header->set_max_int_violation(solution.get_max_int_violation());
  header->set_max_variable_bound_violation(solution.get_max_variable_bound_violation());
  header->set_nodes(solution.get_num_nodes());
  header->set_simplex_iterations(solution.get_num_simplex_iterations());

  add_result_array_descriptor(header,
                              cuopt::remote::RESULT_MIP_SOLUTION,
                              solution.get_solution_host().size(),
                              sizeof(double));
}

// ============================================================================
// Collect solution arrays as raw bytes
// ============================================================================

template <typename i_t, typename f_t>
std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<i_t, f_t>& solution)
{
  std::map<int32_t, std::vector<uint8_t>> arrays;

  const auto& primal       = solution.get_primal_solution_host();
  const auto& dual         = solution.get_dual_solution_host();
  const auto& reduced_cost = solution.get_reduced_cost_host();

  if (!primal.empty()) { arrays[cuopt::remote::RESULT_PRIMAL_SOLUTION] = doubles_to_bytes(primal); }
  if (!dual.empty()) { arrays[cuopt::remote::RESULT_DUAL_SOLUTION] = doubles_to_bytes(dual); }
  if (!reduced_cost.empty()) {
    arrays[cuopt::remote::RESULT_REDUCED_COST] = doubles_to_bytes(reduced_cost);
  }

  if (solution.has_warm_start_data()) {
    const auto& ws = solution.get_cpu_pdlp_warm_start_data();
    if (!ws.current_primal_solution_.empty()) {
      arrays[cuopt::remote::RESULT_WS_CURRENT_PRIMAL] =
        doubles_to_bytes(ws.current_primal_solution_);
    }
    if (!ws.current_dual_solution_.empty()) {
      arrays[cuopt::remote::RESULT_WS_CURRENT_DUAL] = doubles_to_bytes(ws.current_dual_solution_);
    }
    if (!ws.initial_primal_average_.empty()) {
      arrays[cuopt::remote::RESULT_WS_INITIAL_PRIMAL_AVG] =
        doubles_to_bytes(ws.initial_primal_average_);
    }
    if (!ws.initial_dual_average_.empty()) {
      arrays[cuopt::remote::RESULT_WS_INITIAL_DUAL_AVG] =
        doubles_to_bytes(ws.initial_dual_average_);
    }
    if (!ws.current_ATY_.empty()) {
      arrays[cuopt::remote::RESULT_WS_CURRENT_ATY] = doubles_to_bytes(ws.current_ATY_);
    }
    if (!ws.sum_primal_solutions_.empty()) {
      arrays[cuopt::remote::RESULT_WS_SUM_PRIMAL] = doubles_to_bytes(ws.sum_primal_solutions_);
    }
    if (!ws.sum_dual_solutions_.empty()) {
      arrays[cuopt::remote::RESULT_WS_SUM_DUAL] = doubles_to_bytes(ws.sum_dual_solutions_);
    }
    if (!ws.last_restart_duality_gap_primal_solution_.empty()) {
      arrays[cuopt::remote::RESULT_WS_LAST_RESTART_GAP_PRIMAL] =
        doubles_to_bytes(ws.last_restart_duality_gap_primal_solution_);
    }
    if (!ws.last_restart_duality_gap_dual_solution_.empty()) {
      arrays[cuopt::remote::RESULT_WS_LAST_RESTART_GAP_DUAL] =
        doubles_to_bytes(ws.last_restart_duality_gap_dual_solution_);
    }
  }

  return arrays;
}

template <typename i_t, typename f_t>
std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<i_t, f_t>& solution)
{
  std::map<int32_t, std::vector<uint8_t>> arrays;
  const auto& sol_vec = solution.get_solution_host();
  if (!sol_vec.empty()) { arrays[cuopt::remote::RESULT_MIP_SOLUTION] = doubles_to_bytes(sol_vec); }
  return arrays;
}

// ============================================================================
// MessageStream conversion functions (unified conversion layer)
// ============================================================================

template <typename i_t, typename f_t>
MessageStream lp_solution_to_messages(const cpu_lp_solution_t<i_t, f_t>& sol)
{
  MessageStream stream;
  cuopt::remote::PipeEnvelope hdr_env;
  auto* h = hdr_env.mutable_result_header();

  h->set_is_mip(false);
  h->set_lp_termination_status(to_proto_pdlp_status(sol.get_termination_status()));
  h->set_error_message(sol.get_error_status().what());
  h->set_l2_primal_residual(sol.get_l2_primal_residual());
  h->set_l2_dual_residual(sol.get_l2_dual_residual());
  h->set_primal_objective(sol.get_objective_value());
  h->set_dual_objective(sol.get_dual_objective_value());
  h->set_gap(sol.get_gap());
  h->set_nb_iterations(sol.get_num_iterations());
  h->set_solve_time(sol.get_solve_time());
  h->set_solved_by_pdlp(sol.is_solved_by_pdlp());

  const auto& primal       = sol.get_primal_solution_host();
  const auto& dual         = sol.get_dual_solution_host();
  const auto& reduced_cost = sol.get_reduced_cost_host();

  add_result_array_descriptor(
    h, cuopt::remote::RESULT_PRIMAL_SOLUTION, primal.size(), sizeof(double));
  add_result_array_descriptor(h, cuopt::remote::RESULT_DUAL_SOLUTION, dual.size(), sizeof(double));
  add_result_array_descriptor(
    h, cuopt::remote::RESULT_REDUCED_COST, reduced_cost.size(), sizeof(double));

  if (sol.has_warm_start_data()) {
    const auto& ws = sol.get_cpu_pdlp_warm_start_data();
    h->set_ws_initial_primal_weight(static_cast<double>(ws.initial_primal_weight_));
    h->set_ws_initial_step_size(static_cast<double>(ws.initial_step_size_));
    h->set_ws_total_pdlp_iterations(static_cast<int32_t>(ws.total_pdlp_iterations_));
    h->set_ws_total_pdhg_iterations(static_cast<int32_t>(ws.total_pdhg_iterations_));
    h->set_ws_last_candidate_kkt_score(static_cast<double>(ws.last_candidate_kkt_score_));
    h->set_ws_last_restart_kkt_score(static_cast<double>(ws.last_restart_kkt_score_));
    h->set_ws_sum_solution_weight(static_cast<double>(ws.sum_solution_weight_));
    h->set_ws_iterations_since_last_restart(
      static_cast<int32_t>(ws.iterations_since_last_restart_));

    add_result_array_descriptor(h,
                                cuopt::remote::RESULT_WS_CURRENT_PRIMAL,
                                ws.current_primal_solution_.size(),
                                sizeof(double));
    add_result_array_descriptor(
      h, cuopt::remote::RESULT_WS_CURRENT_DUAL, ws.current_dual_solution_.size(), sizeof(double));
    add_result_array_descriptor(h,
                                cuopt::remote::RESULT_WS_INITIAL_PRIMAL_AVG,
                                ws.initial_primal_average_.size(),
                                sizeof(double));
    add_result_array_descriptor(h,
                                cuopt::remote::RESULT_WS_INITIAL_DUAL_AVG,
                                ws.initial_dual_average_.size(),
                                sizeof(double));
    add_result_array_descriptor(
      h, cuopt::remote::RESULT_WS_CURRENT_ATY, ws.current_ATY_.size(), sizeof(double));
    add_result_array_descriptor(
      h, cuopt::remote::RESULT_WS_SUM_PRIMAL, ws.sum_primal_solutions_.size(), sizeof(double));
    add_result_array_descriptor(
      h, cuopt::remote::RESULT_WS_SUM_DUAL, ws.sum_dual_solutions_.size(), sizeof(double));
    add_result_array_descriptor(h,
                                cuopt::remote::RESULT_WS_LAST_RESTART_GAP_PRIMAL,
                                ws.last_restart_duality_gap_primal_solution_.size(),
                                sizeof(double));
    add_result_array_descriptor(h,
                                cuopt::remote::RESULT_WS_LAST_RESTART_GAP_DUAL,
                                ws.last_restart_duality_gap_dual_solution_.size(),
                                sizeof(double));
  }

  stream.push_back(std::move(hdr_env));

  append_array(stream, cuopt::remote::RESULT_PRIMAL_SOLUTION, primal);
  append_array(stream, cuopt::remote::RESULT_DUAL_SOLUTION, dual);
  append_array(stream, cuopt::remote::RESULT_REDUCED_COST, reduced_cost);

  if (sol.has_warm_start_data()) {
    const auto& ws = sol.get_cpu_pdlp_warm_start_data();
    append_array(stream, cuopt::remote::RESULT_WS_CURRENT_PRIMAL, ws.current_primal_solution_);
    append_array(stream, cuopt::remote::RESULT_WS_CURRENT_DUAL, ws.current_dual_solution_);
    append_array(stream, cuopt::remote::RESULT_WS_INITIAL_PRIMAL_AVG, ws.initial_primal_average_);
    append_array(stream, cuopt::remote::RESULT_WS_INITIAL_DUAL_AVG, ws.initial_dual_average_);
    append_array(stream, cuopt::remote::RESULT_WS_CURRENT_ATY, ws.current_ATY_);
    append_array(stream, cuopt::remote::RESULT_WS_SUM_PRIMAL, ws.sum_primal_solutions_);
    append_array(stream, cuopt::remote::RESULT_WS_SUM_DUAL, ws.sum_dual_solutions_);
    append_array(stream,
                 cuopt::remote::RESULT_WS_LAST_RESTART_GAP_PRIMAL,
                 ws.last_restart_duality_gap_primal_solution_);
    append_array(stream,
                 cuopt::remote::RESULT_WS_LAST_RESTART_GAP_DUAL,
                 ws.last_restart_duality_gap_dual_solution_);
  }

  return stream;
}

template <typename i_t, typename f_t>
MessageStream mip_solution_to_messages(const cpu_mip_solution_t<i_t, f_t>& sol)
{
  MessageStream stream;
  cuopt::remote::PipeEnvelope hdr_env;
  auto* h = hdr_env.mutable_result_header();

  h->set_is_mip(true);
  h->set_mip_termination_status(to_proto_mip_status(sol.get_termination_status()));
  h->set_mip_error_message(sol.get_error_status().what());
  h->set_mip_objective(sol.get_objective_value());
  h->set_mip_gap(sol.get_mip_gap());
  h->set_solution_bound(sol.get_solution_bound());
  h->set_total_solve_time(sol.get_solve_time());
  h->set_presolve_time(sol.get_presolve_time());
  h->set_max_constraint_violation(sol.get_max_constraint_violation());
  h->set_max_int_violation(sol.get_max_int_violation());
  h->set_max_variable_bound_violation(sol.get_max_variable_bound_violation());
  h->set_nodes(sol.get_num_nodes());
  h->set_simplex_iterations(sol.get_num_simplex_iterations());

  const auto& sol_vec = sol.get_solution_host();
  add_result_array_descriptor(
    h, cuopt::remote::RESULT_MIP_SOLUTION, sol_vec.size(), sizeof(double));

  stream.push_back(std::move(hdr_env));

  append_array(stream, cuopt::remote::RESULT_MIP_SOLUTION, sol_vec);

  return stream;
}

template <typename i_t, typename f_t>
cpu_lp_solution_t<i_t, f_t> messages_to_lp_solution(const MessageStream& msgs)
{
  const auto& h = find_result_header(msgs);

  auto primal       = extract_array<f_t>(msgs, cuopt::remote::RESULT_PRIMAL_SOLUTION);
  auto dual         = extract_array<f_t>(msgs, cuopt::remote::RESULT_DUAL_SOLUTION);
  auto reduced_cost = extract_array<f_t>(msgs, cuopt::remote::RESULT_REDUCED_COST);

  auto status   = from_proto_pdlp_status(h.lp_termination_status());
  auto obj      = static_cast<f_t>(h.primal_objective());
  auto dual_obj = static_cast<f_t>(h.dual_objective());
  auto solve_t  = h.solve_time();
  auto l2_pr    = static_cast<f_t>(h.l2_primal_residual());
  auto l2_dr    = static_cast<f_t>(h.l2_dual_residual());
  auto g        = static_cast<f_t>(h.gap());
  auto iters    = static_cast<i_t>(h.nb_iterations());
  auto by_pdlp  = h.solved_by_pdlp();

  auto ws_primal = extract_array<f_t>(msgs, cuopt::remote::RESULT_WS_CURRENT_PRIMAL);
  if (!ws_primal.empty()) {
    cpu_pdlp_warm_start_data_t<i_t, f_t> ws;
    ws.current_primal_solution_ = std::move(ws_primal);
    ws.current_dual_solution_   = extract_array<f_t>(msgs, cuopt::remote::RESULT_WS_CURRENT_DUAL);
    ws.initial_primal_average_ =
      extract_array<f_t>(msgs, cuopt::remote::RESULT_WS_INITIAL_PRIMAL_AVG);
    ws.initial_dual_average_ = extract_array<f_t>(msgs, cuopt::remote::RESULT_WS_INITIAL_DUAL_AVG);
    ws.current_ATY_          = extract_array<f_t>(msgs, cuopt::remote::RESULT_WS_CURRENT_ATY);
    ws.sum_primal_solutions_ = extract_array<f_t>(msgs, cuopt::remote::RESULT_WS_SUM_PRIMAL);
    ws.sum_dual_solutions_   = extract_array<f_t>(msgs, cuopt::remote::RESULT_WS_SUM_DUAL);
    ws.last_restart_duality_gap_primal_solution_ =
      extract_array<f_t>(msgs, cuopt::remote::RESULT_WS_LAST_RESTART_GAP_PRIMAL);
    ws.last_restart_duality_gap_dual_solution_ =
      extract_array<f_t>(msgs, cuopt::remote::RESULT_WS_LAST_RESTART_GAP_DUAL);

    ws.initial_primal_weight_         = static_cast<f_t>(h.ws_initial_primal_weight());
    ws.initial_step_size_             = static_cast<f_t>(h.ws_initial_step_size());
    ws.total_pdlp_iterations_         = static_cast<i_t>(h.ws_total_pdlp_iterations());
    ws.total_pdhg_iterations_         = static_cast<i_t>(h.ws_total_pdhg_iterations());
    ws.last_candidate_kkt_score_      = static_cast<f_t>(h.ws_last_candidate_kkt_score());
    ws.last_restart_kkt_score_        = static_cast<f_t>(h.ws_last_restart_kkt_score());
    ws.sum_solution_weight_           = static_cast<f_t>(h.ws_sum_solution_weight());
    ws.iterations_since_last_restart_ = static_cast<i_t>(h.ws_iterations_since_last_restart());

    return cpu_lp_solution_t<i_t, f_t>(std::move(primal),
                                       std::move(dual),
                                       std::move(reduced_cost),
                                       status,
                                       obj,
                                       dual_obj,
                                       solve_t,
                                       l2_pr,
                                       l2_dr,
                                       g,
                                       iters,
                                       by_pdlp,
                                       std::move(ws));
  }

  return cpu_lp_solution_t<i_t, f_t>(std::move(primal),
                                     std::move(dual),
                                     std::move(reduced_cost),
                                     status,
                                     obj,
                                     dual_obj,
                                     solve_t,
                                     l2_pr,
                                     l2_dr,
                                     g,
                                     iters,
                                     by_pdlp);
}

template <typename i_t, typename f_t>
cpu_mip_solution_t<i_t, f_t> messages_to_mip_solution(const MessageStream& msgs)
{
  const auto& h = find_result_header(msgs);

  auto sol_vec = extract_array<f_t>(msgs, cuopt::remote::RESULT_MIP_SOLUTION);

  return cpu_mip_solution_t<i_t, f_t>(std::move(sol_vec),
                                      from_proto_mip_status(h.mip_termination_status()),
                                      static_cast<f_t>(h.mip_objective()),
                                      static_cast<f_t>(h.mip_gap()),
                                      static_cast<f_t>(h.solution_bound()),
                                      h.total_solve_time(),
                                      h.presolve_time(),
                                      static_cast<f_t>(h.max_constraint_violation()),
                                      static_cast<f_t>(h.max_int_violation()),
                                      static_cast<f_t>(h.max_variable_bound_violation()),
                                      static_cast<i_t>(h.nodes()),
                                      static_cast<i_t>(h.simplex_iterations()));
}

// ============================================================================
// Transport adaptors: MessageStream -> gRPC protobuf
// ============================================================================

namespace {
void copy_doubles_to_repeated(const MessageStream& msgs,
                              int32_t field_id,
                              google::protobuf::RepeatedField<double>* out)
{
  for (const auto& env : msgs) {
    if (env.has_array_data() && env.array_data().field_id() == field_id) {
      const auto& raw = env.array_data().data();
      if (raw.empty()) return;
      size_t n = raw.size() / sizeof(double);
      out->Resize(static_cast<int>(n), 0.0);
      std::memcpy(out->mutable_data(), raw.data(), raw.size());
      return;
    }
  }
}
}  // namespace

template <typename i_t, typename f_t>
void stream_to_lp_solution_proto(const MessageStream& msgs, cuopt::remote::LPSolution* proto)
{
  const auto& h = find_result_header(msgs);

  proto->set_termination_status(h.lp_termination_status());
  proto->set_error_message(h.error_message());
  proto->set_l2_primal_residual(h.l2_primal_residual());
  proto->set_l2_dual_residual(h.l2_dual_residual());
  proto->set_primal_objective(h.primal_objective());
  proto->set_dual_objective(h.dual_objective());
  proto->set_gap(h.gap());
  proto->set_nb_iterations(h.nb_iterations());
  proto->set_solve_time(h.solve_time());
  proto->set_solved_by_pdlp(h.solved_by_pdlp());

  copy_doubles_to_repeated(
    msgs, cuopt::remote::RESULT_PRIMAL_SOLUTION, proto->mutable_primal_solution());
  copy_doubles_to_repeated(
    msgs, cuopt::remote::RESULT_DUAL_SOLUTION, proto->mutable_dual_solution());
  copy_doubles_to_repeated(msgs, cuopt::remote::RESULT_REDUCED_COST, proto->mutable_reduced_cost());

  auto ws_primal = extract_array<double>(msgs, cuopt::remote::RESULT_WS_CURRENT_PRIMAL);
  if (!ws_primal.empty()) {
    auto* ws = proto->mutable_warm_start_data();
    ws->set_initial_primal_weight(h.ws_initial_primal_weight());
    ws->set_initial_step_size(h.ws_initial_step_size());
    ws->set_total_pdlp_iterations(h.ws_total_pdlp_iterations());
    ws->set_total_pdhg_iterations(h.ws_total_pdhg_iterations());
    ws->set_last_candidate_kkt_score(h.ws_last_candidate_kkt_score());
    ws->set_last_restart_kkt_score(h.ws_last_restart_kkt_score());
    ws->set_sum_solution_weight(h.ws_sum_solution_weight());
    ws->set_iterations_since_last_restart(h.ws_iterations_since_last_restart());

    copy_doubles_to_repeated(
      msgs, cuopt::remote::RESULT_WS_CURRENT_PRIMAL, ws->mutable_current_primal_solution());
    copy_doubles_to_repeated(
      msgs, cuopt::remote::RESULT_WS_CURRENT_DUAL, ws->mutable_current_dual_solution());
    copy_doubles_to_repeated(
      msgs, cuopt::remote::RESULT_WS_INITIAL_PRIMAL_AVG, ws->mutable_initial_primal_average());
    copy_doubles_to_repeated(
      msgs, cuopt::remote::RESULT_WS_INITIAL_DUAL_AVG, ws->mutable_initial_dual_average());
    copy_doubles_to_repeated(msgs, cuopt::remote::RESULT_WS_CURRENT_ATY, ws->mutable_current_aty());
    copy_doubles_to_repeated(
      msgs, cuopt::remote::RESULT_WS_SUM_PRIMAL, ws->mutable_sum_primal_solutions());
    copy_doubles_to_repeated(
      msgs, cuopt::remote::RESULT_WS_SUM_DUAL, ws->mutable_sum_dual_solutions());
    copy_doubles_to_repeated(msgs,
                             cuopt::remote::RESULT_WS_LAST_RESTART_GAP_PRIMAL,
                             ws->mutable_last_restart_duality_gap_primal_solution());
    copy_doubles_to_repeated(msgs,
                             cuopt::remote::RESULT_WS_LAST_RESTART_GAP_DUAL,
                             ws->mutable_last_restart_duality_gap_dual_solution());
  }
}

template <typename i_t, typename f_t>
void stream_to_mip_solution_proto(const MessageStream& msgs, cuopt::remote::MIPSolution* proto)
{
  const auto& h = find_result_header(msgs);

  proto->set_termination_status(h.mip_termination_status());
  proto->set_error_message(h.mip_error_message());
  proto->set_objective(h.mip_objective());
  proto->set_mip_gap(h.mip_gap());
  proto->set_solution_bound(h.solution_bound());
  proto->set_total_solve_time(h.total_solve_time());
  proto->set_presolve_time(h.presolve_time());
  proto->set_max_constraint_violation(h.max_constraint_violation());
  proto->set_max_int_violation(h.max_int_violation());
  proto->set_max_variable_bound_violation(h.max_variable_bound_violation());
  proto->set_nodes(h.nodes());
  proto->set_simplex_iterations(h.simplex_iterations());

  copy_doubles_to_repeated(msgs, cuopt::remote::RESULT_MIP_SOLUTION, proto->mutable_solution());
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template void map_lp_solution_to_proto(const cpu_lp_solution_t<int32_t, float>& solution,
                                       cuopt::remote::LPSolution* pb_solution);
template cpu_lp_solution_t<int32_t, float> map_proto_to_lp_solution(
  const cuopt::remote::LPSolution& pb_solution);
template void map_mip_solution_to_proto(const cpu_mip_solution_t<int32_t, float>& solution,
                                        cuopt::remote::MIPSolution* pb_solution);
template cpu_mip_solution_t<int32_t, float> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution);
template size_t estimate_lp_solution_proto_size(const cpu_lp_solution_t<int32_t, float>& solution);
template size_t estimate_mip_solution_proto_size(
  const cpu_mip_solution_t<int32_t, float>& solution);
template void populate_chunked_result_header_lp(const cpu_lp_solution_t<int32_t, float>& solution,
                                                cuopt::remote::ChunkedResultHeader* header);
template void populate_chunked_result_header_mip(const cpu_mip_solution_t<int32_t, float>& solution,
                                                 cuopt::remote::ChunkedResultHeader* header);
template std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<int32_t, float>& solution);
template std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<int32_t, float>& solution);
template MessageStream lp_solution_to_messages(const cpu_lp_solution_t<int32_t, float>& solution);
template MessageStream mip_solution_to_messages(const cpu_mip_solution_t<int32_t, float>& solution);
template cpu_lp_solution_t<int32_t, float> messages_to_lp_solution(const MessageStream& messages);
template cpu_mip_solution_t<int32_t, float> messages_to_mip_solution(const MessageStream& messages);
template void stream_to_lp_solution_proto<int32_t, float>(const MessageStream& messages,
                                                          cuopt::remote::LPSolution* proto);
template void stream_to_mip_solution_proto<int32_t, float>(const MessageStream& messages,
                                                           cuopt::remote::MIPSolution* proto);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template void map_lp_solution_to_proto(const cpu_lp_solution_t<int32_t, double>& solution,
                                       cuopt::remote::LPSolution* pb_solution);
template cpu_lp_solution_t<int32_t, double> map_proto_to_lp_solution(
  const cuopt::remote::LPSolution& pb_solution);
template void map_mip_solution_to_proto(const cpu_mip_solution_t<int32_t, double>& solution,
                                        cuopt::remote::MIPSolution* pb_solution);
template cpu_mip_solution_t<int32_t, double> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution);
template size_t estimate_lp_solution_proto_size(const cpu_lp_solution_t<int32_t, double>& solution);
template size_t estimate_mip_solution_proto_size(
  const cpu_mip_solution_t<int32_t, double>& solution);
template void populate_chunked_result_header_lp(const cpu_lp_solution_t<int32_t, double>& solution,
                                                cuopt::remote::ChunkedResultHeader* header);
template void populate_chunked_result_header_mip(
  const cpu_mip_solution_t<int32_t, double>& solution, cuopt::remote::ChunkedResultHeader* header);
template std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<int32_t, double>& solution);
template std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<int32_t, double>& solution);
template MessageStream lp_solution_to_messages(const cpu_lp_solution_t<int32_t, double>& solution);
template MessageStream mip_solution_to_messages(
  const cpu_mip_solution_t<int32_t, double>& solution);
template cpu_lp_solution_t<int32_t, double> messages_to_lp_solution(const MessageStream& messages);
template cpu_mip_solution_t<int32_t, double> messages_to_mip_solution(
  const MessageStream& messages);
template void stream_to_lp_solution_proto<int32_t, double>(const MessageStream& messages,
                                                           cuopt::remote::LPSolution* proto);
template void stream_to_mip_solution_proto<int32_t, double>(const MessageStream& messages,
                                                            cuopt::remote::MIPSolution* proto);
#endif

}  // namespace cuopt::linear_programming
