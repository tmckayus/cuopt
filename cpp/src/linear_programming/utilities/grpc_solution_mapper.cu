/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/utilities/grpc_solution_mapper.hpp>

#include <cuopt/linear_programming/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>

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

  // TODO: Add warmstart data support
  // if (solution.has_warm_start_data()) {
  //   auto* pb_warmstart = pb_solution->mutable_warm_start_data();
  //   const auto& ws_data = solution.get_cpu_pdlp_warm_start_data();
  //   // Map warmstart data fields...
  // }
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

  // Create CPU solution with complete termination info
  return cpu_lp_solution_t<i_t, f_t>(std::move(primal),
                                     std::move(dual),
                                     std::move(reduced_cost),
                                     from_proto_pdlp_status(pb_solution.termination_status()),
                                     static_cast<f_t>(pb_solution.primal_objective()),
                                     static_cast<f_t>(pb_solution.dual_objective()),
                                     pb_solution.solve_time(),
                                     static_cast<f_t>(pb_solution.l2_primal_residual()),
                                     static_cast<f_t>(pb_solution.l2_dual_residual()),
                                     static_cast<f_t>(pb_solution.gap()),
                                     static_cast<i_t>(pb_solution.nb_iterations()),
                                     pb_solution.solved_by_pdlp());

  // TODO: Add warmstart data support
  // if (pb_solution.has_warm_start_data()) {
  //   const auto& pb_ws = pb_solution.warm_start_data();
  //   // Extract warmstart data and create solution with warmstart...
  // }
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
#endif

}  // namespace cuopt::linear_programming
