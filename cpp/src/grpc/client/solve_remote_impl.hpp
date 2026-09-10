/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/cpu_optimization_problem.hpp>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>

namespace cuopt::mathematical_optimization {

/**
 * @brief Whether a feature combination the remote server cannot honour must be dropped.
 *
 * Some client-side requests cannot be forwarded to cuopt_grpc_server. Rather than failing
 * the solve, solve_mip_remote() drops the unsupported part and warns. This predicate is that
 * decision, kept out of the RPC plumbing so it is unit-testable without a live connection.
 *
 * Takes the problem and settings themselves rather than pre-extracted fields, so new rules
 * can consult anything either object exposes without changing this signature or adding
 * branches at the call site.
 *
 * Currently one rule: MIP get/set callbacks are not supported for semi-continuous models.
 *
 * @param problem  The problem being submitted.
 * @param settings The MIP settings the caller configured.
 * @return true when the unsupported request (today, the callbacks) must be dropped.
 */
template <typename i_t, typename f_t>
bool should_disable_unsupported(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                const mip_solver_settings_t<i_t, f_t>& settings);

/**
 * @brief Whether MIP/PDLP settings hold an initial solution that gRPC does not serialize.
 *
 * map_mip_settings_to_proto / map_pdlp_settings_to_proto omit these device arrays.
 * A remote solve therefore silently drops a start that only lives on settings.
 * solve_lp_remote / solve_mip_remote log a warning when this is true.
 */
template <typename i_t, typename f_t>
inline bool mip_settings_have_unsent_initial_solutions(
  const mip_solver_settings_t<i_t, f_t>& settings)
{
  return !settings.initial_solutions.empty();
}

template <typename i_t, typename f_t>
inline bool pdlp_settings_have_unsent_initial_solutions(
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  return settings.has_initial_primal_solution() || settings.has_initial_dual_solution();
}

}  // namespace cuopt::mathematical_optimization
