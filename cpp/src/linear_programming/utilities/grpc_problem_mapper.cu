/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/utilities/grpc_problem_mapper.hpp>

#include <cuopt/linear_programming/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>

#include <limits>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
void map_problem_to_proto(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                          cuopt::remote::OptimizationProblem* pb_problem)
{
  // Basic problem metadata
  pb_problem->set_problem_name(cpu_problem.get_problem_name());
  pb_problem->set_objective_name(cpu_problem.get_objective_name());
  pb_problem->set_maximize(cpu_problem.get_sense());
  pb_problem->set_objective_scaling_factor(cpu_problem.get_objective_scaling_factor());
  pb_problem->set_objective_offset(cpu_problem.get_objective_offset());

  // Get constraint matrix data from host memory
  auto values  = cpu_problem.get_constraint_matrix_values_host();
  auto indices = cpu_problem.get_constraint_matrix_indices_host();
  auto offsets = cpu_problem.get_constraint_matrix_offsets_host();

  // Constraint matrix A in CSR format
  for (const auto& val : values) {
    pb_problem->add_a(static_cast<double>(val));
  }
  for (const auto& idx : indices) {
    pb_problem->add_a_indices(static_cast<int32_t>(idx));
  }
  for (const auto& off : offsets) {
    pb_problem->add_a_offsets(static_cast<int32_t>(off));
  }

  // Objective coefficients
  auto obj_coeffs = cpu_problem.get_objective_coefficients_host();
  for (const auto& c : obj_coeffs) {
    pb_problem->add_c(static_cast<double>(c));
  }

  // Variable bounds
  auto var_lb = cpu_problem.get_variable_lower_bounds_host();
  auto var_ub = cpu_problem.get_variable_upper_bounds_host();
  for (const auto& lb : var_lb) {
    pb_problem->add_variable_lower_bounds(static_cast<double>(lb));
  }
  for (const auto& ub : var_ub) {
    pb_problem->add_variable_upper_bounds(static_cast<double>(ub));
  }

  // Constraint bounds
  auto con_lb = cpu_problem.get_constraint_lower_bounds_host();
  auto con_ub = cpu_problem.get_constraint_upper_bounds_host();

  if (!con_lb.empty() && !con_ub.empty()) {
    for (const auto& lb : con_lb) {
      pb_problem->add_constraint_lower_bounds(static_cast<double>(lb));
    }
    for (const auto& ub : con_ub) {
      pb_problem->add_constraint_upper_bounds(static_cast<double>(ub));
    }
  }

  // Row types (if available)
  auto row_types = cpu_problem.get_row_types_host();
  if (!row_types.empty()) {
    pb_problem->set_row_types(std::string(row_types.begin(), row_types.end()));
  }

  // Constraint bounds (RHS) - if available
  auto b = cpu_problem.get_constraint_bounds_host();
  if (!b.empty()) {
    for (const auto& rhs : b) {
      pb_problem->add_b(static_cast<double>(rhs));
    }
  }

  // Variable names
  const auto& var_names = cpu_problem.get_variable_names();
  for (const auto& name : var_names) {
    pb_problem->add_variable_names(name);
  }

  // Row names
  const auto& row_names = cpu_problem.get_row_names();
  for (const auto& name : row_names) {
    pb_problem->add_row_names(name);
  }

  // Variable types (for MIP problems)
  auto var_types = cpu_problem.get_variable_types_host();
  if (!var_types.empty()) {
    // Convert var_t enum to char representation
    std::string var_types_str;
    var_types_str.reserve(var_types.size());
    for (const auto& vt : var_types) {
      switch (vt) {
        case var_t::CONTINUOUS: var_types_str.push_back('C'); break;
        case var_t::INTEGER: var_types_str.push_back('I'); break;
        default: var_types_str.push_back('C'); break;
      }
    }
    pb_problem->set_variable_types(var_types_str);
  }

  // Quadratic objective matrix Q (for QPS problems)
  if (cpu_problem.has_quadratic_objective()) {
    const auto& q_values  = cpu_problem.get_quadratic_objective_values();
    const auto& q_indices = cpu_problem.get_quadratic_objective_indices();
    const auto& q_offsets = cpu_problem.get_quadratic_objective_offsets();

    for (const auto& val : q_values) {
      pb_problem->add_q_values(static_cast<double>(val));
    }
    for (const auto& idx : q_indices) {
      pb_problem->add_q_indices(static_cast<int32_t>(idx));
    }
    for (const auto& off : q_offsets) {
      pb_problem->add_q_offsets(static_cast<int32_t>(off));
    }
  }
}

template <typename i_t, typename f_t>
void map_proto_to_problem(const cuopt::remote::OptimizationProblem& pb_problem,
                          cpu_optimization_problem_t<i_t, f_t>& cpu_problem)
{
  // Basic problem metadata
  cpu_problem.set_problem_name(pb_problem.problem_name());
  cpu_problem.set_objective_name(pb_problem.objective_name());
  cpu_problem.set_maximize(pb_problem.maximize());
  cpu_problem.set_objective_scaling_factor(pb_problem.objective_scaling_factor());
  cpu_problem.set_objective_offset(pb_problem.objective_offset());

  // Constraint matrix A in CSR format
  std::vector<f_t> values(pb_problem.a().begin(), pb_problem.a().end());
  std::vector<i_t> indices(pb_problem.a_indices().begin(), pb_problem.a_indices().end());
  std::vector<i_t> offsets(pb_problem.a_offsets().begin(), pb_problem.a_offsets().end());

  cpu_problem.set_csr_constraint_matrix(values.data(),
                                        static_cast<i_t>(values.size()),
                                        indices.data(),
                                        static_cast<i_t>(indices.size()),
                                        offsets.data(),
                                        static_cast<i_t>(offsets.size()));

  // Objective coefficients
  std::vector<f_t> obj(pb_problem.c().begin(), pb_problem.c().end());
  cpu_problem.set_objective_coefficients(obj.data(), static_cast<i_t>(obj.size()));

  // Variable bounds
  std::vector<f_t> var_lb(pb_problem.variable_lower_bounds().begin(),
                          pb_problem.variable_lower_bounds().end());
  std::vector<f_t> var_ub(pb_problem.variable_upper_bounds().begin(),
                          pb_problem.variable_upper_bounds().end());
  cpu_problem.set_variable_lower_bounds(var_lb.data(), static_cast<i_t>(var_lb.size()));
  cpu_problem.set_variable_upper_bounds(var_ub.data(), static_cast<i_t>(var_ub.size()));

  // Constraint bounds (prefer lower/upper bounds if available)
  if (pb_problem.constraint_lower_bounds_size() > 0) {
    std::vector<f_t> con_lb(pb_problem.constraint_lower_bounds().begin(),
                            pb_problem.constraint_lower_bounds().end());
    std::vector<f_t> con_ub(pb_problem.constraint_upper_bounds().begin(),
                            pb_problem.constraint_upper_bounds().end());
    cpu_problem.set_constraint_lower_bounds(con_lb.data(), static_cast<i_t>(con_lb.size()));
    cpu_problem.set_constraint_upper_bounds(con_ub.data(), static_cast<i_t>(con_ub.size()));
  } else if (pb_problem.b_size() > 0) {
    // Use b (RHS) + row_types format
    std::vector<f_t> b(pb_problem.b().begin(), pb_problem.b().end());
    cpu_problem.set_constraint_bounds(b.data(), static_cast<i_t>(b.size()));

    if (!pb_problem.row_types().empty()) {
      const std::string& row_types_str = pb_problem.row_types();
      cpu_problem.set_row_types(row_types_str.data(), static_cast<i_t>(row_types_str.size()));
    }
  }

  // Variable names
  if (pb_problem.variable_names_size() > 0) {
    std::vector<std::string> var_names(pb_problem.variable_names().begin(),
                                       pb_problem.variable_names().end());
    cpu_problem.set_variable_names(var_names);
  }

  // Row names
  if (pb_problem.row_names_size() > 0) {
    std::vector<std::string> row_names(pb_problem.row_names().begin(),
                                       pb_problem.row_names().end());
    cpu_problem.set_row_names(row_names);
  }

  // Variable types
  if (!pb_problem.variable_types().empty()) {
    const std::string& var_types_str = pb_problem.variable_types();
    // Convert char representation to var_t enum
    std::vector<var_t> var_types;
    var_types.reserve(var_types_str.size());
    for (char c : var_types_str) {
      switch (c) {
        case 'C': var_types.push_back(var_t::CONTINUOUS); break;
        case 'I':
        case 'B': var_types.push_back(var_t::INTEGER); break;  // Binary treated as integer
        default: var_types.push_back(var_t::CONTINUOUS); break;
      }
    }
    cpu_problem.set_variable_types(var_types.data(), static_cast<i_t>(var_types.size()));
  }

  // Quadratic objective matrix Q (for QPS problems)
  if (pb_problem.q_values_size() > 0) {
    std::vector<f_t> q_values(pb_problem.q_values().begin(), pb_problem.q_values().end());
    std::vector<i_t> q_indices(pb_problem.q_indices().begin(), pb_problem.q_indices().end());
    std::vector<i_t> q_offsets(pb_problem.q_offsets().begin(), pb_problem.q_offsets().end());

    cpu_problem.set_quadratic_objective_matrix(q_values.data(),
                                               static_cast<i_t>(q_values.size()),
                                               q_indices.data(),
                                               static_cast<i_t>(q_indices.size()),
                                               q_offsets.data(),
                                               static_cast<i_t>(q_offsets.size()));
  }

  // Infer problem category from variable types
  if (!pb_problem.variable_types().empty()) {
    const std::string& var_types_str = pb_problem.variable_types();
    bool has_integers                = false;
    for (char c : var_types_str) {
      if (c == 'I' || c == 'B') {
        has_integers = true;
        break;
      }
    }
    cpu_problem.set_problem_category(has_integers ? problem_category_t::MIP
                                                  : problem_category_t::LP);
  } else {
    cpu_problem.set_problem_category(problem_category_t::LP);
  }
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template void map_problem_to_proto(const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
                                   cuopt::remote::OptimizationProblem* pb_problem);
template void map_proto_to_problem(const cuopt::remote::OptimizationProblem& pb_problem,
                                   cpu_optimization_problem_t<int32_t, float>& cpu_problem);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template void map_problem_to_proto(const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
                                   cuopt::remote::OptimizationProblem* pb_problem);
template void map_proto_to_problem(const cuopt::remote::OptimizationProblem& pb_problem,
                                   cpu_optimization_problem_t<int32_t, double>& cpu_problem);
#endif

}  // namespace cuopt::linear_programming
