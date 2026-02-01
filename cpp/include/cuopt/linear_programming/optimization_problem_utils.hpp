/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <mps_parser/data_model_view.hpp>
#include <mps_parser/mps_data_model.hpp>

namespace cuopt::linear_programming {

/**
 * @brief Helper function to populate optimization_problem_interface_t from mps_data_model_t
 *
 * This avoids creating a temporary optimization_problem_t which requires GPU memory allocation.
 * Instead, it directly populates the interface which can use either CPU or GPU memory.
 *
 * @tparam i_t Integer type for indices
 * @tparam f_t Floating point type for values
 * @param[out] problem The optimization problem interface to populate
 * @param[in] data_model The MPS data model containing the problem data
 */
template <typename i_t, typename f_t>
void populate_from_mps_data_model(optimization_problem_interface_t<i_t, f_t>* problem,
                                  const mps_parser::mps_data_model_t<i_t, f_t>& data_model)
{
  // Set scalar values
  problem->set_maximize(data_model.get_sense());
  problem->set_objective_scaling_factor(data_model.get_objective_scaling_factor());
  problem->set_objective_offset(data_model.get_objective_offset());

  // Set string values
  if (!data_model.get_objective_name().empty())
    problem->set_objective_name(data_model.get_objective_name());
  if (!data_model.get_problem_name().empty())
    problem->set_problem_name(data_model.get_problem_name());
  if (!data_model.get_variable_names().empty())
    problem->set_variable_names(data_model.get_variable_names());
  if (!data_model.get_row_names().empty()) problem->set_row_names(data_model.get_row_names());

  // Set array values
  i_t n_vars        = data_model.get_n_variables();
  i_t n_constraints = data_model.get_n_constraints();

  const auto& obj_coeffs = data_model.get_objective_coefficients();
  if (!obj_coeffs.empty()) { problem->set_objective_coefficients(obj_coeffs.data(), n_vars); }

  const auto& A_offsets = data_model.get_constraint_matrix_offsets();
  if (!A_offsets.empty() && A_offsets.size() > static_cast<size_t>(n_constraints)) {
    i_t n_nonzeros = A_offsets[n_constraints];
    if (n_nonzeros > 0) {
      problem->set_csr_constraint_matrix(data_model.get_constraint_matrix_values().data(),
                                         n_nonzeros,
                                         data_model.get_constraint_matrix_indices().data(),
                                         n_nonzeros,
                                         A_offsets.data(),
                                         n_constraints + 1);
    }
  }

  const auto& con_bounds = data_model.get_constraint_bounds();
  if (!con_bounds.empty()) { problem->set_constraint_bounds(con_bounds.data(), n_constraints); }
  const auto& con_lb = data_model.get_constraint_lower_bounds();
  if (!con_lb.empty()) { problem->set_constraint_lower_bounds(con_lb.data(), n_constraints); }

  const auto& con_ub = data_model.get_constraint_upper_bounds();
  if (!con_ub.empty()) { problem->set_constraint_upper_bounds(con_ub.data(), n_constraints); }

  const auto& row_types = data_model.get_row_types();
  if (!row_types.empty()) { problem->set_row_types(row_types.data(), n_constraints); }

  const auto& var_lb = data_model.get_variable_lower_bounds();
  if (!var_lb.empty()) { problem->set_variable_lower_bounds(var_lb.data(), n_vars); }

  const auto& var_ub = data_model.get_variable_upper_bounds();
  if (!var_ub.empty()) { problem->set_variable_upper_bounds(var_ub.data(), n_vars); }

  // Convert variable types from char to enum
  const auto& char_variable_types = data_model.get_variable_types();
  if (!char_variable_types.empty()) {
    std::vector<var_t> enum_variable_types(char_variable_types.size());
    for (size_t i = 0; i < char_variable_types.size(); ++i) {
      enum_variable_types[i] = (char_variable_types[i] == 'I' || char_variable_types[i] == 'B')
                                 ? var_t::INTEGER
                                 : var_t::CONTINUOUS;
    }
    problem->set_variable_types(enum_variable_types.data(), n_vars);
  }

  // Set problem category
  bool has_integers = false;
  if (!char_variable_types.empty()) {
    for (const auto& vt : char_variable_types) {
      if (vt == 'I' || vt == 'B') {
        has_integers = true;
        break;
      }
    }
  }

  if (has_integers) {
    problem->set_problem_category(problem_category_t::MIP);
  } else {
    problem->set_problem_category(problem_category_t::LP);
  }

  // Handle quadratic objective if present
  if (data_model.has_quadratic_objective()) {
    i_t q_nonzeros = data_model.get_quadratic_objective_offsets()[n_vars];
    problem->set_quadratic_objective_matrix(data_model.get_quadratic_objective_values().data(),
                                            q_nonzeros,
                                            data_model.get_quadratic_objective_indices().data(),
                                            q_nonzeros,
                                            data_model.get_quadratic_objective_offsets().data(),
                                            n_vars + 1);
  }
}

/**
 * @brief Helper function to populate optimization_problem_interface_t from data_model_view_t
 *
 * This is used by the Python Cython interface which provides data_model_view_t.
 * Similar to populate_from_mps_data_model but works with data_model_view_t instead.
 *
 * @tparam i_t Integer type for indices
 * @tparam f_t Floating point type for values
 * @param[out] problem The optimization problem interface to populate
 * @param[in] data_model The data model view containing the problem data
 */
template <typename i_t, typename f_t>
void populate_from_data_model_view(optimization_problem_interface_t<i_t, f_t>* problem,
                                   cuopt::mps_parser::data_model_view_t<i_t, f_t>* data_model)
{
  problem->set_maximize(data_model->get_sense());

  if (data_model->get_constraint_matrix_values().size() != 0 &&
      data_model->get_constraint_matrix_indices().size() != 0 &&
      data_model->get_constraint_matrix_offsets().size() != 0) {
    problem->set_csr_constraint_matrix(data_model->get_constraint_matrix_values().data(),
                                       data_model->get_constraint_matrix_values().size(),
                                       data_model->get_constraint_matrix_indices().data(),
                                       data_model->get_constraint_matrix_indices().size(),
                                       data_model->get_constraint_matrix_offsets().data(),
                                       data_model->get_constraint_matrix_offsets().size());
  }

  if (data_model->get_constraint_bounds().size() != 0) {
    problem->set_constraint_bounds(data_model->get_constraint_bounds().data(),
                                   data_model->get_constraint_bounds().size());
  }

  if (data_model->get_objective_coefficients().size() != 0) {
    problem->set_objective_coefficients(data_model->get_objective_coefficients().data(),
                                        data_model->get_objective_coefficients().size());
  }

  problem->set_objective_scaling_factor(data_model->get_objective_scaling_factor());
  problem->set_objective_offset(data_model->get_objective_offset());

  if (data_model->get_quadratic_objective_values().size() != 0 &&
      data_model->get_quadratic_objective_indices().size() != 0 &&
      data_model->get_quadratic_objective_offsets().size() != 0) {
    problem->set_quadratic_objective_matrix(data_model->get_quadratic_objective_values().data(),
                                            data_model->get_quadratic_objective_values().size(),
                                            data_model->get_quadratic_objective_indices().data(),
                                            data_model->get_quadratic_objective_indices().size(),
                                            data_model->get_quadratic_objective_offsets().data(),
                                            data_model->get_quadratic_objective_offsets().size());
  }

  if (data_model->get_variable_lower_bounds().size() != 0) {
    problem->set_variable_lower_bounds(data_model->get_variable_lower_bounds().data(),
                                       data_model->get_variable_lower_bounds().size());
  }

  if (data_model->get_variable_upper_bounds().size() != 0) {
    problem->set_variable_upper_bounds(data_model->get_variable_upper_bounds().data(),
                                       data_model->get_variable_upper_bounds().size());
  }

  if (data_model->get_row_types().size() != 0) {
    problem->set_row_types(data_model->get_row_types().data(), data_model->get_row_types().size());
  }

  if (data_model->get_constraint_lower_bounds().size() != 0) {
    problem->set_constraint_lower_bounds(data_model->get_constraint_lower_bounds().data(),
                                         data_model->get_constraint_lower_bounds().size());
  }

  if (data_model->get_constraint_upper_bounds().size() != 0) {
    problem->set_constraint_upper_bounds(data_model->get_constraint_upper_bounds().data(),
                                         data_model->get_constraint_upper_bounds().size());
  }

  if (data_model->get_variable_types().size() != 0) {
    std::vector<var_t> enum_variable_types(data_model->get_variable_types().size());
    std::transform(
      data_model->get_variable_types().data(),
      data_model->get_variable_types().data() + data_model->get_variable_types().size(),
      enum_variable_types.begin(),
      [](const auto val) -> var_t {
        return (val == 'I' || val == 'B') ? var_t::INTEGER : var_t::CONTINUOUS;
      });
    problem->set_variable_types(enum_variable_types.data(), enum_variable_types.size());
  }

  // Set problem category based on variable types
  bool has_integers = false;
  if (data_model->get_variable_types().size() != 0) {
    for (size_t i = 0; i < data_model->get_variable_types().size(); ++i) {
      char vt = data_model->get_variable_types().data()[i];
      if (vt == 'I' || vt == 'B') {
        has_integers = true;
        break;
      }
    }
  }

  if (has_integers) {
    problem->set_problem_category(problem_category_t::MIP);
  } else {
    problem->set_problem_category(problem_category_t::LP);
  }

  if (data_model->get_variable_names().size() != 0) {
    problem->set_variable_names(data_model->get_variable_names());
  }

  if (data_model->get_row_names().size() != 0) {
    problem->set_row_names(data_model->get_row_names());
  }
}

}  // namespace cuopt::linear_programming
