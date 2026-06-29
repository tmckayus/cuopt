/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/cuopt_c.h>
#include <cuopt/mathematical_optimization/optimization_problem_interface.hpp>
#include <cuopt/mathematical_optimization/optimization_problem_utils.hpp>
#include <pdlp/cuopt_c_internal.hpp>

#include <string>
#include <vector>

using namespace cuopt::mathematical_optimization;

namespace {

problem_and_stream_view_t* as_problem(cuOptOptimizationProblem problem)
{
  return static_cast<problem_and_stream_view_t*>(problem);
}

optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>* get_iface(
  cuOptOptimizationProblem problem)
{
  return as_problem(problem)->get_problem();
}

bool is_read_only_int_attribute(cuOptProblemAttribute_t attribute)
{
  switch (attribute) {
    case CUOPT_PROBLEM_ATTR_NUM_VARIABLES:
    case CUOPT_PROBLEM_ATTR_NUM_CONSTRAINTS:
    case CUOPT_PROBLEM_ATTR_NUM_NONZEROS:
    case CUOPT_PROBLEM_ATTR_NUM_INTEGERS:
    case CUOPT_PROBLEM_ATTR_PROBLEM_CATEGORY:
    case CUOPT_PROBLEM_ATTR_IS_MIP:
    case CUOPT_PROBLEM_ATTR_HAS_QUADRATIC_OBJECTIVE:
    case CUOPT_PROBLEM_ATTR_HAS_QUADRATIC_CONSTRAINTS: return true;
    default: return false;
  }
}

bool is_int_attribute(cuOptProblemAttribute_t attribute)
{
  switch (attribute) {
    case CUOPT_PROBLEM_ATTR_NUM_VARIABLES:
    case CUOPT_PROBLEM_ATTR_NUM_CONSTRAINTS:
    case CUOPT_PROBLEM_ATTR_NUM_NONZEROS:
    case CUOPT_PROBLEM_ATTR_NUM_INTEGERS:
    case CUOPT_PROBLEM_ATTR_OBJECTIVE_SENSE:
    case CUOPT_PROBLEM_ATTR_PROBLEM_CATEGORY:
    case CUOPT_PROBLEM_ATTR_IS_MIP:
    case CUOPT_PROBLEM_ATTR_HAS_QUADRATIC_OBJECTIVE:
    case CUOPT_PROBLEM_ATTR_HAS_QUADRATIC_CONSTRAINTS: return true;
    default: return false;
  }
}

bool is_float_attribute(cuOptProblemAttribute_t attribute)
{
  return attribute == CUOPT_PROBLEM_ATTR_OBJECTIVE_OFFSET ||
         attribute == CUOPT_PROBLEM_ATTR_OBJECTIVE_SCALING_FACTOR;
}

bool is_string_attribute(cuOptProblemAttribute_t attribute)
{
  return attribute == CUOPT_PROBLEM_ATTR_PROBLEM_NAME ||
         attribute == CUOPT_PROBLEM_ATTR_OBJECTIVE_NAME;
}

cuopt_int_t get_array_size(optimization_problem_interface_t<cuopt_int_t, cuopt_float_t>* problem,
                           cuOptProblemArrayAttribute_t attribute)
{
  switch (attribute) {
    case CUOPT_PROBLEM_ARRAY_ATTR_OBJECTIVE_COEFFICIENTS:
    case CUOPT_PROBLEM_ARRAY_ATTR_VARIABLE_LOWER_BOUNDS:
    case CUOPT_PROBLEM_ARRAY_ATTR_VARIABLE_UPPER_BOUNDS:
    case CUOPT_PROBLEM_ARRAY_ATTR_VARIABLE_TYPES: return problem->get_n_variables();
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_LOWER_BOUNDS:
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_UPPER_BOUNDS:
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_RHS:
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_SENSE: return problem->get_n_constraints();
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_MATRIX_VALUES:
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_MATRIX_INDICES: return problem->get_nnz();
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_MATRIX_OFFSETS:
      return problem->get_n_constraints() + 1;
    default: return -1;
  }
}

}  // namespace

cuopt_int_t cuOptGetProblemIntAttribute(cuOptOptimizationProblem problem,
                                        cuOptProblemAttribute_t attribute,
                                        cuopt_int_t* value_out)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (value_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (!is_int_attribute(attribute)) { return CUOPT_INVALID_ARGUMENT; }

  auto* iface = get_iface(problem);
  switch (attribute) {
    case CUOPT_PROBLEM_ATTR_NUM_VARIABLES:
      *value_out = iface->get_n_variables();
      return CUOPT_SUCCESS;
    case CUOPT_PROBLEM_ATTR_NUM_CONSTRAINTS:
      *value_out = iface->get_n_constraints();
      return CUOPT_SUCCESS;
    case CUOPT_PROBLEM_ATTR_NUM_NONZEROS: *value_out = iface->get_nnz(); return CUOPT_SUCCESS;
    case CUOPT_PROBLEM_ATTR_NUM_INTEGERS:
      *value_out = iface->get_n_integers();
      return CUOPT_SUCCESS;
    case CUOPT_PROBLEM_ATTR_OBJECTIVE_SENSE:
      *value_out = iface->get_sense() ? CUOPT_MAXIMIZE : CUOPT_MINIMIZE;
      return CUOPT_SUCCESS;
    case CUOPT_PROBLEM_ATTR_PROBLEM_CATEGORY:
      *value_out = static_cast<cuopt_int_t>(iface->get_problem_category());
      return CUOPT_SUCCESS;
    case CUOPT_PROBLEM_ATTR_IS_MIP: {
      const auto category = iface->get_problem_category();
      *value_out =
        (category == problem_category_t::MIP || category == problem_category_t::IP) ? 1 : 0;
      return CUOPT_SUCCESS;
    }
    case CUOPT_PROBLEM_ATTR_HAS_QUADRATIC_OBJECTIVE:
      *value_out = iface->has_quadratic_objective() ? 1 : 0;
      return CUOPT_SUCCESS;
    case CUOPT_PROBLEM_ATTR_HAS_QUADRATIC_CONSTRAINTS:
      *value_out = iface->has_quadratic_constraints() ? 1 : 0;
      return CUOPT_SUCCESS;
    default: return CUOPT_INVALID_ARGUMENT;
  }
}

cuopt_int_t cuOptSetProblemIntAttribute(cuOptOptimizationProblem problem,
                                        cuOptProblemAttribute_t attribute,
                                        cuopt_int_t value)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (is_read_only_int_attribute(attribute)) { return CUOPT_INVALID_ARGUMENT; }
  if (attribute != CUOPT_PROBLEM_ATTR_OBJECTIVE_SENSE) { return CUOPT_INVALID_ARGUMENT; }
  if (value != CUOPT_MINIMIZE && value != CUOPT_MAXIMIZE) { return CUOPT_INVALID_ARGUMENT; }

  auto* problem_view = as_problem(problem);
  auto* iface        = problem_view->get_problem();
  iface->set_maximize(value == CUOPT_MAXIMIZE);
  problem_view->invalidate_c_api_caches();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetProblemFloatAttribute(cuOptOptimizationProblem problem,
                                          cuOptProblemAttribute_t attribute,
                                          cuopt_float_t* value_out)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (value_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (!is_float_attribute(attribute)) { return CUOPT_INVALID_ARGUMENT; }

  auto* iface = get_iface(problem);
  if (attribute == CUOPT_PROBLEM_ATTR_OBJECTIVE_OFFSET) {
    *value_out = iface->get_objective_offset();
  } else {
    *value_out = iface->get_objective_scaling_factor();
  }
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetProblemFloatAttribute(cuOptOptimizationProblem problem,
                                          cuOptProblemAttribute_t attribute,
                                          cuopt_float_t value)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (!is_float_attribute(attribute)) { return CUOPT_INVALID_ARGUMENT; }

  auto* problem_view = as_problem(problem);
  auto* iface        = problem_view->get_problem();
  if (attribute == CUOPT_PROBLEM_ATTR_OBJECTIVE_OFFSET) {
    iface->set_objective_offset(value);
  } else {
    iface->set_objective_scaling_factor(value);
  }
  problem_view->invalidate_c_api_caches();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetProblemStringAttribute(cuOptOptimizationProblem problem,
                                           cuOptProblemAttribute_t attribute,
                                           const char** value_out)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (value_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (!is_string_attribute(attribute)) { return CUOPT_INVALID_ARGUMENT; }

  auto* problem_view = as_problem(problem);
  auto* iface        = problem_view->get_problem();
  if (attribute == CUOPT_PROBLEM_ATTR_PROBLEM_NAME) {
    problem_view->scalar_string_cache_ = iface->get_problem_name();
  } else {
    problem_view->scalar_string_cache_ = iface->get_objective_name();
  }
  *value_out = problem_view->scalar_string_cache_.c_str();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetProblemStringAttribute(cuOptOptimizationProblem problem,
                                           cuOptProblemAttribute_t attribute,
                                           const char* value)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (value == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (!is_string_attribute(attribute)) { return CUOPT_INVALID_ARGUMENT; }

  auto* problem_view = as_problem(problem);
  auto* iface        = problem_view->get_problem();
  if (attribute == CUOPT_PROBLEM_ATTR_PROBLEM_NAME) {
    iface->set_problem_name(value);
  } else {
    iface->set_objective_name(value);
  }
  problem_view->invalidate_c_api_caches();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetProblemArraySize(cuOptOptimizationProblem problem,
                                     cuOptProblemArrayAttribute_t attribute,
                                     cuopt_int_t* size_out)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (size_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  const cuopt_int_t size = get_array_size(get_iface(problem), attribute);
  if (size < 0) { return CUOPT_INVALID_ARGUMENT; }
  *size_out = size;
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetProblemFloatArrayAttribute(cuOptOptimizationProblem problem,
                                               cuOptProblemArrayAttribute_t attribute,
                                               const cuopt_float_t** data_out,
                                               cuopt_int_t* count_out)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (data_out == nullptr || count_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  auto* problem_view         = as_problem(problem);
  auto* iface                = problem_view->get_problem();
  const cuopt_int_t expected = get_array_size(iface, attribute);
  if (expected < 0) { return CUOPT_INVALID_ARGUMENT; }

  switch (attribute) {
    case CUOPT_PROBLEM_ARRAY_ATTR_OBJECTIVE_COEFFICIENTS:
      problem_view->float_array_cache_ = iface->get_objective_coefficients_host();
      break;
    case CUOPT_PROBLEM_ARRAY_ATTR_VARIABLE_LOWER_BOUNDS:
      problem_view->float_array_cache_ = iface->get_variable_lower_bounds_host();
      break;
    case CUOPT_PROBLEM_ARRAY_ATTR_VARIABLE_UPPER_BOUNDS:
      problem_view->float_array_cache_ = iface->get_variable_upper_bounds_host();
      break;
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_LOWER_BOUNDS:
      problem_view->float_array_cache_ = iface->get_constraint_lower_bounds_host();
      break;
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_UPPER_BOUNDS:
      problem_view->float_array_cache_ = iface->get_constraint_upper_bounds_host();
      break;
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_RHS:
      problem_view->float_array_cache_ = iface->get_constraint_bounds_host();
      break;
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_MATRIX_VALUES:
      problem_view->float_array_cache_ = iface->get_constraint_matrix_values_host();
      break;
    default: return CUOPT_INVALID_ARGUMENT;
  }

  if (static_cast<cuopt_int_t>(problem_view->float_array_cache_.size()) != expected) {
    return CUOPT_VALIDATION_ERROR;
  }

  *count_out = expected;
  *data_out =
    problem_view->float_array_cache_.empty() ? nullptr : problem_view->float_array_cache_.data();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetProblemFloatArrayAttribute(cuOptOptimizationProblem problem,
                                               cuOptProblemArrayAttribute_t attribute,
                                               const cuopt_float_t* data,
                                               cuopt_int_t count)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (data == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  auto* problem_view         = as_problem(problem);
  auto* iface                = problem_view->get_problem();
  const cuopt_int_t expected = get_array_size(iface, attribute);
  if (expected < 0 || count != expected) { return CUOPT_INVALID_ARGUMENT; }

  switch (attribute) {
    case CUOPT_PROBLEM_ARRAY_ATTR_OBJECTIVE_COEFFICIENTS:
      iface->set_objective_coefficients(data, count);
      break;
    case CUOPT_PROBLEM_ARRAY_ATTR_VARIABLE_LOWER_BOUNDS:
      iface->set_variable_lower_bounds(data, count);
      break;
    case CUOPT_PROBLEM_ARRAY_ATTR_VARIABLE_UPPER_BOUNDS:
      iface->set_variable_upper_bounds(data, count);
      break;
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_LOWER_BOUNDS:
      iface->set_constraint_lower_bounds(data, count);
      break;
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_UPPER_BOUNDS:
      iface->set_constraint_upper_bounds(data, count);
      break;
    case CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_RHS: iface->set_constraint_bounds(data, count); break;
    default: return CUOPT_INVALID_ARGUMENT;
  }

  problem_view->invalidate_c_api_caches();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetProblemIntArrayAttribute(cuOptOptimizationProblem problem,
                                             cuOptProblemArrayAttribute_t attribute,
                                             const cuopt_int_t** data_out,
                                             cuopt_int_t* count_out)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (data_out == nullptr || count_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  auto* problem_view         = as_problem(problem);
  auto* iface                = problem_view->get_problem();
  const cuopt_int_t expected = get_array_size(iface, attribute);
  if (expected < 0) { return CUOPT_INVALID_ARGUMENT; }

  if (attribute == CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_MATRIX_INDICES) {
    problem_view->int_array_cache_ = iface->get_constraint_matrix_indices_host();
  } else if (attribute == CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_MATRIX_OFFSETS) {
    problem_view->int_array_cache_ = iface->get_constraint_matrix_offsets_host();
  } else {
    return CUOPT_INVALID_ARGUMENT;
  }

  if (static_cast<cuopt_int_t>(problem_view->int_array_cache_.size()) != expected) {
    return CUOPT_VALIDATION_ERROR;
  }

  *count_out = expected;
  *data_out =
    problem_view->int_array_cache_.empty() ? nullptr : problem_view->int_array_cache_.data();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetProblemIntArrayAttribute(cuOptOptimizationProblem problem,
                                             cuOptProblemArrayAttribute_t attribute,
                                             const cuopt_int_t* data,
                                             cuopt_int_t count)
{
  (void)problem;
  (void)attribute;
  (void)data;
  (void)count;
  return CUOPT_INVALID_ARGUMENT;
}

cuopt_int_t cuOptGetProblemCharArrayAttribute(cuOptOptimizationProblem problem,
                                              cuOptProblemArrayAttribute_t attribute,
                                              const char** data_out,
                                              cuopt_int_t* count_out)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (data_out == nullptr || count_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  auto* problem_view         = as_problem(problem);
  auto* iface                = problem_view->get_problem();
  const cuopt_int_t expected = get_array_size(iface, attribute);
  if (expected < 0) { return CUOPT_INVALID_ARGUMENT; }

  if (attribute == CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_SENSE) {
    problem_view->char_array_cache_ = iface->get_row_types_host();
  } else if (attribute == CUOPT_PROBLEM_ARRAY_ATTR_VARIABLE_TYPES) {
    const auto var_types = iface->get_variable_types_host();
    problem_view->char_array_cache_.resize(var_types.size());
    for (std::size_t i = 0; i < var_types.size(); ++i) {
      problem_view->char_array_cache_[i] = var_type_to_char(var_types[i]);
    }
  } else {
    return CUOPT_INVALID_ARGUMENT;
  }

  if (static_cast<cuopt_int_t>(problem_view->char_array_cache_.size()) != expected) {
    return CUOPT_VALIDATION_ERROR;
  }

  *count_out = expected;
  *data_out =
    problem_view->char_array_cache_.empty() ? nullptr : problem_view->char_array_cache_.data();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetProblemCharArrayAttribute(cuOptOptimizationProblem problem,
                                              cuOptProblemArrayAttribute_t attribute,
                                              const char* data,
                                              cuopt_int_t count)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (data == nullptr) { return CUOPT_INVALID_ARGUMENT; }

  auto* problem_view         = as_problem(problem);
  auto* iface                = problem_view->get_problem();
  const cuopt_int_t expected = get_array_size(iface, attribute);
  if (expected < 0 || count != expected) { return CUOPT_INVALID_ARGUMENT; }

  if (attribute == CUOPT_PROBLEM_ARRAY_ATTR_CONSTRAINT_SENSE) {
    iface->set_row_types(data, count);
  } else if (attribute == CUOPT_PROBLEM_ARRAY_ATTR_VARIABLE_TYPES) {
    std::vector<var_t> var_types(static_cast<std::size_t>(count));
    for (cuopt_int_t i = 0; i < count; ++i) {
      var_types[static_cast<std::size_t>(i)] = char_to_var_type(data[i]);
    }
    iface->set_variable_types(var_types.data(), count);
  } else {
    return CUOPT_INVALID_ARGUMENT;
  }

  problem_view->invalidate_c_api_caches();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptGetProblemStringArrayAttribute(cuOptOptimizationProblem problem,
                                                cuOptProblemStringArrayAttribute_t attribute,
                                                const char*** strings_out,
                                                cuopt_int_t* count_out)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (strings_out == nullptr || count_out == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (attribute != CUOPT_PROBLEM_STRING_ARRAY_VARIABLE_NAMES &&
      attribute != CUOPT_PROBLEM_STRING_ARRAY_ROW_NAMES) {
    return CUOPT_INVALID_ARGUMENT;
  }

  auto* problem_view = as_problem(problem);
  auto* iface        = problem_view->get_problem();
  const auto& names  = (attribute == CUOPT_PROBLEM_STRING_ARRAY_VARIABLE_NAMES)
                         ? iface->get_variable_names()
                         : iface->get_row_names();

  problem_view->refresh_string_array_view(names);
  *count_out   = static_cast<cuopt_int_t>(names.size());
  *strings_out = names.empty() ? nullptr : problem_view->string_array_view_cache_.data();
  return CUOPT_SUCCESS;
}

cuopt_int_t cuOptSetProblemStringArrayAttribute(cuOptOptimizationProblem problem,
                                                cuOptProblemStringArrayAttribute_t attribute,
                                                const char* const* strings,
                                                cuopt_int_t count)
{
  if (problem == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (strings == nullptr) { return CUOPT_INVALID_ARGUMENT; }
  if (attribute != CUOPT_PROBLEM_STRING_ARRAY_VARIABLE_NAMES &&
      attribute != CUOPT_PROBLEM_STRING_ARRAY_ROW_NAMES) {
    return CUOPT_INVALID_ARGUMENT;
  }

  auto* problem_view         = as_problem(problem);
  auto* iface                = problem_view->get_problem();
  const cuopt_int_t expected = (attribute == CUOPT_PROBLEM_STRING_ARRAY_VARIABLE_NAMES)
                                 ? iface->get_n_variables()
                                 : iface->get_n_constraints();
  if (count != expected) { return CUOPT_INVALID_ARGUMENT; }

  std::vector<std::string> names(static_cast<std::size_t>(count));
  for (cuopt_int_t i = 0; i < count; ++i) {
    if (strings[i] == nullptr) { return CUOPT_INVALID_ARGUMENT; }
    names[static_cast<std::size_t>(i)] = strings[i];
  }

  if (attribute == CUOPT_PROBLEM_STRING_ARRAY_VARIABLE_NAMES) {
    iface->set_variable_names(names);
  } else {
    iface->set_row_names(names);
  }

  problem_view->invalidate_c_api_caches();
  return CUOPT_SUCCESS;
}
