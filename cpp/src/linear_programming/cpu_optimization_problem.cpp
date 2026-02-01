/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/optimization_problem_interface.hpp>

#include <mip/mip_constants.hpp>
#include <mps_parser/writer.hpp>
#include <utilities/logger.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

namespace cuopt::linear_programming {

// ==============================================================================
// Constructor
// ==============================================================================

template <typename i_t, typename f_t>
cpu_optimization_problem_t<i_t, f_t>::cpu_optimization_problem_t(raft::handle_t const* handle_ptr)
  : handle_ptr_(handle_ptr)
{
  fprintf(stderr, "cpu_optimization_problem_t constructor: Using CPU backend\n");
}

// ==============================================================================
// Setters
// ==============================================================================

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_maximize(bool maximize)
{
  maximize_ = maximize;
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_csr_constraint_matrix(const f_t* A_values,
                                                                     i_t size_values,
                                                                     const i_t* A_indices,
                                                                     i_t size_indices,
                                                                     const i_t* A_offsets,
                                                                     i_t size_offsets)
{
  n_constraints_ = size_offsets - 1;

  A_.resize(size_values);
  A_indices_.resize(size_indices);
  A_offsets_.resize(size_offsets);

  std::copy(A_values, A_values + size_values, A_.begin());
  std::copy(A_indices, A_indices + size_indices, A_indices_.begin());
  std::copy(A_offsets, A_offsets + size_offsets, A_offsets_.begin());
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_constraint_bounds(const f_t* b, i_t size)
{
  b_.resize(size);
  std::copy(b, b + size, b_.begin());
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_objective_coefficients(const f_t* c, i_t size)
{
  n_vars_ = size;
  c_.resize(size);
  std::copy(c, c + size, c_.begin());
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_objective_scaling_factor(
  f_t objective_scaling_factor)
{
  objective_scaling_factor_ = objective_scaling_factor;
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_objective_offset(f_t objective_offset)
{
  objective_offset_ = objective_offset;
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_quadratic_objective_matrix(
  const f_t* Q_values,
  i_t size_values,
  const i_t* Q_indices,
  i_t size_indices,
  const i_t* Q_offsets,
  i_t size_offsets,
  bool validate_positive_semi_definite)
{
  Q_values_.resize(size_values);
  Q_indices_.resize(size_indices);
  Q_offsets_.resize(size_offsets);

  std::copy(Q_values, Q_values + size_values, Q_values_.begin());
  std::copy(Q_indices, Q_indices + size_indices, Q_indices_.begin());
  std::copy(Q_offsets, Q_offsets + size_offsets, Q_offsets_.begin());
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_variable_lower_bounds(
  const f_t* variable_lower_bounds, i_t size)
{
  variable_lower_bounds_.resize(size);
  std::copy(variable_lower_bounds, variable_lower_bounds + size, variable_lower_bounds_.begin());
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_variable_upper_bounds(
  const f_t* variable_upper_bounds, i_t size)
{
  variable_upper_bounds_.resize(size);
  std::copy(variable_upper_bounds, variable_upper_bounds + size, variable_upper_bounds_.begin());
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_variable_types(const var_t* variable_types, i_t size)
{
  variable_types_.resize(size);
  std::copy(variable_types, variable_types + size, variable_types_.begin());
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_problem_category(const problem_category_t& category)
{
  problem_category_ = category;
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_constraint_lower_bounds(
  const f_t* constraint_lower_bounds, i_t size)
{
  constraint_lower_bounds_.resize(size);
  std::copy(
    constraint_lower_bounds, constraint_lower_bounds + size, constraint_lower_bounds_.begin());
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_constraint_upper_bounds(
  const f_t* constraint_upper_bounds, i_t size)
{
  constraint_upper_bounds_.resize(size);
  std::copy(
    constraint_upper_bounds, constraint_upper_bounds + size, constraint_upper_bounds_.begin());
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_row_types(const char* row_types, i_t size)
{
  row_types_.resize(size);
  std::copy(row_types, row_types + size, row_types_.begin());
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_objective_name(const std::string& objective_name)
{
  objective_name_ = objective_name;
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_problem_name(const std::string& problem_name)
{
  problem_name_ = problem_name;
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_variable_names(
  const std::vector<std::string>& variable_names)
{
  var_names_ = variable_names;
}

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::set_row_names(const std::vector<std::string>& row_names)
{
  row_names_ = row_names;
}

// ==============================================================================
// Device Getters - Throw exceptions (not supported for CPU implementation)
// ==============================================================================

namespace {
[[noreturn]] void throw_gpu_not_supported(const char* method_name)
{
  throw std::runtime_error(std::string("cpu_optimization_problem_t::") + method_name +
                           "(): GPU memory access is not supported in CPU implementation. "
                           "Use the corresponding _host() method instead.");
}
}  // namespace

template <typename i_t, typename f_t>
i_t cpu_optimization_problem_t<i_t, f_t>::get_n_variables() const
{
  return n_vars_;
}

template <typename i_t, typename f_t>
i_t cpu_optimization_problem_t<i_t, f_t>::get_n_constraints() const
{
  return n_constraints_;
}

template <typename i_t, typename f_t>
i_t cpu_optimization_problem_t<i_t, f_t>::get_nnz() const
{
  return A_.size();
}

template <typename i_t, typename f_t>
i_t cpu_optimization_problem_t<i_t, f_t>::get_n_integers() const
{
  i_t count = 0;
  for (const auto& type : variable_types_) {
    if (type == var_t::INTEGER) count++;
  }
  return count;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_values()
  const
{
  throw_gpu_not_supported("get_constraint_matrix_values");
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_values()
{
  throw_gpu_not_supported("get_constraint_matrix_values");
}

template <typename i_t, typename f_t>
const rmm::device_uvector<i_t>&
cpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_indices() const
{
  throw_gpu_not_supported("get_constraint_matrix_indices");
}

template <typename i_t, typename f_t>
rmm::device_uvector<i_t>& cpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_indices()
{
  throw_gpu_not_supported("get_constraint_matrix_indices");
}

template <typename i_t, typename f_t>
const rmm::device_uvector<i_t>&
cpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_offsets() const
{
  throw_gpu_not_supported("get_constraint_matrix_offsets");
}

template <typename i_t, typename f_t>
rmm::device_uvector<i_t>& cpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_offsets()
{
  throw_gpu_not_supported("get_constraint_matrix_offsets");
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_constraint_bounds() const
{
  throw_gpu_not_supported("get_constraint_bounds");
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_constraint_bounds()
{
  throw_gpu_not_supported("get_constraint_bounds");
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_objective_coefficients()
  const
{
  throw_gpu_not_supported("get_objective_coefficients");
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_objective_coefficients()
{
  throw_gpu_not_supported("get_objective_coefficients");
}

template <typename i_t, typename f_t>
f_t cpu_optimization_problem_t<i_t, f_t>::get_objective_scaling_factor() const
{
  return objective_scaling_factor_;
}

template <typename i_t, typename f_t>
f_t cpu_optimization_problem_t<i_t, f_t>::get_objective_offset() const
{
  return objective_offset_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_variable_lower_bounds()
  const
{
  throw_gpu_not_supported("get_variable_lower_bounds");
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_variable_lower_bounds()
{
  throw_gpu_not_supported("get_variable_lower_bounds");
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_variable_upper_bounds()
  const
{
  throw_gpu_not_supported("get_variable_upper_bounds");
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_variable_upper_bounds()
{
  throw_gpu_not_supported("get_variable_upper_bounds");
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_constraint_lower_bounds()
  const
{
  throw_gpu_not_supported("get_constraint_lower_bounds");
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_constraint_lower_bounds()
{
  throw_gpu_not_supported("get_constraint_lower_bounds");
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_constraint_upper_bounds()
  const
{
  throw_gpu_not_supported("get_constraint_upper_bounds");
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_constraint_upper_bounds()
{
  throw_gpu_not_supported("get_constraint_upper_bounds");
}

template <typename i_t, typename f_t>
const rmm::device_uvector<char>& cpu_optimization_problem_t<i_t, f_t>::get_row_types() const
{
  throw_gpu_not_supported("get_row_types");
}

template <typename i_t, typename f_t>
const rmm::device_uvector<var_t>& cpu_optimization_problem_t<i_t, f_t>::get_variable_types() const
{
  throw_gpu_not_supported("get_variable_types");
}

template <typename i_t, typename f_t>
bool cpu_optimization_problem_t<i_t, f_t>::get_sense() const
{
  return maximize_;
}

template <typename i_t, typename f_t>
bool cpu_optimization_problem_t<i_t, f_t>::empty() const
{
  return n_vars_ == 0 || n_constraints_ == 0;
}

template <typename i_t, typename f_t>
std::string cpu_optimization_problem_t<i_t, f_t>::get_objective_name() const
{
  return objective_name_;
}

template <typename i_t, typename f_t>
std::string cpu_optimization_problem_t<i_t, f_t>::get_problem_name() const
{
  return problem_name_;
}

template <typename i_t, typename f_t>
problem_category_t cpu_optimization_problem_t<i_t, f_t>::get_problem_category() const
{
  return problem_category_;
}

template <typename i_t, typename f_t>
const std::vector<std::string>& cpu_optimization_problem_t<i_t, f_t>::get_variable_names() const
{
  return var_names_;
}

template <typename i_t, typename f_t>
const std::vector<std::string>& cpu_optimization_problem_t<i_t, f_t>::get_row_names() const
{
  return row_names_;
}

template <typename i_t, typename f_t>
const std::vector<i_t>& cpu_optimization_problem_t<i_t, f_t>::get_quadratic_objective_offsets()
  const
{
  return Q_offsets_;
}

template <typename i_t, typename f_t>
const std::vector<i_t>& cpu_optimization_problem_t<i_t, f_t>::get_quadratic_objective_indices()
  const
{
  return Q_indices_;
}

template <typename i_t, typename f_t>
const std::vector<f_t>& cpu_optimization_problem_t<i_t, f_t>::get_quadratic_objective_values() const
{
  return Q_values_;
}

template <typename i_t, typename f_t>
bool cpu_optimization_problem_t<i_t, f_t>::has_quadratic_objective() const
{
  return !Q_values_.empty();
}

// ==============================================================================
// Host Getters (return references to CPU memory)
// ==============================================================================

template <typename i_t, typename f_t>
std::vector<f_t> cpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_values_host() const
{
  return A_;
}

template <typename i_t, typename f_t>
std::vector<i_t> cpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_indices_host() const
{
  return A_indices_;
}

template <typename i_t, typename f_t>
std::vector<i_t> cpu_optimization_problem_t<i_t, f_t>::get_constraint_matrix_offsets_host() const
{
  return A_offsets_;
}

template <typename i_t, typename f_t>
std::vector<f_t> cpu_optimization_problem_t<i_t, f_t>::get_constraint_bounds_host() const
{
  return b_;
}

template <typename i_t, typename f_t>
std::vector<f_t> cpu_optimization_problem_t<i_t, f_t>::get_objective_coefficients_host() const
{
  return c_;
}

template <typename i_t, typename f_t>
std::vector<f_t> cpu_optimization_problem_t<i_t, f_t>::get_variable_lower_bounds_host() const
{
  return variable_lower_bounds_;
}

template <typename i_t, typename f_t>
std::vector<f_t> cpu_optimization_problem_t<i_t, f_t>::get_variable_upper_bounds_host() const
{
  return variable_upper_bounds_;
}

template <typename i_t, typename f_t>
std::vector<f_t> cpu_optimization_problem_t<i_t, f_t>::get_constraint_lower_bounds_host() const
{
  return constraint_lower_bounds_;
}

template <typename i_t, typename f_t>
std::vector<f_t> cpu_optimization_problem_t<i_t, f_t>::get_constraint_upper_bounds_host() const
{
  return constraint_upper_bounds_;
}

template <typename i_t, typename f_t>
std::vector<char> cpu_optimization_problem_t<i_t, f_t>::get_row_types_host() const
{
  return row_types_;
}

template <typename i_t, typename f_t>
std::vector<var_t> cpu_optimization_problem_t<i_t, f_t>::get_variable_types_host() const
{
  return variable_types_;
}

// ==============================================================================
// Conversion to optimization_problem_t
// ==============================================================================

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> cpu_optimization_problem_t<i_t, f_t>::to_optimization_problem()
{
  if (handle_ptr_ == nullptr) {
    // NOTE: We could theoretically allocate GPU resources here, but we are not
    // currently supporting local solve of a problem that has been built on the CPU.
    // CPU problems are intended for remote execution only.
    throw std::runtime_error(
      "cpu_optimization_problem_t::to_optimization_problem(): "
      "handle_ptr is null. Cannot convert to GPU-backed optimization_problem_t without CUDA "
      "resources. "
      "CPU problems are intended for remote execution only. "
      "For local solving, create the problem with GPU backend from the start.");
  }

  optimization_problem_t<i_t, f_t> problem(handle_ptr_);

  // Set scalar values
  problem.set_maximize(maximize_);
  problem.set_objective_scaling_factor(objective_scaling_factor_);
  problem.set_objective_offset(objective_offset_);
  problem.set_problem_category(problem_category_);

  // Set string values
  if (!objective_name_.empty()) problem.set_objective_name(objective_name_);
  if (!problem_name_.empty()) problem.set_problem_name(problem_name_);
  if (!var_names_.empty()) problem.set_variable_names(var_names_);
  if (!row_names_.empty()) problem.set_row_names(row_names_);

  // Set CSR constraint matrix (data will be copied to GPU by optimization_problem_t setters)
  if (!A_.empty()) {
    problem.set_csr_constraint_matrix(A_.data(),
                                      A_.size(),
                                      A_indices_.data(),
                                      A_indices_.size(),
                                      A_offsets_.data(),
                                      A_offsets_.size());
  }

  // Set constraint bounds
  if (!b_.empty()) { problem.set_constraint_bounds(b_.data(), b_.size()); }

  // Set objective coefficients
  if (!c_.empty()) { problem.set_objective_coefficients(c_.data(), c_.size()); }

  // Set quadratic objective if present
  if (!Q_values_.empty()) {
    problem.set_quadratic_objective_matrix(Q_values_.data(),
                                           Q_values_.size(),
                                           Q_indices_.data(),
                                           Q_indices_.size(),
                                           Q_offsets_.data(),
                                           Q_offsets_.size());
  }

  // Set variable bounds
  if (!variable_lower_bounds_.empty()) {
    problem.set_variable_lower_bounds(variable_lower_bounds_.data(), variable_lower_bounds_.size());
  }
  if (!variable_upper_bounds_.empty()) {
    problem.set_variable_upper_bounds(variable_upper_bounds_.data(), variable_upper_bounds_.size());
  }

  // Set variable types
  if (!variable_types_.empty()) {
    problem.set_variable_types(variable_types_.data(), variable_types_.size());
  }

  // Set constraint bounds
  if (!constraint_lower_bounds_.empty()) {
    problem.set_constraint_lower_bounds(constraint_lower_bounds_.data(),
                                        constraint_lower_bounds_.size());
  }
  if (!constraint_upper_bounds_.empty()) {
    problem.set_constraint_upper_bounds(constraint_upper_bounds_.data(),
                                        constraint_upper_bounds_.size());
  }

  // Set row types
  if (!row_types_.empty()) { problem.set_row_types(row_types_.data(), row_types_.size()); }

  return problem;
}

// ==============================================================================
// File I/O
// ==============================================================================

template <typename i_t, typename f_t>
void cpu_optimization_problem_t<i_t, f_t>::write_to_mps(const std::string& mps_file_path)
{
  // Data is already in host memory, so we can directly create a view and write
  cuopt::mps_parser::data_model_view_t<i_t, f_t> data_model_view;

  // Set optimization sense
  data_model_view.set_maximize(maximize_);

  // Set constraint matrix in CSR format
  if (!A_.empty()) {
    data_model_view.set_csr_constraint_matrix(A_.data(),
                                              A_.size(),
                                              A_indices_.data(),
                                              A_indices_.size(),
                                              A_offsets_.data(),
                                              A_offsets_.size());
  }

  // Set constraint bounds (RHS)
  if (!b_.empty()) { data_model_view.set_constraint_bounds(b_.data(), b_.size()); }

  // Set objective coefficients
  if (!c_.empty()) { data_model_view.set_objective_coefficients(c_.data(), c_.size()); }

  // Set objective scaling and offset
  data_model_view.set_objective_scaling_factor(objective_scaling_factor_);
  data_model_view.set_objective_offset(objective_offset_);

  // Set variable bounds
  if (!variable_lower_bounds_.empty()) {
    data_model_view.set_variable_lower_bounds(variable_lower_bounds_.data(),
                                              variable_lower_bounds_.size());
    data_model_view.set_variable_upper_bounds(variable_upper_bounds_.data(),
                                              variable_upper_bounds_.size());
  }

  // Set row types (constraint types)
  if (!row_types_.empty()) { data_model_view.set_row_types(row_types_.data(), row_types_.size()); }

  // Set constraint bounds (lower and upper)
  if (!constraint_lower_bounds_.empty() && !constraint_upper_bounds_.empty()) {
    data_model_view.set_constraint_lower_bounds(constraint_lower_bounds_.data(),
                                                constraint_lower_bounds_.size());
    data_model_view.set_constraint_upper_bounds(constraint_upper_bounds_.data(),
                                                constraint_upper_bounds_.size());
  }

  // Set problem and variable names if available
  if (!problem_name_.empty()) { data_model_view.set_problem_name(problem_name_); }
  if (!objective_name_.empty()) { data_model_view.set_objective_name(objective_name_); }
  if (!var_names_.empty()) { data_model_view.set_variable_names(var_names_); }
  if (!row_names_.empty()) { data_model_view.set_row_names(row_names_); }

  // Set variable types (convert from enum to char)
  // CRITICAL: Declare var_types_char OUTSIDE the if block so it stays alive
  // until after write_mps() is called, since data_model_view stores a span (pointer) to it
  std::vector<char> var_types_char;
  if (!variable_types_.empty()) {
    var_types_char.resize(variable_types_.size());

    for (size_t i = 0; i < var_types_char.size(); ++i) {
      var_types_char[i] = (variable_types_[i] == var_t::INTEGER) ? 'I' : 'C';
    }

    data_model_view.set_variable_types(var_types_char.data(), var_types_char.size());
  }

  cuopt::mps_parser::write_mps(data_model_view, mps_file_path);
}

// ==============================================================================
// Comparison
// ==============================================================================

template <typename i_t, typename f_t>
bool cpu_optimization_problem_t<i_t, f_t>::is_equivalent(
  const optimization_problem_interface_t<i_t, f_t>& other) const
{
  // Compare scalar properties
  if (maximize_ != other.get_sense()) return false;
  if (n_vars_ != other.get_n_variables()) return false;
  if (n_constraints_ != other.get_n_constraints()) return false;
  if (objective_scaling_factor_ != other.get_objective_scaling_factor()) return false;
  if (objective_offset_ != other.get_objective_offset()) return false;
  if (problem_category_ != other.get_problem_category()) return false;

  // Get host data from both problems
  auto other_c = other.get_objective_coefficients_host();
  if (c_.size() != other_c.size()) return false;

  auto other_var_lb = other.get_variable_lower_bounds_host();
  if (variable_lower_bounds_.size() != other_var_lb.size()) return false;

  auto other_var_ub = other.get_variable_upper_bounds_host();
  if (variable_upper_bounds_.size() != other_var_ub.size()) return false;

  auto other_var_types = other.get_variable_types_host();
  if (variable_types_.size() != other_var_types.size()) return false;

  auto other_b = other.get_constraint_bounds_host();
  if (b_.size() != other_b.size()) return false;

  auto other_A_values = other.get_constraint_matrix_values_host();
  if (A_.size() != other_A_values.size()) return false;

  // Check if we have variable and row names for permutation matching
  const auto& other_var_names = other.get_variable_names();
  const auto& other_row_names = other.get_row_names();

  if (var_names_.empty() || other_var_names.empty()) return false;
  if (row_names_.empty() || other_row_names.empty()) return false;

  // Build variable permutation map
  std::unordered_map<std::string, i_t> other_var_idx;
  for (size_t j = 0; j < other_var_names.size(); ++j) {
    other_var_idx[other_var_names[j]] = static_cast<i_t>(j);
  }

  std::vector<i_t> var_perm(n_vars_);
  for (i_t i = 0; i < n_vars_; ++i) {
    auto it = other_var_idx.find(var_names_[i]);
    if (it == other_var_idx.end()) return false;
    var_perm[i] = it->second;
  }

  // Build row permutation map
  std::unordered_map<std::string, i_t> other_row_idx;
  for (size_t j = 0; j < other_row_names.size(); ++j) {
    other_row_idx[other_row_names[j]] = static_cast<i_t>(j);
  }

  std::vector<i_t> row_perm(n_constraints_);
  for (i_t i = 0; i < n_constraints_; ++i) {
    auto it = other_row_idx.find(row_names_[i]);
    if (it == other_row_idx.end()) return false;
    row_perm[i] = it->second;
  }

  // Compare variable-indexed arrays with permutation
  for (i_t i = 0; i < n_vars_; ++i) {
    i_t j = var_perm[i];
    if (std::abs(c_[i] - other_c[j]) > 1e-9) return false;
    if (std::abs(variable_lower_bounds_[i] - other_var_lb[j]) > 1e-9) return false;
    if (std::abs(variable_upper_bounds_[i] - other_var_ub[j]) > 1e-9) return false;
    if (variable_types_[i] != other_var_types[j]) return false;
  }

  // Compare constraint-indexed arrays with permutation
  for (i_t i = 0; i < n_constraints_; ++i) {
    i_t j = row_perm[i];
    if (std::abs(b_[i] - other_b[j]) > 1e-9) return false;
  }

  // For CSR matrix, we'd need more complex comparison - for now just check size matches
  // A full implementation would need to compare matrix entries with row/column permutations
  if (A_.size() != other_A_values.size()) return false;

  return true;
}

// ==============================================================================
// Template instantiations matching optimization_problem_t
// ==============================================================================

#if MIP_INSTANTIATE_FLOAT
template class cpu_optimization_problem_t<int32_t, float>;
template class cpu_optimization_problem_t<int64_t, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class cpu_optimization_problem_t<int32_t, double>;
template class cpu_optimization_problem_t<int64_t, double>;
#endif

}  // namespace cuopt::linear_programming
