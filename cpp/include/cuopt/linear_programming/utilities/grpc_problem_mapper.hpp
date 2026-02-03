/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt_remote.pb.h>

#include <cstdint>

namespace cuopt::linear_programming {

// Forward declarations
template <typename i_t, typename f_t>
class cpu_optimization_problem_t;

/**
 * @brief Map cpu_optimization_problem_t to protobuf OptimizationProblem message.
 *
 * This function populates a protobuf message object using the generated protobuf C++ API.
 * It does NOT perform serialization - that is handled by the protobuf library.
 *
 * Security note: This function only uses the protobuf-generated C++ API to populate
 * message objects. All actual serialization/deserialization is performed by the
 * protobuf library, not by custom code.
 *
 * @tparam i_t Index type (int32_t or int64_t)
 * @tparam f_t Float type (float or double)
 * @param cpu_problem The cuOpt CPU problem to map from
 * @param pb_problem The protobuf message to populate (output parameter)
 */
template <typename i_t, typename f_t>
void map_problem_to_proto(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                          cuopt::remote::OptimizationProblem* pb_problem);

/**
 * @brief Map protobuf OptimizationProblem message to cpu_optimization_problem_t.
 *
 * This function reads from a protobuf message object using the generated protobuf C++ API
 * and populates a cpu_optimization_problem_t. It does NOT perform deserialization - that
 * is handled by the protobuf library before this function is called.
 *
 * Security note: This function only uses the protobuf-generated C++ API to read
 * message objects. All actual deserialization is performed by the protobuf library
 * before this function is called.
 *
 * @tparam i_t Index type (int32_t or int64_t)
 * @tparam f_t Float type (float or double)
 * @param pb_problem The protobuf message to read from
 * @param cpu_problem The cuOpt CPU problem to populate (output parameter)
 */
template <typename i_t, typename f_t>
void map_proto_to_problem(const cuopt::remote::OptimizationProblem& pb_problem,
                          cpu_optimization_problem_t<i_t, f_t>& cpu_problem);

}  // namespace cuopt::linear_programming
