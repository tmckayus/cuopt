/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt_remote.pb.h>

#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>

#include <cstdint>

namespace cuopt::linear_programming {

/**
 * @brief Map cpu_lp_solution_t to protobuf LPSolution message.
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
 * @param solution The cuOpt CPU LP solution to map from
 * @param pb_solution The protobuf message to populate (output parameter)
 */
template <typename i_t, typename f_t>
void map_lp_solution_to_proto(const cpu_lp_solution_t<i_t, f_t>& solution,
                              cuopt::remote::LPSolution* pb_solution);

/**
 * @brief Map protobuf LPSolution message to cpu_lp_solution_t.
 *
 * This function reads from a protobuf message object using the generated protobuf C++ API
 * and creates a cpu_lp_solution_t. It does NOT perform deserialization - that
 * is handled by the protobuf library before this function is called.
 *
 * Security note: This function only uses the protobuf-generated C++ API to read
 * message objects. All actual deserialization is performed by the protobuf library
 * before this function is called.
 *
 * @tparam i_t Index type (int32_t or int64_t)
 * @tparam f_t Float type (float or double)
 * @param pb_solution The protobuf message to read from
 * @return The cuOpt CPU LP solution
 */
template <typename i_t, typename f_t>
cpu_lp_solution_t<i_t, f_t> map_proto_to_lp_solution(const cuopt::remote::LPSolution& pb_solution);

/**
 * @brief Map cpu_mip_solution_t to protobuf MIPSolution message.
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
 * @param solution The cuOpt CPU MIP solution to map from
 * @param pb_solution The protobuf message to populate (output parameter)
 */
template <typename i_t, typename f_t>
void map_mip_solution_to_proto(const cpu_mip_solution_t<i_t, f_t>& solution,
                               cuopt::remote::MIPSolution* pb_solution);

/**
 * @brief Map protobuf MIPSolution message to cpu_mip_solution_t.
 *
 * This function reads from a protobuf message object using the generated protobuf C++ API
 * and creates a cpu_mip_solution_t. It does NOT perform deserialization - that
 * is handled by the protobuf library before this function is called.
 *
 * Security note: This function only uses the protobuf-generated C++ API to read
 * message objects. All actual deserialization is performed by the protobuf library
 * before this function is called.
 *
 * @tparam i_t Index type (int32_t or int64_t)
 * @tparam f_t Float type (float or double)
 * @param pb_solution The protobuf message to read from
 * @return The cuOpt CPU MIP solution
 */
template <typename i_t, typename f_t>
cpu_mip_solution_t<i_t, f_t> map_proto_to_mip_solution(
  const cuopt::remote::MIPSolution& pb_solution);

/**
 * @brief Convert cuOpt termination status to protobuf enum.
 * @param status cuOpt PDLP termination status
 * @return Protobuf PDLPTerminationStatus enum
 */
cuopt::remote::PDLPTerminationStatus to_proto_pdlp_status(pdlp_termination_status_t status);

/**
 * @brief Convert protobuf enum to cuOpt termination status.
 * @param status Protobuf PDLPTerminationStatus enum
 * @return cuOpt PDLP termination status
 */
pdlp_termination_status_t from_proto_pdlp_status(cuopt::remote::PDLPTerminationStatus status);

/**
 * @brief Convert cuOpt MIP termination status to protobuf enum.
 * @param status cuOpt MIP termination status
 * @return Protobuf MIPTerminationStatus enum
 */
cuopt::remote::MIPTerminationStatus to_proto_mip_status(mip_termination_status_t status);

/**
 * @brief Convert protobuf enum to cuOpt MIP termination status.
 * @param status Protobuf MIPTerminationStatus enum
 * @return cuOpt MIP termination status
 */
mip_termination_status_t from_proto_mip_status(cuopt::remote::MIPTerminationStatus status);

}  // namespace cuopt::linear_programming
