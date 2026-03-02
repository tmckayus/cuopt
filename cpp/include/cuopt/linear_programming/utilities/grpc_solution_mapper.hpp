/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>

#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/utilities/proto_message_stream.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

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

// ============================================================================
// Chunked result support (for results exceeding gRPC max message size)
// ============================================================================

/**
 * @brief Estimate serialized protobuf size of an LP solution.
 */
template <typename i_t, typename f_t>
size_t estimate_lp_solution_proto_size(const cpu_lp_solution_t<i_t, f_t>& solution);

/**
 * @brief Estimate serialized protobuf size of a MIP solution.
 */
template <typename i_t, typename f_t>
size_t estimate_mip_solution_proto_size(const cpu_mip_solution_t<i_t, f_t>& solution);

/**
 * @brief Populate a ChunkedResultHeader from an LP solution (scalar fields + array descriptors).
 */
template <typename i_t, typename f_t>
void populate_chunked_result_header_lp(const cpu_lp_solution_t<i_t, f_t>& solution,
                                       cuopt::remote::ChunkedResultHeader* header);

/**
 * @brief Populate a ChunkedResultHeader from a MIP solution (scalar fields + array descriptors).
 */
template <typename i_t, typename f_t>
void populate_chunked_result_header_mip(const cpu_mip_solution_t<i_t, f_t>& solution,
                                        cuopt::remote::ChunkedResultHeader* header);

/**
 * @brief Collect LP solution arrays as raw bytes keyed by ResultFieldId.
 *
 * Returns a map of ResultFieldId -> raw byte data (doubles packed as bytes).
 * Used by the worker to send chunked result data.
 */
template <typename i_t, typename f_t>
std::map<int32_t, std::vector<uint8_t>> collect_lp_solution_arrays(
  const cpu_lp_solution_t<i_t, f_t>& solution);

/**
 * @brief Collect MIP solution arrays as raw bytes keyed by ResultFieldId.
 */
template <typename i_t, typename f_t>
std::map<int32_t, std::vector<uint8_t>> collect_mip_solution_arrays(
  const cpu_mip_solution_t<i_t, f_t>& solution);

// ============================================================================
// MessageStream conversion functions (unified conversion layer)
// ============================================================================

/**
 * @brief Convert an LP solution to a MessageStream.
 *
 * The first envelope carries scalar fields in a ChunkedResultHeader.
 * Subsequent envelopes carry array data (primal, dual, reduced_cost, warm start).
 * Adding a new field is a one-line change in this function.
 */
template <typename i_t, typename f_t>
MessageStream lp_solution_to_messages(const cpu_lp_solution_t<i_t, f_t>& solution);

/**
 * @brief Convert a MIP solution to a MessageStream.
 */
template <typename i_t, typename f_t>
MessageStream mip_solution_to_messages(const cpu_mip_solution_t<i_t, f_t>& solution);

/**
 * @brief Reconstruct a cpu_lp_solution_t from a MessageStream.
 */
template <typename i_t, typename f_t>
cpu_lp_solution_t<i_t, f_t> messages_to_lp_solution(const MessageStream& messages);

/**
 * @brief Reconstruct a cpu_mip_solution_t from a MessageStream.
 */
template <typename i_t, typename f_t>
cpu_mip_solution_t<i_t, f_t> messages_to_mip_solution(const MessageStream& messages);

// ============================================================================
// Transport adaptors: MessageStream <-> gRPC protobuf messages
// ============================================================================

/**
 * @brief Reconstruct a full LPSolution protobuf from a MessageStream.
 * Used by the server's GetResult RPC to serve unary responses.
 */
template <typename i_t, typename f_t>
void stream_to_lp_solution_proto(const MessageStream& messages, cuopt::remote::LPSolution* proto);

/**
 * @brief Reconstruct a full MIPSolution protobuf from a MessageStream.
 */
template <typename i_t, typename f_t>
void stream_to_mip_solution_proto(const MessageStream& messages, cuopt::remote::MIPSolution* proto);

}  // namespace cuopt::linear_programming
