/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>

#include <cuopt/linear_programming/utilities/proto_message_stream.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace cuopt::remote {
class ChunkedProblemHeader;
}

namespace cuopt::linear_programming {

// Forward declarations
template <typename i_t, typename f_t>
class cpu_optimization_problem_t;

template <typename i_t, typename f_t>
struct pdlp_solver_settings_t;

template <typename i_t, typename f_t>
struct mip_solver_settings_t;

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

/**
 * @brief Estimate the serialized protobuf size of a SolveLPRequest/SolveMIPRequest.
 *
 * Computes an approximate upper bound on the serialized size without actually building
 * the protobuf message. Used to decide whether to use chunked array transfer.
 *
 * @return Estimated size in bytes
 */
template <typename i_t, typename f_t>
size_t estimate_problem_proto_size(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem);

/**
 * @brief Populate a ChunkedProblemHeader from a cpu_optimization_problem_t and LP settings.
 *
 * Fills the header with problem scalars, string arrays, and LP settings.
 * Numeric arrays are NOT included (they are sent as ArrayChunk messages).
 */
template <typename i_t, typename f_t>
void populate_chunked_header_lp(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                                const pdlp_solver_settings_t<i_t, f_t>& settings,
                                cuopt::remote::ChunkedProblemHeader* header);

/**
 * @brief Populate a ChunkedProblemHeader from a cpu_optimization_problem_t and MIP settings.
 *
 * Fills the header with problem scalars, string arrays, and MIP settings.
 * Numeric arrays are NOT included (they are sent as ArrayChunk messages).
 */
template <typename i_t, typename f_t>
void populate_chunked_header_mip(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                                 const mip_solver_settings_t<i_t, f_t>& settings,
                                 bool enable_incumbents,
                                 cuopt::remote::ChunkedProblemHeader* header);

/**
 * @brief Reconstruct a cpu_optimization_problem_t from a ChunkedProblemHeader.
 *
 * Populates problem scalars and string arrays from the header. Numeric arrays
 * must be populated separately from ArrayChunk data.
 */
template <typename i_t, typename f_t>
void map_chunked_header_to_problem(const cuopt::remote::ChunkedProblemHeader& header,
                                   cpu_optimization_problem_t<i_t, f_t>& cpu_problem);

/**
 * @brief Reconstruct a cpu_optimization_problem_t from a ChunkedProblemHeader and raw array data.
 *
 * This is the single entry point for reconstructing a problem from chunked transfer data.
 * It calls map_chunked_header_to_problem() for scalars/strings, then populates all numeric
 * arrays from the raw byte data keyed by ArrayFieldId.
 *
 * @param header The chunked problem header (scalars, settings metadata, string arrays)
 * @param arrays Map of ArrayFieldId (as int32_t) to raw byte data for each array field
 * @param cpu_problem The cpu_optimization_problem_t to populate (output parameter)
 */
template <typename i_t, typename f_t>
void map_chunked_arrays_to_problem(const cuopt::remote::ChunkedProblemHeader& header,
                                   const std::map<int32_t, std::vector<uint8_t>>& arrays,
                                   cpu_optimization_problem_t<i_t, f_t>& cpu_problem);

// ============================================================================
// MessageStream conversion functions (unified conversion layer)
// ============================================================================

/**
 * @brief Convert a cpu_optimization_problem_t + settings to a MessageStream.
 *
 * The first envelope carries scalar fields and settings in a ChunkedProblemHeader.
 * Subsequent envelopes carry array data. Adding a new field is a one-line change.
 *
 * @param lp_settings  LP settings (nullptr if MIP)
 * @param mip_settings MIP settings (nullptr if LP)
 */
template <typename i_t, typename f_t>
MessageStream problem_to_messages(const cpu_optimization_problem_t<i_t, f_t>& problem,
                                  const pdlp_solver_settings_t<i_t, f_t>* lp_settings,
                                  const mip_solver_settings_t<i_t, f_t>* mip_settings,
                                  bool enable_incumbents = false);

/**
 * @brief Reconstruct a cpu_optimization_problem_t + settings from a MessageStream.
 */
template <typename i_t, typename f_t>
void messages_to_problem(const MessageStream& messages,
                         cpu_optimization_problem_t<i_t, f_t>& problem,
                         pdlp_solver_settings_t<i_t, f_t>& lp_settings,
                         mip_solver_settings_t<i_t, f_t>& mip_settings,
                         bool& enable_incumbents);

// ============================================================================
// Transport adaptors: MessageStream <-> gRPC protobuf messages
// ============================================================================

/**
 * @brief Convert a SolveLPRequest protobuf into a MessageStream.
 * Used by the server worker to unify the unary and chunked code paths.
 */
template <typename i_t, typename f_t>
MessageStream lp_request_proto_to_stream(const cuopt::remote::SolveLPRequest& request);

/**
 * @brief Convert a SolveMIPRequest protobuf into a MessageStream.
 */
template <typename i_t, typename f_t>
MessageStream mip_request_proto_to_stream(const cuopt::remote::SolveMIPRequest& request);

}  // namespace cuopt::linear_programming
