/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cuopt::linear_programming {

// Canonical intermediate representation for all data transfer.
// First envelope carries scalar fields (in a header), subsequent envelopes carry arrays.
using MessageStream = std::vector<cuopt::remote::PipeEnvelope>;

// ============================================================================
// Array primitives
// ============================================================================

/**
 * @brief Pack a typed vector into a PipeArrayData envelope and append to stream.
 *
 * The data is stored as raw bytes. For doubles the wire representation is
 * IEEE-754 little-endian; for int32 it is native 4-byte little-endian.
 */
template <typename T>
void append_array(MessageStream& stream, int32_t field_id, const std::vector<T>& data);

/**
 * @brief Gather all bytes for a given field_id across envelopes and reconstruct a typed vector.
 */
template <typename T>
std::vector<T> extract_array(const MessageStream& messages, int32_t field_id);

// ============================================================================
// Serialization (MessageStream <-> length-delimited bytes for pipe IPC)
// ============================================================================

/**
 * @brief Serialize a MessageStream to length-delimited bytes.
 *
 * Wire format per envelope: [4 bytes LE uint32 size][size bytes serialized PipeEnvelope]
 */
std::vector<uint8_t> serialize_stream(const MessageStream& messages);

/**
 * @brief Deserialize length-delimited bytes back into a MessageStream.
 */
MessageStream deserialize_stream(const uint8_t* data, size_t size);

inline MessageStream deserialize_stream(const std::vector<uint8_t>& data)
{
  return deserialize_stream(data.data(), data.size());
}

// ============================================================================
// Header lookup helpers
// ============================================================================

const cuopt::remote::ChunkedResultHeader& find_result_header(const MessageStream& messages);
const cuopt::remote::ChunkedProblemHeader& find_problem_header(const MessageStream& messages);

}  // namespace cuopt::linear_programming
