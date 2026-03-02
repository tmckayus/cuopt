/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/utilities/proto_message_stream.hpp>

#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>

#include <cstring>
#include <stdexcept>

namespace cuopt::linear_programming {

// ============================================================================
// append_array
// ============================================================================

template <typename T>
void append_array(MessageStream& stream, int32_t field_id, const std::vector<T>& data)
{
  if (data.empty()) return;

  // For double arrays, always store as double (convert if source is float)
  if constexpr (std::is_same_v<T, float>) {
    std::vector<double> tmp(data.begin(), data.end());
    cuopt::remote::PipeEnvelope env;
    auto* ad = env.mutable_array_data();
    ad->set_field_id(field_id);
    ad->set_data(tmp.data(), tmp.size() * sizeof(double));
    stream.push_back(std::move(env));
  } else {
    cuopt::remote::PipeEnvelope env;
    auto* ad = env.mutable_array_data();
    ad->set_field_id(field_id);
    ad->set_data(data.data(), data.size() * sizeof(T));
    stream.push_back(std::move(env));
  }
}

// ============================================================================
// extract_array
// ============================================================================

template <typename T>
std::vector<T> extract_array(const MessageStream& messages, int32_t field_id)
{
  for (const auto& env : messages) {
    if (env.has_array_data() && env.array_data().field_id() == field_id) {
      const auto& raw = env.array_data().data();
      if (raw.empty()) return {};

      if constexpr (std::is_same_v<T, float>) {
        // Wire is double, convert to float
        size_t n = raw.size() / sizeof(double);
        std::vector<double> tmp(n);
        std::memcpy(tmp.data(), raw.data(), raw.size());
        return std::vector<T>(tmp.begin(), tmp.end());
      } else if constexpr (std::is_same_v<T, double>) {
        size_t n = raw.size() / sizeof(double);
        std::vector<double> v(n);
        std::memcpy(v.data(), raw.data(), raw.size());
        return v;
      } else if constexpr (std::is_same_v<T, int32_t>) {
        size_t n = raw.size() / sizeof(int32_t);
        std::vector<int32_t> v(n);
        std::memcpy(v.data(), raw.data(), raw.size());
        return v;
      } else if constexpr (std::is_same_v<T, uint8_t>) {
        return std::vector<uint8_t>(raw.begin(), raw.end());
      } else {
        size_t n = raw.size() / sizeof(T);
        std::vector<T> v(n);
        std::memcpy(v.data(), raw.data(), raw.size());
        return v;
      }
    }
  }
  return {};
}

// ============================================================================
// serialize_stream / deserialize_stream
// ============================================================================

std::vector<uint8_t> serialize_stream(const MessageStream& messages)
{
  std::vector<uint8_t> blob;
  for (const auto& env : messages) {
    uint32_t size = static_cast<uint32_t>(env.ByteSizeLong());
    size_t offset = blob.size();
    blob.resize(offset + 4 + size);
    std::memcpy(blob.data() + offset, &size, 4);
    env.SerializeToArray(blob.data() + offset + 4, size);
  }
  return blob;
}

MessageStream deserialize_stream(const uint8_t* data, size_t size)
{
  MessageStream messages;
  const uint8_t* ptr = data;
  const uint8_t* end = data + size;

  while (ptr + 4 <= end) {
    uint32_t msg_size;
    std::memcpy(&msg_size, ptr, 4);
    ptr += 4;
    if (ptr + msg_size > end) break;
    cuopt::remote::PipeEnvelope env;
    if (!env.ParseFromArray(ptr, msg_size)) break;
    ptr += msg_size;
    messages.push_back(std::move(env));
  }
  return messages;
}

// ============================================================================
// Header lookup helpers
// ============================================================================

const cuopt::remote::ChunkedResultHeader& find_result_header(const MessageStream& messages)
{
  for (const auto& env : messages) {
    if (env.has_result_header()) { return env.result_header(); }
  }
  throw std::runtime_error("MessageStream contains no result_header envelope");
}

const cuopt::remote::ChunkedProblemHeader& find_problem_header(const MessageStream& messages)
{
  for (const auto& env : messages) {
    if (env.has_chunked_header()) { return env.chunked_header(); }
  }
  throw std::runtime_error("MessageStream contains no chunked_header envelope");
}

// ============================================================================
// Explicit template instantiations
// ============================================================================

template void append_array<double>(MessageStream&, int32_t, const std::vector<double>&);
template void append_array<float>(MessageStream&, int32_t, const std::vector<float>&);
template void append_array<int32_t>(MessageStream&, int32_t, const std::vector<int32_t>&);
template void append_array<uint8_t>(MessageStream&, int32_t, const std::vector<uint8_t>&);

template std::vector<double> extract_array<double>(const MessageStream&, int32_t);
template std::vector<float> extract_array<float>(const MessageStream&, int32_t);
template std::vector<int32_t> extract_array<int32_t>(const MessageStream&, int32_t);
template std::vector<uint8_t> extract_array<uint8_t>(const MessageStream&, int32_t);

}  // namespace cuopt::linear_programming
