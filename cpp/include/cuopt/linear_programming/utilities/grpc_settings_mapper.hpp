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
struct pdlp_solver_settings_t;

template <typename i_t, typename f_t>
struct mip_solver_settings_t;

/**
 * @brief Map pdlp_solver_settings_t to protobuf PDLPSolverSettings message.
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
 * @param settings The cuOpt PDLP solver settings to map from
 * @param pb_settings The protobuf message to populate (output parameter)
 */
template <typename i_t, typename f_t>
void map_pdlp_settings_to_proto(const pdlp_solver_settings_t<i_t, f_t>& settings,
                                cuopt::remote::PDLPSolverSettings* pb_settings);

/**
 * @brief Map protobuf PDLPSolverSettings message to pdlp_solver_settings_t.
 *
 * This function reads from a protobuf message object using the generated protobuf C++ API
 * and populates a pdlp_solver_settings_t. It does NOT perform deserialization - that
 * is handled by the protobuf library before this function is called.
 *
 * Security note: This function only uses the protobuf-generated C++ API to read
 * message objects. All actual deserialization is performed by the protobuf library
 * before this function is called.
 *
 * @tparam i_t Index type (int32_t or int64_t)
 * @tparam f_t Float type (float or double)
 * @param pb_settings The protobuf message to read from
 * @param settings The cuOpt PDLP solver settings to populate (output parameter)
 */
template <typename i_t, typename f_t>
void map_proto_to_pdlp_settings(const cuopt::remote::PDLPSolverSettings& pb_settings,
                                pdlp_solver_settings_t<i_t, f_t>& settings);

/**
 * @brief Map mip_solver_settings_t to protobuf MIPSolverSettings message.
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
 * @param settings The cuOpt MIP solver settings to map from
 * @param pb_settings The protobuf message to populate (output parameter)
 */
template <typename i_t, typename f_t>
void map_mip_settings_to_proto(const mip_solver_settings_t<i_t, f_t>& settings,
                               cuopt::remote::MIPSolverSettings* pb_settings);

/**
 * @brief Map protobuf MIPSolverSettings message to mip_solver_settings_t.
 *
 * This function reads from a protobuf message object using the generated protobuf C++ API
 * and populates a mip_solver_settings_t. It does NOT perform deserialization - that
 * is handled by the protobuf library before this function is called.
 *
 * Security note: This function only uses the protobuf-generated C++ API to read
 * message objects. All actual deserialization is performed by the protobuf library
 * before this function is called.
 *
 * @tparam i_t Index type (int32_t or int64_t)
 * @tparam f_t Float type (float or double)
 * @param pb_settings The protobuf message to read from
 * @param settings The cuOpt MIP solver settings to populate (output parameter)
 */
template <typename i_t, typename f_t>
void map_proto_to_mip_settings(const cuopt::remote::MIPSolverSettings& pb_settings,
                               mip_solver_settings_t<i_t, f_t>& settings);

}  // namespace cuopt::linear_programming
