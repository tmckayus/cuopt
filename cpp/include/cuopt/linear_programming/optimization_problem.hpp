/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>

namespace cuopt::linear_programming {

enum class var_t { CONTINUOUS = 0, INTEGER };
enum class problem_category_t : int8_t { LP = 0, MIP = 1, IP = 2 };

// Forward declaration - actual definition in optimization_problem_interface.hpp
template <typename i_t, typename f_t>
class optimization_problem_t;

}  // namespace cuopt::linear_programming
