#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build c_api_mps_benchmark from the cuOpt cpp tree.
# Run from repo root or from this directory.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CPP_DIR="$REPO_ROOT/cpp"
BUILD_DIR="$CPP_DIR/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
if [[ ! -f Makefile && ! -f build.ninja ]]; then
  echo "Configuring with cmake..."
  cmake ..
fi
make -j"${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 2)}" c_api_mps_benchmark
echo "Built: $BUILD_DIR/c_api_mps_benchmark"
echo "Usage: $BUILD_DIR/c_api_mps_benchmark <mps_directory> <output_csv> [--iterations N] [--time-limit SEC] [--method M]"
