# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


def _has_local_gpu():
    """Check for local GPU via CUDA driver API (no context creation, ~1-5ms)."""
    try:
        from cuda.bindings.driver import cuInit, cuDeviceGetCount
        (err,) = cuInit(0)
        if err != 0:
            return False
        err, count = cuDeviceGetCount()
        return err == 0 and count > 0
    except (ImportError, Exception):
        return False


def _eager_cuda_init():
    """
    Eagerly initialize CUDA runtime for local GPU execution.

    This eliminates the lazy-init penalty (~140ms) that would otherwise
    occur on the first solve.  Sets CUDA_MODULE_LOADING=EAGER (matching
    the C++ CLI behaviour) and forces context creation with a trivial
    device allocation.
    """
    # If remote execution is configured, skip — no local GPU needed
    if os.environ.get("CUOPT_REMOTE_HOST") and os.environ.get("CUOPT_REMOTE_PORT"):
        return

    if not _has_local_gpu():
        return

    # Set eager module loading BEFORE any CUDA runtime call
    # (mirrors set_cuda_module_loading() in cuopt_cli.cpp).
    # setdefault so we don't override an explicit user setting.
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")

    try:
        import cudf
        cudf.Series([1, 2, 3])  # Force CUDA context + RMM device memory initialization
    except Exception:
        pass


try:
    import libcuopt
except ModuleNotFoundError:
    pass
else:
    libcuopt.load_library()
    del libcuopt

_eager_cuda_init()

from cuopt import distance_engine, linear_programming, routing
from cuopt._version import __git_commit__, __version__, __version_major_minor__

# Lazy imports for linear_programming, routing, and distance_engine modules
# This allows cuopt to be imported on CPU-only hosts when remote solve is configured
_submodules = ["linear_programming", "routing", "distance_engine"]


def __getattr__(name):
    """Lazy import submodules to support CPU-only hosts with remote solve."""
    if name in _submodules:
        import importlib
        return importlib.import_module(f"cuopt.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(dict.fromkeys(__all__ + _submodules))


__all__ = ["__git_commit__", "__version__", "__version_major_minor__"]
