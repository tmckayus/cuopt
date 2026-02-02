# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for CPU-only execution mode.

These tests verify that cuOpt can run on a CPU host without GPU access,
simulating remote execution scenarios. The tests set CUDA_VISIBLE_DEVICES=""
and enable remote execution mode to force CPU-only operation.

This tests:
- Problem creation on CPU backend
- LP solving with CPU backend
- MIP solving with CPU backend
- Solution retrieval without CUDA errors
- Warmstart data handling on CPU
"""

import os
import pytest
import subprocess
import sys


# Get the path to the test MPS files
RAPIDS_DATASET_ROOT_DIR = os.environ.get(
    "RAPIDS_DATASET_ROOT_DIR", "/home/datasets/cuopt"
)


class TestCPUOnlyExecution:
    """Tests that run with CUDA_VISIBLE_DEVICES="" to simulate CPU-only hosts."""

    @pytest.fixture
    def cpu_only_env(self):
        """Create environment with no GPU access and remote execution enabled."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["CUOPT_REMOTE_HOST"] = "localhost"
        env["CUOPT_REMOTE_PORT"] = (
            "12345"  # Fake port, remote not actually called
        )
        return env

    def test_lp_solve_cpu_only(self, cpu_only_env):
        """Test LP solving works in CPU-only mode."""
        test_script = f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUOPT_REMOTE_HOST"] = "localhost"
os.environ["CUOPT_REMOTE_PORT"] = "12345"

from cuopt import linear_programming
import cuopt_mps_parser

# Parse MPS file
mps_file = "{RAPIDS_DATASET_ROOT_DIR}/linear_programming/afiro_original.mps"
data_model = cuopt_mps_parser.ParseMps(mps_file)

# Create solver settings
settings = linear_programming.SolverSettings()

# Solve - this should use CPU backend due to remote execution mode
try:
    solution = linear_programming.Solve(data_model, settings)

    # Verify we can access solution properties
    status = solution.get_termination_status()
    objective = solution.get_primal_objective()
    solve_time = solution.get_solve_time()
    primal = solution.get_primal_solution()

    print(f"Status: {{status}}")
    print(f"Objective: {{objective}}")
    print(f"Solve time: {{solve_time}}")
    print(f"Primal size: {{len(primal)}}")
    print("CPU-only LP solve: PASSED")
except Exception as e:
    print(f"CPU-only LP solve: FAILED - {{e}}")
    raise
'''
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            env=cpu_only_env,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "PASSED" in result.stdout

    def test_mip_solve_cpu_only(self, cpu_only_env):
        """Test MIP solving works in CPU-only mode."""
        test_script = f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUOPT_REMOTE_HOST"] = "localhost"
os.environ["CUOPT_REMOTE_PORT"] = "12345"

from cuopt import linear_programming
from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT
import cuopt_mps_parser

# Parse MPS file (actual MIP problem with integer variables)
mps_file = "{RAPIDS_DATASET_ROOT_DIR}/mip/bb_optimality.mps"
data_model = cuopt_mps_parser.ParseMps(mps_file)

# Create solver settings
settings = linear_programming.SolverSettings()
settings.set_parameter(CUOPT_TIME_LIMIT, 60.0)

# Solve - MIP will be detected from integer variables in the problem
try:
    solution = linear_programming.Solve(data_model, settings)

    # Verify we can access solution properties
    status = solution.get_termination_status()
    objective = solution.get_primal_objective()
    solve_time = solution.get_solve_time()
    solution_values = solution.get_primal_solution()

    print(f"Status: {{status}}")
    print(f"Objective: {{objective}}")
    print(f"Solve time: {{solve_time}}")
    print(f"Solution size: {{len(solution_values)}}")
    print("CPU-only MIP solve: PASSED")
except Exception as e:
    print(f"CPU-only MIP solve: FAILED - {{e}}")
    raise
'''
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            env=cpu_only_env,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "PASSED" in result.stdout

    def test_lp_dual_solution_cpu_only(self, cpu_only_env):
        """Test LP dual solution retrieval works in CPU-only mode."""
        test_script = f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUOPT_REMOTE_HOST"] = "localhost"
os.environ["CUOPT_REMOTE_PORT"] = "12345"

from cuopt import linear_programming
import cuopt_mps_parser

# Parse MPS file
mps_file = "{RAPIDS_DATASET_ROOT_DIR}/linear_programming/afiro_original.mps"
data_model = cuopt_mps_parser.ParseMps(mps_file)

# Create solver settings
settings = linear_programming.SolverSettings()

try:
    solution = linear_programming.Solve(data_model, settings)

    # Verify LP-specific properties are accessible
    dual_solution = solution.get_dual_solution()
    dual_objective = solution.get_dual_objective()
    reduced_costs = solution.get_reduced_cost()

    print(f"Dual solution size: {{len(dual_solution)}}")
    print(f"Dual objective: {{dual_objective}}")
    print(f"Reduced costs size: {{len(reduced_costs)}}")
    print("CPU-only LP dual: PASSED")
except Exception as e:
    print(f"CPU-only LP dual: FAILED - {{e}}")
    raise
'''
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            env=cpu_only_env,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "PASSED" in result.stdout

    def test_warmstart_cpu_only(self, cpu_only_env):
        """Test warmstart data handling works in CPU-only mode."""
        test_script = f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUOPT_REMOTE_HOST"] = "localhost"
os.environ["CUOPT_REMOTE_PORT"] = "12345"

from cuopt import linear_programming
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_METHOD,
    CUOPT_ITERATION_LIMIT,
)
from cuopt.linear_programming.solver_settings import SolverMethod
import cuopt_mps_parser

# Parse MPS file
mps_file = "{RAPIDS_DATASET_ROOT_DIR}/linear_programming/afiro_original.mps"
data_model = cuopt_mps_parser.ParseMps(mps_file)

# Create solver settings with iteration limit to get warmstart data
settings = linear_programming.SolverSettings()
settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
settings.set_parameter(CUOPT_ITERATION_LIMIT, 100)

try:
    # First solve
    solution1 = linear_programming.Solve(data_model, settings)
    warmstart_data = solution1.get_pdlp_warm_start_data()

    if warmstart_data is not None:
        print(f"Got warmstart data")

        # Second solve with warmstart
        settings.set_pdlp_warm_start_data(warmstart_data)
        settings.set_parameter(CUOPT_ITERATION_LIMIT, 200)
        solution2 = linear_programming.Solve(data_model, settings)

        print(f"Second solve completed")
        print("CPU-only warmstart: PASSED")
    else:
        print("No warmstart data available (expected for small iteration count)")
        print("CPU-only warmstart: PASSED")
except Exception as e:
    print(f"CPU-only warmstart: FAILED - {{e}}")
    raise
'''
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            env=cpu_only_env,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "PASSED" in result.stdout


class TestCuoptCliCPUOnly:
    """Tests that cuopt_cli can run without CUDA resources in remote execution mode."""

    @pytest.fixture
    def cpu_only_env(self):
        """Create environment with no GPU access and remote execution enabled."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["CUOPT_REMOTE_HOST"] = "localhost"
        env["CUOPT_REMOTE_PORT"] = "12345"
        return env

    def _find_cuopt_cli(self):
        """Try to find cuopt_cli executable."""
        import shutil

        # Check common locations
        locations = [
            shutil.which("cuopt_cli"),  # In PATH
            "./cuopt_cli",  # Current directory
            "../cpp/build/cuopt_cli",  # Build directory from python tests
            "../../cpp/build/cuopt_cli",  # Another common relative path
        ]

        for loc in locations:
            if loc and os.path.isfile(loc) and os.access(loc, os.X_OK):
                return os.path.abspath(loc)

        # Try to find in conda environment
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            conda_cli = os.path.join(conda_prefix, "bin", "cuopt_cli")
            if os.path.isfile(conda_cli):
                return conda_cli

        return None

    def test_cuopt_cli_lp_cpu_only(self, cpu_only_env):
        """Test cuopt_cli LP solve runs without CUDA initialization in CPU-only mode."""
        cuopt_cli = self._find_cuopt_cli()
        if cuopt_cli is None:
            pytest.skip("cuopt_cli not found")

        mps_file = (
            f"{RAPIDS_DATASET_ROOT_DIR}/linear_programming/afiro_original.mps"
        )
        if not os.path.exists(mps_file):
            pytest.skip(f"Test file not found: {mps_file}")

        cmd = [cuopt_cli, mps_file, "--time-limit", "60"]
        print(f"Running: {' '.join(cmd)}")
        print(
            "Environment: CUDA_VISIBLE_DEVICES='' CUOPT_REMOTE_HOST=localhost"
        )

        result = subprocess.run(
            cmd,
            env=cpu_only_env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)

        # Check that there are no CUDA initialization errors
        combined_output = result.stdout + result.stderr
        cuda_errors = [
            "CUDA error",
            "cudaErrorNoDevice",
            "no CUDA-capable device",
            "CUDA driver version is insufficient",
            "CUDA initialization failed",
        ]

        for error in cuda_errors:
            assert error not in combined_output, (
                f"Found CUDA error '{error}' in output - "
                f"cuopt_cli should not require CUDA in remote execution mode"
            )

        # The solve should complete (with dummy remote results)
        # Exit code 0 means success
        assert result.returncode == 0, (
            f"cuopt_cli failed with return code {result.returncode}"
        )

        print("cuopt_cli LP CPU-only test: PASSED")

    def test_cuopt_cli_mip_cpu_only(self, cpu_only_env):
        """Test cuopt_cli MIP solve runs without CUDA initialization in CPU-only mode."""
        cuopt_cli = self._find_cuopt_cli()
        if cuopt_cli is None:
            pytest.skip("cuopt_cli not found")

        mps_file = f"{RAPIDS_DATASET_ROOT_DIR}/mip/bb_optimality.mps"
        if not os.path.exists(mps_file):
            pytest.skip(f"Test file not found: {mps_file}")

        cmd = [cuopt_cli, mps_file, "--time-limit", "60"]
        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            env=cpu_only_env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)

        # Check for CUDA errors
        combined_output = result.stdout + result.stderr
        cuda_errors = [
            "CUDA error",
            "cudaErrorNoDevice",
            "no CUDA-capable device",
        ]

        for error in cuda_errors:
            assert error not in combined_output, (
                f"Found CUDA error '{error}' - should not require CUDA in remote mode"
            )

        assert result.returncode == 0
        print("cuopt_cli MIP CPU-only test: PASSED")


class TestSolutionInterfacePolymorphism:
    """Tests for solution interface polymorphic methods."""

    def test_lp_solution_has_dual_info(self):
        """Test that LP solutions have dual solution info."""
        test_script = f'''
from cuopt import linear_programming
import cuopt_mps_parser

mps_file = "{RAPIDS_DATASET_ROOT_DIR}/linear_programming/afiro_original.mps"
data_model = cuopt_mps_parser.ParseMps(mps_file)
settings = linear_programming.SolverSettings()

solution = linear_programming.Solve(data_model, settings)

# LP solutions should have dual info
dual = solution.get_dual_solution()
assert dual is not None, "Dual solution is None"
assert len(dual) > 0, "Dual solution is empty"

reduced_costs = solution.get_reduced_cost()
assert reduced_costs is not None, "Reduced costs is None"
assert len(reduced_costs) > 0, "Reduced costs is empty"

print("LP solution dual info test: PASSED")
'''
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "PASSED" in result.stdout

    def test_mip_solution_has_mip_info(self):
        """Test that MIP solutions have MIP-specific info."""
        test_script = f'''
from cuopt import linear_programming
from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT
import cuopt_mps_parser

# Use actual MIP file with integer variables
mps_file = "{RAPIDS_DATASET_ROOT_DIR}/mip/bb_optimality.mps"
data_model = cuopt_mps_parser.ParseMps(mps_file)
settings = linear_programming.SolverSettings()
settings.set_parameter(CUOPT_TIME_LIMIT, 60.0)

# Solve - MIP will be detected from integer variables
solution = linear_programming.Solve(data_model, settings)

# MIP solutions should have MIP-specific info via get_milp_stats()
milp_stats = solution.get_milp_stats()
mip_gap = milp_stats["mip_gap"]
solution_bound = milp_stats["solution_bound"]

# Just verify we can call these without errors
print(f"MIP gap: {{mip_gap}}")
print(f"Solution bound: {{solution_bound}}")
print("MIP solution info test: PASSED")
'''
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "PASSED" in result.stdout
