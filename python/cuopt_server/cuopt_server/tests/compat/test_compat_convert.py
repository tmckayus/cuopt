# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for proxy validation with shared cuOpt converters."""

import copy

import numpy as np
import pytest

from cuopt.linear_programming import toDataModelAndSettings
from cuopt_server.compat.validate import (
    LegacyJsonValidationError,
    parse_lp_data,
    validate_lp_data,
)
from cuopt_server.utils.linear_programming.data_definition import (
    lp_example_data,
)


def test_parse_and_normalize_inf():
    data = copy.deepcopy(lp_example_data)
    lp = parse_lp_data(data)
    assert isinstance(lp.variable_bounds.upper_bounds, np.ndarray)
    assert np.isinf(lp.variable_bounds.upper_bounds).all()
    assert np.isneginf(lp.constraint_bounds.lower_bounds).all()


def test_validate_ok():
    lp = parse_lp_data(copy.deepcopy(lp_example_data))
    validate_lp_data(lp)


def test_validate_rejects_bad_csr():
    data = copy.deepcopy(lp_example_data)
    data["csr_constraint_matrix"]["indices"] = [0]  # length mismatch
    lp = parse_lp_data(data)
    with pytest.raises(LegacyJsonValidationError):
        validate_lp_data(lp)


def test_shared_converter_without_validate():
    dm, settings = toDataModelAndSettings(copy.deepcopy(lp_example_data))
    assert dm is not None
    assert settings is not None
    assert len(dm.get_objective_coefficients()) == 2


def test_shared_converter_after_proxy_validation():
    data = copy.deepcopy(lp_example_data)
    lp = parse_lp_data(data)
    validate_lp_data(lp)
    dm, settings = toDataModelAndSettings(data)
    assert dm is not None
    assert settings is not None
