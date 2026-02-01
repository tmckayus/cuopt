# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from pylibraft.common.handle cimport *

from datetime import date, datetime

from dateutil.relativedelta import relativedelta

from cuopt.utilities import type_cast

from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, strcpy, strlen
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.device_buffer cimport DeviceBuffer

from cuopt.linear_programming.data_model.data_model cimport data_model_view_t
from cuopt.linear_programming.data_model.data_model_wrapper cimport DataModel
from cuopt.linear_programming.solver.solver cimport (
    call_batch_solve,
    call_solve,
    delete_lp_solution,
    delete_mip_solution,
    error_type_t,
    mip_termination_status_t,
    pdlp_solver_mode_t,
    pdlp_termination_status_t,
    problem_category_t,
    solver_ret_t,
    solver_settings_t,
)

import math
import sys
import warnings
from enum import IntEnum

import cupy as cp
import numpy as np
from numba import cuda

import cudf

from cuopt.linear_programming.solver_settings.solver_settings import (
    PDLPSolverMode,
    SolverSettings,
)
from cuopt.utilities import InputValidationError, series_from_buf

import pyarrow as pa


cdef extern from "cuopt/linear_programming/utilities/internals.hpp" namespace "cuopt::internals": # noqa
    cdef cppclass base_solution_callback_t


class MILPTerminationStatus(IntEnum):
    NoTermination = mip_termination_status_t.NoTermination
    Optimal = mip_termination_status_t.Optimal
    FeasibleFound = mip_termination_status_t.FeasibleFound
    Infeasible = mip_termination_status_t.Infeasible
    Unbounded = mip_termination_status_t.Unbounded
    TimeLimit = mip_termination_status_t.TimeLimit


class LPTerminationStatus(IntEnum):
    NoTermination = pdlp_termination_status_t.NoTermination
    NumericalError = pdlp_termination_status_t.NumericalError
    Optimal = pdlp_termination_status_t.Optimal
    PrimalInfeasible = pdlp_termination_status_t.PrimalInfeasible
    DualInfeasible = pdlp_termination_status_t.DualInfeasible
    IterationLimit = pdlp_termination_status_t.IterationLimit
    TimeLimit = pdlp_termination_status_t.TimeLimit
    PrimalFeasible = pdlp_termination_status_t.PrimalFeasible


class ErrorStatus(IntEnum):
    Success = error_type_t.Success
    ValidationError = error_type_t.ValidationError
    OutOfMemoryError = error_type_t.OutOfMemoryError
    RuntimeError = error_type_t.RuntimeError


class ProblemCategory(IntEnum):
    LP = problem_category_t.LP
    MIP = problem_category_t.MIP
    IP = problem_category_t.IP


cdef char* c_get_string(string in_str):
    cdef char* c_string = <char *> malloc((in_str.length()+1) * sizeof(char))
    if not c_string:
        return NULL  # malloc failed
    # copy except the terminating char
    strcpy(c_string, in_str.c_str())
    return c_string


cdef object _vector_to_numpy(const vector[double]& vec):
    """Convert C++ std::vector<double> to numpy array"""
    cdef Py_ssize_t size = vec.size()
    if size == 0:
        return np.array([], dtype=np.float64)
    cdef const double* data_ptr = vec.data()
    return np.asarray(<double[:size]> data_ptr, dtype=np.float64).copy()


def get_data_ptr(array):
    if isinstance(array, cudf.Series):
        return array.__cuda_array_interface__['data'][0]
    elif isinstance(array, np.ndarray):
        return array.__array_interface__['data'][0]
    else:
        raise Exception(
            "get_data_ptr must be called with cudf.Series or np.ndarray"
        )


def type_cast(cudf_obj, np_type, name):
    if isinstance(cudf_obj, cudf.Series):
        cudf_type = cudf_obj.dtype
    elif isinstance(cudf_obj, np.ndarray):
        cudf_type = cudf_obj.dtype
    elif isinstance(cudf_obj, cudf.DataFrame):
        if all([np.issubdtype(dtype, np.number) for dtype in cudf_obj.dtypes]):  # noqa
            cudf_type = cudf_obj.dtypes[0]
        else:
            msg = "All columns in " + name + " should be numeric"
            raise Exception(msg)
    if ((np.issubdtype(np_type, np.floating) and
         (not np.issubdtype(cudf_type, np.floating)))
       or (np.issubdtype(np_type, np.integer) and
           (not np.issubdtype(cudf_type, np.integer)))
       or (np.issubdtype(np_type, np.bool_) and
           (not np.issubdtype(cudf_type, np.bool_)))
       or (np.issubdtype(np_type, np.int8) and
           (not np.issubdtype(cudf_type, np.int8)))):
        msg = "Casting " + name + " from " + str(cudf_type) + " to " + str(np.dtype(np_type))  # noqa
        warnings.warn(msg)
    cudf_obj = cudf_obj.astype(np.dtype(np_type))
    return cudf_obj


cdef set_solver_setting(
        unique_ptr[solver_settings_t[int, double]]& unique_solver_settings,
        settings,
        DataModel data_model_obj=None,
        mip=False):
    cdef solver_settings_t[int, double]* c_solver_settings = (
        unique_solver_settings.get()
    )
    # Set initial solution on the C++ side if set on the Python side
    cdef uintptr_t c_initial_primal_solution = (
        0 if data_model_obj is None else get_data_ptr(data_model_obj.get_initial_primal_solution())  # noqa
    )
    cdef uintptr_t c_initial_dual_solution = (
        0 if data_model_obj is None else get_data_ptr(data_model_obj.get_initial_dual_solution())  # noqa
    )

    cdef uintptr_t c_current_primal_solution
    cdef uintptr_t c_current_dual_solution
    cdef uintptr_t c_initial_primal_average
    cdef uintptr_t c_initial_dual_average
    cdef uintptr_t c_current_ATY
    cdef uintptr_t c_sum_primal_solutions
    cdef uintptr_t c_sum_dual_solutions
    cdef uintptr_t c_last_restart_duality_gap_primal_solution
    cdef uintptr_t c_last_restart_duality_gap_dual_solution
    cdef uintptr_t callback_ptr = 0
    cdef uintptr_t callback_user_data = 0
    if mip:
        if data_model_obj is not None and data_model_obj.get_initial_primal_solution().shape[0] != 0:  # noqa
            c_solver_settings.add_initial_mip_solution(
                <const double *> c_initial_primal_solution,
                data_model_obj.get_initial_primal_solution().shape[0]
            )

        for name, value in settings.settings_dict.items():
            c_solver_settings.set_parameter_from_string(
                name.encode('utf-8'),
                str(value).encode('utf-8')
            )

        callbacks = settings.get_mip_callbacks()
        for callback in callbacks:
            if callback:
                callback_ptr = callback.get_native_callback()
                callback_user_data = (
                    callback.get_user_data_ptr()
                    if hasattr(callback, "get_user_data_ptr")
                    else 0
                )

                c_solver_settings.set_mip_callback(
                    <base_solution_callback_t*>callback_ptr,
                    <void*>callback_user_data
                )
    else:
        if data_model_obj is not None and data_model_obj.get_initial_primal_solution().shape[0] != 0:  # noqa
            c_solver_settings.set_initial_pdlp_primal_solution(
                <const double *> c_initial_primal_solution,
                data_model_obj.get_initial_primal_solution().shape[0]
            )
        if data_model_obj is not None and data_model_obj.get_initial_dual_solution().shape[0] != 0: # noqa
            c_solver_settings.set_initial_pdlp_dual_solution(
                <const double *> c_initial_dual_solution,
                data_model_obj.get_initial_dual_solution().shape[0]
            )

        for name, value in settings.settings_dict.items():
            c_solver_settings.set_parameter_from_string(
                name.encode('utf-8'),
                str(value).encode('utf-8')
            )


    if settings.get_pdlp_warm_start_data() is not None:  # noqa
        if len(data_model_obj.get_objective_coefficients()) != len(
            settings.get_pdlp_warm_start_data().current_primal_solution
        ):
            raise Exception(
                "Invalid PDLPWarmStart data. Passed problem and PDLPWarmStart " # noqa
                "data should have the same amount of variables."
            )
        if len(data_model_obj.get_constraint_matrix_offsets()) - 1 != len( # noqa
            settings.get_pdlp_warm_start_data().current_dual_solution
        ):
            raise Exception(
                "Invalid PDLPWarmStart data. Passed problem and PDLPWarmStart " # noqa
                "data should have the same amount of constraints."
            )
        c_current_primal_solution = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().current_primal_solution # noqa
            )
        )
        c_current_dual_solution = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().current_dual_solution
            )
        )
        c_initial_primal_average = (
            get_data_ptr(
               settings.get_pdlp_warm_start_data().initial_primal_average # noqa
            )
        )
        c_initial_dual_average = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().initial_dual_average
            )
        )
        c_current_ATY = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().current_ATY
            )
        )
        c_sum_primal_solutions = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().sum_primal_solutions
            )
        )
        c_sum_dual_solutions = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().sum_dual_solutions
            )
        )
        c_last_restart_duality_gap_primal_solution = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution # noqa
            )
        )
        c_last_restart_duality_gap_dual_solution = (
            get_data_ptr(
                settings.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution # noqa
            )
        )
        c_solver_settings.set_pdlp_warm_start_data(
            <const double *> c_current_primal_solution,
            <const double *> c_current_dual_solution,
            <const double *> c_initial_primal_average,
            <const double *> c_initial_dual_average,
            <const double *> c_current_ATY,
            <const double *> c_sum_primal_solutions,
            <const double *> c_sum_dual_solutions,
            <const double *> c_last_restart_duality_gap_primal_solution,
            <const double *> c_last_restart_duality_gap_dual_solution,
            settings.get_pdlp_warm_start_data().last_restart_duality_gap_primal_solution.shape[0], # Primal size # noqa
            settings.get_pdlp_warm_start_data().last_restart_duality_gap_dual_solution.shape[0], # Dual size # noqa
            settings.get_pdlp_warm_start_data().initial_primal_weight,
            settings.get_pdlp_warm_start_data().initial_step_size,
            settings.get_pdlp_warm_start_data().total_pdlp_iterations,
            settings.get_pdlp_warm_start_data().total_pdhg_iterations,
            settings.get_pdlp_warm_start_data().last_candidate_kkt_score,
            settings.get_pdlp_warm_start_data().last_restart_kkt_score,
            settings.get_pdlp_warm_start_data().sum_solution_weight,
            settings.get_pdlp_warm_start_data().iterations_since_last_restart # noqa
        )

cdef create_solution(unique_ptr[solver_ret_t] sol_ret_ptr,
                     DataModel data_model_obj,
                     is_batch=False):

    from cuopt.linear_programming.solution.solution import Solution

    # Declare all cdef variables at the top (Cython requirement)
    cdef solver_ret_t* sol_ret

    # Access the solver_ret_t struct
    sol_ret = sol_ret_ptr.get()

    if sol_ret.problem_type == ProblemCategory.MIP or sol_ret.problem_type == ProblemCategory.IP: # noqa
        # Get solution data from interface (works for both GPU and CPU)
        solution = _vector_to_numpy(sol_ret.mip_solution.get_solution_host())

        result = Solution(
            ProblemCategory(sol_ret.problem_type),
            dict(zip(data_model_obj.get_variable_names(), solution)),
            sol_ret.mip_solution.get_solve_time(),
            primal_solution=solution,
            termination_status=MILPTerminationStatus(sol_ret.mip_solution.get_termination_status()),
            error_status=ErrorStatus(int(sol_ret.mip_solution.get_error_status().get_error_type())),
            error_message=sol_ret.mip_solution.get_error_status().what().decode('utf-8'),
            primal_objective=sol_ret.mip_solution.get_objective_value(),
            mip_gap=sol_ret.mip_solution.get_mip_gap(),
            solution_bound=sol_ret.mip_solution.get_solution_bound(),
            presolve_time=sol_ret.mip_solution.get_presolve_time(),
            max_variable_bound_violation=sol_ret.mip_solution.get_max_variable_bound_violation(),
            max_int_violation=sol_ret.mip_solution.get_max_int_violation(),
            max_constraint_violation=sol_ret.mip_solution.get_max_constraint_violation(),
            num_nodes=sol_ret.mip_solution.get_num_nodes(),
            num_simplex_iterations=sol_ret.mip_solution.get_num_simplex_iterations()
        )

        # Clean up C++ solution object now that we've copied all data to Python
        delete_mip_solution(sol_ret.mip_solution)
        return result

    else:
        # LP problem - get solution data from interface (works for both GPU and CPU)
        primal_solution = _vector_to_numpy(sol_ret.lp_solution.get_primal_solution_host())
        dual_solution = _vector_to_numpy(sol_ret.lp_solution.get_dual_solution_host())
        reduced_cost = _vector_to_numpy(sol_ret.lp_solution.get_reduced_cost_host())

        # Extract warm start data if available (works for both GPU and CPU)
        if not is_batch:
            if sol_ret.lp_solution.has_warm_start_data():
                current_primal = _vector_to_numpy(sol_ret.lp_solution.get_current_primal_solution_host())
                current_dual = _vector_to_numpy(sol_ret.lp_solution.get_current_dual_solution_host())
                initial_primal_avg = _vector_to_numpy(sol_ret.lp_solution.get_initial_primal_average_host())
                initial_dual_avg = _vector_to_numpy(sol_ret.lp_solution.get_initial_dual_average_host())
                current_ATY = _vector_to_numpy(sol_ret.lp_solution.get_current_ATY_host())
                sum_primal = _vector_to_numpy(sol_ret.lp_solution.get_sum_primal_solutions_host())
                sum_dual = _vector_to_numpy(sol_ret.lp_solution.get_sum_dual_solutions_host())
                last_restart_primal = _vector_to_numpy(sol_ret.lp_solution.get_last_restart_duality_gap_primal_solution_host())
                last_restart_dual = _vector_to_numpy(sol_ret.lp_solution.get_last_restart_duality_gap_dual_solution_host())
                initial_primal_weight = sol_ret.lp_solution.get_initial_primal_weight()
                initial_step_size = sol_ret.lp_solution.get_initial_step_size()
                total_pdlp_iters = sol_ret.lp_solution.get_total_pdlp_iterations()
                total_pdhg_iters = sol_ret.lp_solution.get_total_pdhg_iterations()
                last_candidate_kkt = sol_ret.lp_solution.get_last_candidate_kkt_score()
                last_restart_kkt = sol_ret.lp_solution.get_last_restart_kkt_score()
                sum_weight = sol_ret.lp_solution.get_sum_solution_weight()
                iters_since_restart = sol_ret.lp_solution.get_iterations_since_last_restart()
            else:
                current_primal = None
                current_dual = None
                initial_primal_avg = None
                initial_dual_avg = None
                current_ATY = None
                sum_primal = None
                sum_dual = None
                last_restart_primal = None
                last_restart_dual = None
                initial_primal_weight = 0.0
                initial_step_size = 0.0
                total_pdlp_iters = 0
                total_pdhg_iters = 0
                last_candidate_kkt = 0.0
                last_restart_kkt = 0.0
                sum_weight = 0.0
                iters_since_restart = 0

            result = Solution(
                ProblemCategory(sol_ret.problem_type),
                dict(zip(data_model_obj.get_variable_names(), primal_solution)), # noqa
                sol_ret.lp_solution.get_solve_time(),
                primal_solution,
                dual_solution,
                reduced_cost,
                current_primal,
                current_dual,
                initial_primal_avg,
                initial_dual_avg,
                current_ATY,
                sum_primal,
                sum_dual,
                last_restart_primal,
                last_restart_dual,
                initial_primal_weight,
                initial_step_size,
                total_pdlp_iters,
                total_pdhg_iters,
                last_candidate_kkt,
                last_restart_kkt,
                sum_weight,
                iters_since_restart,
                LPTerminationStatus(sol_ret.lp_solution.get_termination_status(0)),
                ErrorStatus(int(sol_ret.lp_solution.get_error_status().get_error_type())),
                sol_ret.lp_solution.get_error_status().what().decode('utf-8'),
                sol_ret.lp_solution.get_l2_primal_residual(0),
                sol_ret.lp_solution.get_l2_dual_residual(0),
                sol_ret.lp_solution.get_objective_value(0),
                sol_ret.lp_solution.get_dual_objective_value(0),
                sol_ret.lp_solution.get_gap(0),
                sol_ret.lp_solution.get_num_iterations(0),
                sol_ret.lp_solution.is_solved_by_pdlp(0),
            )
            # Clean up C++ solution object now that we've copied all data to Python
            delete_lp_solution(sol_ret.lp_solution)
            return result
        else:
            # Batch mode - simpler return structure
            result = Solution(
                problem_category=ProblemCategory(sol_ret.problem_type),
                vars=dict(zip(data_model_obj.get_variable_names(), primal_solution)), # noqa
                solve_time=sol_ret.lp_solution.get_solve_time(),
                primal_solution=primal_solution,
                dual_solution=dual_solution,
                reduced_cost=reduced_cost,
                termination_status=LPTerminationStatus(sol_ret.lp_solution.get_termination_status(0)),
                error_status=ErrorStatus(int(sol_ret.lp_solution.get_error_status().get_error_type())),
                error_message=sol_ret.lp_solution.get_error_status().what().decode('utf-8'),
                primal_residual=sol_ret.lp_solution.get_l2_primal_residual(0),
                dual_residual=sol_ret.lp_solution.get_l2_dual_residual(0),
                primal_objective=sol_ret.lp_solution.get_objective_value(0),
                dual_objective=sol_ret.lp_solution.get_dual_objective_value(0),
                gap=sol_ret.lp_solution.get_gap(0),
                nb_iterations=sol_ret.lp_solution.get_num_iterations(0),
                solved_by_pdlp=sol_ret.lp_solution.is_solved_by_pdlp(0),
            )

        # Clean up C++ solution object now that we've copied all data to Python
        delete_lp_solution(sol_ret.lp_solution)
        return result


def Solve(py_data_model_obj, settings, mip=False):

    cdef DataModel data_model_obj = <DataModel>py_data_model_obj
    cdef unique_ptr[solver_settings_t[int, double]] unique_solver_settings

    unique_solver_settings.reset(new solver_settings_t[int, double]())

    data_model_obj.variable_types = type_cast(
        data_model_obj.variable_types, "S1", "variable_types"
    )

    set_solver_setting(
        unique_solver_settings, settings, data_model_obj, mip
    )
    data_model_obj.set_data_model_view()

    return create_solution(move(call_solve(
        data_model_obj.c_data_model_view.get(),
        unique_solver_settings.get(),
    )), data_model_obj)


cdef set_and_insert_vector(
        DataModel data_model_obj,
        vector[data_model_view_t[int, double] *]& data_model_views):
    data_model_obj.set_data_model_view()
    data_model_views.push_back(data_model_obj.c_data_model_view.get())


def BatchSolve(py_data_model_list, settings):
    cdef unique_ptr[solver_settings_t[int, double]] unique_solver_settings
    unique_solver_settings.reset(new solver_settings_t[int, double]())

    if settings.get_pdlp_warm_start_data() is not None:  # noqa
        raise Exception("Cannot use warmstart data with Batch Solve")
    set_solver_setting(unique_solver_settings, settings)

    cdef vector[data_model_view_t[int, double] *] data_model_views

    for data_model_obj in py_data_model_list:
        set_and_insert_vector(<DataModel>data_model_obj, data_model_views)

    cdef pair[
        vector[unique_ptr[solver_ret_t]],
        double] batch_solve_result = (
        move(call_batch_solve(data_model_views, unique_solver_settings.get())) # noqa
    )

    cdef vector[unique_ptr[solver_ret_t]] c_solutions = (
        move(batch_solve_result.first)
    )
    cdef double solve_time = batch_solve_result.second

    solutions = [] * len(py_data_model_list)
    for i in range(c_solutions.size()):
        solutions.append(
            create_solution(
                move(c_solutions[i]),
                <DataModel>py_data_model_list[i],
                True
            )
        )

    return solutions, solve_time
