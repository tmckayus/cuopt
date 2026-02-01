# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector

from pylibraft.common.handle cimport *
from rmm.librmm.device_buffer cimport device_buffer

from cuopt.linear_programming.data_model.data_model cimport data_model_view_t


cdef extern from "cuopt/linear_programming/utilities/internals.hpp" namespace "cuopt::internals": # noqa
    cdef cppclass base_solution_callback_t

cdef extern from "cuopt/linear_programming/pdlp/solver_settings.hpp" namespace "cuopt::linear_programming": # noqa
    ctypedef enum pdlp_solver_mode_t "cuopt::linear_programming::pdlp_solver_mode_t": # noqa
        Stable1 "cuopt::linear_programming::pdlp_solver_mode_t::Stable1" # noqa
        Stable2 "cuopt::linear_programming::pdlp_solver_mode_t::Stable2" # noqa
        Methodical1 "cuopt::linear_programming::pdlp_solver_mode_t::Methodical1" # noqa
        Fast1 "cuopt::linear_programming::pdlp_solver_mode_t::Fast1" # noqa
        Stable3 "cuopt::linear_programming::pdlp_solver_mode_t::Stable3" # noqa

cdef extern from "cuopt/linear_programming/solver_settings.hpp" namespace "cuopt::linear_programming": # noqa

    cdef cppclass solver_settings_t[i_t, f_t]:
        solver_settings_t() except +

        void set_pdlp_warm_start_data(
            const f_t* current_primal_solution,
            const f_t* current_dual_solution,
            const f_t* initial_primal_average,
            const f_t* initial_dual_average,
            const f_t* current_ATY,
            const f_t* sum_primal_solutions,
            const f_t* sum_dual_solutions,
            const f_t* last_restart_duality_gap_primal_solution,
            const f_t* last_restart_duality_gap_dual_solution,
            i_t primal_size,
            i_t dual_size,
            f_t initial_primal_weight_,
            f_t initial_step_size_,
            i_t total_pdlp_iterations_,
            i_t total_pdhg_iterations_,
            f_t last_candidate_kkt_score_,
            f_t last_restart_kkt_score_,
            f_t sum_solution_weight_,
            i_t iterations_since_last_restart_) except +

        void set_parameter_from_string(
            const string& name,
            const string& value
        ) except +

        string get_parameter_as_string(const string& name) except +

        vector[string] get_parameter_names() except +

        # LP settings
        void set_initial_pdlp_primal_solution(
            const f_t* initial_primal_solution,
            i_t size
        ) except +
        void set_initial_pdlp_dual_solution(
            const f_t* initial_dual_solution,
            i_t size
        ) except +

        # MIP settings
        void add_initial_mip_solution(
            const f_t* initial_solution,
            i_t size
        ) except +
        void set_mip_callback(
            base_solution_callback_t* callback,
            void* user_data
        ) except +


cdef extern from "cuopt/linear_programming/optimization_problem.hpp" namespace "cuopt::linear_programming": # noqa
    ctypedef enum problem_category_t "cuopt::linear_programming::problem_category_t": # noqa
        LP "cuopt::linear_programming::problem_category_t::LP"
        MIP "cuopt::linear_programming::problem_category_t::MIP"
        IP "cuopt::linear_programming::problem_category_t::IP"

cdef extern from "cuopt/error.hpp" namespace "cuopt": # noqa
    ctypedef enum error_type_t "cuopt::error_type_t": # noqa
        Success "cuopt::error_type_t::Success" # noqa
        ValidationError "cuopt::error_type_t::ValidationError" # noqa
        OutOfMemoryError "cuopt::error_type_t::OutOfMemoryError" # noqa
        RuntimeError "cuopt::error_type_t::RuntimeError" # noqa

    cdef cppclass logic_error:
        error_type_t get_error_type() except +
        const char* what() except +

cdef extern from "cuopt/linear_programming/mip/solver_solution.hpp" namespace "cuopt::linear_programming": # noqa
    ctypedef enum mip_termination_status_t "cuopt::linear_programming::mip_termination_status_t": # noqa
        NoTermination "cuopt::linear_programming::mip_termination_status_t::NoTermination" # noqa
        Optimal "cuopt::linear_programming::mip_termination_status_t::Optimal"
        FeasibleFound "cuopt::linear_programming::mip_termination_status_t::FeasibleFound" # noqa
        Infeasible "cuopt::linear_programming::mip_termination_status_t::Infeasible" # noqa
        Unbounded "cuopt::linear_programming::mip_termination_status_t::Unbounded" # noqa
        TimeLimit "cuopt::linear_programming::mip_termination_status_t::TimeLimit" # noqa


cdef extern from "cuopt/linear_programming/pdlp/solver_solution.hpp" namespace "cuopt::linear_programming": # noqa
    ctypedef enum pdlp_termination_status_t "cuopt::linear_programming::pdlp_termination_status_t": # noqa
        NoTermination "cuopt::linear_programming::pdlp_termination_status_t::NoTermination" # noqa
        NumericalError "cuopt::linear_programming::pdlp_termination_status_t::NumericalError" # noqa
        Optimal "cuopt::linear_programming::pdlp_termination_status_t::Optimal" # noqa
        PrimalInfeasible "cuopt::linear_programming::pdlp_termination_status_t::PrimalInfeasible" # noqa
        DualInfeasible "cuopt::linear_programming::pdlp_termination_status_t::DualInfeasible" # noqa
        IterationLimit "cuopt::linear_programming::pdlp_termination_status_t::IterationLimit" # noqa
        TimeLimit "cuopt::linear_programming::pdlp_termination_status_t::TimeLimit" # noqa
        ConcurrentLimit "cuopt::linear_programming::pdlp_termination_status_t::ConcurrentLimit" # noqa
        PrimalFeasible "cuopt::linear_programming::pdlp_termination_status_t::PrimalFeasible" # noqa


cdef extern from "cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp" namespace "cuopt::linear_programming": # noqa
    cdef cppclass pdlp_warm_start_data_t[i_t, f_t]:
        pass  # Opaque type for warm start data

cdef extern from "cuopt/linear_programming/optimization_problem_solution_interface.hpp" namespace "cuopt::linear_programming": # noqa
    # LP Solution Interface
    cdef cppclass lp_solution_interface_t[i_t, f_t]:
        # Host memory accessors (work for both CPU and GPU solutions)
        const vector[f_t]& get_primal_solution_host() except +
        const vector[f_t]& get_dual_solution_host() except +
        const vector[f_t]& get_reduced_cost_host() except +

        # Error and metadata accessors
        logic_error get_error_status() except +
        pdlp_termination_status_t get_termination_status(i_t id) except +
        f_t get_solve_time() except +
        f_t get_objective_value(i_t id) except +
        f_t get_dual_objective_value(i_t id) except +
        i_t get_primal_solution_size() except +
        i_t get_dual_solution_size() except +
        i_t get_reduced_cost_size() except +

        # Additional termination info
        f_t get_l2_primal_residual(i_t id) except +
        f_t get_l2_dual_residual(i_t id) except +
        f_t get_gap(i_t id) except +
        i_t get_num_iterations(i_t id) except +
        bool is_solved_by_pdlp(i_t id) except +

        # Warm start data (throws for CPU solutions)
        const pdlp_warm_start_data_t[i_t, f_t]& get_pdlp_warm_start_data() except +

        # Individual warm start accessors (work for both GPU and CPU)
        bool has_warm_start_data() except +
        vector[f_t] get_current_primal_solution_host() except +
        vector[f_t] get_current_dual_solution_host() except +
        vector[f_t] get_initial_primal_average_host() except +
        vector[f_t] get_initial_dual_average_host() except +
        vector[f_t] get_current_ATY_host() except +
        vector[f_t] get_sum_primal_solutions_host() except +
        vector[f_t] get_sum_dual_solutions_host() except +
        vector[f_t] get_last_restart_duality_gap_primal_solution_host() except +
        vector[f_t] get_last_restart_duality_gap_dual_solution_host() except +
        f_t get_initial_primal_weight() except +
        f_t get_initial_step_size() except +
        i_t get_total_pdlp_iterations() except +
        i_t get_total_pdhg_iterations() except +
        f_t get_last_candidate_kkt_score() except +
        f_t get_last_restart_kkt_score() except +
        f_t get_sum_solution_weight() except +
        i_t get_iterations_since_last_restart() except +

    # MIP Solution Interface
    cdef cppclass mip_solution_interface_t[i_t, f_t]:
        # Host memory accessors (work for both CPU and GPU solutions)
        const vector[f_t]& get_solution_host() except +

        # Error and metadata accessors
        logic_error get_error_status() except +
        mip_termination_status_t get_termination_status() except +
        f_t get_objective_value() except +
        f_t get_mip_gap() except +
        f_t get_solution_bound() except +
        f_t get_solve_time() except +
        i_t get_solution_size() except +

        # Additional MIP stats
        f_t get_presolve_time() except +
        f_t get_max_constraint_violation() except +
        f_t get_max_int_violation() except +
        f_t get_max_variable_bound_violation() except +
        i_t get_num_nodes() except +
        i_t get_num_simplex_iterations() except +

cdef extern from "cuopt/linear_programming/utilities/cython_solve.hpp" namespace "cuopt::cython": # noqa
    cdef cppclass solver_ret_t:
        problem_category_t problem_type
        lp_solution_interface_t[int, double]* lp_solution
        mip_solution_interface_t[int, double]* mip_solution

    cdef unique_ptr[solver_ret_t] call_solve(
        data_model_view_t[int, double]* data_model,
        solver_settings_t[int, double]* solver_settings,
    ) except +

    # Helper functions to delete solution pointers (avoids Cython template issues)
    cdef void delete_lp_solution(lp_solution_interface_t[int, double]* ptr)
    cdef void delete_mip_solution(mip_solution_interface_t[int, double]* ptr)

    cdef pair[vector[unique_ptr[solver_ret_t]], double] call_batch_solve( # noqa
        vector[data_model_view_t[int, double] *] data_models,
        solver_settings_t[int, double]* solver_settings,
    ) except +
