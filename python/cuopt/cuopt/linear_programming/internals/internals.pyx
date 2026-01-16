# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libc.stdint cimport uintptr_t

import numpy as np
import ctypes
from numba.cuda.api import from_cuda_array_interface


cdef extern from "Python.h":
    cdef cppclass PyObject


cdef extern from "cuopt/linear_programming/utilities/callbacks_implems.hpp" namespace "cuopt::internals":  # noqa
    cdef cppclass Callback:
        pass

    cdef cppclass default_get_solution_callback_t(Callback):
        void setup() except +
        void get_solution(void* data, void* objective_value) except +
        PyObject* pyCallbackClass

    cdef cppclass default_set_solution_callback_t(Callback):
        void setup() except +
        void set_solution(void* data, void* objective_value) except +
        PyObject* pyCallbackClass


cdef class PyCallback:

    def get_numba_matrix(self, data, shape, typestr):

        sizeofType = 4 if typestr == "float32" else 8
        desc = {
            'shape': (shape,),
            'strides': None,
            'typestr': typestr,
            'data': (data, True),
            'version': 3,
        }

        data = from_cuda_array_interface(desc, None, False)
        return data

    def get_numpy_array(self, data, shape, typestr):
        ctype = ctypes.c_float if typestr == "float32" else ctypes.c_double
        buf_type = ctype * shape
        buf = buf_type.from_address(data)
        numpy_array = np.ctypeslib.as_array(buf)
        return numpy_array

cdef class GetSolutionCallback(PyCallback):

    cdef default_get_solution_callback_t native_callback

    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject *><void*>self

    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)


cdef class SetSolutionCallback(PyCallback):

    cdef default_set_solution_callback_t native_callback

    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject *><void*>self

    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)
