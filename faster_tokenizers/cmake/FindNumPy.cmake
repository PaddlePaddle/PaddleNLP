# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Find the Python NumPy package
# PYTHON_NUMPY_INCLUDE_DIR
# NUMPY_FOUND
# will be set by this script

cmake_minimum_required(VERSION 2.6)

if(NOT PYTHON_EXECUTABLE)
  if(NumPy_FIND_QUIETLY)
    find_package(PythonInterp QUIET)
  else()
    find_package(PythonInterp)
    set(_numpy_out 1)
  endif()
endif()

if (PYTHON_EXECUTABLE)
  # write a python script that finds the numpy path
  file(WRITE ${PROJECT_BINARY_DIR}/FindNumpyPath.py
      "try: import numpy; print(numpy.get_include())\nexcept:pass\n")

  # execute the find script
  exec_program("${PYTHON_EXECUTABLE}" ${PROJECT_BINARY_DIR}
    ARGS "FindNumpyPath.py"
    OUTPUT_VARIABLE NUMPY_PATH)
elseif(_numpy_out)
  message(STATUS "Python executable not found.")
endif(PYTHON_EXECUTABLE)

find_path(PYTHON_NUMPY_INCLUDE_DIR numpy/arrayobject.h
  HINTS "${NUMPY_PATH}" "${PYTHON_INCLUDE_PATH}")

if(PYTHON_NUMPY_INCLUDE_DIR)
  set(PYTHON_NUMPY_FOUND 1 CACHE INTERNAL "Python numpy found")
endif(PYTHON_NUMPY_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NumPy DEFAULT_MSG PYTHON_NUMPY_INCLUDE_DIR)
