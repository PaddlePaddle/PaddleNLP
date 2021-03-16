cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(FasterTransformer LANGUAGES CXX CUDA)

find_package(CUDA 10.1 REQUIRED)

INCLUDE(ExternalProject)

set(CXX_STD "11" CACHE STRING "C++ standard")

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

# Setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")    
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

if (SM STREQUAL 80 OR
    SM STREQUAL 86 OR
    SM STREQUAL 70 OR
    SM STREQUAL 75 OR
    SM STREQUAL 61 OR
    SM STREQUAL 60)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM},code=\\\"sm_${SM},compute_${SM}\\\"")
  if (SM STREQUAL 70 OR SM STREQUAL 75 OR SM STREQUAL 80 OR SM STREQUAL 86)
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
  endif()
message("-- Assign GPU architecture (sm=${SM})")

else()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                      -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                      -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                      ")

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")

message("-- Assign GPU architecture (sm=70,75)")
endif()

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include)

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

set(THIRD_PARTY_PATH "third-party")
set(THIRD_PARTY_NAME "fastertransformer")

ExternalProject_Add(
  extern_${THIRD_PARTY_NAME}
  GIT_REPOSITORY    https://github.com/NVIDIA/DeepLearningExamples.git
  GIT_TAG           master
  PREFIX            ${THIRD_PARTY_PATH}
  SOURCE_DIR        ${THIRD_PARTY_PATH}/${THIRD_PARTY_NAME}
  SOURCE_SUBDIR     "FasterTransformer/v3.1"
  BINARY_DIR        ${THIRD_PARTY_PATH}/build
  INSTALL_COMMAND   ""
  CMAKE_ARGS        -DCMAKE_BUILD_TYPE=Release -DSM=${SM}
)
ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} BINARY_DIR)
ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} SOURCE_DIR)
ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} SOURCE_SUBDIR)

set(FT_INCLUDE_PATH ${SOURCE_DIR}/${SOURCE_SUBDIR})
set(FT_LIB_PATH ${BINARY_DIR}/lib)

include_directories(
  ${FT_INCLUDE_PATH}
)

if(NOT PY_CMD)
  set(PYTHON_PATH "python" CACHE STRING "Python path")
else()
  set(PYTHON_PATH ${PY_CMD} CACHE STRING "Python path")
endif()

execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import paddle; print(paddle.sysconfig.get_include())"
                RESULT_VARIABLE _INC_PYTHON_SUCCESS
                OUTPUT_VARIABLE _INC_PYTHON_VALUES)
if (NOT _INC_PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Python config Error.")
endif()
string(REGEX REPLACE ";" "\\\\;" _INC_PYTHON_VALUES ${_INC_PYTHON_VALUES})
string(REGEX REPLACE "\n" ";" _INC_PYTHON_VALUES ${_INC_PYTHON_VALUES})
list(GET _INC_PYTHON_VALUES 0 PY_INCLUDE_DIR)

list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})
list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR}/third_party)

include_directories(
  ${COMMON_HEADER_DIRS}
)

execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import paddle; print(paddle.sysconfig.get_lib())"
                RESULT_VARIABLE _LIB_PYTHON_SUCCESS
                OUTPUT_VARIABLE _LIB_PYTHON_VALUES)
if (NOT _LIB_PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Python config Error.")
endif()
string(REGEX REPLACE ";" "\\\\;" _LIB_PYTHON_VALUES ${_LIB_PYTHON_VALUES})
string(REGEX REPLACE "\n" ";" _LIB_PYTHON_VALUES ${_LIB_PYTHON_VALUES})
list(GET _LIB_PYTHON_VALUES 0 PY_LIB_DIR)
list(APPEND COMMON_LIB_DIRS ${PY_LIB_DIR})

link_directories(
  ${COMMON_LIB_DIRS}
)

link_directories(
  ${FT_LIB_PATH}
)

add_subdirectory(src)
