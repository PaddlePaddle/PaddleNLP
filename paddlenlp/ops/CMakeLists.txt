# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(FasterTransformer LANGUAGES C CXX CUDA)

find_package(CUDA 10.1 REQUIRED)

INCLUDE(ExternalProject)

set(CXX_STD "11" CACHE STRING "C++ standard")

option(ON_INFER        "Compile with inference. "                                  OFF)
option(WITH_GPU        "Compile with GPU/CPU, default use CPU."                    ON)
option(USE_TENSORRT    "Compile with TensorRT."                                    OFF)

if(NOT WITH_GPU)
  message(FATAL_ERROR "Faster transformer custom op doesn't support CPU. Please add the flag -DWITH_GPU=ON to use GPU. ")
endif()

set(decoding_op_files fusion_decoding_op.cc fusion_decoding_op.cu)

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

set(THIRD_PATH "third-party")
set(THIRD_PARTY_NAME "fastertransformer")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(external/boost)

set(OPS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/allocator.h allocator_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/${THIRD_PARTY_NAME}/fastertransformer/allocator.h allocator_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/common.h common_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/${THIRD_PARTY_NAME}/fastertransformer/common.h common_dst)

file(TO_NATIVE_PATH ${OPS_SOURCE_DIR}/patches/FasterTransformer/CMakeLists.txt cmakelists_src)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${THIRD_PATH}/${THIRD_PARTY_NAME}/CMakeLists.txt cmakelists_dst)

set(FT_PATCH_COMMAND cp ${allocator_src} ${allocator_dst} | cp ${common_src} ${common_dst} | cp ${cmakelists_src} ${cmakelists_dst})

ExternalProject_Add(
  extern_${THIRD_PARTY_NAME}
  GIT_REPOSITORY    https://github.com/NVIDIA/FasterTransformer.git
  GIT_TAG           v3.1
  PREFIX            ${THIRD_PATH}
  SOURCE_DIR        ${THIRD_PATH}/${THIRD_PARTY_NAME}
  PATCH_COMMAND     ${FT_PATCH_COMMAND}
  BINARY_DIR        ${THIRD_PATH}/build
  INSTALL_COMMAND   ""
  CMAKE_ARGS        -DCMAKE_BUILD_TYPE=Release -DSM=${SM} -DBUILD_PD=ON -DPY_CMD=${PY_CMD}
)
ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} BINARY_DIR)
ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} SOURCE_DIR)
ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} SOURCE_SUBDIR)

set(FT_INCLUDE_PATH ${SOURCE_DIR}/${SOURCE_SUBDIR})
set(FT_LIB_PATH ${BINARY_DIR}/lib)

include_directories(
  ${FT_INCLUDE_PATH}
)

link_directories(
  ${FT_LIB_PATH}
)

add_subdirectory(src)
