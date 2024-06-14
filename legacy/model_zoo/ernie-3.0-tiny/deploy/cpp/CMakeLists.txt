# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")
include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)
include(${FASTDEPLOY_INSTALL_DIR}/utils/gflags.cmake)

include_directories(${FASTDEPLOY_INCS})

add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer_demo.cc)
add_dependencies(infer_demo gflags)

if(UNIX AND (NOT APPLE) AND (NOT ANDROID))
  target_link_libraries(infer_demo ${FASTDEPLOY_LIBS} gflags pthread)
else()
  target_link_libraries(infer_demo ${FASTDEPLOY_LIBS} gflags)
endif()