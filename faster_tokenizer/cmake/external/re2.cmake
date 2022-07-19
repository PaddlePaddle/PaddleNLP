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
INCLUDE(ExternalProject)

SET(RE2_PREFIX_DIR    ${THIRD_PARTY_PATH}/re2)
SET(RE2_INSTALL_DIR   ${THIRD_PARTY_PATH}/install/re2)
# As we add extra features for utf8proc, we use the non-official repo
SET(RE2_REPOSITORY    ${GIT_URL}/google/re2.git)
SET(RE2_TAG           2022-04-01)

IF(WIN32)
  SET(RE2_LIBRARIES     "${RE2_INSTALL_DIR}/lib/re2.lib")
  add_definitions(-DRE2_STATIC)
ELSE(WIN32)
  SET(RE2_LIBRARIES     "${RE2_INSTALL_DIR}/lib64/libre2.a")
ENDIF(WIN32)

SET(RE2_INCLUDE_DIR ${RE2_INSTALL_DIR}/include)
INCLUDE_DIRECTORIES(${RE2_INCLUDE_DIR})

ExternalProject_Add(
  extern_re2
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  GIT_REPOSITORY        ${RE2_REPOSITORY}
  GIT_TAG               ${RE2_TAG}
  PREFIX                ${RE2_PREFIX_DIR}
  UPDATE_COMMAND        ""
  CMAKE_ARGS            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                        -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                        -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                        -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
                        -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
                        -DCMAKE_INSTALL_PREFIX:PATH=${RE2_INSTALL_DIR}
                        -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
  BUILD_BYPRODUCTS     ${RE2_LIBRARIES}
)

ADD_LIBRARY(re2 STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET re2 PROPERTY IMPORTED_LOCATION ${RE2_LIBRARIES})
ADD_DEPENDENCIES(re2 extern_re2)
