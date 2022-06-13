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

include(ExternalProject)

set(BOOST_PROJECT       "extern_boost")
set(BOOST_VER   "1.79.0")
set(BOOST_URL   "https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.zip" CACHE STRING "" FORCE)

MESSAGE(STATUS "BOOST_VERSION: ${BOOST_VER}, BOOST_URL: ${BOOST_URL}")

set(BOOST_PREFIX_DIR ${THIRD_PARTY_PATH}/boost)

set(BOOST_INCLUDE_DIR "${THIRD_PARTY_PATH}/boost/src/extern_boost" CACHE PATH "boost include directory." FORCE)
set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)

include_directories(${BOOST_INCLUDE_DIR})

if(WIN32 AND MSVC_VERSION GREATER_EQUAL 1600)
    add_definitions(-DBOOST_HAS_STATIC_ASSERT)
endif()

ExternalProject_Add(
    ${BOOST_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    URL                   ${BOOST_URL}
    URL_HASH              SHA256=3634f9a85759311f321e587eace21799c0d0c946ff933e477a2f98885c54bbff
    PREFIX                ${BOOST_PREFIX_DIR}
    CONFIGURE_COMMAND     ""
    BUILD_COMMAND         ""
    INSTALL_COMMAND       ""
    UPDATE_COMMAND        ""
    )

add_library(boost INTERFACE)
add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)
add_dependencies(boost ${BOOST_PROJECT})
set(Boost_INCLUDE_DIR ${BOOST_INCLUDE_DIR})
