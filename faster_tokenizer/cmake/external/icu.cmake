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
include(CMakeParseArguments)
include(ExternalProject)
include (ByproductsICU)
SET(ICU_PREFIX_DIR    ${THIRD_PARTY_PATH}/icu)
SET(ICU_INSTALL_DIR   ${THIRD_PARTY_PATH}/install/icu)
SET(ICU_REPOSITORY    ${GIT_URL}/unicode-org/icu.git)
SET(ICU_TAG           release-70-1)
set(FIND_OR_BUILD_ICU_DIR ${CMAKE_CURRENT_LIST_DIR})

set(HOST_CFLAGS "${CMAKE_C_FLAGS}")
set(HOST_CXXFLAGS "${CMAKE_CXX_FLAGS}")
set(HOST_CC "${CMAKE_C_COMPILER}")
set(HOST_CXX "${CMAKE_CXX_COMPILER}")
set(HOST_LDFLAGS "${CMAKE_MODULE_LINKER_FLAGS}")

set(HOST_ENV_CMAKE ${CMAKE_COMMAND} -E env
        CC=${HOST_CC}
        CXX=${HOST_CXX}
        CFLAGS=${HOST_CFLAGS}
        CXXFLAGS=${HOST_CXXFLAGS}
        LDFLAGS=${HOST_LDFLAGS}
)

# predict host libraries
set(ICU_STATIC TRUE)
GetICUByproducts(${ICU_INSTALL_DIR} ICU_LIBRARIES ICU_INCLUDE_DIRS ICU_BASE_NAMES)
INCLUDE_DIRECTORIES(${ICU_INCLUDE_DIRS})

if(NOT WIN32)
ExternalProject_Add(
        extern_icu
        ${EXTERNAL_PROJECT_LOG_ARGS}
        ${SHALLOW_CLONE}
        GIT_REPOSITORY    ${ICU_REPOSITORY}
        GIT_TAG           ${ICU_TAG}
        GIT_PROGRESS      1
        PREFIX            ${ICU_PREFIX_DIR}
        UPDATE_COMMAND    ""
        CONFIGURE_COMMAND ${HOST_ENV_CMAKE} ../extern_icu/icu4c/source/runConfigureICU "Linux/gcc" --enable-static --disable-shared --enable-rpath
        BUILD_COMMAND make -j4
        INSTALL_COMMAND make install prefix="" DESTDIR=${ICU_INSTALL_DIR} install
        BUILD_BYPRODUCTS ${ICU_LIBRARIES}
)
else()
ExternalProject_Add(
        extern_icu
        ${EXTERNAL_PROJECT_LOG_ARGS}
        ${SHALLOW_CLONE}
        GIT_REPOSITORY    ${ICU_REPOSITORY}
        GIT_TAG           ${ICU_TAG}
        GIT_PROGRESS      1
        PREFIX            ${ICU_PREFIX_DIR}
        UPDATE_COMMAND    ""
        CONFIGURE_COMMAND msbuild ..\\extern_icu\\icu4c\\source\\allinone\\allinone.sln /p:Configuration=Release /p:Platform=x64 /p:RuntimeLibrary=MT_StaticRelease /p:SkipUWP=true
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ../extern_icu/icu4c/include ${ICU_INSTALL_DIR}/include
                     && ${CMAKE_COMMAND} -E copy_directory ../extern_icu/icu4c/lib64 ${ICU_INSTALL_DIR}/lib64
        BUILD_BYPRODUCTS ${ICU_LIBRARIES}
)
endif()

list(LENGTH ICU_LIBRARIES ICU_LIB_LEN)
MATH(EXPR ICU_LIB_LEN "${ICU_LIB_LEN}-1")

# icui18n icudata icuuc icuio icutu
foreach(ICU_IDX RANGE ${ICU_LIB_LEN})
  list(GET ICU_LIBRARIES ${ICU_IDX} ICU_LIB)
  list(GET ICU_BASE_NAMES ${ICU_IDX} ICU_BASE_NAME)
  ADD_LIBRARY("icu${ICU_BASE_NAME}" STATIC IMPORTED GLOBAL)
  SET_PROPERTY(TARGET "icu${ICU_BASE_NAME}" PROPERTY IMPORTED_LOCATION ${ICU_LIB})
  ADD_DEPENDENCIES("icu${ICU_BASE_NAME}" extern_icu)
  list(APPEND ICU_INTERFACE_LINK_LIBRARIES "icu${ICU_BASE_NAME}")
endforeach()

if(WIN32)
ADD_LIBRARY("icudata" ALIAS "icudt")
endif()