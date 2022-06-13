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

SET(ICU_PREFIX_DIR    ${THIRD_PARTY_PATH}/icu)
SET(ICU_INSTALL_DIR   ${THIRD_PARTY_PATH}/install/icu)
SET(ICU_REPOSITORY    ${GIT_URL}/unicode-org/icu.git)
SET(ICU_TAG           release-70-1)
SET(ICU_INCLUDE_DIR   "${ICU_INSTALL_DIR}/include" CACHE PATH "icu include directory." FORCE)
set(FIND_OR_BUILD_ICU_DIR ${CMAKE_CURRENT_LIST_DIR})

INCLUDE_DIRECTORIES(${ICU_INCLUDE_DIR})


IF(WIN32)
SET(ICUUC_LIBRARIES     "${ICU_INSTALL_DIR}/lib/icuuc_static.lib")
add_definitions(-DUTF8PROC_STATIC)
ExternalProject_Add(
        extern_icu
        ${EXTERNAL_PROJECT_LOG_ARGS}
        ${SHALLOW_CLONE}
        GIT_REPOSITORY    ${ICU_REPOSITORY}
        GIT_TAG           ${ICU_TAG}
        GIT_PROGRESS      1
        PREFIX            ${ICU_PREFIX_DIR}
        CONFIGURE_COMMAND "msbuild ../extern_icu/icu4c/source\allinone\allinone.sln /p:Configuration=Release /p:Platform=x64"
      )
ELSE(WIN32)
SET(ICUUC_LIBRARIES     "${ICU_INSTALL_DIR}/lib/libicuuc.so")
SET(ICUDATA_LIBRARIES     "${ICU_INSTALL_DIR}/lib/libicudata.so")

ExternalProject_Add(
        extern_icu
        ${EXTERNAL_PROJECT_LOG_ARGS}
        ${SHALLOW_CLONE}
        GIT_REPOSITORY    ${ICU_REPOSITORY}
        GIT_TAG           ${ICU_TAG}
        PREFIX            ${ICU_PREFIX_DIR}
        CONFIGURE_COMMAND ../extern_icu/icu4c/source/runConfigureICU Linux/gcc
        UPDATE_COMMAND    ""
        BUILD_COMMAND make -j4
        INSTALL_COMMAND make prefix="" DESTDIR=${ICU_INSTALL_DIR} install
      )
ENDIF(WIN32)

ADD_LIBRARY(icuuc SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET icuuc PROPERTY IMPORTED_LOCATION ${ICUUC_LIBRARIES})

ADD_LIBRARY(icudata SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET icudata PROPERTY IMPORTED_LOCATION ${ICUDATA_LIBRARIES})

ADD_LIBRARY(icu INTERFACE IMPORTED GLOBAL)
SET_PROPERTY(TARGET icu PROPERTY INTERFACE_LINK_LIBRARIES icuuc icudata)
ADD_DEPENDENCIES(icu extern_icu)
