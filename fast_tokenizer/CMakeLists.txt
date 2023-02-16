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

cmake_minimum_required(VERSION 3.10)

project(tokenizers LANGUAGES CXX C VERSION 1.0)

option(WITH_TESTING     "Compile PaddleNLP fast_tokenizer with unit testing"        OFF)
option(WITH_PYTHON      "Compile PaddleNLP fast_tokenizer with python interpreter"   ON)
option(WITH_ICU_LITE      "Compile PaddleNLP fast_tokenizer with lite icu library"   OFF)
option(USE_ABI0      "Set -D_GLIBCXX_USE_CXX11_ABI to 0"   OFF)

add_definitions(-DFASTTOKENIZER_LIB)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set (PUBLIC_DEPEND_LIBS "")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
    endif()
    if (USE_ABI0)
        message(STATUS "-D_GLIBCXX_USE_CXX11_ABI will be set to 0.")
        add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    else()
        message(STATUS "-D_GLIBCXX_USE_CXX11_ABI will be set to 1.")
        add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    endif()
else()
    set(CMAKE_CXX_STANDARD 11)
endif()

IF(WIN32)
# Need to add flags for windows
foreach(
    flag_var
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_MINSIZEREL
    CMAKE_C_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    elseif(NOT (${flag_var} MATCHES "/MT"))
      set(${flag_var} "${${flag_var}} /MT")
    endif()
    set(${flag_var} "${${flag_var}}" CACHE STRING "msvc compiler flags" FORCE)
endforeach()

add_definitions("-DNOMINMAX")
# windows build turn off warnings, use parallel compiling.

foreach(
  flag_var
  CMAKE_CXX_FLAGS
  CMAKE_CXX_FLAGS_DEBUG
  CMAKE_CXX_FLAGS_RELEASE
  CMAKE_CXX_FLAGS_MINSIZEREL
  CMAKE_CXX_FLAGS_RELWITHDEBINFO
  CMAKE_C_FLAGS
  CMAKE_C_FLAGS_DEBUG
  CMAKE_C_FLAGS_RELEASE
  CMAKE_C_FLAGS_MINSIZEREL
  CMAKE_C_FLAGS_RELWITHDEBINFO)
  string(REGEX REPLACE "/W[1-4]" " /W0 " ${flag_var} "${${flag_var}}")
endforeach()

foreach(flag_var CMAKE_CXX_FLAGS CMAKE_C_FLAGS)
    set(${flag_var} "${${flag_var}} /w")
endforeach()

# Windows Remove /Zi, /ZI for Release, MinSizeRel builds
foreach(flag_var
        CMAKE_C_FLAGS CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL)
if(${flag_var} MATCHES "/Z[iI]")
    string(REGEX REPLACE "/Z[iI]" "" ${flag_var} "${${flag_var}}")
endif()
endforeach()

set(CMAKE_SUPPRESS_REGENERATION ON)
set(CMAKE_STATIC_LIBRARY_PREFIX lib)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /bigobj")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /bigobj")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")

if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Zc:inline")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zc:inline")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zc:inline")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zc:inline")
endif()

foreach(flag_var CMAKE_SHARED_LINKER_FLAGS CMAKE_STATIC_LINKER_FLAGS
                CMAKE_EXE_LINKER_FLAGS CMAKE_LINKER_FLAGS)
set(${flag_var}
    "${${flag_var}} /ignore:4049 /ignore:4217 /ignore:4006 /ignore:4221")
set(${flag_var} "${${flag_var}} /NODEFAULTLIB:MSVCRT.LIB")
endforeach()

ELSE(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
    IF (NOT APPLE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl")
      IF (NOT ANDROID)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread")
      ELSE()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os")
      ENDIF()
    ENDIF()
    set (PUBLIC_DEPEND_LIBS ${CMAKE_DL_LIBS})
ENDIF(WIN32)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(TOKENIZERS_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR})

message("CMAKE_BUILD_TYPE = " ${CMAKE_BUILD_TYPE})
message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message("CMAKE_EXE_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}")

# config GIT_URL with github mirrors to speed up dependent repos clone
option(GIT_URL "Git URL to clone dependent repos" ${GIT_URL})
if(NOT GIT_URL)
    set(GIT_URL "https://github.com")
endif()

include_directories(${TOKENIZERS_INSTALL_INCLUDE_DIR})

include(generic)
include(third_party)

add_subdirectory(fast_tokenizer)

if(WITH_PYTHON)

add_subdirectory(python)

if (NOT APPLE AND NOT WIN32) # Linux
add_custom_target(build_tokenizers_bdist_wheel ALL
    COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel --plat-name=manylinux1_x86_64
    COMMENT "Packing whl packages------>>>"
    DEPENDS copy_python_tokenizers)
else()
add_custom_target(build_tokenizers_bdist_wheel ALL
    COMMAND ${CMAKE_COMMAND} -E env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    COMMENT "Packing whl packages------>>>"
    DEPENDS copy_python_tokenizers)
endif()

else(WITH_PYTHON) # Pack fast_tokenizer cpp lib

set(CPP_PACKAGE_DIR ${CMAKE_BINARY_DIR}/cpp/fast_tokenizer)
add_custom_target(build_cpp_package_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CPP_PACKAGE_DIR}/lib ${CPP_PACKAGE_DIR}/include ${CPP_PACKAGE_DIR}/third_party/include ${CPP_PACKAGE_DIR}/third_party/lib
    DEPENDS core_tokenizers)

# copy cmake
configure_file(${PROJECT_SOURCE_DIR}/FastTokenizer.cmake.in ${PROJECT_SOURCE_DIR}/FastTokenizer.cmake @ONLY)
file(COPY ${PROJECT_SOURCE_DIR}/FastTokenizer.cmake DESTINATION ${CPP_PACKAGE_DIR}/)

# copy headers
file(COPY ${PROJECT_SOURCE_DIR}/fast_tokenizer/ DESTINATION ${CPP_PACKAGE_DIR}/include/fast_tokenizer/
    FILES_MATCHING PATTERN "*.h"
    PATTERN "test" EXCLUDE
    PATTERN "demo" EXCLUDE
    PATTERN "pybind" EXCLUDE)

add_custom_target(copy_third_party_headers ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${GFLAGS_INCLUDE_DIR} ${ICU_INCLUDE_DIR} ${DART_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIR} ${JSON_INCLUDE_DIR} ${RE2_INCLUDE_DIR}
    ${CPP_PACKAGE_DIR}/third_party/include
    DEPENDS build_cpp_package_dir)

# copy library
set(TOKENIZER_CORE_NAME "core_tokenizers")
set(TOKENIZER_CORE_PATH ${CMAKE_BINARY_DIR}/fast_tokenizer)
if (WIN32)
    set(ICU_DLL_DIR ${CMAKE_BINARY_DIR}/third_party/icu/src/extern_icu/icu4c/bin64)
    set(ICU_LIB_DIR ${CMAKE_BINARY_DIR}/third_party/icu/src/extern_icu/icu4c/lib64)
    add_custom_target(copy_shared_library ALL
        COMMAND ${CMAKE_COMMAND} -E copy ${TOKENIZER_CORE_PATH}/${TOKENIZER_CORE_NAME}.dll ${TOKENIZER_CORE_PATH}/${TOKENIZER_CORE_NAME}.lib ${CPP_PACKAGE_DIR}/lib
        COMMAND ${CMAKE_COMMAND} -E copy ${ICU_DLL_DIR}/icudt70.dll ${ICU_DLL_DIR}/icuuc70.dll ${ICU_LIB_DIR}/icudt.lib ${ICU_LIB_DIR}/icuuc.lib ${CPP_PACKAGE_DIR}/third_party/lib
        DEPENDS build_cpp_package_dir core_tokenizers)
elseif(APPLE)
    set(TOKENIZER_CORE_LIBS_PATH "${TOKENIZER_CORE_PATH}/lib${TOKENIZER_CORE_NAME}.dylib")
    add_custom_target(copy_shared_library ALL
        COMMAND ${CMAKE_COMMAND} -E copy ${TOKENIZER_CORE_LIBS_PATH} ${CPP_PACKAGE_DIR}/lib
        DEPENDS build_cpp_package_dir core_tokenizers)
elseif(ANDROID)
        set(TOKENIZER_CORE_LIBS_PATH "${TOKENIZER_CORE_PATH}/lib${TOKENIZER_CORE_NAME}.so")
        add_custom_target(copy_shared_library ALL
            COMMAND ${CMAKE_COMMAND} -E copy ${TOKENIZER_CORE_LIBS_PATH} ${CPP_PACKAGE_DIR}/lib
            COMMAND ${CMAKE_STRIP} --strip-all ${CPP_PACKAGE_DIR}/lib/lib${TOKENIZER_CORE_NAME}.so
            DEPENDS build_cpp_package_dir core_tokenizers)
else()
    set(TOKENIZER_CORE_LIBS_PATH "${TOKENIZER_CORE_PATH}/lib${TOKENIZER_CORE_NAME}.so")
    add_custom_target(copy_shared_library ALL
        COMMAND ${CMAKE_COMMAND} -E copy ${TOKENIZER_CORE_LIBS_PATH} ${CPP_PACKAGE_DIR}/lib
        DEPENDS build_cpp_package_dir core_tokenizers)
endif()

add_custom_target(create_commit_id_file ALL
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%H > ${CPP_PACKAGE_DIR}/commit.log
    DEPENDS copy_shared_library)
endif(WITH_PYTHON)

