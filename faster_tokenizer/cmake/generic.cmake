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

function(cc_library TARGET_NAME)
  set(options STATIC static SHARED shared INTERFACE interface)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(WIN32)
      # add libxxx.lib prefix in windows
      set(${TARGET_NAME}_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${TARGET_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE STRING "output library name for target ${TARGET_NAME}")
  endif(WIN32)
  if(cc_library_SRCS)
      if(cc_library_SHARED OR cc_library_shared) # build *.so
        add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
      elseif(cc_library_INTERFACE OR cc_library_interface)
        generate_dummy_static_lib(LIB_NAME ${TARGET_NAME} FILE_PATH ${target_SRCS} GENERATOR "generic.cmake:cc_library")
      else()
        add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
      endif()
    if(cc_library_DEPS)
      # remove link to python, see notes at:
      # https://github.com/pybind/pybind11/blob/master/docs/compiling.rst#building-manually
      if("${cc_library_DEPS};" MATCHES "python;")
        list(REMOVE_ITEM cc_library_DEPS python)
        add_dependencies(${TARGET_NAME} python)
        if(WIN32)
          target_link_libraries(${TARGET_NAME} ${PYTHON_LIBRARIES})
        else()
          target_link_libraries(${TARGET_NAME} "-Wl,-undefined,dynamic_lookup")
        endif(WIN32)
      endif()
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
    endif()
    # For C++ 17 filesystem
    # target_link_libraries(${TARGET_NAME} stdc++fs)

    # cpplint code style
    foreach(source_file ${cc_library_SRCS})
      string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        list(APPEND cc_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
      endif()
    endforeach()

  else(cc_library_SRCS)
    if(cc_library_DEPS)
      list(REMOVE_DUPLICATES cc_library_DEPS)
      set(dummy_FILE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.c")
      configure_file(${PROJECT_SOURCE_DIR}/cmake/dummy.c.in ${dummy_FILE_PATH} @ONLY)
      if(cc_library_SHARED OR cc_library_shared) # build *.so
        add_library(${TARGET_NAME} SHARED ${dummy_FILE_PATH})
      elseif(cc_library_INTERFACE OR cc_library_interface)
        generate_dummy_static_lib(LIB_NAME ${TARGET_NAME} FILE_PATH ${dummy_FILE_PATH} GENERATOR "generic.cmake:cc_library")
      else()
        add_library(${TARGET_NAME} STATIC ${dummy_FILE_PATH})
      endif()
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
    else()
      message(FATAL_ERROR "Please specify source files or libraries in cc_library(${TARGET_NAME} ...).")
    endif()
  endif(cc_library_SRCS)
endfunction(cc_library)

function(cc_test_build TARGET_NAME)
  if(WITH_TESTING AND NOT "$ENV{CI_SKIP_CPP_TEST}" STREQUAL "ON")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET_NAME} ${cc_test_SRCS})
    if(WIN32)
      if("${cc_test_DEPS};" MATCHES "python;")
        list(REMOVE_ITEM cc_test_DEPS python)
        target_link_libraries(${TARGET_NAME} ${PYTHON_LIBRARIES})
      endif()
    endif(WIN32)
    get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
    target_link_libraries(${TARGET_NAME} ${cc_test_DEPS} ${os_dependency_modules} tokenizers_gtest_main gtest glog)
    add_dependencies(${TARGET_NAME} ${cc_test_DEPS} gtest)
  endif()
endfunction()

function(cc_test_run TARGET_NAME)
  if(WITH_TESTING AND NOT "$ENV{CI_SKIP_CPP_TEST}" STREQUAL "ON")
    set(oneValueArgs "")
    set(multiValueArgs COMMAND ARGS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_test(NAME ${TARGET_NAME}
      COMMAND ${cc_test_COMMAND} ${cc_test_ARGS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # No unit test should exceed 2 minutes.
    if (WIN32)
        set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 150)
    endif()
    if (APPLE)
        set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 20)
    endif()
  elseif(WITH_TESTING AND NOT TEST ${TARGET_NAME})
    add_test(NAME ${TARGET_NAME} COMMAND ${CMAKE_COMMAND} -E echo CI skip ${TARGET_NAME}.)
  endif()
endfunction()

function(cc_test TARGET_NAME)
    # The environment variable `CI_SKIP_CPP_TEST` is used to skip the compilation
    # and execution of test in CI. `CI_SKIP_CPP_TEST` is set to ON when no files
  # other than *.py are modified.
  if(WITH_TESTING)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cc_test_build(${TARGET_NAME}
      SRCS ${cc_test_SRCS}
      DEPS ${cc_test_DEPS})
    add_test(NAME ${TARGET_NAME} COMMAND ${CMAKE_COMMAND} -E echo CI skip ${TARGET_NAME}.)
  endif()
endfunction(cc_test)

# create a dummy source file, then create a static library.
# LIB_NAME should be the static lib name.
# FILE_PATH should be the dummy source file path.
# GENERATOR should be the file name invoke this function.
# CONTENT should be some helpful info.
# example: generate_dummy_static_lib(mylib FILE_PATH /path/to/dummy.c GENERATOR mylib.cmake CONTENT "helpful info")
function(generate_dummy_static_lib)
  set(options "")
  set(oneValueArgs LIB_NAME FILE_PATH GENERATOR CONTENT)
  set(multiValueArgs "")
  cmake_parse_arguments(dummy "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT dummy_LIB_NAME)
    message(FATAL_ERROR "You must provide a static lib name.")
  endif()
  if(NOT dummy_FILE_PATH)
    set(dummy_FILE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${dummy_LIB_NAME}_dummy.c")
  endif()
  if(NOT dummy_GENERATOR)
    message(FATAL_ERROR "You must provide a generator file name.")
  endif()
  if(NOT dummy_CONTENT)
    set(dummy_CONTENT "${dummy_LIB_NAME}_dummy.c for lib ${dummy_LIB_NAME}")
  endif()

  configure_file(${PROJECT_SOURCE_DIR}/cmake/dummy.c.in ${dummy_FILE_PATH} @ONLY)
  add_library(${dummy_LIB_NAME} STATIC ${dummy_FILE_PATH})
endfunction()

function(paddle_protobuf_generate_cpp SRCS HDRS)
  if(NOT ARGN)
    message(
      SEND_ERROR
        "Error: paddle_protobuf_generate_cpp() called without any proto files")
    return()
  endif()

  set(${SRCS})
  set(${HDRS})

  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    set(_protobuf_protoc_src "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    set(_protobuf_protoc_hdr "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")
    list(APPEND ${SRCS} "${_protobuf_protoc_src}")
    list(APPEND ${HDRS} "${_protobuf_protoc_hdr}")

    add_custom_command(
      OUTPUT "${_protobuf_protoc_src}" "${_protobuf_protoc_hdr}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} -I${CMAKE_CURRENT_SOURCE_DIR}
              --cpp_out "${CMAKE_CURRENT_BINARY_DIR}" ${ABS_FIL}
      # Set `EXTERN_PROTOBUF_DEPEND` only if need to compile `protoc.exe`.
      DEPENDS ${ABS_FIL} ${EXTERN_PROTOBUF_DEPEND}
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM)
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS}
      ${${SRCS}}
      PARENT_SCOPE)
  set(${HDRS}
      ${${HDRS}}
      PARENT_SCOPE)
endfunction()

function(proto_library TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(proto_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  set(proto_srcs)
  set(proto_hdrs)
  paddle_protobuf_generate_cpp(proto_srcs proto_hdrs ${proto_library_SRCS})
  cc_library(
    ${TARGET_NAME}
    SRCS ${proto_srcs}
    DEPS ${proto_library_DEPS} protobuf)
endfunction()