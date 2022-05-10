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
    target_link_libraries(${TARGET_NAME} stdc++fs)

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

      generate_dummy_static_lib(LIB_NAME ${TARGET_NAME} FILE_PATH ${target_SRCS} GENERATOR "generic.cmake:cc_library")

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
