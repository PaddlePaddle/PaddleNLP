cmake_minimum_required(VERSION 3.10)

project(tokenizers LANGUAGES CXX C VERSION 1.0)

option(WITH_TESTING     "Compile PaddleNLP tokenizers with unit testing"        ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
else()
    set(CMAKE_CXX_STANDARD 11)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
IF(WIN32)
# Need to add flags for windows
ELSE(WIN32)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread")
ENDIF(WIN32)

# For OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    add_compile_definitions(WITH_OMP)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(TOKENIZERS_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(TOKENIZERS_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)
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

add_subdirectory(faster_tokenizers)
add_subdirectory(python)

add_custom_target(build_tokenizers_bdist_wheel ALL
    COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    DEPENDS copy_python_tokenizers)
