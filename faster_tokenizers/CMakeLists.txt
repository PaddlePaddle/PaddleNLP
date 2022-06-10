cmake_minimum_required(VERSION 3.10)

project(tokenizers LANGUAGES CXX C VERSION 1.0)

option(WITH_TESTING     "Compile PaddleNLP tokenizers with unit testing"        ON)
option(WITH_PYTHON      "Compile PaddlePaddle with python interpreter"  ON)

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
    add_definitions(-DWITH_OMP)
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

if(WITH_PYTHON)

add_subdirectory(python)
add_custom_target(build_tokenizers_bdist_wheel ALL
    COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    DEPENDS copy_python_tokenizers)

else(WITH_PYTHON)

set(CPP_PACKAGE_DIR ${CMAKE_BINARY_DIR}/faster_tokenizers_cpp)
add_custom_target(build_cpp_package_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CPP_PACKAGE_DIR}/lib ${CPP_PACKAGE_DIR}/include ${CPP_PACKAGE_DIR}/third_party/include)

    # copy headers
add_custom_target(copy_headers ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
    ${CMAKE_SOURCE_DIR}/faster_tokenizers/include ${CPP_PACKAGE_DIR}/include
    DEPENDS build_cpp_package_dir)

add_custom_target(copy_third_party_headers ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
    ${GFLAGS_INCLUDE_DIR} ${ICU_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIR} ${JSON_INCLUDE_DIR} ${RE2_INCLUDE_DIR}
    ${CPP_PACKAGE_DIR}/third_party/include
    DEPENDS build_cpp_package_dir)

add_custom_target(copy_boost_headers ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
    ${BOOST_INCLUDE_DIR}/boost ${CPP_PACKAGE_DIR}/third_party/include/boost
    DEPENDS build_cpp_package_dir)

add_custom_target(remove_pybind_headers ALL
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CPP_PACKAGE_DIR}/include/pybind
    DEPENDS copy_headers)


# copy library
add_custom_target(copy_shared_library ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/faster_tokenizers/src/libcore_tokenizers.so ${CPP_PACKAGE_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/third_party/install/icu/lib/libicuuc.so.70 ${CPP_PACKAGE_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/third_party/install/icu/lib/libicudata.so.70 ${CPP_PACKAGE_DIR}/lib
    DEPENDS build_cpp_package_dir core_tokenizers)

add_custom_target(create_commit_id_file ALL
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%H > ${CPP_PACKAGE_DIR}/commit.log
    DEPENDS copy_shared_library)
endif(WITH_PYTHON)
    