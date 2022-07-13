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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -ldl -lpthread")
ENDIF(WIN32)

# For OpenMP
# openmp not support well for now on windows
if (NOT WIN32)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        add_definitions(-DWITH_OMP)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
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

add_subdirectory(faster_tokenizer)

if(WITH_PYTHON)

add_subdirectory(python)

if(WIN32)
add_custom_target(build_tokenizers_bdist_wheel ALL
    COMMAND ${CMAKE_COMMAND} -E env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    COMMENT "Packing whl packages------>>>"
    DEPENDS copy_python_tokenizers)
else()
add_custom_target(build_tokenizers_bdist_wheel ALL
    COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel --plat-name=manylinux1_x86_64
    COMMENT "Packing whl packages------>>>"
    DEPENDS copy_python_tokenizers)
endif()

else(WITH_PYTHON)

set(CPP_PACKAGE_DIR ${CMAKE_BINARY_DIR}/faster_tokenizer_cpp)
add_custom_target(build_cpp_package_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CPP_PACKAGE_DIR}/lib ${CPP_PACKAGE_DIR}/include ${CPP_PACKAGE_DIR}/third_party/include)

# copy headers
add_custom_target(copy_headers ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
    ${CMAKE_SOURCE_DIR}/faster_tokenizer/include ${CPP_PACKAGE_DIR}/include
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
set(TOKENIZER_CORE_NAME "core_tokenizers")
set(TOKENIZER_CORE_PATH ${CMAKE_BINARY_DIR}/faster_tokenizer/src/)
if (WIN32)
    set(TOKENIZER_CORE_LIBS_PATH "${TOKENIZER_CORE_PATH}/${TOKENIZER_CORE_NAME}.dll ${TOKENIZER_CORE_PATH}/lib${TOKENIZER_CORE_NAME}.lib")
else()
    set(TOKENIZER_CORE_LIBS_PATH "${TOKENIZER_CORE_PATH}/lib${TOKENIZER_CORE_NAME}.so")
endif()

add_custom_target(copy_shared_library ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${TOKENIZER_CORE_LIBS_PATH} ${CPP_PACKAGE_DIR}/lib
    DEPENDS build_cpp_package_dir core_tokenizers)

add_custom_target(create_commit_id_file ALL
    COMMAND ${GIT_EXECUTABLE} log -1 --format=%H > ${CPP_PACKAGE_DIR}/commit.log
    DEPENDS copy_shared_library)
endif(WITH_PYTHON)
    