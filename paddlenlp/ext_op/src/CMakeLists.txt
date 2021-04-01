cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(FasterTransformer LANGUAGES C CXX CUDA)

find_package(CUDA 10.1 REQUIRED)

INCLUDE(ExternalProject)

set(CXX_STD "11" CACHE STRING "C++ standard")

option(ON_INFER        "Compile with inference. "                                  OFF)
option(WITH_GPU        "Compile with GPU/CPU, default use CPU."                    ON)
option(USE_TENSORRT    "Compile with TensorRT."                                    OFF)

if(NOT WITH_GPU)
  message(FATAL_ERROR "Faster transformer custom op doesn't support CPU. Please add the flag -DWITH_GPU=ON to use GPU. ")
endif()

set(decoding_op_files fusion_decoding_op.cc fusion_decoding_op.cu)

if(ON_INFER)
  set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

  list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

  # Setting compiler flags
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")    
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

  if (SM STREQUAL 80 OR
      SM STREQUAL 86 OR
      SM STREQUAL 70 OR
      SM STREQUAL 75 OR
      SM STREQUAL 61 OR
      SM STREQUAL 60)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM},code=\\\"sm_${SM},compute_${SM}\\\"")
    if (SM STREQUAL 70 OR SM STREQUAL 75 OR SM STREQUAL 80 OR SM STREQUAL 86)
      set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
      set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
    endif()
  message("-- Assign GPU architecture (sm=${SM})")

  else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        ")

  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")

  message("-- Assign GPU architecture (sm=70,75)")
  endif()

  set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
  set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

  set(CMAKE_CXX_STANDARD "${CXX_STD}")
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}")

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")

  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  list(APPEND COMMON_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}
    ${CUDA_PATH}/include)

  set(COMMON_LIB_DIRS
    ${CUDA_PATH}/lib64
  )

  link_directories(${COMMON_LIB_DIRS})

  add_definitions(-DPADDLE_ON_INFERENCE)

  set(FT_PATH "third-party")
  set(THIRD_PARTY_NAME "fastertransformer")

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(external/boost)

  ExternalProject_Add(
    extern_${THIRD_PARTY_NAME}
    GIT_REPOSITORY    https://github.com/NVIDIA/DeepLearningExamples.git
    GIT_TAG           master
    PREFIX            ${FT_PATH}
    SOURCE_DIR        ${FT_PATH}/${THIRD_PARTY_NAME}
    SOURCE_SUBDIR     "FasterTransformer/v3.1"
    BINARY_DIR        ${FT_PATH}/build
    INSTALL_COMMAND   ""
    CMAKE_ARGS        -DCMAKE_BUILD_TYPE=Release -DSM=${SM}
  )
  ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} BINARY_DIR)
  ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} SOURCE_DIR)
  ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} SOURCE_SUBDIR)

  set(FT_INCLUDE_PATH ${SOURCE_DIR}/${SOURCE_SUBDIR})
  set(FT_LIB_PATH ${BINARY_DIR}/lib)

  include_directories(
    ${FT_INCLUDE_PATH}
  )

  link_directories(
    ${FT_LIB_PATH}
  )

  set(ft_lib_link
    decoder decoding topk cuda_int8_kernels cuda_kernels online_softmax_beamsearch
  )

  if(WITH_GPU)
    # find_package(CUDA REQUIRED)
    add_definitions("-DPADDLE_WITH_CUDA")
  endif()

  if(NOT WITH_STATIC_LIB)
    add_definitions("-DPADDLE_WITH_SHARED_LIB")
  else()
    # PD_INFER_DECL is mainly used to set the dllimport/dllexport attribute in dynamic library mode.
    # Set it to empty in static library mode to avoid compilation issues.
    add_definitions("/DPD_INFER_DECL=")
  endif()

  macro(safe_set_static_flag)
      foreach(flag_var
          CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
          string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/MD")
      endforeach(flag_var)
  endmacro()

  if(NOT DEFINED PADDLE_LIB)
    message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
  endif()
  if(NOT DEFINED DEMO_NAME)
    message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
  endif()

  include_directories("${PADDLE_LIB}/")
  set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install/")
  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
  include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")

  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
  link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
  link_directories("${PADDLE_LIB}/paddle/lib")

  if (WIN32)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
    option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)
    if (MSVC_STATIC_CRT)
      if (WITH_MKL)
        set(FLAG_OPENMP "/openmp")
      endif()
      set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
      set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
      set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
      set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
      safe_set_static_flag()
      if (WITH_STATIC_LIB)
        add_definitions(-DSTATIC_LIB)
      endif()
    endif()
  else()
    if(WITH_MKL)
      set(FLAG_OPENMP "-fopenmp")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${FLAG_OPENMP}")
  endif()
#[[
  if(WITH_GPU)
    if(NOT WIN32)
      set(CUDA_LIB "/usr/local/cuda/lib64/" CACHE STRING "CUDA Library")
    else()
      if(CUDA_LIB STREQUAL "")
        set(CUDA_LIB "C:\\Program\ Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64")
      endif()
    endif(NOT WIN32)
  endif()
]]
  if (USE_TENSORRT AND WITH_GPU)
    set(TENSORRT_ROOT "" CACHE STRING "The root directory of TensorRT library")
    if("${TENSORRT_ROOT}" STREQUAL "")
        message(FATAL_ERROR "The TENSORRT_ROOT is empty, you must assign it a value with CMake command. Such as: -DTENSORRT_ROOT=TENSORRT_ROOT_PATH ")
    endif()
    set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
  endif()

  if (NOT WIN32)
    if (USE_TENSORRT AND WITH_GPU)
        include_directories("${TENSORRT_INCLUDE_DIR}")
        link_directories("${TENSORRT_LIB_DIR}")
    endif()
  endif(NOT WIN32)

  if(WITH_MKL)
    set(MATH_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mklml")
    include_directories("${MATH_LIB_PATH}/include")
    if(WIN32)
      set(MATH_LIB ${MATH_LIB_PATH}/lib/mklml${CMAKE_STATIC_LIBRARY_SUFFIX}
                  ${MATH_LIB_PATH}/lib/libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
      set(MATH_LIB ${MATH_LIB_PATH}/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
                  ${MATH_LIB_PATH}/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    set(MKLDNN_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mkldnn")
    if(EXISTS ${MKLDNN_PATH})
      include_directories("${MKLDNN_PATH}/include")
      if(WIN32)
        set(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
      else(WIN32)
        set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libmkldnn.so.0)
      endif(WIN32)
    endif()
  else()
    set(OPENBLAS_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}openblas")
    include_directories("${OPENBLAS_LIB_PATH}/include/openblas")
    if(WIN32)
      set(MATH_LIB ${OPENBLAS_LIB_PATH}/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
      set(MATH_LIB ${OPENBLAS_LIB_PATH}/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
  endif()

  if(WITH_STATIC_LIB)
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    if(WIN32)
      set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
      set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
  endif()

  if (NOT WIN32)
    set(EXTERNAL_LIB "-lrt -ldl -lpthread")
    set(DEPS ${DEPS}
        ${MATH_LIB} ${MKLDNN_LIB}
        glog gflags protobuf  xxhash
        ${EXTERNAL_LIB})
  else()
    set(DEPS ${DEPS}
        ${MATH_LIB} ${MKLDNN_LIB}
        glog gflags_static libprotobuf xxhash ${EXTERNAL_LIB})
    set(DEPS ${DEPS} shlwapi.lib)
  endif(NOT WIN32)
#[[
  if(WITH_GPU)
    if(NOT WIN32)
      if (USE_TENSORRT)
        set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
      endif()
      set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
    else()
      if(USE_TENSORRT)
        set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/nvinfer${CMAKE_STATIC_LIBRARY_SUFFIX})
        set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/nvinfer_plugin${CMAKE_STATIC_LIBRARY_SUFFIX})
      endif()
      set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
      set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
      set(DEPS ${DEPS} ${CUDA_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX} )
    endif()
  endif()
]]
  cuda_add_library(pd_infer_custom_op ${decoding_op_files} SHARED)
  add_dependencies(pd_infer_custom_op extern_${THIRD_PARTY_NAME})
  add_executable(transformer_e2e ${DEMO_NAME})
  set(DEPS pd_infer_custom_op ${ft_lib_link} boost ${DEPS} cublas cudart)

  if(WIN32)
    if(USE_TENSORRT)
      add_custom_command(TARGET transformer_e2e POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_LIB_DIR}/nvinfer${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
              COMMAND ${CMAKE_COMMAND} -E copy ${TENSORRT_LIB_DIR}/nvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
      )
    endif()
    if(WITH_MKL)
      add_custom_command(TARGET transformer_e2e POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/mklml.dll ${CMAKE_BINARY_DIR}/Release
            COMMAND ${CMAKE_COMMAND} -E copy ${MATH_LIB_PATH}/lib/libiomp5md.dll ${CMAKE_BINARY_DIR}/Release
            COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_PATH}/lib/mkldnn.dll  ${CMAKE_BINARY_DIR}/Release
      )
    else()
      add_custom_command(TARGET transformer_e2e POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_LIB_PATH}/lib/openblas.dll ${CMAKE_BINARY_DIR}/Release
      )
    endif()
    if(NOT WITH_STATIC_LIB)
        add_custom_command(TARGET transformer_e2e POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy "${PADDLE_LIB}/paddle/lib/paddle_fluid.dll" ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
        )
    endif()
  endif()

  target_link_libraries(transformer_e2e ${DEPS})

else(ON_INFER)

  set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

  list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

  # Setting compiler flags
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")    
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

  if (SM STREQUAL 80 OR
      SM STREQUAL 86 OR
      SM STREQUAL 70 OR
      SM STREQUAL 75 OR
      SM STREQUAL 61 OR
      SM STREQUAL 60)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${SM},code=\\\"sm_${SM},compute_${SM}\\\"")
    if (SM STREQUAL 70 OR SM STREQUAL 75 OR SM STREQUAL 80 OR SM STREQUAL 86)
      set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
      set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
    endif()
  message("-- Assign GPU architecture (sm=${SM})")

  else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        ")

  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")

  message("-- Assign GPU architecture (sm=70,75)")
  endif()

  set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
  set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

  set(CMAKE_CXX_STANDARD "${CXX_STD}")
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}")

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")

  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  list(APPEND COMMON_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}
    ${CUDA_PATH}/include)

  set(COMMON_LIB_DIRS
    ${CUDA_PATH}/lib64
  )

  set(THIRD_PARTY_PATH "third-party")
  set(THIRD_PARTY_NAME "fastertransformer")

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(external/boost)

  ExternalProject_Add(
    extern_${THIRD_PARTY_NAME}
    GIT_REPOSITORY    https://github.com/NVIDIA/DeepLearningExamples.git
    GIT_TAG           master
    PREFIX            ${THIRD_PARTY_PATH}
    SOURCE_DIR        ${THIRD_PARTY_PATH}/${THIRD_PARTY_NAME}
    SOURCE_SUBDIR     "FasterTransformer/v3.1"
    BINARY_DIR        ${THIRD_PARTY_PATH}/build
    INSTALL_COMMAND   ""
    CMAKE_ARGS        -DCMAKE_BUILD_TYPE=Release -DSM=${SM}
  )
  ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} BINARY_DIR)
  ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} SOURCE_DIR)
  ExternalProject_Get_property(extern_${THIRD_PARTY_NAME} SOURCE_SUBDIR)

  set(FT_INCLUDE_PATH ${SOURCE_DIR}/${SOURCE_SUBDIR})
  set(FT_LIB_PATH ${BINARY_DIR}/lib)

  include_directories(
    ${FT_INCLUDE_PATH}
  )

  if(NOT PY_CMD)
    set(PYTHON_PATH "python" CACHE STRING "Python path")
  else()
    set(PYTHON_PATH ${PY_CMD} CACHE STRING "Python path")
  endif()

  execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import paddle; print(paddle.sysconfig.get_include())"
                  RESULT_VARIABLE _INC_PYTHON_SUCCESS
                  OUTPUT_VARIABLE _INC_PYTHON_VALUES)
  if (NOT _INC_PYTHON_SUCCESS MATCHES 0)
      message(FATAL_ERROR "Python config Error.")
  endif()
  string(REGEX REPLACE ";" "\\\\;" _INC_PYTHON_VALUES ${_INC_PYTHON_VALUES})
  string(REGEX REPLACE "\n" ";" _INC_PYTHON_VALUES ${_INC_PYTHON_VALUES})
  list(GET _INC_PYTHON_VALUES 0 PY_INCLUDE_DIR)

  list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})
  list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR}/third_party)

  include_directories(
    ${COMMON_HEADER_DIRS}
  )

  execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import paddle; print(paddle.sysconfig.get_lib())"
                  RESULT_VARIABLE _LIB_PYTHON_SUCCESS
                  OUTPUT_VARIABLE _LIB_PYTHON_VALUES)
  if (NOT _LIB_PYTHON_SUCCESS MATCHES 0)
      message(FATAL_ERROR "Python config Error.")
  endif()
  string(REGEX REPLACE ";" "\\\\;" _LIB_PYTHON_VALUES ${_LIB_PYTHON_VALUES})
  string(REGEX REPLACE "\n" ";" _LIB_PYTHON_VALUES ${_LIB_PYTHON_VALUES})
  list(GET _LIB_PYTHON_VALUES 0 PY_LIB_DIR)
  list(APPEND COMMON_LIB_DIRS ${PY_LIB_DIR})

  link_directories(
    ${COMMON_LIB_DIRS}
  )

  link_directories(
    ${FT_LIB_PATH}
  )

  include_directories(${PY_INCLUDE_DIR})
  include_directories(${PY_INCLUDE_DIR}\third_party)

  if(EXISTS ${PY_LIB_DIR}/libpaddle_custom_op.so)
    set(lib_link
      -lpaddle_custom_op
    )
  endif()

  set(ft_lib_link
    -ldecoder -ldecoding -ltopk -lcuda_int8_kernels -lcuda_kernels -lonline_softmax_beamsearch
  )

  add_definitions(-DPADDLE_WITH_CUDA)
  add_definitions(-DEIGEN_USE_GPU)
  add_definitions(-DPADDLE_USE_DSO)
  add_definitions(-DPADDLE_WITH_MKLDNN)

  add_library(decoding_op SHARED ${decoding_op_files})
  add_dependencies(decoding_op extern_${THIRD_PARTY_NAME} boost)
  target_link_libraries(decoding_op PRIVATE -lcublas -lcudart ${lib_link} ${ft_lib_link})
endif()
