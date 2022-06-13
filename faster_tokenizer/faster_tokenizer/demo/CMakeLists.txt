cmake_minimum_required(VERSION 3.10)
project(cpp_faster_tokenizer_demo CXX C)

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ")

if(NOT DEFINED FASTER_TOKENIZER_LIB)
  message(FATAL_ERROR "please set FASTER_TOKENIZER_LIB with -DFASTER_TOKENIZER_LIB=/path/to/faster_tokenizer_cpp")
endif()

# Download ernie vocab for demo
set(ERNIE_VOCAB_PATH ${CMAKE_CURRENT_BINARY_DIR}/ernie_vocab.txt)
if (EXISTS ${ERNIE_VOCAB_PATH})
message("The ${ERNIE_VOCAB_PATH} exists already.")
else()
file(DOWNLOAD "https://bj.bcebos.com/paddlenlp/models/transformers/ernie/vocab.txt" ${ERNIE_VOCAB_PATH} SHOW_PROGRESS)
message("Already download the vocab.txt of ernie to ${CMAKE_CURRENT_BINARY_DIR} for demo.")
endif()

# Include all necessary headers path
include_directories("${FASTER_TOKENIZER_LIB}/include"
                    "${FASTER_TOKENIZER_LIB}/third_party/include")
add_executable(ernie_faster_tokenizer_demo ernie_faster_tokenizer_demo.cc)

# Include the path of core_tokenizers library
link_directories("${FASTER_TOKENIZER_LIB}/lib")

ADD_LIBRARY(core_tokenizers SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET core_tokenizers PROPERTY IMPORTED_LOCATION ${FASTER_TOKENIZER_LIB}/lib/libcore_tokenizers.so)

target_link_libraries(ernie_faster_tokenizer_demo core_tokenizers)