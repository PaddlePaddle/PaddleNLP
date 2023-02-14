cmake_minimum_required(VERSION 3.10)
project(cpp_fast_tokenizer_demo CXX C)

option(FAST_TOKENIZER_INSTALL_DIR "Path of downloaded fast_tokenizer sdk.")

# Download ernie vocab for demo
set(ERNIE_VOCAB_PATH ${CMAKE_CURRENT_BINARY_DIR}/ernie_vocab.txt)
if (EXISTS ${ERNIE_VOCAB_PATH})
  message(STATUS "The ${ERNIE_VOCAB_PATH} exists already.")
else()
  file(DOWNLOAD "https://bj.bcebos.com/paddlenlp/models/transformers/ernie/vocab.txt" ${ERNIE_VOCAB_PATH} SHOW_PROGRESS)
  message(STATUS "Already download the vocab.txt of ernie to ${CMAKE_CURRENT_BINARY_DIR} for demo.")
endif()

# Get FAST_TOKENIZER_INCS and FAST_TOKENIZER_LIBS
message(STATUS "The fast_tokenizer install dir: ${FAST_TOKENIZER_INSTALL_DIR}")
include(${FAST_TOKENIZER_INSTALL_DIR}/FastTokenizer.cmake)

include_directories(${FAST_TOKENIZER_INCS})

add_executable(demo ${PROJECT_SOURCE_DIR}/demo.cc)
target_link_libraries(demo ${FAST_TOKENIZER_LIBS})
