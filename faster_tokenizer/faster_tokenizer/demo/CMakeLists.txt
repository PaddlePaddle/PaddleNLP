cmake_minimum_required(VERSION 3.10)
project(cpp_faster_tokenizer_demo CXX C)

option(FASTER_TOKENIZER_INSTALL_DIR "Path of downloaded faster_tokenizer sdk.")

# Download ernie vocab for demo
set(ERNIE_VOCAB_PATH ${CMAKE_CURRENT_BINARY_DIR}/ernie_vocab.txt)
if (EXISTS ${ERNIE_VOCAB_PATH})
message("The ${ERNIE_VOCAB_PATH} exists already.")
else()
file(DOWNLOAD "https://bj.bcebos.com/paddlenlp/models/transformers/ernie/vocab.txt" ${ERNIE_VOCAB_PATH} SHOW_PROGRESS)
message("Already download the vocab.txt of ernie to ${CMAKE_CURRENT_BINARY_DIR} for demo.")
endif()

include(${FASTER_TOKENIZER_INSTALL_DIR}/FasterTokenizer.cmake)

include_directories(${FASTER_TOKENIZER_INCS})

add_executable(ernie_faster_tokenizer_demo ${PROJECT_SOURCE_DIR}/ernie_faster_tokenizer_demo.cc)
target_link_libraries(ernie_faster_tokenizer_demo ${FASTER_TOKENIZER_LIBS})
