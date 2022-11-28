cmake_minimum_required(VERSION 3.10)
project(cpp_fast_tokenizer_demo CXX C)
option(FAST_TOKENIZER_INSTALL_DIR "Path of downloaded fast_tokenizer sdk.")

# Download clip vocab and merge files
set(CLIP_VOCAB_PATH ${CMAKE_CURRENT_BINARY_DIR}/clip_vocab.json)
set(CLIP_MERGES_PATH ${CMAKE_CURRENT_BINARY_DIR}/clip_merges.txt)

if (EXISTS ${CLIP_VOCAB_PATH})
  message("The ${CLIP_VOCAB_PATH} exists already.")
else()
  file(DOWNLOAD "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-vit-large-patch14/vocab.json" ${CLIP_VOCAB_PATH} SHOW_PROGRESS)
  message("Already download the vocab.json of clip to ${CMAKE_CURRENT_BINARY_DIR} for test.")
endif()

if (EXISTS ${CLIP_MERGES_PATH})
  message("The ${CLIP_MERGES_PATH} exists already.")
else()
  file(DOWNLOAD "http://bj.bcebos.com/paddlenlp/models/community/openai/clip-vit-large-patch14/merges.txt" ${CLIP_MERGES_PATH} SHOW_PROGRESS)
  message("Already download the merges.txt of clip to ${CMAKE_CURRENT_BINARY_DIR} for test.")
endif()

# Get FAST_TOKENIZER_INCS and FAST_TOKENIZER_LIBS
include(${FAST_TOKENIZER_INSTALL_DIR}/FastTokenizer.cmake)
include_directories(${FAST_TOKENIZER_INCS})

add_executable(demo ${PROJECT_SOURCE_DIR}/demo.cc)
target_link_libraries(demo ${FAST_TOKENIZER_LIBS})
