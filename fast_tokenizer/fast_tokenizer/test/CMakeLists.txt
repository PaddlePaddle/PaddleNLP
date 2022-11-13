if(WITH_TESTING)
cc_library(tokenizers_gtest_main SRCS gtest_main.cc DEPS gtest gflags)

# Test Normalizers modules
cc_test(test_normalizer SRCS test_normalizer.cc DEPS normalizers)
cc_test(test_unicode SRCS test_unicode.cc DEPS normalizers)
cc_test(test_replace SRCS test_replace.cc DEPS normalizers)
cc_test(test_strip SRCS test_strip.cc DEPS normalizers)
cc_test(test_utils SRCS test_utils.cc DEPS normalizers)

# Test PreTokenizers modules
cc_test(test_whitespace SRCS test_whitespace.cc DEPS pretokenizers)
cc_test(test_bert_pretokenizer SRCS test_bert_pretokenizer.cc DEPS pretokenizers)
cc_test(test_split_pretokenizer SRCS test_split_pretokenizer.cc DEPS pretokenizers)

# Test Model
cc_test(test_wordpiece SRCS test_wordpiece.cc DEPS models)
cc_test(test_fast_wordpiece SRCS test_fast_wordpiece.cc DEPS models)

# Download ernie vocab for test
set(ERNIE_VOCAB_PATH ${CMAKE_CURRENT_BINARY_DIR}/ernie_vocab.txt)
if (EXISTS ${ERNIE_VOCAB_PATH})
  message("The ${ERNIE_VOCAB_PATH} exists already.")
else()
  file(DOWNLOAD "https://bj.bcebos.com/paddlenlp/models/transformers/ernie/vocab.txt" ${ERNIE_VOCAB_PATH} SHOW_PROGRESS)
  message("Already download the vocab.txt of ernie to ${CMAKE_CURRENT_BINARY_DIR} for test.")
endif()

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

# Test Tokenizer
cc_test(test_bert_tokenizer SRCS test_bert_tokenizer.cc DEPS normalizers pretokenizers models postprocessors tokenizer)

# Test PostProcessor
cc_test(test_roberta_postprocessor SRCS test_roberta_postprocessor.cc DEPS normalizers pretokenizers models postprocessors tokenizer)

if(NOT WITH_PYTHON)
  cc_test(test_ernie_fast_tokenizer SRCS test_ernie_fast_tokenizer.cc DEPS normalizers pretokenizers models postprocessors tokenizer core_tokenizers)
  cc_test(test_clip_fast_tokenizer SRCS test_clip_fast_tokenizer.cc DEPS normalizers pretokenizers models postprocessors tokenizer core_tokenizers)
endif()

endif()
