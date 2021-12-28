add_library(Tokenizer tokenizer.cc)
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}utf8proc/lib")
target_link_libraries(Tokenizer utf8proc)