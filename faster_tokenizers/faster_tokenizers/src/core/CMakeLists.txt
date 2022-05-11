cc_library(added_vocabulary SRCS added_vocabulary.cc DEPS normalizers pretokenizers)
cc_library(tokenizer SRCS tokenizer.cc DEPS added_vocabulary)
cc_library(core SRCS encoding.cc)
add_dependencies(tokenizer extern_boost)
