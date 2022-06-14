cc_library(added_vocabulary SRCS added_vocabulary.cc DEPS normalizers pretokenizers json)
cc_library(tokenizer SRCS tokenizer.cc DEPS added_vocabulary json decoders trie models postprocessors)
cc_library(core SRCS encoding.cc DEPS json)
add_dependencies(tokenizer extern_boost)
