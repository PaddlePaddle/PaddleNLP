cc_library(added_vocabulary SRCS added_vocabulary.cc DEPS normalizers pretokenizers json)
cc_library(base SRCS base.cc DEPS json)
cc_library(tokenizer SRCS tokenizer.cc DEPS added_vocabulary json decoders trie models postprocessors base)
cc_library(core SRCS encoding.cc DEPS json base)
