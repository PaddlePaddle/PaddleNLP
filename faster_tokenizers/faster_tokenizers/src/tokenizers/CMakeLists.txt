cc_library(core_tokenizers SHARED
           SRCS ernie_faster_tokenizer.cc
           DEPS normalizers pretokenizers models
                postprocessors core added_vocabulary tokenizer json)