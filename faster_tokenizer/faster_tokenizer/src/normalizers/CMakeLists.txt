cc_library(normalizers 
      SRCS normalizer.cc unicode.cc
           utils.cc strip.cc replace.cc bert.cc
           precompiled.cc 
      DEPS re2 json sentencepiece_normalizer icuuc icudata)
