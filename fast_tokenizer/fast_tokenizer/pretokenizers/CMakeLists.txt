cc_library(pretokenizers
      SRCS pretokenizer.cc whitespace.cc bert.cc metaspace.cc
           sequence.cc byte_level.cc split.cc whitespace_split.cc
      DEPS normalizers core json utils)
