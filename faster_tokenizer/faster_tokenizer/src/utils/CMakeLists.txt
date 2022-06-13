cc_library(trie SRCS trie.cc DEPS dart icu)
cc_library(failure SRCS failure.cc DEPS trie icu)
