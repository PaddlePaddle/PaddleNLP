cc_library(trie SRCS trie.cc DEPS dart icuuc icudata)
cc_library(failure SRCS failure.cc DEPS trie icuuc icudata)
cc_library(sentencepiece_normalizer SRCS sentencepiece_normalizer.cc DEPS trie icuuc icudata)
cc_library(lattice SRCS lattice.cc)