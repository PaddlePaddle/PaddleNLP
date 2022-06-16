cc_library(models SRCS wordpiece.cc faster_wordpiece.cc bpe.cc DEPS core json trie failure icu)
add_dependencies(models extern_boost)
