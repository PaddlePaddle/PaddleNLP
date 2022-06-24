cc_library(models SRCS wordpiece.cc faster_wordpiece.cc bpe.cc DEPS core json trie failure icuuc icudata)
add_dependencies(models extern_boost)
