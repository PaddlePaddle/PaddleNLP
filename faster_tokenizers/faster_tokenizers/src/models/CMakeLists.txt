cc_library(models SRCS wordpiece.cc faster_wordpiece.cc DEPS core json trie)
add_dependencies(models extern_boost)
