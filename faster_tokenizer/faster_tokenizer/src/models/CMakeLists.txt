cc_library(models SRCS wordpiece.cc faster_wordpiece.cc DEPS core json trie failure)
add_dependencies(models extern_boost)
