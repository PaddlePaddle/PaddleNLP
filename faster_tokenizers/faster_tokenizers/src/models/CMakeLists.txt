cc_library(models SRCS wordpiece.cc faster_wordpiece DEPS core json)
    add_dependencies(models extern_boost)
