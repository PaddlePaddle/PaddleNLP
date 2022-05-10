cc_library(models SRCS wordpiece.cc DEPS core json)
add_dependencies(models extern_boost)
