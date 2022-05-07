cc_library(models SRCS wordpiece.cc DEPS core)
add_dependencies(models extern_boost)
