cc_library(decoders SRCS wordpiece.cc DEPS json utils)
add_dependencies(decoders extern_boost)