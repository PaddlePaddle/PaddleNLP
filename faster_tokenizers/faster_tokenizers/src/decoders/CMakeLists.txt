cc_library(decoders SRCS wordpiece.cc DEPS json)
add_dependencies(decoders extern_boost)