add_subdirectory(decoders)
add_subdirectory(models)
add_subdirectory(normalizers)
add_subdirectory(pretokenizers)
add_subdirectory(postprocessors)
add_subdirectory(core)
# set the relative path of shared library
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath='$ORIGIN'")

if (WITH_PYTHON)
add_subdirectory(pybind)
else(WITH_PYTHON)
add_subdirectory(tokenizers)
endif(WITH_PYTHON)
