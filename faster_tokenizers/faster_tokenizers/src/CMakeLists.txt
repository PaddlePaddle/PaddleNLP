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
cc_library(core_tokenizers SHARED 
           DEPS pybind_all normalizers pretokenizers core models tokenizer added_vocabulary postprocessors json)
# For python
set_target_properties(core_tokenizers PROPERTIES PREFIX "")
else(WITH_PYTHON)
cc_library(core_tokenizers SHARED
           DEPS normalizers pretokenizers models postprocessors core added_vocabulary tokenizer json)
endif(WITH_PYTHON)
