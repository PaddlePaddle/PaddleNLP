cc_library(pybind_utils SRCS utils.cc DEPS pybind python json)
cc_library(pybind_normalizers SRCS normalizers.cc DEPS pybind python json)
cc_library(pybind_pretokenizers SRCS pretokenizers.cc DEPS pybind python json)
cc_library(pybind_models SRCS models.cc DEPS pybind python json)
cc_library(pybind_postprocessors SRCS postprocessors.cc DEPS pybind python core json)
cc_library(pybind_tokenizers SRCS tokenizers.cc DEPS pybind python pybind_utils json)
cc_library(pybind_exception SRCS exception.cc DEPS pybind python)
cc_library(pybind_core SRCS core.cc DEPS pybind python json)
cc_library(core_tokenizers SHARED
           SRCS pybind.cc
           DEPS pybind python pybind_normalizers pybind_utils
                pybind_pretokenizers pybind_models
                pybind_postprocessors pybind_tokenizers pybind_exception
                pybind_core normalizers pretokenizers core models
                tokenizer added_vocabulary postprocessors json)
# For python
set_target_properties(core_tokenizers PROPERTIES PREFIX "")