/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <Python.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "fast_tokenizer/pybind/core.h"
#include "fast_tokenizer/pybind/decoders.h"
#include "fast_tokenizer/pybind/models.h"
#include "fast_tokenizer/pybind/normalizers.h"
#include "fast_tokenizer/pybind/postprocessors.h"
#include "fast_tokenizer/pybind/pretokenizers.h"
#include "fast_tokenizer/pybind/tokenizers.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace pybind {

PYBIND11_MODULE(core_tokenizers, m) {
  m.doc() = "The paddlenlp fast_tokenizer core module.";
  // 1. Bind normalizers submodule
  BindNormalizers(&m);
  // 2. Bind pre_tokenizers submodule
  BindPreTokenizers(&m);
  // 3. Bind models submodule
  BindModels(&m);
  // 4. Bind processors submodule
  BindPostProcessors(&m);
  // 5. Bind tokenizers submodule
  BindTokenizers(&m);
  // 6. Bind core
  BindCore(&m);
  // 7. Bind decoder submodule
  BindDecoders(&m);
}

}  // namespace pybind
}  // namespace fast_tokenizer
}  // namespace paddlenlp
