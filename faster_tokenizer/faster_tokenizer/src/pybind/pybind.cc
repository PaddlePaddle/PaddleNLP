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

#include "pybind/core.h"
#include "pybind/decoders.h"
#include "pybind/models.h"
#include "pybind/normalizers.h"
#include "pybind/postprocessors.h"
#include "pybind/pretokenizers.h"
#include "pybind/tokenizers.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace pybind {

PYBIND11_MODULE(core_tokenizers, m) {
  m.doc() = "The paddlenlp tokenizers core module.";
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
}  // namespace faster_tokenizer
}  // namespace paddlenlp
