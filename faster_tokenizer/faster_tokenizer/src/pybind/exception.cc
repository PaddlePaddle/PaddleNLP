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

#include "pybind/exception.h"

namespace py = pybind11;

namespace paddlenlp {
namespace faster_tokenizer {
namespace pybind {

void ThrowExceptionToPython(std::exception_ptr p) {
  static PyObject* EnforceNotMetException =
      PyErr_NewException("tokenizer.EnforceNotMet", PyExc_Exception, NULL);
  try {
    if (p) std::rethrow_exception(p);
  } catch (const std::runtime_error& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
}

}  // namespace pybind
}  // namespace faster_tokenizer
}  // namespace paddlenlp
