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

#include "decoders/decoders.h"
#include <Python.h>
#include "pybind/decoders.h"

namespace py = pybind11;

namespace paddlenlp {
namespace faster_tokenizer {
namespace pybind {

class PyDecoder : public decoders::Decoder {
public:
  using Decoder::Decoder;
  virtual void operator()(const std::vector<std::string> tokens,
                          std::string* result) const override {
    PYBIND11_OVERLOAD_PURE_NAME(
        void, Decoder, "__call__", operator(), tokens, result);
  }
};

class PyWordPieceDecoder : public decoders::WordPiece {
public:
  using WordPiece::WordPiece;
  virtual void operator()(const std::vector<std::string> tokens,
                          std::string* result) const override {
    PYBIND11_OVERLOAD_NAME(
        void, WordPiece, "__call__", operator(), tokens, result);
  }
};

void BindDecoders(pybind11::module* m) {
  auto submodule = m->def_submodule("decoders", "The decoders module");
  py::class_<decoders::Decoder, PyDecoder>(submodule, "Decoder")
      .def(py::init<>())
      .def("decode",
           [](const decoders::Decoder& self,
              const std::vector<std::string>& tokens) {
             std::string result;
             self(tokens, &result);
             return result;
           },
           py::arg("tokens"));

  py::class_<decoders::WordPiece, PyWordPieceDecoder>(submodule, "WordPiece")
      .def(py::init<std::string, bool>(),
           py::arg("prefix") = "##",
           py::arg("cleanup") = true)
      .def("decode",
           [](const decoders::Decoder& self,
              const std::vector<std::string>& tokens) {
             std::string result;
             self(tokens, &result);
             return result;
           },
           py::arg("tokens"));
}

}  // namespace pybind
}  // namespace faster_tokenizer
}  // namespace paddlenlp
