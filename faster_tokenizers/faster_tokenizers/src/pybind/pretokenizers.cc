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

#include "pretokenizers/pretokenizers.h"
#include "pybind/pretokenizers.h"

namespace py = pybind11;

namespace tokenizers {
namespace pybind {

class PyPreTokenizer : public pretokenizers::PreTokenizer {
public:
  using PreTokenizer::PreTokenizer;
  virtual void operator()(
      pretokenizers::PreTokenizedString* pretokenized) const override {
    PYBIND11_OVERLOAD_PURE_NAME(
        void, PreTokenizer, "__call__", operator(), pretokenized);
  }
};

class PyWhitespace : public pretokenizers::Whitespace {
public:
  using Whitespace::Whitespace;
  virtual void operator()(
      pretokenizers::PreTokenizedString* pretokenized) const override {
    PYBIND11_OVERLOAD_NAME(
        void, Whitespace, "__call__", operator(), pretokenized);
  }
};

class PyBertPreTokenizer : public pretokenizers::BertPreTokenizer {
public:
  using BertPreTokenizer::BertPreTokenizer;
  virtual void operator()(
      pretokenizers::PreTokenizedString* pretokenized) const override {
    PYBIND11_OVERLOAD_NAME(
        void, BertPreTokenizer, "__call__", operator(), pretokenized);
  }
};

void BindPreTokenizers(pybind11::module* m) {
  auto sub_module =
      m->def_submodule("pretokenizers", "The pretokenizers module");
  py::class_<pretokenizers::StringSplit>(sub_module, "StringSplit")
      .def(py::init<const normalizers::NormalizedString&>(),
           py::arg("nomalized_text"))
      .def(py::init<const normalizers::NormalizedString&,
                    const std::vector<core::Token>&>(),
           py::arg("nomalized_text"),
           py::arg("tokens"))
      .def_readwrite("normalized", &pretokenizers::StringSplit::normalized_)
      .def_readwrite("tokens", &pretokenizers::StringSplit::tokens_);
  py::class_<pretokenizers::PreTokenizedString>(sub_module,
                                                "PreTokenizedString")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("raw_text"))
      .def(py::init<const normalizers::NormalizedString&>(),
           py::arg("nomalized_text"))
      .def("get_string_split", &pretokenizers::PreTokenizedString::GetSplit)
      .def("get_string_splits_size",
           &pretokenizers::PreTokenizedString::GetSplitsSize)
      .def("get_original_text",
           &pretokenizers::PreTokenizedString::GetOriginStr)
      .def("to_encoding",
           [](const pretokenizers::PreTokenizedString& self,
              const std::vector<uint>& word_idx,
              uint type_id,
              core::OffsetType offset_type) {
             core::Encoding encoding;
             self.TransformToEncoding(
                 word_idx, type_id, offset_type, &encoding);
             return encoding;
           });
  py::class_<pretokenizers::PreTokenizer, PyPreTokenizer>(sub_module,
                                                          "PreTokenizer")
      .def(py::init<>())
      .def("__call__", &pretokenizers::PreTokenizer::operator());
  py::class_<pretokenizers::Whitespace, PyWhitespace>(sub_module,
                                                      "WhitespacePreTokenizer")
      .def(py::init<>())
      .def("__call__", &pretokenizers::Whitespace::operator());
  py::class_<pretokenizers::BertPreTokenizer, PyBertPreTokenizer>(
      sub_module, "BertPreTokenizer")
      .def(py::init<>())
      .def("__call__", &pretokenizers::BertPreTokenizer::operator());
}

}  // pybind
}  // tokenizers
