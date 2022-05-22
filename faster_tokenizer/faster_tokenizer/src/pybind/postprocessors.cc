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

#include "postprocessors/postprocessors.h"
#include <Python.h>
#include "pybind/postprocessors.h"

#include "core/encoding.h"

namespace py = pybind11;

namespace tokenizers {
namespace pybind {

class PyPostProcessor : public postprocessors::PostProcessor {
public:
  using PostProcessor::PostProcessor;
  virtual void operator()(core::Encoding* encoding,
                          core::Encoding* pair_encoding,
                          bool add_special_tokens,
                          core::Encoding* result_encoding) const override {
    PYBIND11_OVERLOAD_PURE_NAME(void,
                                PostProcessor,
                                "__call__",
                                operator(),
                                encoding,
                                pair_encoding,
                                add_special_tokens,
                                result_encoding);
  }
  virtual size_t AddedTokensNum(bool is_pair) const override {
    PYBIND11_OVERLOAD_PURE_NAME(size_t,
                                PostProcessor,
                                "num_special_tokens_to_add",
                                AddedTokensNum,
                                is_pair);
  }
};

class PyBertPostProcessor : public postprocessors::BertPostProcessor {
public:
  using BertPostProcessor::BertPostProcessor;
  virtual void operator()(core::Encoding* encoding,
                          core::Encoding* pair_encoding,
                          bool add_special_tokens,
                          core::Encoding* result_encoding) const override {
    PYBIND11_OVERLOAD_NAME(void,
                           BertPostProcessor,
                           "__call__",
                           operator(),
                           encoding,
                           pair_encoding,
                           add_special_tokens,
                           result_encoding);
  }
  virtual size_t AddedTokensNum(bool is_pair) const override {
    PYBIND11_OVERLOAD_NAME(size_t,
                           BertPostProcessor,
                           "num_special_tokens_to_add",
                           AddedTokensNum,
                           is_pair);
  }
};

class PyTemplatePostProcessor : public postprocessors::TemplatePostProcessor {
public:
  using TemplatePostProcessor::TemplatePostProcessor;
  virtual void operator()(core::Encoding* encoding,
                          core::Encoding* pair_encoding,
                          bool add_special_tokens,
                          core::Encoding* result_encoding) const override {
    PYBIND11_OVERLOAD_NAME(void,
                           TemplatePostProcessor,
                           "__call__",
                           operator(),
                           encoding,
                           pair_encoding,
                           add_special_tokens,
                           result_encoding);
  }
  virtual size_t AddedTokensNum(bool is_pair) const override {
    PYBIND11_OVERLOAD_NAME(size_t,
                           TemplatePostProcessor,
                           "num_special_tokens_to_add",
                           AddedTokensNum,
                           is_pair);
  }
};


void BindPostProcessors(pybind11::module* m) {
  auto submodule =
      m->def_submodule("postprocessors", "The postprocessors module");
  py::class_<postprocessors::PostProcessor, PyPostProcessor>(submodule,
                                                             "PostProcessor")
      .def(py::init<>())
      .def("num_special_tokens_to_add",
           &postprocessors::PostProcessor::AddedTokensNum,
           py::arg("is_pair"))
      .def("__call__",
           [](const postprocessors::PostProcessor& self,
              core::Encoding* encoding,
              core::Encoding* pair_encoding,
              bool add_special_tokens) {
             core::Encoding result_encoding;
             self(
                 encoding, pair_encoding, add_special_tokens, &result_encoding);
             return result_encoding;
           },
           py::arg("encoding"),
           py::arg("pair_encoding"),
           py::arg("add_special_tokens"));
  py::class_<postprocessors::BertPostProcessor, PyBertPostProcessor>(
      submodule, "BertPostProcessor")
      .def(py::init<>())
      .def(py::init<const std::pair<std::string, uint>&,
                    const std::pair<std::string, uint>&>(),
           py::arg("sep"),
           py::arg("cls"))
      .def("num_special_tokens_to_add",
           &postprocessors::BertPostProcessor::AddedTokensNum,
           py::arg("is_pair"))
      .def("__call__",
           [](const postprocessors::BertPostProcessor& self,
              core::Encoding* encoding,
              core::Encoding* pair_encoding,
              bool add_special_tokens) {
             core::Encoding result_encoding;
             self(
                 encoding, pair_encoding, add_special_tokens, &result_encoding);
             return result_encoding;
           },
           py::arg("encoding"),
           py::arg("pair_encoding"),
           py::arg("add_special_tokens"));

  // For Template Processing
  py::class_<postprocessors::SpecialToken>(submodule, "SpecialToken")
      .def(py::init<>())
      .def(py::init<const std::string&,
                    const std::vector<uint>&,
                    const std::vector<std::string>&>(),
           py::arg("id"),
           py::arg("ids"),
           py::arg("tokens"))
      .def(py::init<const std::string&, uint>(),
           py::arg("token"),
           py::arg("id"));

  py::class_<postprocessors::Template>(submodule, "Template")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("template"))
      .def(py::init<const std::vector<std::string>&>(), py::arg("pieces"))
      .def(py::init<const std::vector<postprocessors::TemplatePiece>&>(),
           py::arg("pieces"));

  py::class_<postprocessors::TemplatePostProcessor, PyTemplatePostProcessor>(
      submodule, "TemplatePostProcessor")
      .def(py::init<const postprocessors::Template&,
                    const postprocessors::Template&,
                    const std::vector<postprocessors::SpecialToken>&>(),
           py::arg("single"),
           py::arg("pair"),
           py::arg("special_tokens"))
      .def("num_special_tokens_to_add",
           &postprocessors::TemplatePostProcessor::AddedTokensNum,
           py::arg("is_pair"))
      .def("__call__",
           [](const postprocessors::TemplatePostProcessor& self,
              core::Encoding* encoding,
              core::Encoding* pair_encoding,
              bool add_special_tokens) {
             core::Encoding result_encoding;
             self(
                 encoding, pair_encoding, add_special_tokens, &result_encoding);
             return result_encoding;
           },
           py::arg("encoding"),
           py::arg("pair_encoding"),
           py::arg("add_special_tokens"));
}

}  // pybind
}  // tokenizers
