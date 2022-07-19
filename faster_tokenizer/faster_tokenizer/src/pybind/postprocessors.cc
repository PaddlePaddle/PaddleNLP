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
#include "core/encoding.h"
#include "glog/logging.h"
#include "pybind/postprocessors.h"
#include "pybind/utils.h"

namespace py = pybind11;

namespace paddlenlp {
namespace faster_tokenizer {
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
      .def(py::init<const std::pair<std::string, uint32_t>&,
                    const std::pair<std::string, uint32_t>&>(),
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
                    const std::vector<uint32_t>&,
                    const std::vector<std::string>&>(),
           py::arg("id"),
           py::arg("ids"),
           py::arg("tokens"))
      .def(py::init<const std::string&, uint32_t>(),
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
      .def(
          py::init([](const py::object& single_obj,
                      const py::object& pair_obj,
                      const py::object& special_tokens_obj) {
            postprocessors::TemplatePostProcessor self;
            // Setting single
            if (py::isinstance<py::list>(single_obj)) {
              std::vector<std::string> template_piece =
                  CastPyArg2VectorOfStr(single_obj.ptr(), 0);
              self.UpdateSinglePieces(template_piece);
            } else if (py::isinstance<py::str>(single_obj)) {
              self.UpdateSinglePieces(
                  CastPyArg2AttrString(single_obj.ptr(), 0));
            } else {
              throw py::value_error(
                  "Type of args single need to be List[str] or str.");
            }
            // Setting pair
            if (py::isinstance<py::list>(pair_obj)) {
              std::vector<std::string> template_piece =
                  CastPyArg2VectorOfStr(pair_obj.ptr(), 0);
              self.UpdatePairPieces(template_piece);
            } else if (py::isinstance<py::str>(pair_obj)) {
              self.UpdatePairPieces(CastPyArg2AttrString(pair_obj.ptr(), 0));
            } else {
              throw py::value_error(
                  "Type of args pair need to be List[str] or str.");
            }
            // Setting special_tokens
            if (py::isinstance<py::list>(special_tokens_obj)) {
              std::vector<postprocessors::SpecialToken> special_tokens;
              for (auto& str : special_tokens_obj.cast<py::list>()) {
                if (py::isinstance<py::tuple>(str)) {
                  auto token_tuple = str.cast<py::tuple>();
                  uint32_t id;
                  std::string token;
                  if (token_tuple.size() == 2) {
                    if (py::isinstance<py::str>(token_tuple[0]) &&
                        py::isinstance<py::int_>(token_tuple[1])) {
                      token = token_tuple[0].cast<std::string>();
                      id = token_tuple[1].cast<uint32_t>();
                    } else if (py::isinstance<py::str>(token_tuple[1]) &&
                               py::isinstance<py::int_>(token_tuple[0])) {
                      token = token_tuple[1].cast<std::string>();
                      id = token_tuple[0].cast<uint32_t>();
                    } else {
                      throw py::value_error(
                          "`Tuple` with both a token and its associated ID, in "
                          "any order");
                    }
                    special_tokens.emplace_back(token, id);
                  } else {
                    throw py::value_error(
                        "Type of args special_tokens need to be "
                        "List[Union[Tuple[int, str], Tuple[str, int], dict]]");
                  }
                } else if (py::isinstance<py::dict>(str)) {
                  auto token_dict = str.cast<py::dict>();
                  std::string id;
                  std::vector<uint32_t> ids;
                  std::vector<std::string> tokens;
                  if (token_dict.contains("id") &&
                      py::isinstance<py::str>(token_dict["id"])) {
                    id = token_dict["id"].cast<std::string>();
                  } else {
                    throw py::value_error(
                        "Type of args special_tokens dict need to have key 'id'"
                        "and the respective value should be `str`");
                  }
                  if (token_dict.contains("ids") &&
                      py::isinstance<py::list>(token_dict["ids"])) {
                    for (auto py_id : token_dict["ids"].cast<py::list>()) {
                      if (py::isinstance<py::int_>(py_id)) {
                        ids.push_back(py_id.cast<uint32_t>());
                      } else {
                        throw py::value_error(
                            "Type of args special_tokens dict need to have key "
                            "'ids'"
                            "and the respective value should be List[int]");
                      }
                    }
                  } else {
                    throw py::value_error(
                        "Type of args special_tokens dict need to have key "
                        "'ids'"
                        "and the respective value should be List[int]");
                  }
                  if (token_dict.contains("tokens") &&
                      py::isinstance<py::list>(token_dict["tokens"])) {
                    for (auto& py_token :
                         token_dict["tokens"].cast<py::list>()) {
                      if (py::isinstance<py::str>(py_token)) {
                        tokens.push_back(py_token.cast<std::string>());
                      } else {
                        throw py::value_error(
                            "Type of args special_tokens dict need to have key "
                            "'tokens'"
                            "and the respective value should be List[str]");
                      }
                    }
                  } else {
                    throw py::value_error(
                        "Type of args special_tokens dict need to have key "
                        "'tokens'"
                        "and the respective value should be List[str]");
                  }
                  special_tokens.emplace_back(id, ids, tokens);
                } else {
                  throw py::value_error(
                      "Type of args special_tokens need to be "
                      "List[Union[Tuple[int, str], Tuple[str, int], dict]]");
                }
              }
              self.SetTokensMap(special_tokens);
            } else {
              throw py::value_error(
                  "Type of args special_tokens need to be "
                  "List[Union[Tuple[int, str], Tuple[str, int], dict]]");
            }
            return self;
          }),
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

}  // namespace pybind
}  // namespace faster_tokenizer
}  // namespace paddlenlp
