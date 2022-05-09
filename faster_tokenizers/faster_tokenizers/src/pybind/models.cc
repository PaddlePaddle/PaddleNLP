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

#include "models/models.h"
#include "pybind/models.h"

namespace py = pybind11;

namespace tokenizers {
namespace pybind {

class PyModel : public models::Model {
public:
  using Model::Model;
  virtual std::vector<core::Token> Tokenize(
      const std::string& tokens) const override {
    PYBIND11_OVERLOAD_PURE_NAME(
        std::vector<core::Token>, Model, "tokenize", Tokenize, tokens);
  }

  virtual bool TokenToId(const std::string& token, uint* id) const override {
    PYBIND11_OVERLOAD_PURE_NAME(
        bool, Model, "token_to_id", TokenToId, token, id);
  }

  virtual bool IdToToken(uint id, std::string* token) const override {
    PYBIND11_OVERLOAD_PURE_NAME(
        bool, Model, "id_to_token", IdToToken, id, token);
  }

  virtual core::Vocab GetVocab() const override {
    PYBIND11_OVERLOAD_PURE_NAME(core::Vocab, Model, "get_vocab", GetVocab);
  }

  virtual size_t GetVocabSize() const override {
    PYBIND11_OVERLOAD_PURE_NAME(size_t, Model, "get_vocab_size", GetVocabSize);
  }

  virtual std::string Save(const std::string& folder,
                           const std::string& filename_prefix) const override {
    PYBIND11_OVERLOAD_PURE_NAME(
        std::string, Model, "save", Save, folder, filename_prefix);
  }
};

class PyWordPiece : public models::WordPiece {
  using WordPiece::WordPiece;
  virtual std::vector<core::Token> Tokenize(
      const std::string& tokens) const override {
    PYBIND11_OVERLOAD_NAME(
        std::vector<core::Token>, WordPiece, "tokenize", Tokenize, tokens);
  }

  virtual bool TokenToId(const std::string& token, uint* id) const override {
    PYBIND11_OVERLOAD_NAME(
        bool, WordPiece, "token_to_id", TokenToId, token, id);
  }

  virtual bool IdToToken(uint id, std::string* token) const override {
    PYBIND11_OVERLOAD_NAME(
        bool, WordPiece, "id_to_token", IdToToken, id, token);
  }

  virtual core::Vocab GetVocab() const override {
    PYBIND11_OVERLOAD_NAME(core::Vocab, WordPiece, "get_vocab", GetVocab);
  }

  virtual size_t GetVocabSize() const override {
    PYBIND11_OVERLOAD_NAME(size_t, WordPiece, "get_vocab_size", GetVocabSize);
  }

  virtual std::string Save(const std::string& folder,
                           const std::string& filename_prefix) const override {
    PYBIND11_OVERLOAD_NAME(
        std::string, WordPiece, "save", Save, folder, filename_prefix);
  }
};

void BindModels(pybind11::module* m) {
  auto submodule = m->def_submodule("models", "The models module");
  py::class_<models::Model, PyModel>(submodule, "Model")
      .def(py::init<>())
      .def("tokenize", &models::Model::Tokenize)
      .def("token_to_id", &models::Model::TokenToId)
      .def("id_to_token", &models::Model::IdToToken)
      .def("get_vocab", &models::Model::GetVocab)
      .def("get_vocab_size", &models::Model::GetVocabSize)
      .def("save", &models::Model::Save);
  py::class_<models::WordPiece, PyWordPiece>(submodule, "WordPiece")
      .def(py::init<>())
      .def(py::init<const core::Vocab&,
                    const std::string&,
                    size_t,
                    const std::string&>(),
           py::arg("vocab"),
           py::arg("unk_token") = "[UNK]",
           py::arg("max_input_chars_per_word") = 100,
           py::arg("continuing_subword_prefix") = "##")
      .def("tokenize", &models::WordPiece::Tokenize)
      .def("token_to_id", &models::WordPiece::TokenToId)
      .def("id_to_token", &models::WordPiece::IdToToken)
      .def("get_vocab", &models::WordPiece::GetVocab)
      .def("get_vocab_size", &models::WordPiece::GetVocabSize)
      .def_static(
          "read_file", &models::WordPiece::GetVocabFromFile, py::arg("vocab"))
      .def_static("from_file",
                  &models::WordPiece::GetWordPieceFromFile,
                  py::arg("vocab"),
                  py::arg("unk_token") = "[UNK]",
                  py::arg("max_input_chars_per_word") = 100,
                  py::arg("continuing_subword_prefix") = "##")
      .def("save",
           [](const models::WordPiece& wordpiece,
              const std::string& folder,
              const py::object& py_obj) {
             std::string prefix = "";
             if (!py_obj.is(py::none())) {
               prefix = py_obj.cast<std::string>();
             }
             return wordpiece.Save(folder, prefix);
           },
           py::arg("folder"),
           py::arg("prefix") = py::none());
}
}  // pybind
}  // tokenizers
