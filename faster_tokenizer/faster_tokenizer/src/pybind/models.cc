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

#include "models/models.h"
#include <Python.h>
#include "glog/logging.h"
#include "pybind/models.h"
#include "pybind/utils.h"

namespace py = pybind11;

namespace paddlenlp {
namespace faster_tokenizer {
namespace pybind {

class PyModel : public models::Model {
public:
  using Model::Model;
  virtual std::vector<core::Token> Tokenize(
      const std::string& tokens) override {
    PYBIND11_OVERLOAD_PURE_NAME(
        std::vector<core::Token>, Model, "tokenize", Tokenize, tokens);
  }

  virtual bool TokenToId(const std::string& token,
                         uint32_t* id) const override {
    PYBIND11_OVERLOAD_PURE_NAME(
        bool, Model, "token_to_id", TokenToId, token, id);
  }

  virtual bool IdToToken(uint32_t id, std::string* token) const override {
    PYBIND11_OVERLOAD_PURE_NAME(
        bool, Model, "id_to_token", IdToToken, id, token);
  }

  virtual core::Vocab GetVocab() const override {
    PYBIND11_OVERLOAD_PURE_NAME(core::Vocab, Model, "get_vocab", GetVocab);
  }

  virtual size_t GetVocabSize() const override {
    PYBIND11_OVERLOAD_PURE_NAME(size_t, Model, "get_vocab_size", GetVocabSize);
  }

  virtual std::vector<std::string> Save(
      const std::string& folder,
      const std::string& filename_prefix) const override {
    PYBIND11_OVERLOAD_PURE_NAME(
        std::vector<std::string>, Model, "save", Save, folder, filename_prefix);
  }
};

class PyWordPiece : public models::WordPiece {
  using WordPiece::WordPiece;
  virtual std::vector<core::Token> Tokenize(
      const std::string& tokens) override {
    PYBIND11_OVERLOAD_NAME(
        std::vector<core::Token>, WordPiece, "tokenize", Tokenize, tokens);
  }

  virtual bool TokenToId(const std::string& token,
                         uint32_t* id) const override {
    PYBIND11_OVERLOAD_NAME(
        bool, WordPiece, "token_to_id", TokenToId, token, id);
  }

  virtual bool IdToToken(uint32_t id, std::string* token) const override {
    PYBIND11_OVERLOAD_NAME(
        bool, WordPiece, "id_to_token", IdToToken, id, token);
  }

  virtual core::Vocab GetVocab() const override {
    PYBIND11_OVERLOAD_NAME(core::Vocab, WordPiece, "get_vocab", GetVocab);
  }

  virtual size_t GetVocabSize() const override {
    PYBIND11_OVERLOAD_NAME(size_t, WordPiece, "get_vocab_size", GetVocabSize);
  }

  virtual std::vector<std::string> Save(
      const std::string& folder,
      const std::string& filename_prefix) const override {
    PYBIND11_OVERLOAD_NAME(std::vector<std::string>,
                           WordPiece,
                           "save",
                           Save,
                           folder,
                           filename_prefix);
  }
};

class PyFasterWordPiece : public models::FasterWordPiece {
  using FasterWordPiece::FasterWordPiece;
  virtual std::vector<core::Token> Tokenize(
      const std::string& tokens) override {
    PYBIND11_OVERLOAD_NAME(std::vector<core::Token>,
                           FasterWordPiece,
                           "tokenize",
                           Tokenize,
                           tokens);
  }

  virtual bool TokenToId(const std::string& token,
                         uint32_t* id) const override {
    PYBIND11_OVERLOAD_NAME(
        bool, FasterWordPiece, "token_to_id", TokenToId, token, id);
  }

  virtual bool IdToToken(uint32_t id, std::string* token) const override {
    PYBIND11_OVERLOAD_NAME(
        bool, FasterWordPiece, "id_to_token", IdToToken, id, token);
  }

  virtual core::Vocab GetVocab() const override {
    PYBIND11_OVERLOAD_NAME(core::Vocab, FasterWordPiece, "get_vocab", GetVocab);
  }

  virtual size_t GetVocabSize() const override {
    PYBIND11_OVERLOAD_NAME(
        size_t, FasterWordPiece, "get_vocab_size", GetVocabSize);
  }

  virtual std::vector<std::string> Save(
      const std::string& folder,
      const std::string& filename_prefix) const override {
    PYBIND11_OVERLOAD_NAME(std::vector<std::string>,
                           FasterWordPiece,
                           "save",
                           Save,
                           folder,
                           filename_prefix);
  }
};

class PyBPE : public models::BPE {
  using BPE::BPE;
  virtual std::vector<core::Token> Tokenize(
      const std::string& tokens) override {
    PYBIND11_OVERLOAD_NAME(
        std::vector<core::Token>, BPE, "tokenize", Tokenize, tokens);
  }

  virtual bool TokenToId(const std::string& token,
                         uint32_t* id) const override {
    PYBIND11_OVERLOAD_NAME(bool, BPE, "token_to_id", TokenToId, token, id);
  }

  virtual bool IdToToken(uint32_t id, std::string* token) const override {
    PYBIND11_OVERLOAD_NAME(bool, BPE, "id_to_token", IdToToken, id, token);
  }

  virtual core::Vocab GetVocab() const override {
    PYBIND11_OVERLOAD_NAME(core::Vocab, BPE, "get_vocab", GetVocab);
  }

  virtual size_t GetVocabSize() const override {
    PYBIND11_OVERLOAD_NAME(size_t, BPE, "get_vocab_size", GetVocabSize);
  }

  virtual std::vector<std::string> Save(
      const std::string& folder,
      const std::string& filename_prefix) const override {
    PYBIND11_OVERLOAD_NAME(
        std::vector<std::string>, BPE, "save", Save, folder, filename_prefix);
  }
};

class PyUnigram : public models::Unigram {
  using Unigram::Unigram;
  virtual std::vector<core::Token> Tokenize(
      const std::string& tokens) override {
    PYBIND11_OVERLOAD_NAME(
        std::vector<core::Token>, Unigram, "tokenize", Tokenize, tokens);
  }

  virtual bool TokenToId(const std::string& token,
                         uint32_t* id) const override {
    PYBIND11_OVERLOAD_NAME(bool, Unigram, "token_to_id", TokenToId, token, id);
  }

  virtual bool IdToToken(uint32_t id, std::string* token) const override {
    PYBIND11_OVERLOAD_NAME(bool, Unigram, "id_to_token", IdToToken, id, token);
  }

  virtual core::Vocab GetVocab() const override {
    PYBIND11_OVERLOAD_NAME(core::Vocab, Unigram, "get_vocab", GetVocab);
  }

  virtual size_t GetVocabSize() const override {
    PYBIND11_OVERLOAD_NAME(size_t, Unigram, "get_vocab_size", GetVocabSize);
  }

  virtual std::vector<std::string> Save(
      const std::string& folder,
      const std::string& filename_prefix) const override {
    PYBIND11_OVERLOAD_NAME(std::vector<std::string>,
                           Unigram,
                           "save",
                           Save,
                           folder,
                           filename_prefix);
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
  py::class_<models::FasterWordPiece, PyFasterWordPiece>(submodule,
                                                         "FasterWordPiece")
      .def(py::init<>())
      .def(py::init<const core::Vocab&,
                    const std::string&,
                    size_t,
                    const std::string&,
                    bool>(),
           py::arg("vocab"),
           py::arg("unk_token") = "[UNK]",
           py::arg("max_input_chars_per_word") = 100,
           py::arg("continuing_subword_prefix") = "##",
           py::arg("with_pretokenization") = false)
      .def("tokenize", &models::FasterWordPiece::Tokenize)
      .def("token_to_id", &models::FasterWordPiece::TokenToId)
      .def("id_to_token", &models::FasterWordPiece::IdToToken)
      .def("get_vocab", &models::FasterWordPiece::GetVocab)
      .def("get_vocab_size", &models::FasterWordPiece::GetVocabSize)
      .def_static("read_file",
                  &models::FasterWordPiece::GetVocabFromFile,
                  py::arg("vocab"))
      .def_static("from_file",
                  &models::FasterWordPiece::GetWordPieceFromFile,
                  py::arg("vocab"),
                  py::arg("unk_token") = "[UNK]",
                  py::arg("max_input_chars_per_word") = 100,
                  py::arg("continuing_subword_prefix") = "##")
      .def("save",
           [](const models::FasterWordPiece& wordpiece,
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
  py::class_<models::BPE, PyBPE>(submodule, "BPE")
      .def(py::init([](const py::object& py_vocab,
                       const py::object& py_merges,
                       const py::object& py_cache_capacity,
                       const py::object& py_dropout,
                       const py::object& py_unk_token,
                       const py::object& py_continuing_subword_prefix,
                       const py::object& py_end_of_word_suffix,
                       const py::object& py_fuse_unk) {
             core::Vocab vocab;
             if (!py_vocab.is(py::none())) {
               vocab = py_vocab.cast<core::Vocab>();
             }

             core::Merges merges;
             if (!py_merges.is(py::none())) {
               merges = py_merges.cast<core::Merges>();
             }

             size_t cache_capacity = utils::DEFAULT_CACHE_CAPACITY;
             if (!py_cache_capacity.is(py::none())) {
               cache_capacity = py_cache_capacity.cast<size_t>();
             }

             std::vector<float> dropout;
             if (!py_dropout.is(py::none())) {
               dropout.emplace_back(py_dropout.cast<float>());
             }

             std::vector<std::string> unk_token;
             if (!py_unk_token.is(py::none())) {
               unk_token.emplace_back(py_unk_token.cast<std::string>());
             }

             std::vector<std::string> continuing_subword_prefix;
             if (!py_continuing_subword_prefix.is(py::none())) {
               continuing_subword_prefix.emplace_back(
                   py_continuing_subword_prefix.cast<std::string>());
             }

             std::vector<std::string> end_of_word_suffix;
             if (!py_end_of_word_suffix.is(py::none())) {
               end_of_word_suffix.emplace_back(
                   py_end_of_word_suffix.cast<std::string>());
             }

             bool fuse_unk = false;
             if (!py_fuse_unk.is(py::none())) {
               fuse_unk = py_fuse_unk.cast<bool>();
             }
             models::BPE self(vocab,
                              merges,
                              cache_capacity,
                              dropout,
                              unk_token,
                              continuing_subword_prefix,
                              end_of_word_suffix,
                              fuse_unk);
             return self;
           }),
           py::arg("vocab") = py::none(),
           py::arg("merges") = py::none(),
           py::arg("cache_capacity") = py::none(),
           py::arg("dropout") = py::none(),
           py::arg("unk_token") = py::none(),
           py::arg("continuing_subword_prefix") = py::none(),
           py::arg("end_of_word_suffix") = py::none(),
           py::arg("fuse_unk") = py::none())
      .def("tokenize", &models::BPE::Tokenize)
      .def("token_to_id", &models::BPE::TokenToId)
      .def("id_to_token", &models::BPE::IdToToken)
      .def("get_vocab", &models::BPE::GetVocab)
      .def("get_vocab_size", &models::BPE::GetVocabSize)
      .def("save",
           [](const models::BPE& bpe,
              const std::string& folder,
              const py::object& py_obj) {
             std::string prefix = "";
             if (!py_obj.is(py::none())) {
               prefix = py_obj.cast<std::string>();
             }
             return bpe.Save(folder, prefix);
           },
           py::arg("folder"),
           py::arg("prefix") = py::none())
      .def_static(
          "read_file",
          [](const std::string& vocab_path, const std::string& merges_path) {
            core::Vocab vocab;
            core::Merges merges;
            models::BPE::GetVocabAndMergesFromFile(
                vocab_path, merges_path, &vocab, &merges);
            return py::make_tuple(vocab, merges);
          },
          py::arg("vocab"),
          py::arg("merges"))
      .def_static(
          "from_file",
          [](const std::string& vocab_path,
             const std::string& merges_path,
             const py::kwargs& kwargs) {
            core::Vocab vocab;
            core::Merges merges;
            models::BPE::GetVocabAndMergesFromFile(
                vocab_path, merges_path, &vocab, &merges);
            VLOG(6) << "In BPE from_file:";
            size_t cache_capacity = utils::DEFAULT_CACHE_CAPACITY;
            if (kwargs.contains("cache_capacity")) {
              cache_capacity = kwargs["cache_capacity"].cast<size_t>();
              VLOG(6) << "cache_capacity = " << cache_capacity;
            }
            std::vector<float> dropout;
            if (kwargs.contains("dropout")) {
              dropout.emplace_back(kwargs["dropout"].cast<float>());
              VLOG(6) << "dropout = " << kwargs["dropout"].cast<float>();
            }

            std::vector<std::string> unk_token;
            if (kwargs.contains("unk_token")) {
              unk_token.emplace_back(kwargs["unk_token"].cast<std::string>());
              VLOG(6) << "unk_token = "
                      << kwargs["unk_token"].cast<std::string>();
            }

            std::vector<std::string> continuing_subword_prefix;
            if (kwargs.contains("continuing_subword_prefix")) {
              continuing_subword_prefix.emplace_back(
                  kwargs["continuing_subword_prefix"].cast<std::string>());
              VLOG(6)
                  << "continuing_subword_prefix = "
                  << kwargs["continuing_subword_prefix"].cast<std::string>();
            }

            std::vector<std::string> end_of_word_suffix;
            if (kwargs.contains("end_of_word_suffix")) {
              end_of_word_suffix.emplace_back(
                  kwargs["end_of_word_suffix"].cast<std::string>());
              VLOG(6) << "end_of_word_suffix = "
                      << kwargs["end_of_word_suffix"].cast<std::string>();
            }

            bool fuse_unk = false;
            if (kwargs.contains("fuse_unk")) {
              fuse_unk = kwargs["fuse_unk"].cast<bool>();
              VLOG(6) << "fuse_unk = " << kwargs["fuse_unk"].cast<bool>();
            }
            return models::BPE(vocab,
                               merges,
                               cache_capacity,
                               dropout,
                               unk_token,
                               continuing_subword_prefix,
                               end_of_word_suffix,
                               fuse_unk);
          },
          py::arg("vocab"),
          py::arg("merges"));
  py::class_<models::Unigram, PyUnigram>(submodule, "Unigram")
      .def(py::init([](const py::object& py_vocab_list,
                       const py::object& py_unk_token_id) {
             if (py_vocab_list.is(py::none()) &&
                 py_unk_token_id.is(py::none())) {
               return models::Unigram();
             } else if (!py_vocab_list.is(py::none()) &&
                        !py_unk_token_id.is(py::none())) {
               try {
                 core::VocabList vocab_list =
                     py_vocab_list.cast<core::VocabList>();
                 size_t unk_id = py_unk_token_id.cast<size_t>();
                 return models::Unigram(vocab_list, {unk_id});
               } catch (std::exception& e) {
                 VLOG(0) << "Init Unigram error:" << e.what();
                 goto error;
               }
             }
           error:
             throw py::value_error(
                 "`vocab` and `unk_id` must be both specified");
           }),
           py::arg("vocab") = py::none(),
           py::arg("unk_id") = py::none())
      .def("tokenize", &models::Unigram::Tokenize)
      .def("token_to_id", &models::Unigram::TokenToId)
      .def("id_to_token", &models::Unigram::IdToToken)
      .def("get_vocab", &models::Unigram::GetVocab)
      .def("get_vocab_size", &models::Unigram::GetVocabSize)
      .def("set_filter_token",
           &models::Unigram::SetFilterToken,
           py::arg("filter_token") = "")
      .def("set_split_rule",
           &models::Unigram::SetSplitRule,
           py::arg("split_rule") = "")
      .def("save",
           [](const models::Unigram& unigram,
              const std::string& folder,
              const py::object& py_obj) {
             std::string prefix = "";
             if (!py_obj.is(py::none())) {
               prefix = py_obj.cast<std::string>();
             }
             return unigram.Save(folder, prefix);
           },
           py::arg("folder"),
           py::arg("prefix") = py::none());
}
}  // namespace pybind
}  // namespace faster_tokenizer
}  // namespace paddlenlp
