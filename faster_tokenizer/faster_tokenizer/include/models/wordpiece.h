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

#pragma once

#include "models/model.h"
#include "nlohmann/json.hpp"

namespace paddlenlp {
namespace faster_tokenizer {
namespace models {

struct WordPiece : public Model {
  WordPiece();
  WordPiece(const core::Vocab& vocab,
            const std::string& unk_token = "[UNK]",
            size_t max_input_chars_per_word = 100,
            const std::string& continuing_subword_prefix = "##");
  // Move version
  WordPiece(core::Vocab&& vocab,
            std::string&& unk_token,
            size_t max_input_chars_per_word,
            std::string&& continuing_subword_prefix);
  virtual std::vector<core::Token> Tokenize(
      const std::string& sequence) override;
  virtual bool TokenToId(const std::string& token, uint32_t* id) const override;
  virtual bool IdToToken(uint32_t id, std::string* token) const override;
  virtual core::Vocab GetVocab() const override;
  virtual size_t GetVocabSize() const override;
  // Return the saved voacb full path
  virtual std::vector<std::string> Save(
      const std::string& folder,
      const std::string& filename_prefix) const override;
  static core::Vocab GetVocabFromFile(const std::string& file);
  static WordPiece GetWordPieceFromFile(
      const std::string& file,
      const std::string& unk_token = "[UNK]",
      size_t max_input_chars_per_word = 100,
      const std::string& continuing_subword_prefix = "##");

protected:
  core::Vocab vocab_;
  core::VocabReversed vocab_reversed_;
  std::string unk_token_;
  uint32_t unk_token_id_;
  size_t max_input_chars_per_word_;
  std::string continuing_subword_prefix_;
  friend void to_json(nlohmann::json& j, const WordPiece& model);
  friend void from_json(const nlohmann::json& j, WordPiece& model);
};

struct WordPieceConfig {
  WordPieceConfig();
  std::string files_;
  core::Vocab vocab_;
  std::string unk_token_;
  size_t max_input_chars_per_word_;
  std::string continuing_subword_prefix_;
};


struct WordPieceFactory {
  WordPieceConfig config_;
  void SetFiles(const std::string& files);
  void SetUNKToken(const std::string& unk_token);
  void SetMaxInputCharsPerWord(size_t max_input_chars_per_word);
  void SetContinuingSubwordPrefix(const std::string& continuing_subword_prefix);
  WordPiece CreateWordPieceModel();
  void GetVocabFromFiles(const std::string& files);
};

}  // namespace models
}  // namespace faster_tokenizer
}  // namespace paddlenlp
