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

#include <algorithm>
#include <codecvt>
#include <fstream>
#include <locale>
#include <map>

#include "glog/logging.h"
#include "models/faster_wordpiece.h"
#include "models/wordpiece.h"
#include "utils/path.h"
#include "utils/utf8.h"

namespace tokenizers {
namespace models {

const std::string WHITESPACE = " \n\r\t\f\v";

FasterWordPiece::FasterWordPiece() : WordPiece() {}

FasterWordPiece::FasterWordPiece(const core::Vocab& vocab,
                                 const std::string& unk_token,
                                 size_t max_input_chars_per_word,
                                 const std::string& continuing_subword_prefix)
    : WordPiece(vocab,
                unk_token,
                max_input_chars_per_word,
                continuing_subword_prefix),
      trie_(vocab) {}

bool FasterWordPiece::TokenToId(const std::string& token, uint* id) const {
  return true;
}

std::vector<core::Token> FasterWordPiece::Tokenize(
    const std::string& sequence) const {
  std::vector<core::Token> all_tokens;
  size_t unicode_len =
      utils::GetUnicodeLenFromUTF8(sequence.data(), sequence.length());
  if (unicode_len > max_input_chars_per_word_) {
    all_tokens.emplace_back(
        vocab_.at(unk_token_), unk_token_, core::Offset{0, sequence.length()});
  } else {
  }
  return all_tokens;
}

void to_json(nlohmann::json& j, const FasterWordPiece& model) {
  j = {
      {"type", "FasterWordPiece"},
      {"vocab", model.vocab_},
      {"unk_token", model.unk_token_},
      {"max_input_chars_per_word", model.max_input_chars_per_word_},
      {"continuing_subword_prefix", model.continuing_subword_prefix_},
  };
}

void from_json(const nlohmann::json& j, FasterWordPiece& model) {
  j["vocab"].get_to(model.vocab_);
  j["unk_token"].get_to(model.unk_token_);
  j["max_input_chars_per_word"].get_to(model.max_input_chars_per_word_);
  j["continuing_subword_prefix"].get_to(model.continuing_subword_prefix_);
}

}  // models
}  // tokenizers
