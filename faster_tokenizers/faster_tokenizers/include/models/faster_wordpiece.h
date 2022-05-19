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
#include "models/wordpiece.h"
#include "nlohmann/json.hpp"

namespace tokenizers {
namespace models {

struct FasterWordPiece : public WordPiece {
  FasterWordPiece();
  FasterWordPiece(const core::Vocab& vocab,
                  const std::string& unk_token = "[UNK]",
                  size_t max_input_chars_per_word = 100,
                  const std::string& continuing_subword_prefix = "##");
  // Move version
  FasterWordPiece(core::Vocab&& vocab,
                  std::string&& unk_token,
                  size_t max_input_chars_per_word,
                  std::string&& continuing_subword_prefix);

  virtual std::vector<core::Token> Tokenize(
      const std::string& sequence) const override;

private:
  void ContructTrie(const core::Vocab&) const;
  friend void to_json(nlohmann::json& j, const FasterWordPiece& model);
  friend void from_json(const nlohmann::json& j, FasterWordPiece& model);
};

}  // models
}  // tokenizers
