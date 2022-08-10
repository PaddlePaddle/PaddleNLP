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

#include "decoders/wordpiece.h"
#include "utils/utils.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace decoders {

WordPiece::WordPiece(const std::string prefix, bool cleanup)
    : prefix_(prefix), cleanup_(cleanup) {}

void WordPiece::CleanUp(std::string* result) const {
  utils::StringReplaceAll(result, " .", ".");
  utils::StringReplaceAll(result, " !", "!");
  utils::StringReplaceAll(result, " ?", "?");
  utils::StringReplaceAll(result, " ,", ",");
  utils::StringReplaceAll(result, " ' ", "'");
  utils::StringReplaceAll(result, " n't", "n't");
  utils::StringReplaceAll(result, " 'm", "'m");
  utils::StringReplaceAll(result, " do not", " don't");
  utils::StringReplaceAll(result, " 's", "'s");
  utils::StringReplaceAll(result, " 've", "'ve");
  utils::StringReplaceAll(result, " 're", "'re");
}

void WordPiece::operator()(const std::vector<std::string> tokens,
                           std::string* result) const {
  *result = "";
  for (int i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      *result += " ";
    }
    *result += tokens[i];
  }
  utils::StringReplaceAll(result, " " + prefix_, "");
  if (cleanup_) {
    CleanUp(result);
  }
}

void to_json(nlohmann::json& j, const WordPiece& decoder) {
  j = {
      {"type", "WordPiece"},
      {"cleanup", decoder.cleanup_},
      {"prefix", decoder.prefix_},
  };
}

void from_json(const nlohmann::json& j, WordPiece& decoder) {
  j["cleanup"].get_to(decoder.cleanup_);
  j["prefix"].get_to(decoder.prefix_);
}

}  // namespace decoders
}  // namespace faster_tokenizer
}  // namespace paddlenlp
