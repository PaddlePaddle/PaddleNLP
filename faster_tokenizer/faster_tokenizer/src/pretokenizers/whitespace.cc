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

#include "pretokenizers/whitespace.h"
#include "normalizers/normalizer.h"
#include "re2/re2.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace pretokenizers {
static re2::RE2 pattern("[\\s\\p{Zs}]+");

void WhitespacePreTokenizer::operator()(
    PreTokenizedString* pretokenized) const {
  pretokenized->Split([&](int idx,
                          normalizers::NormalizedString* normalized,
                          std::vector<StringSplit>* string_splits) {
    std::vector<normalizers::NormalizedString> normalized_splits;
    normalized->Split(pattern, normalizers::REMOVED, &normalized_splits);
    for (auto& normalize : normalized_splits) {
      string_splits->push_back(StringSplit(normalize));
    }
  });
}

void to_json(nlohmann::json& j,
             const WhitespacePreTokenizer& whitespace_pretokenizer) {
  j = {
      {"type", "WhitespacePreTokenizer"},
  };
}

void from_json(const nlohmann::json& j,
               WhitespacePreTokenizer& whitespace_pretokenizer) {}

}  // namespace pretokenizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
