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

#include "pretokenizers/metaspace.h"
#include "re2/re2.h"
#include "utils/utf8.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace pretokenizers {

static re2::RE2 pattern(" ");

void MetaSpacePreTokenizer::UpdateReplacementChar() {
  uint32_t ch;
  utils::UTF8ToUInt32(replacement_.data(), &ch);
  replacement_char_ = utils::UTF8ToUnicode(ch);
}

MetaSpacePreTokenizer::MetaSpacePreTokenizer(const std::string& replacement,
                                             bool add_prefix_space)
    : replacement_(replacement), add_prefix_space_(add_prefix_space) {
  UpdateReplacementChar();
}

std::string MetaSpacePreTokenizer::GetReplacement() const {
  return replacement_;
}

void MetaSpacePreTokenizer::SetReplacement(const std::string& replacement) {
  replacement_ = replacement;
  UpdateReplacementChar();
}

void MetaSpacePreTokenizer::operator()(PreTokenizedString* pretokenized) const {
  std::vector<normalizers::NormalizedString> normalized_splits;
  pretokenized->Split([&](int idx,
                          normalizers::NormalizedString* normalized,
                          std::vector<StringSplit>* string_splits) {
    normalized->Replace(pattern, replacement_);
    if (add_prefix_space_ && normalized->GetStr().find(replacement_) != 0) {
      normalized->Prepend(replacement_);
    }
    normalized->Split(
        [&](char32_t ch) -> bool { return ch == replacement_char_; },
        normalizers::MERGED_WITH_NEXT,
        &normalized_splits);
    for (auto&& normalize : normalized_splits) {
      if (!normalize.IsEmpty()) {
        VLOG(6) << "After pretokenized: " << normalize.GetStr();
        string_splits->emplace_back(std::move(normalize));
      }
    }
  });
}

void to_json(nlohmann::json& j,
             const MetaSpacePreTokenizer& meta_pretokenizer) {
  j = {
      {"type", "MetaSpacePreTokenizer"},
      {"replacement", meta_pretokenizer.replacement_},
      {"add_prefix_space", meta_pretokenizer.add_prefix_space_},
  };
}

void from_json(const nlohmann::json& j,
               MetaSpacePreTokenizer& meta_pretokenizer) {
  j.at("add_prefix_space").get_to(meta_pretokenizer.add_prefix_space_);
  meta_pretokenizer.SetReplacement(j.at("replacement").get<std::string>());
}

}  // namespace pretokenizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
