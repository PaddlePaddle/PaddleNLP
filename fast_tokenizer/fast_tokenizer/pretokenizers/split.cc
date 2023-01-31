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

#include "fast_tokenizer/pretokenizers/split.h"

#include "fast_tokenizer/core/base.h"
#include "fast_tokenizer/normalizers/normalizer.h"
#include "fast_tokenizer/utils/unique_ptr.h"
#include "re2/re2.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace pretokenizers {

SplitPreTokenizer::SplitPreTokenizer(
    const SplitPreTokenizer& split_pretokenizer)
    : pattern_(new re2::RE2(split_pretokenizer.pattern_->pattern())) {
  split_mode_ = split_pretokenizer.split_mode_;
  invert_ = split_pretokenizer.invert_;
}

SplitPreTokenizer::SplitPreTokenizer(const std::string& pattern,
                                     core::SplitMode split_mode,
                                     bool invert)
    : invert_(invert), split_mode_(split_mode) {
  pattern_ = utils::make_unique<re2::RE2>(pattern);
}

void SplitPreTokenizer::operator()(PreTokenizedString* pretokenized) const {
  pretokenized->Split([&](int idx,
                          normalizers::NormalizedString* normalized,
                          std::vector<StringSplit>* string_splits) {
    std::vector<normalizers::NormalizedString> normalized_splits;
    normalized->Split(*pattern_, split_mode_, &normalized_splits, invert_);
    for (auto& normalize : normalized_splits) {
      string_splits->push_back(StringSplit(normalize));
    }
  });
}


void to_json(nlohmann::json& j, const SplitPreTokenizer& split_pretokenizer) {
  j = {
      {"type", "SplitPreTokenizer"},
      {"pattern", split_pretokenizer.pattern_->pattern()},
      {"split_mode", split_pretokenizer.split_mode_},
      {"invert", split_pretokenizer.invert_},
  };
}

void from_json(const nlohmann::json& j, SplitPreTokenizer& split_pretokenizer) {
  split_pretokenizer.pattern_ =
      utils::make_unique<re2::RE2>(j.at("pattern").get<std::string>());
  j.at("split_mode").get_to(split_pretokenizer.split_mode_);
  j.at("invert").get_to(split_pretokenizer.invert_);
}


}  // namespace pretokenizers
}  // namespace fast_tokenizer
}  // namespace paddlenlp