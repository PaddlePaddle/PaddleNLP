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

#include "normalizers/replace.h"
#include "utils/unique_ptr.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace normalizers {

ReplaceNormalizer::ReplaceNormalizer(const std::string& pattern,
                                     const std::string& content)
    : pattern_(new re2::RE2(pattern)), content_(content) {}

ReplaceNormalizer::ReplaceNormalizer(
    const ReplaceNormalizer& replace_normalizer)
    : pattern_(new re2::RE2(replace_normalizer.pattern_->pattern())),
      content_(replace_normalizer.content_) {}

void ReplaceNormalizer::operator()(NormalizedString* input) const {
  input->Replace(*pattern_, content_);
}

void to_json(nlohmann::json& j, const ReplaceNormalizer& replace_normalizer) {
  j = {
      {"type", "ReplaceNormalizer"},
      {"pattern", replace_normalizer.pattern_->pattern()},
      {"content", replace_normalizer.content_},
  };
}

void from_json(const nlohmann::json& j, ReplaceNormalizer& replace_normalizer) {
  replace_normalizer.pattern_ =
      utils::make_unique<re2::RE2>(std::string(j.at("pattern")));
  j.at("content").get_to(replace_normalizer.content_);
}

}  // namespace normalizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
