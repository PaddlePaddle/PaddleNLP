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

#include "pretokenizers/bert.h"
#include "glog/logging.h"
#include "re2/re2.h"
#include "unicode/uchar.h"
#include "utils/utils.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace pretokenizers {

// Note (zhoushunjie): Init re2::RE2 objects cost too much,
// ensure not init in the operator()
static re2::RE2 pattern("[\\s\\p{Zs}]+");
static re2::RE2 punc_pattern("[[:punct:]]|[\\pP]");

void BertPreTokenizer::operator()(PreTokenizedString* pretokenized) const {
  std::vector<normalizers::NormalizedString> normalized_splits;
  pretokenized->Split([&normalized_splits](
      int idx,
      normalizers::NormalizedString* normalized,
      std::vector<StringSplit>* string_splits) {
    // Use single character match instead of regex to improve performance
    normalized->Split([](char32_t ch) -> bool { return u_isUWhiteSpace(ch); },
                      normalizers::REMOVED,
                      &normalized_splits);
    for (auto&& normalize : normalized_splits) {
      if (!normalize.IsEmpty()) {
        string_splits->emplace_back(std::move(normalize));
      }
    }
  });
  normalized_splits.clear();
  pretokenized->Split(
      [&normalized_splits](int idx,
                           normalizers::NormalizedString* normalized,
                           std::vector<StringSplit>* string_splits) {
        // Use single character match instead of regex to improve performance
        normalized->Split(
            utils::IsPunctuation, normalizers::ISOLATED, &normalized_splits);
        for (auto&& normalize : normalized_splits) {
          if (!normalize.IsEmpty()) {
            VLOG(6) << "After pretokenized: " << normalize.GetStr();
            string_splits->emplace_back(std::move(normalize));
          }
        }
      });
}

void to_json(nlohmann::json& j, const BertPreTokenizer& bert_pre_tokenizer) {
  j = {
      {"type", "BertPreTokenizer"},
  };
}

void from_json(const nlohmann::json& j, BertPreTokenizer& bert_pre_tokenizer) {}

}  // namespace pretokenizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
