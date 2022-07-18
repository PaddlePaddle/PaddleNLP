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

#include <string>
#include "nlohmann/json.hpp"
#include "normalizers/normalizer.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace normalizers {
struct BertNormalizer : public Normalizer {
  BertNormalizer(bool clean_text = true,
                 bool handle_chinese_chars = true,
                 bool strip_accents = true,
                 bool lowercase = true);
  virtual void operator()(NormalizedString* input) const override;
  BertNormalizer(const BertNormalizer&) = default;
  BertNormalizer(BertNormalizer&&) = default;

private:
  bool clean_text_;
  bool handle_chinese_chars_;
  bool strip_accents_;
  bool lowercase_;
  void DoCleanText(NormalizedString* input) const;
  void DoHandleChineseChars(NormalizedString* input) const;
  friend void to_json(nlohmann::json& j, const BertNormalizer& bert_normalizer);
  friend void from_json(const nlohmann::json& j,
                        BertNormalizer& bert_normalizer);
};
}  // namespace normalizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
