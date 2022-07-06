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

#include "normalizers/strip.h"
#include "unicode/translit.h"
#include "unicode/unistr.h"
#include "unicode/utypes.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace normalizers {
StripNormalizer::StripNormalizer(bool left /* = true*/, bool right /* = true*/)
    : left_(left), right_(right) {}

void StripNormalizer::operator()(NormalizedString* input) const {
  if (left_) {
    input->LStrip();
  }
  if (right_) {
    input->RStrip();
  }
}

void to_json(nlohmann::json& j, const StripNormalizer& strip_normalizer) {
  j = {
      {"type", "StripNormalizer"},
      {"left", strip_normalizer.left_},
      {"right", strip_normalizer.right_},
  };
}

void from_json(const nlohmann::json& j, StripNormalizer& strip_normalizer) {
  j.at("left").get_to(strip_normalizer.left_);
  j.at("right").get_to(strip_normalizer.right_);
}

void StripAccentsNormalizer::operator()(NormalizedString* input) const {
  input->NFD();
  input->FilterChar([](char32_t ch) -> bool {
    // equals to `unicodedata.category(char) == 'Mn'`
    return u_charType(ch) != U_NON_SPACING_MARK;
  });
}

void to_json(nlohmann::json& j,
             const StripAccentsNormalizer& strip_normalizer) {
  j = {
      {"type", "StripAccentsNormalizer"},
  };
}

void from_json(const nlohmann::json& j,
               StripAccentsNormalizer& strip_normalizer) {}

}  // namespace normalizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
