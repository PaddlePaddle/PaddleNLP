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
#include <locale>
#include <string>

#include "normalizers/unicode.h"
#include "unicode/edits.h"
#include "unicode/errorcode.h"
#include "unicode/normalizer2.h"
#include "unicode/utypes.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace normalizers {

void NFCNormalizer::operator()(NormalizedString* input) const { input->NFC(); }

void to_json(nlohmann::json& j, const NFCNormalizer& normalizer) {
  j = {
      {"type", "NFCNormalizer"},
  };
}

void from_json(const nlohmann::json& j, NFCNormalizer& normalizer) {}

void NFKCNormalizer::operator()(NormalizedString* input) const {
  input->NFKC();
}

void to_json(nlohmann::json& j, const NFKCNormalizer& normalizer) {
  j = {
      {"type", "NFKCNormalizer"},
  };
}

void from_json(const nlohmann::json& j, NFKCNormalizer& normalizer) {}

void NFDNormalizer::operator()(NormalizedString* input) const { input->NFD(); }

void to_json(nlohmann::json& j, const NFDNormalizer& normalizer) {
  j = {
      {"type", "NFDNormalizer"},
  };
}

void from_json(const nlohmann::json& j, NFDNormalizer& normalizer) {}

void NFKDNormalizer::operator()(NormalizedString* input) const {
  input->NFKD();
}

void to_json(nlohmann::json& j, const NFKDNormalizer& normalizer) {
  j = {
      {"type", "NFKDNormalizer"},
  };
}

void from_json(const nlohmann::json& j, NFKDNormalizer& normalizer) {}

void NmtNormalizer::operator()(NormalizedString* input) const {
  input->FilterChar([](char32_t ch) -> bool {
    if ((ch >= 0x0001 && ch <= 0x0008) || (ch == 0x000B) ||
        (ch >= 0x000E && ch <= 0x001F) || (ch == 0x007F) || (ch == 0x008F) ||
        (ch == 0x009F)) {
      return false;
    }
    return true;
  });
  input->MapChar([](char32_t ch) -> char32_t {
    if ((ch == 0x0009) || (ch == 0x000A) || (ch == 0x000C) || (ch == 0x000D) ||
        (ch == 0x1680) || (ch >= 0x200B && ch <= 0x200F) || (ch == 0x2028) ||
        (ch == 0x2029) || (ch == 0x2581) || (ch == 0xFEFF) || (ch == 0xFFFD)) {
      return ' ';
    }
    return ch;
  });
}

void to_json(nlohmann::json& j, const NmtNormalizer& normalizer) {
  j = {
      {"type", "NmtNormalizer"},
  };
}

void from_json(const nlohmann::json& j, NmtNormalizer& normalizer) {}

}  // namespace normalizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
