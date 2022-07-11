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

#include "normalizers/bert.h"
#include <algorithm>
#include <codecvt>
#include <locale>
#include "glog/logging.h"
#include "normalizers/strip.h"
#include "normalizers/utils.h"
#include "unicode/uchar.h"
#include "unicode/unistr.h"
#include "utils/utils.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace normalizers {
BertNormalizer::BertNormalizer(bool clean_text,
                               bool handle_chinese_chars,
                               bool strip_accents,
                               bool lowercase)
    : clean_text_(clean_text),
      handle_chinese_chars_(handle_chinese_chars),
      strip_accents_(strip_accents),
      lowercase_(lowercase) {}

static bool IsWhiteSpace(int ch) {
  const std::string WHITESPACE = " \n\r\t\f\v";
  for (int i = 0; i < WHITESPACE.length(); ++i) {
    if (ch == WHITESPACE[i]) return true;
  }
  return u_isspace(ch);
}

static bool IsControl(int ch) {
  if (ch == '\t' || ch == '\n' || ch == '\r') return false;
  // It means (general category "C").
  return !u_isprint(ch);
}

void BertNormalizer::DoCleanText(NormalizedString* input) const {
  (*input)
      .FilterChar([](char32_t ch) -> bool {
        return !(ch == 0 || ch == 0xfffd || IsControl(ch));
      })
      .MapChar([](char32_t ch) -> char32_t {
        if (IsWhiteSpace(ch)) {
          return ' ';
        }
        return ch;
      });
}

void BertNormalizer::DoHandleChineseChars(NormalizedString* input) const {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::u32string u32input = conv.from_bytes(input->GetStr());
  std::u32string u32output;
  std::vector<int> changes;
  u32output.reserve(u32input.length() * 3);
  changes.reserve(u32input.length() * 3);
  for (int i = 0; i < u32input.length(); ++i) {
    if (utils::IsChineseChar(u32input[i])) {
      u32output.push_back(' ');
      u32output.push_back(u32input[i]);
      u32output.push_back(' ');
      changes.push_back(0);
      changes.push_back(1);
      changes.push_back(1);
    } else {
      u32output.push_back(u32input[i]);
      changes.push_back(0);
    }
  }
  OffsetMapping new_normalized_offset{u32output, changes};
  input->UpdateNormalized(new_normalized_offset, 0);
}
void BertNormalizer::operator()(NormalizedString* input) const {
  if (clean_text_) {
    DoCleanText(input);
  }
  if (handle_chinese_chars_) {
    DoHandleChineseChars(input);
  }
  if (strip_accents_) {
    StripAccentsNormalizer()(input);
  }
  if (lowercase_) {
    input->Lowercase();
  }
}

void to_json(nlohmann::json& j, const BertNormalizer& bert_normalizer) {
  j = {
      {"type", "BertNormalizer"},
      {"clean_text", bert_normalizer.clean_text_},
      {"handle_chinese_chars", bert_normalizer.handle_chinese_chars_},
      {"strip_accents", bert_normalizer.strip_accents_},
      {"lowercase", bert_normalizer.lowercase_},
  };
}

void from_json(const nlohmann::json& j, BertNormalizer& bert_normalizer) {
  j.at("clean_text").get_to(bert_normalizer.clean_text_);
  j.at("handle_chinese_chars").get_to(bert_normalizer.handle_chinese_chars_);
  j.at("lowercase").get_to(bert_normalizer.lowercase_);
  j.at("strip_accents").get_to(bert_normalizer.strip_accents_);
}

}  // namespace normalizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
