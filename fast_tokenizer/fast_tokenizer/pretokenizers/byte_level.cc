// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fast_tokenizer/pretokenizers/byte_level.h"

#include <codecvt>
#include <locale>

#include "fast_tokenizer/utils/utf8.h"
#include "fast_tokenizer/utils/utils.h"
#include "glog/logging.h"
#include "re2/re2.h"
#include "unicode/uchar.h"


namespace paddlenlp {
namespace fast_tokenizer {
namespace pretokenizers {


static re2::RE2 pattern(
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+)");


static std::unordered_map<uint8_t, uint32_t> BYTES_TO_CHARS =
    utils::CreateBytesToChars();
ByteLevelPreTokenizer::ByteLevelPreTokenizer(bool add_prefix_space,
                                             bool use_regex)
    : add_prefix_space_(add_prefix_space), use_regex_(use_regex) {}


void ByteLevelPreTokenizer::operator()(PreTokenizedString* pretokenized) const {
  std::vector<normalizers::NormalizedString> normalized_splits;
  pretokenized->Split([&normalized_splits, this](
                          int idx,
                          normalizers::NormalizedString* normalized,
                          std::vector<StringSplit>* string_splits) {
    if (this->add_prefix_space_ && normalized->GetStr().find(' ') != 0) {
      normalized->Prepend(" ");
    }
    if (this->use_regex_) {
      normalized->Split(pattern, core::SplitMode::ISOLATED, &normalized_splits);
      for (auto&& normalize : normalized_splits) {
        if (!normalize.IsEmpty()) {
          string_splits->emplace_back(std::move(normalize));
        }
      }
    } else {
      string_splits->emplace_back(*normalized);
    }
  });
  pretokenized->Normalize([](normalizers::NormalizedString* normalized) {
    const std::string& str = normalized->GetStr();
    std::u32string u32normalized;
    std::vector<int> changes;
    size_t utf8_len = 0;
    uint32_t last_char;
    uint32_t curr_char;
    while (utf8_len < str.length()) {
      auto chwidth = utils::UTF8ToUInt32(str.data() + utf8_len, &curr_char);
      curr_char = utils::UTF8ToUnicode(curr_char);
      for (int i = 0; i < chwidth; ++i) {
        u32normalized.push_back(BYTES_TO_CHARS.at(str[i + utf8_len]));
        if (i == 0) {
          changes.push_back(0);
        } else {
          changes.push_back(1);
        }
      }
      utf8_len += chwidth;
    }
    normalized->UpdateNormalized({u32normalized, changes}, 0);
  });
}


void to_json(nlohmann::json& j,
             const ByteLevelPreTokenizer& byte_pre_tokenizer) {
  j = {
      {"type", "ByteLevelPreTokenizer"},
      {"add_prefix_space", byte_pre_tokenizer.add_prefix_space_},
      {"use_regex", byte_pre_tokenizer.use_regex_},
  };
}


void from_json(const nlohmann::json& j,
               ByteLevelPreTokenizer& byte_pre_tokenizer) {
  j.at("add_prefix_space").get_to(byte_pre_tokenizer.add_prefix_space_);
  j.at("use_regex").get_to(byte_pre_tokenizer.add_prefix_space_);
}

void ProcessOffsets(core::Encoding* encoding, bool add_prefix_space) {
  auto process_token_fn =
      [&](uint32_t i, const std::string& token, core::Offset* offset) -> void {
    uint32_t leading_spaces = 0;
    uint32_t trailing_spaces = 0;

    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    std::u32string u32token = conv.from_bytes(token);
    for (int i = 0; i < u32token.size(); ++i) {
      if (utils::IsWhiteSpace(u32token[i]) ||
          u32token[i] == BYTES_TO_CHARS.at(' ')) {
        ++leading_spaces;
      } else {
        break;
      }
    }

    for (int i = u32token.size() - 1; i >= 0; --i) {
      if (utils::IsWhiteSpace(u32token[i]) ||
          u32token[i] == BYTES_TO_CHARS.at(' ')) {
        ++trailing_spaces;
      } else {
        break;
      }
    }

    if (leading_spaces > 0 || trailing_spaces > 0) {
      if (leading_spaces > 0) {
        bool is_first = (i == 0) || (offset->first == 0);
        if (is_first && add_prefix_space && leading_spaces == 1) {
          leading_spaces = 0;
        }
        offset->first =
            (std::min)(offset->first + leading_spaces, offset->second);
      }
    }
    if (trailing_spaces > 0 && offset->second >= trailing_spaces) {
      offset->second =
          (std::max)(offset->second - trailing_spaces, offset->first);
    }
  };
  encoding->ProcessTokenWithOffsets(process_token_fn);
}

}  // namespace pretokenizers
}  // namespace fast_tokenizer
}  // namespace paddlenlp
