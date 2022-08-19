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

#include <codecvt>
#include <exception>
#include <locale>

#include "glog/logging.h"
#include "pretokenizers/pretokenizer.h"
#include "utils/utf8.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace pretokenizers {

BytesToCharOffsetConverter::BytesToCharOffsetConverter(const std::string& seq)
    : OffsetConverter(seq) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::u32string u32seq = conv.from_bytes(seq);
  offset_map_.reserve(u32seq.length() * 4);
  for (int i = 0; i < u32seq.length(); ++i) {
    auto utf8_len = utils::GetUTF8CharLen(u32seq[i]);
    for (int j = 0; j < utf8_len; ++j) {
      offset_map_.push_back(i);
    }
  }
}

bool BytesToCharOffsetConverter::convert(const core::Offset& offset,
                                         core::Offset* result) const {
  size_t byte_start = offset.first;
  size_t byte_end = offset.second;
  if (offset_map_.size() <= byte_start) {
    return false;
  }
  auto char_start = offset_map_.at(byte_start);
  auto char_end = char_start + 1;
  if (offset_map_.size() > byte_end) {
    char_end = offset_map_.at(byte_end);
  } else if (offset_map_.size() > byte_end - 1) {
    char_end = offset_map_.at(byte_end - 1) + 1;
  }
  *result = {char_start, char_end};
  return true;
}

PreTokenizedString::PreTokenizedString(const std::string& original)
    : original_(original) {
  splits_.emplace_back(std::move(StringSplit(original_)));
}

PreTokenizedString::PreTokenizedString(
    const normalizers::NormalizedString& normalized)
    : original_(normalized.GetOrignalStr()) {
  splits_.emplace_back(std::move(StringSplit(original_)));
}

PreTokenizedString& PreTokenizedString::operator=(PreTokenizedString&& other) {
  original_ = std::move(other.original_);
  splits_ = std::move(other.splits_);
  return *this;
}

size_t PreTokenizedString::GetSplitsSize() const { return splits_.size(); }

StringSplit PreTokenizedString::GetSplit(int idx) const { return splits_[idx]; }

const std::string& PreTokenizedString::GetOriginStr() const {
  return original_;
}

void PreTokenizedString::Split(
    std::function<void(int,
                       normalizers::NormalizedString*,
                       std::vector<StringSplit>*)> split_fn) {
  std::vector<StringSplit> new_splits;
  new_splits.reserve(splits_.size());
  for (int i = 0; i < splits_.size(); ++i) {
    if (splits_[i].tokens_.size() > 0) {
      new_splits.emplace_back(std::move(splits_[i]));
      continue;
    }
    split_fn(i, &splits_[i].normalized_, &new_splits);
  }
  splits_ = std::move(new_splits);
}

void PreTokenizedString::Normalize(
    std::function<void(normalizers::NormalizedString*)> normalize_fn) {
  for (auto& split : splits_) {
    if (!split.tokens_.empty()) {
      normalize_fn(&split.normalized_);
    }
  }
}
void PreTokenizedString::Tokenize(
    std::function<std::vector<core::Token>(normalizers::NormalizedString*)>
        tokenize_fn) {
  for (auto& split : splits_) {
    if (split.tokens_.empty()) {
      split.tokens_ = std::move(tokenize_fn(&split.normalized_));
    }
  }
}

bool PreTokenizedString::TransformToEncoding(
    const std::vector<uint32_t>& input_word_idx,
    uint32_t type_id,
    core::OffsetType offset_type,
    core::Encoding* encoding) const {
  if (splits_.empty()) {
    *encoding = core::Encoding();
    return true;
  }
  for (const auto& split : splits_) {
    if (split.tokens_.empty()) {
      throw std::logic_error(
          "The split of PreTokenizedString is empty, please call "
          "PreTokenizedString::Tokenize first before transform to Encoding.");
      return false;
    }
  }

  if (offset_type == core::OffsetType::CHAR) {
    return TransformToEncodingUseConvertor<BytesToCharOffsetConverter>(
        input_word_idx, type_id, encoding);
  }
  return TransformToEncodingUseConvertor<OffsetConverter>(
      input_word_idx, type_id, encoding);
}

template <typename Convertor>
bool PreTokenizedString::TransformToEncodingUseConvertor(
    const std::vector<uint32_t>& input_word_idx,
    uint32_t type_id,
    core::Encoding* encoding) const {
  Convertor converter(original_);
  uint32_t tokens_size = 0;
  for (int i = 0; i < splits_.size(); ++i) {
    tokens_size += splits_[i].tokens_.size();
  }

  std::vector<uint32_t> token_ids(tokens_size);
  std::vector<std::string> tokens(tokens_size);
  std::vector<core::Offset> offsets(tokens_size);
  uint32_t curr_idx = 0;
  for (int i = 0; i < splits_.size(); ++i) {
    const auto& split = splits_[i];
    const auto& normalized = split.normalized_;
    auto offset = normalized.GetOrginalOffset();
    core::Offset tmp_offset;
    bool has_set_offset = false;
    for (const auto& token : split.tokens_) {
      auto token_offset = token.offset_;
      bool flag = normalized.ConvertOffsets(&token_offset, false);
      if (flag) {
        token_offset.first += offset.first;
        token_offset.second += offset.first;
      }
      if (has_set_offset) {
        offset = token_offset;
        has_set_offset = true;
      }
      converter.convert(token_offset, &tmp_offset);
      token_ids[curr_idx] = token.id_;
      tokens[curr_idx] = token.value_;
      offsets[curr_idx] = tmp_offset;
      ++curr_idx;
    }
  }
  // Setting words_idx
  std::vector<uint32_t> words_idx(tokens_size);
  if (input_word_idx.size() == 0) {
    uint32_t word_offset = 0;
    for (uint32_t i = 0; i < splits_.size(); ++i) {
      std::fill_n(
          words_idx.begin() + word_offset, splits_[i].tokens_.size(), i);
      word_offset += splits_[i].tokens_.size();
    }
  } else {
    std::fill(words_idx.begin(), words_idx.end(), input_word_idx[0]);
  }
  *encoding = std::move(core::Encoding(
      std::move(token_ids),
      std::vector<uint32_t>(tokens_size, type_id),  // type_ids
      std::move(tokens),
      std::move(words_idx),
      std::move(offsets),
      std::vector<uint32_t>(tokens_size, 0), /* special_tokens_mask */
      std::vector<uint32_t>(tokens_size, 1), /* attention_mask */
      std::vector<core::Encoding>(),         /* overflowing */
      std::unordered_map<uint32_t, core::Range>() /* sequence_ranges */));
  return true;
}

void PreTokenizedString::SetOriginalStr(const std::string& original) {
  original_ = original;
  splits_.clear();
  splits_.emplace_back(original_);
}

}  // namespace pretokenizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
