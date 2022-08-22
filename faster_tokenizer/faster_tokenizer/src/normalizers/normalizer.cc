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
#include <vector>

#include "normalizers/normalizer.h"
#include "utils/utf8.h"

#include "glog/logging.h"
#include "normalizers/unicode.h"
#include "re2/re2.h"
#include "unicode/edits.h"
#include "unicode/errorcode.h"
#include "unicode/normalizer2.h"
#include "unicode/uchar.h"
#include "unicode/utypes.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace normalizers {

NormalizedString::NormalizedString(const std::string& original)
    : original_(original), normalized_(original), original_shift_(0) {
  // calculate alignments
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::u32string u32normalized = conv.from_bytes(normalized_);
  for (int i = 0; i < u32normalized.length(); ++i) {
    auto new_normalized_char_len = utils::GetUTF8CharLen(u32normalized[i]);
    uint32_t start = 0;
    uint32_t end = 0;
    if (i != 0) {
      start = alignments_.back().second;
    }
    end = start + new_normalized_char_len;
    for (int j = 0; j < new_normalized_char_len; ++j) {
      alignments_.push_back({start, end});
    }
  }
}

NormalizedString::NormalizedString(NormalizedString&& other)
    : original_(std::move(other.original_)),
      normalized_(std::move(other.normalized_)),
      alignments_(std::move(other.alignments_)),
      original_shift_(other.original_shift_) {}

NormalizedString& NormalizedString::operator=(NormalizedString&& other) {
  original_ = std::move(other.original_);
  normalized_ = std::move(other.normalized_);
  alignments_ = std::move(other.alignments_);
  original_shift_ = other.original_shift_;
  return *this;
}

const std::string& NormalizedString::GetStr() const { return normalized_; }

const std::string& NormalizedString::GetOrignalStr() const { return original_; }

uint32_t NormalizedString::GetLen() const { return normalized_.length(); }

uint32_t NormalizedString::GetOriginalLen() const { return original_.length(); }

core::Offset NormalizedString::GetOrginalOffset() const {
  return {original_shift_, GetOriginalLen() + original_shift_};
}

bool NormalizedString::IsEmpty() const { return normalized_.empty(); }

bool NormalizedString::IsOriginalEmpty() const { return original_.empty(); }

void NormalizedString::UpdateNormalized(const OffsetMapping& new_normalized,
                                        uint32_t initial_offset) {
  UpdateNormalizedRange(new_normalized, initial_offset, {0, GetLen()}, true);
}

void NormalizedString::UpdateNormalizedRange(
    const OffsetMapping& new_normalized,
    uint32_t initial_offset,
    const core::Range& range,
    bool origin_range) {
  auto n_range = range;
  if (origin_range) {
    ConvertOffsets(&n_range, origin_range);
  }
  // Retrieve the original characters that are being replaced. This let us
  // compute the change in byte sizes along the way.
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::u32string u32replaced_normalized = conv.from_bytes(
      normalized_.substr(n_range.first, n_range.second - n_range.first));
  uint32_t initial_removed = 0;
  // calculate initial_removed
  for (int i = 0; i < initial_offset; ++i) {
    size_t chwidth = utils::GetUTF8CharLen(u32replaced_normalized[i]);
    initial_removed += chwidth;
  }

  uint32_t offset = initial_removed + n_range.first;
  std::vector<core::Range> alignments;
  alignments.reserve(n_range.second - n_range.first);

  int replaced_normalized_idx = initial_removed;
  // Calculate the new alignments
  for (int i = 0; i < new_normalized.u32normalized.length(); ++i) {
    auto idx = offset;
    core::Range align;
    int curr_changes = new_normalized.changes[i];
    if (curr_changes > 0) {
      // Insert a char
      if (idx < 1) {
        align = {0, 0};
      } else {
        align = alignments_[idx - 1];
      }
    } else {
      align = alignments_[idx];
    }
    char32_t new_normalized_char = new_normalized.u32normalized[i];
    auto new_normalized_char_len = utils::GetUTF8CharLen(new_normalized_char);
    char32_t replaced_char = -1;
    if (curr_changes <= 0) {
      replaced_char = u32replaced_normalized[replaced_normalized_idx++];
    }
    uint32_t replaced_char_size =
        (replaced_char == -1) ? 0 : utils::GetUTF8CharLen(replaced_char);

    uint32_t total_bytes_to_remove = 0;
    if (curr_changes < 0) {
      for (int j = 0; j < -curr_changes; ++j) {
        replaced_char = u32replaced_normalized[replaced_normalized_idx++];
        total_bytes_to_remove += utils::GetUTF8CharLen(replaced_char);
      }
    }
    offset += replaced_char_size + total_bytes_to_remove;
    alignments.insert(alignments.end(), new_normalized_char_len, align);
  }
  // Replace the old alignments in n_range
  if (n_range.second - n_range.first >= alignments.size()) {
    std::memcpy(alignments_.data() + n_range.first,
                alignments.data(),
                alignments.size() * sizeof(core::Range));
    alignments_.erase(alignments_.begin() + n_range.first + alignments.size(),
                      alignments_.begin() + n_range.second);
  } else {
    std::vector<core::Range> new_alignments;
    auto third_len = 0;
    if (alignments_.size() > n_range.second) {
      third_len = alignments_.size() - n_range.second;
    }
    new_alignments.resize(n_range.first + alignments.size() + third_len);
    if (n_range.first > 0) {
      std::copy_n(alignments_.begin(), n_range.first, new_alignments.begin());
    }
    std::copy_n(alignments.begin(),
                alignments.size(),
                new_alignments.begin() + n_range.first);
    if (third_len > 0) {
      std::copy_n(alignments_.begin() + n_range.second,
                  third_len,
                  new_alignments.begin() + n_range.first + alignments.size());
    }
    alignments_ = std::move(new_alignments);
  }
  // Unicode -> UTF8
  uint32_t normalized_utf8_size = 0;
  for (auto& ch : new_normalized.u32normalized) {
    normalized_utf8_size += utils::GetUTF8CharLen(ch);
  }
  std::vector<char> utf8_str(normalized_utf8_size + 1);
  utils::GetUTF8Str(new_normalized.u32normalized.data(),
                    utf8_str.data(),
                    new_normalized.u32normalized.length());

  // Update normalized_
  auto normalized_iter = normalized_.begin();
  normalized_.replace(normalized_iter + n_range.first,
                      normalized_iter + n_range.second,
                      utf8_str.data(),
                      normalized_utf8_size);
}

bool NormalizedString::ConvertOffsets(core::Range* range,
                                      bool origin_range) const {
  auto len_original = GetOriginalLen();
  auto len_normalized = GetLen();
  if (range->first == range->second) {
    return true;
  }
  if (range->first > range->second) {
    return false;
  }
  if (origin_range && original_.empty() &&
      (range->first == 0 && range->second == 0)) {
    range->second = len_normalized;
    return true;
  }
  if (!origin_range && normalized_.empty() &&
      (range->first == 0 && range->second == 0)) {
    range->second = len_original;
    return true;
  }
  if (origin_range) {
    int start = -1;
    int end = -1;
    for (int i = 0; i < alignments_.size(); ++i) {
      if (range->second >= alignments_[i].second) {
        if (start < 0 && range->first <= alignments_[i].first) {
          if (alignments_[i].first != alignments_[i].second) {
            start = i;
          }
        }
        if (range->second >= alignments_[i].second) {
          end = i + 1;
        }
      }
    }
    if (start > 0 && end < 0) {
      *range = {start, start};
    } else if (start < 0 && end > 0) {
      *range = {end, end};
    } else if (start > 0 && end > 0) {
      *range = {start, end};
    } else {
      return false;
    }
  } else {
    range->first = alignments_[range->first].first;
    range->second = alignments_[range->second - 1].second;
  }
  return true;
}

void NormalizedString::RunNormalization(const std::string& mode) {
  icu::ErrorCode icu_error;
  const icu::Normalizer2* normalizer = nullptr;
  if (mode == "NFD") {
    normalizer = icu::Normalizer2::getNFDInstance(icu_error);
  } else if (mode == "NFKD") {
    normalizer = icu::Normalizer2::getNFKDInstance(icu_error);
  } else if (mode == "NFC") {
    normalizer = icu::Normalizer2::getNFCInstance(icu_error);
  } else if (mode == "NFKC") {
    normalizer = icu::Normalizer2::getNFKCInstance(icu_error);
  }
  std::string normalized_result;
  icu::Edits edits;
  icu::StringByteSink<std::string> byte_sink(&normalized_result);
  normalizer->normalizeUTF8(
      0,
      icu::StringPiece(normalized_.data(), normalized_.size()),
      byte_sink,
      &edits,
      icu_error);
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::u32string u32new_normalized = conv.from_bytes(normalized_result);
  // Set changes
  std::vector<int> changes;
  changes.reserve(u32new_normalized.length());
  auto iter = edits.getFineIterator();
  int old_offset = 0;
  int new_offset = 0;
  // The edits record the byte level modification, so need to transform to char
  // level
  // using GetUnicodeLenFromUTF8
  while (iter.next(icu_error)) {
    auto old_length = iter.oldLength();
    auto new_length = iter.newLength();
    auto new_unicode_len = utils::GetUnicodeLenFromUTF8(
        normalized_result.data() + new_offset, new_length);
    auto old_unicode_len = utils::GetUnicodeLenFromUTF8(
        normalized_.data() + old_offset, old_length);
    old_offset += old_length;
    new_offset += new_length;
    if (old_unicode_len == new_unicode_len) {
      // Just replace the char
      changes.insert(changes.end(), old_unicode_len, 0);
    } else if (old_unicode_len < new_unicode_len) {
      // Insert the char
      changes.insert(changes.end(), old_unicode_len, 0);
      changes.insert(changes.end(), new_unicode_len - old_unicode_len, 1);
    } else /* old_length > new_length */ {
      // Remove the char
      if (new_unicode_len > 1) {
        changes.insert(changes.end(), new_unicode_len - 1, 0);
      }
      changes.push_back(new_unicode_len - old_unicode_len);
    }
  }
  OffsetMapping new_normalized_offset{u32new_normalized, changes};
  // Update normalized_ and alignments_
  UpdateNormalized(new_normalized_offset, 0);
}

NormalizedString& NormalizedString::NFD() {
  RunNormalization("NFD");
  return *this;
}

NormalizedString& NormalizedString::NFKD() {
  RunNormalization("NFKD");
  return *this;
}

NormalizedString& NormalizedString::NFC() {
  RunNormalization("NFC");
  return *this;
}

NormalizedString& NormalizedString::NFKC() {
  RunNormalization("NFKC");
  return *this;
}

NormalizedString& NormalizedString::LStrip() { return LRStrip(true, false); }

NormalizedString& NormalizedString::RStrip() { return LRStrip(false, true); }

const std::string WHITESPACE = " \n\r\t\f\v";

NormalizedString& NormalizedString::LRStrip(bool left, bool right) {
  int leading_spaces = 0;
  int trailing_spaces = 0;
  std::string new_normalized = normalized_;
  if (left) {
    leading_spaces = new_normalized.find_first_not_of(WHITESPACE);
    if (leading_spaces != std::string::npos) {
      new_normalized = new_normalized.substr(leading_spaces);
    }
  }
  if (right) {
    trailing_spaces = new_normalized.find_last_not_of(WHITESPACE);
    if (trailing_spaces != std::string::npos) {
      new_normalized = new_normalized.substr(0, trailing_spaces + 1);
    }
  }

  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::u32string u32new_normalized = conv.from_bytes(new_normalized);
  // Set changes
  std::vector<int> changes(u32new_normalized.length(), 0);
  changes.back() = -trailing_spaces;

  OffsetMapping new_normalized_offset{u32new_normalized, changes};
  // Update normalized_ and alignments_
  UpdateNormalized(new_normalized_offset, leading_spaces);
  return *this;
}

NormalizedString& NormalizedString::FilterChar(
    std::function<bool(char32_t)> keep_char_fn) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::u32string u32new_normalized;
  u32new_normalized.reserve(normalized_.length());
  uint32_t removed_start = 0;
  uint32_t removed = 0;
  std::vector<int> changes;
  changes.reserve(normalized_.length());
  bool has_init_ch = false;
  uint32_t last_char;
  uint32_t curr_char;
  size_t utf8_len = 0;
  while (utf8_len < normalized_.length()) {
    auto chwidth =
        utils::UTF8ToUInt32(normalized_.data() + utf8_len, &curr_char);
    curr_char = utils::UTF8ToUnicode(curr_char);
    if (keep_char_fn(curr_char)) {
      if (has_init_ch) {
        u32new_normalized.push_back(last_char);
        changes.push_back(-removed);
      } else {
        has_init_ch = true;
        removed_start = removed;
      }
      last_char = curr_char;
      removed = 0;
    } else {
      removed += 1;
    }
    utf8_len += chwidth;
  }
  if (has_init_ch) {
    u32new_normalized.push_back(last_char);
    changes.push_back(-removed);
  }
  OffsetMapping new_normalized_offset{u32new_normalized, changes};
  // Update normalized_ and alignments_
  UpdateNormalized(new_normalized_offset, removed_start);
  return *this;
}

NormalizedString& NormalizedString::MapChar(
    std::function<char32_t(char32_t)> map_char_fn) {
  size_t utf8_len = 0;
  std::u32string u32normalized;
  uint32_t curr_char;
  u32normalized.reserve(normalized_.length());
  while (utf8_len < normalized_.length()) {
    auto chwidth =
        utils::UTF8ToUInt32(normalized_.data() + utf8_len, &curr_char);
    curr_char = utils::UTF8ToUnicode(curr_char);
    curr_char = map_char_fn(curr_char);
    u32normalized.push_back(curr_char);
    utf8_len += chwidth;
  }
  std::vector<int> changes(u32normalized.size(), 0);
  UpdateNormalized({u32normalized, changes}, 0);
  return *this;
}

NormalizedString& NormalizedString::Lowercase() {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::u32string u32normalized = conv.from_bytes(normalized_);
  // Can cover all single char covert cases
  for (int i = 0; i < u32normalized.length(); ++i) {
    u32normalized[i] = u_tolower(u32normalized[i]);
  }
  // No need to update normalized range
  normalized_ = conv.to_bytes(u32normalized);
  return *this;
}

NormalizedString& NormalizedString::Replace(const re2::RE2& pattern,
                                            const std::string& content) {
  re2::StringPiece result;
  size_t start = 0;
  size_t end = normalized_.length();
  int64_t offset = 0;

  std::u32string u32content;
  u32content.reserve(content.size());
  std::vector<int> changes;
  changes.reserve(content.size());

  size_t content_utf8_len = 0;
  while (content_utf8_len < content.length()) {
    uint32_t content_char;
    auto content_char_width =
        utils::UTF8ToUInt32(content.data() + content_utf8_len, &content_char);
    content_char = utils::UTF8ToUnicode(content_char);
    u32content.push_back(content_char);
    changes.push_back(1);
    content_utf8_len += content_char_width;
  }
  size_t new_len = content.length();

  OffsetMapping new_normalized{u32content, changes};

  while (pattern.Match(normalized_, start, end, RE2::UNANCHORED, &result, 1)) {
    size_t curr_start = result.data() - normalized_.data();
    size_t old_len = result.length();
    size_t curr_end = curr_start + old_len;
    size_t removed_chars =
        utils::GetUnicodeLenFromUTF8(normalized_.data() + curr_start, old_len);
    UpdateNormalizedRange(
        new_normalized, removed_chars, {curr_start, curr_end}, false);
    offset = new_len - old_len;
    // update start
    start = curr_end;
    if (offset >= 0) {
      start = curr_end + offset;
    } else {
      size_t uoffset = -offset;
      start = (curr_end >= uoffset) ? curr_end - uoffset : 0;
    }
    end = normalized_.length();
  }
  return *this;
}

NormalizedString& NormalizedString::Prepend(const std::string& content) {
  // Get the first unicode char of normalized
  uint32_t first_char_of_normalized;
  auto first_char_width =
      utils::UTF8ToUInt32(normalized_.data(), &first_char_of_normalized);
  first_char_of_normalized = utils::UTF8ToUnicode(first_char_of_normalized);

  std::u32string u32content;
  u32content.reserve(content.length());
  std::vector<int> changes;
  changes.reserve(content.length());
  uint32_t utf8_len = 0;
  while (utf8_len < content.length()) {
    uint32_t content_char;
    auto content_char_width =
        utils::UTF8ToUInt32(content.data() + utf8_len, &content_char);
    content_char = utils::UTF8ToUnicode(content_char);
    u32content.push_back(content_char);
    if (utf8_len == 0) {
      changes.push_back(0);
    } else {
      changes.push_back(1);
    }
    utf8_len += content_char_width;
  }
  u32content.push_back(first_char_of_normalized);
  changes.push_back(1);
  UpdateNormalizedRange({u32content, changes}, 0, {0, first_char_width}, false);
  return *this;
}

bool NormalizedString::ValidateRange(const core::Range& range,
                                     bool origin_range) const {
  if (origin_range) {
    return utils::IsCharBoundary(original_.data() + range.first) &&
           utils::IsCharBoundary(original_.data() + range.second - 1);
  }
  return utils::IsCharBoundary(normalized_.data() + range.first) &&
         utils::IsCharBoundary(normalized_.data() + range.second - 1);
}

bool NormalizedString::Slice(core::Range range,
                             NormalizedString* normalized,
                             bool origin_range) const {
  if (ValidateRange(range, origin_range)) {
    core::Range normalized_range = range;
    core::Range original_range = range;
    if (origin_range) {
      ConvertOffsets(&normalized_range, true);
    } else {
      ConvertOffsets(&original_range, false);
    }
    uint32_t n_shift = original_range.first;
    normalized->original_ = this->original_.substr(
        original_range.first, original_range.second - original_range.first);
    normalized->normalized_ = this->normalized_.substr(
        normalized_range.first,
        normalized_range.second - normalized_range.first);
    normalized->alignments_.reserve(normalized_range.second -
                                    normalized_range.first);
    for (uint32_t i = normalized_range.first; i < normalized_range.second;
         ++i) {
      normalized->alignments_.emplace_back(
          this->alignments_[i].first - n_shift,
          this->alignments_[i].second - n_shift);
    }

    normalized->original_shift_ = this->original_shift_ + original_range.first;
    return true;
  }
  return false;
}

uint32_t NormalizedString::GetMatch(
    const std::string& normalized,
    const re2::RE2& pattern,
    std::vector<std::pair<core::Range, bool>>* matches) const {
  size_t start = 0;
  size_t end = normalized.length();
  // Construct the matches whose mode is REMOVED.
  re2::StringPiece result;
  uint32_t reserved_num = 0;
  while (pattern.Match(normalized, start, end, RE2::UNANCHORED, &result, 1)) {
    size_t curr_start = result.data() - normalized.data();
    size_t curr_end = curr_start + result.length();
    if (start != curr_start) {
      matches->push_back({{start, curr_start}, false});
      ++reserved_num;
    }
    matches->push_back({{curr_start, curr_end}, true});
    start = curr_end;
  }
  if (start < end) {
    matches->push_back({{start, end}, false});
    ++reserved_num;
  }
  return reserved_num;
}

uint32_t NormalizedString::GetMatch(
    const std::string& normalized,
    const std::function<bool(char32_t)>& pattern_func,
    std::vector<std::pair<core::Range, bool>>* matches) const {
  size_t utf8_len = 0;
  size_t start = 0;
  size_t curr_start = 0;
  size_t curr_end = 0;
  matches->reserve(normalized.length());
  uint32_t ch;
  uint32_t reserved_num = 0;
  while (utf8_len < normalized.length()) {
    auto chwidth = utils::UTF8ToUInt32(normalized.data() + utf8_len, &ch);
    ch = utils::UTF8ToUnicode(ch);
    if (pattern_func(ch)) {
      curr_start = utf8_len;
      curr_end = curr_start + chwidth;
      if (curr_start != start) {
        matches->emplace_back(core::Range{start, curr_start}, false);
        ++reserved_num;
      }
      matches->emplace_back(core::Range{curr_start, curr_end}, true);
      start = curr_end;
    }
    utf8_len += chwidth;
  }
  if (start < normalized.length()) {
    matches->emplace_back(core::Range{start, normalized.length()}, false);
    ++reserved_num;
  }
  return reserved_num;
}

template void NormalizedString::Split(
    const re2::RE2& pattern,
    SplitMode mode,
    std::vector<NormalizedString>* normalizes) const;
template void NormalizedString::Split(
    const std::function<bool(char32_t)>& pattern_func,
    SplitMode mode,
    std::vector<NormalizedString>* normalizes) const;

}  // namespace normalizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
