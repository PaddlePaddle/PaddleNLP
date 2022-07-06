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

#include "core/encoding.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <sstream>
#include "glog/logging.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace core {

Encoding::Encoding(const std::vector<uint32_t>& ids,
                   const std::vector<uint32_t>& type_ids,
                   const std::vector<std::string>& tokens,
                   const std::vector<uint32_t>& words_idx,
                   const std::vector<Offset>& offsets,
                   const std::vector<uint32_t>& special_tokens_mask,
                   const std::vector<uint32_t>& attention_mask,
                   const std::vector<Encoding>& overflowing,
                   const std::unordered_map<uint32_t, Range>& sequence_ranges)
    : ids_(ids),
      type_ids_(type_ids),
      tokens_(tokens),
      words_idx_(words_idx),
      offsets_(offsets),
      special_tokens_mask_(special_tokens_mask),
      attention_mask_(attention_mask),
      overflowing_(overflowing),
      sequence_ranges_(sequence_ranges) {}
// Move version
Encoding::Encoding(std::vector<uint32_t>&& ids,
                   std::vector<uint32_t>&& type_ids,
                   std::vector<std::string>&& tokens,
                   std::vector<uint32_t>&& words_idx,
                   std::vector<Offset>&& offsets,
                   std::vector<uint32_t>&& special_tokens_mask,
                   std::vector<uint32_t>&& attention_mask,
                   std::vector<Encoding>&& overflowing,
                   std::unordered_map<uint32_t, Range>&& sequence_ranges)
    : ids_(std::move(ids)),
      type_ids_(std::move(type_ids)),
      tokens_(std::move(tokens)),
      words_idx_(std::move(words_idx)),
      offsets_(std::move(offsets)),
      special_tokens_mask_(std::move(special_tokens_mask)),
      attention_mask_(std::move(attention_mask)),
      overflowing_(std::move(overflowing)),
      sequence_ranges_(std::move(sequence_ranges)) {}

Encoding::Encoding(uint32_t capacity) {
  ids_.reserve(capacity);
  type_ids_.reserve(capacity);
  tokens_.reserve(capacity);
  words_idx_.reserve(capacity);
  offsets_.reserve(capacity);
  special_tokens_mask_.reserve(capacity);
  attention_mask_.reserve(capacity);
}

Encoding::Encoding(const std::vector<Token>& tokens, uint32_t type_id)
    : type_ids_(tokens.size(), type_id),
      words_idx_(tokens.size()),
      attention_mask_(tokens.size(), 1),
      special_tokens_mask_(tokens.size(), 0) {
  auto length = tokens.size();
  ids_.reserve(length);
  offsets_.reserve(length);
  tokens_.reserve(length);
  for (const auto& token : tokens) {
    ids_.push_back(token.id_);
    tokens_.push_back(token.value_);
    offsets_.push_back(token.offset_);
  }
}

Encoding::Encoding(Encoding&& other)
    : ids_(std::move(other.ids_)),
      type_ids_(std::move(other.type_ids_)),
      tokens_(std::move(other.tokens_)),
      words_idx_(std::move(other.words_idx_)),
      offsets_(std::move(other.offsets_)),
      special_tokens_mask_(std::move(other.special_tokens_mask_)),
      attention_mask_(std::move(other.attention_mask_)),
      overflowing_(std::move(other.overflowing_)),
      sequence_ranges_(std::move(other.sequence_ranges_)) {}

Encoding& Encoding::operator=(Encoding&& other) {
  ids_ = std::move(other.ids_);
  type_ids_ = std::move(other.type_ids_);
  tokens_ = std::move(other.tokens_);
  words_idx_ = std::move(other.words_idx_);
  offsets_ = std::move(other.offsets_);
  special_tokens_mask_ = std::move(other.special_tokens_mask_);
  attention_mask_ = std::move(other.attention_mask_);
  overflowing_ = std::move(other.overflowing_);
  sequence_ranges_ = std::move(other.sequence_ranges_);
  return *this;
}

bool Encoding::IsEmpty() const { return ids_.empty(); }

int Encoding::GetLen() const { return ids_.size(); }

int Encoding::GetNumSequence() const {
  if (sequence_ranges_.empty()) {
    return 1;
  }
  return sequence_ranges_.size();
}

void Encoding::SetSequenceIds(uint32_t seq_ids) {
  sequence_ranges_[seq_ids] = {0, GetLen()};
}

const std::vector<std::string>& Encoding::GetTokens() const { return tokens_; }

const std::vector<uint32_t>& Encoding::GetWordsIdx() const {
  return words_idx_;
}

std::vector<uint32_t>& Encoding::GetMutableWordsIdx() { return words_idx_; }

std::vector<uint32_t> Encoding::GetSequenceIds() const {
  std::vector<uint32_t> sequences(GetLen());
  for (uint32_t seq_id = 0; seq_id < GetNumSequence(); ++seq_id) {
    Range range = sequence_ranges_.at(seq_id);
    auto seq_len = range.second - range.first;
    for (int i = range.first; i < range.second; ++i) {
      sequences[i] = seq_id;
    }
  }
  return sequences;
}

const std::vector<uint32_t>& Encoding::GetIds() const { return ids_; }

const std::vector<uint32_t>& Encoding::GetTypeIds() const { return type_ids_; }

const std::vector<Offset>& Encoding::GetOffsets() const { return offsets_; }

std::vector<Offset>& Encoding::GetMutableOffsets() { return offsets_; }

const std::vector<uint32_t>& Encoding::GetSpecialTokensMask() const {
  return special_tokens_mask_;
}

const std::vector<uint32_t>& Encoding::GetAttentionMask() const {
  return attention_mask_;
}

const std::vector<Encoding>& Encoding::GetOverflowing() const {
  return overflowing_;
}

std::vector<Encoding>& Encoding::GetMutableOverflowing() {
  return overflowing_;
}

Range Encoding::GetSequenceRange(uint32_t seq_id) const {
  return sequence_ranges_.at(seq_id);
}

void Encoding::ProcessTokenWithOffsets(
    std::function<void(uint32_t, std::string*, Offset*)> process_token_fn) {
  auto length = GetLen();
  for (int i = 0; i < length; ++i) {
    process_token_fn(i, &tokens_[i], &offsets_[i]);
  }
}

std::vector<uint32_t> Encoding::TokenIdxToSequenceIds(
    uint32_t token_idx) const {
  std::vector<uint32_t> seq_ids;
  if (token_idx < GetLen()) {
    if (sequence_ranges_.empty()) {
      seq_ids.push_back(0);
    } else {
      for (auto iter = sequence_ranges_.begin(); iter != sequence_ranges_.end();
           ++iter) {
        if (token_idx >= iter->second.first &&
            token_idx < iter->second.second) {
          seq_ids.push_back(iter->first);
          break;
        }
      }
    }
  }
  return seq_ids;
}

std::vector<Range> Encoding::WordIdxToTokensIdx(uint32_t word_idx,
                                                uint32_t seq_id) const {
  auto seq_range = sequence_ranges_.at(seq_id);
  std::vector<Range> ranges;
  int start = -1;
  int end = -1;
  for (uint32_t i = seq_range.first; i < seq_range.second; ++i) {
    // -1 is the word index of special token
    if (words_idx_[i] > word_idx &&
        words_idx_[i] != static_cast<uint32_t>(-1)) {
      break;
    }
    if (words_idx_[i] == word_idx) {
      if (start < 0 || i < start) {
        start = i;
      }
      if (end < 0 || i >= end) {
        end = i + 1;
      }
    }
  }
  if (start >= 0 && end >= 0) {
    seq_range.first += start;
    seq_range.second += end;
    ranges.push_back(seq_range);
  }
  return ranges;
}

std::vector<Offset> Encoding::WordIdxToCharOffsets(uint32_t word_idx,
                                                   uint32_t seq_id) const {
  std::vector<Offset> offsets;
  std::vector<Range> ranges = WordIdxToTokensIdx(word_idx, seq_id);
  if (ranges.size() > 0) {
    auto start = ranges[0].first;
    auto end = ranges[0].second;
    if (end > 0) {
      offsets.push_back({offsets_[start].first, offsets_[end - 1].second});
    }
  }
  return offsets;
}

std::vector<std::pair<uint32_t, Offset>> Encoding::TokenIdxToCharOffsets(
    uint32_t token_idx) const {
  std::vector<std::pair<uint32_t, Offset>> results;
  auto seq_ids = TokenIdxToSequenceIds(token_idx);
  if (seq_ids.size() > 0) {
    results.push_back({seq_ids[0], offsets_[token_idx]});
  }
  return results;
}

std::vector<std::pair<uint32_t, uint32_t>> Encoding::TokenIdxToWordIdx(
    uint32_t token_idx) const {
  std::vector<std::pair<uint32_t, uint32_t>> results;
  auto seq_ids = TokenIdxToSequenceIds(token_idx);
  if (seq_ids.size() > 0) {
    results.push_back({seq_ids[0], words_idx_[token_idx]});
  }
  return results;
}

std::vector<uint32_t> Encoding::CharOffsetsToTokenIdx(uint32_t char_pos,
                                                      uint32_t seq_id) const {
  std::vector<uint32_t> token_idx;
  auto seq_range = sequence_ranges_.at(seq_id);
  for (int i = seq_range.first; i < seq_range.second; ++i) {
    if (char_pos >= offsets_[i].first && char_pos < offsets_[i].second) {
      token_idx.push_back(i);
      break;
    }
  }
  return token_idx;
}

std::vector<uint32_t> Encoding::CharOffsetsToWordIdx(uint32_t char_pos,
                                                     uint32_t seq_id) const {
  std::vector<uint32_t> token_idx = CharOffsetsToTokenIdx(char_pos, seq_id);
  std::vector<uint32_t> word_idx;
  if (token_idx.size() > 0) {
    auto words_idx = TokenIdxToWordIdx(token_idx[0]);
    if (words_idx.size() > 0) {
      word_idx.push_back(words_idx[0].second);
    }
  }
  return word_idx;
}

void Encoding::Truncate(size_t max_len, size_t stride, Direction direction) {
  size_t encoding_len = ids_.size();
  if (max_len < encoding_len) {
    if (max_len == 0) {
      *this = Encoding(0);
      overflowing_.push_back(*this);
      return;
    }
    assert(stride < max_len);
    sequence_ranges_.clear();

    size_t step_len = max_len - stride;
    bool found_end = false;
    std::vector<Range> part_ranges;
    // Get PartRanges
    if (direction == RIGHT) {
      for (size_t start = 0; start < encoding_len && !found_end;
           start += step_len) {
        size_t stop = std::min(start + max_len, encoding_len);
        found_end = (stop == encoding_len);
        part_ranges.push_back({start, stop});
      }
    } else {
      for (size_t i = 0; i < encoding_len; i += step_len) {
        size_t stop = encoding_len - i;
        size_t start = (stop < max_len) ? 0 : stop - max_len;
        if (start < stop && !found_end) {
          found_end = (start == 0);
          part_ranges.push_back({start, stop});
        } else {
          break;
        }
      }
    }
    // Create new encoding
    auto new_encoding_len = part_ranges[0].second - part_ranges[0].first;
    Encoding new_encoding(
        std::vector<uint32_t>(ids_.begin(), ids_.begin() + new_encoding_len),
        std::vector<uint32_t>(type_ids_.begin(),
                              type_ids_.begin() + new_encoding_len),
        std::vector<std::string>(tokens_.begin(),
                                 tokens_.begin() + new_encoding_len),
        std::vector<uint32_t>(words_idx_.begin(),
                              words_idx_.begin() + new_encoding_len),
        std::vector<Offset>(offsets_.begin(),
                            offsets_.begin() + new_encoding_len),
        std::vector<uint32_t>(special_tokens_mask_.begin(),
                              special_tokens_mask_.begin() + new_encoding_len),
        std::vector<uint32_t>(attention_mask_.begin(),
                              attention_mask_.begin() + new_encoding_len),
        std::vector<Encoding>(),
        std::unordered_map<uint32_t, Range>());
    // Set overflowing
    for (size_t i = 1; i < part_ranges.size() - 1; ++i) {
      auto start = part_ranges[i].first;
      auto end = part_ranges[i].second;
      new_encoding.overflowing_.emplace_back(Encoding(
          std::vector<uint32_t>(ids_.begin() + start, ids_.begin() + end),
          std::vector<uint32_t>(type_ids_.begin() + start,
                                type_ids_.begin() + end),
          std::vector<std::string>(tokens_.begin() + start,
                                   tokens_.begin() + end),
          std::vector<uint32_t>(words_idx_.begin() + start,
                                words_idx_.begin() + end),
          std::vector<Offset>(offsets_.begin() + start, offsets_.begin() + end),
          std::vector<uint32_t>(special_tokens_mask_.begin() + start,
                                special_tokens_mask_.begin() + end),
          std::vector<uint32_t>(attention_mask_.begin() + start,
                                attention_mask_.begin() + end),
          std::vector<Encoding>(),
          std::unordered_map<uint32_t, Range>()));
    }
    *this = std::move(new_encoding);
  }
}


void Encoding::MergeWith(const Encoding& pair, bool growing_offsets) {
  std::vector<Encoding> overflowings;

  for (const auto& this_o : overflowing_) {
    auto n_encoding = this_o;
    n_encoding.MergeWith(pair, growing_offsets);
    overflowings.emplace_back(n_encoding);
    for (const auto& pair_o : pair.overflowing_) {
      auto n_encoding = this_o;
      n_encoding.MergeWith(pair_o, growing_offsets);
      overflowings.emplace_back(n_encoding);
    }
  }
  for (const auto& pair_o : pair.overflowing_) {
    auto n_encoding = *this;
    n_encoding.MergeWith(pair_o, growing_offsets);
    overflowings.emplace_back(n_encoding);
  }

  auto orignal_len = GetLen();
  for (const auto& pair_seq_range : pair.sequence_ranges_) {
    sequence_ranges_.insert({pair_seq_range.first,
                             {pair_seq_range.second.first + orignal_len,
                              pair_seq_range.second.second + orignal_len}});
  }
#define EXTEND_VECTOR(member) \
  member.insert(member.end(), pair.member.begin(), pair.member.end())
  EXTEND_VECTOR(ids_);
  EXTEND_VECTOR(type_ids_);
  EXTEND_VECTOR(tokens_);
  EXTEND_VECTOR(words_idx_);
  EXTEND_VECTOR(special_tokens_mask_);
  EXTEND_VECTOR(attention_mask_);
#undef EXTEND_VECTOR
  // Setting offet
  uint32_t starting_offset = 0;
  if (growing_offsets && offsets_.size() > 0) {
    starting_offset = offsets_.back().second;
  }
  for (const auto& pair_offset : pair.offsets_) {
    offsets_.push_back({pair_offset.first + starting_offset,
                        pair_offset.second + starting_offset});
  }

  overflowing_ = std::move(overflowings);
}

void Encoding::Pad(uint32_t target_length,
                   uint32_t pad_id,
                   uint32_t pad_type_id,
                   const std::string& pad_token,
                   Direction direction) {
  for (auto& overflowing : overflowing_) {
    overflowing.Pad(target_length, pad_id, pad_type_id, pad_token, direction);
  }
  // Need to be padded in this situation
  if (GetLen() < target_length) {
    auto pad_len = target_length - GetLen();
    if (direction == LEFT) {
      ids_.insert(ids_.begin(), pad_len, pad_id);
      type_ids_.insert(type_ids_.begin(), pad_len, pad_type_id);
      tokens_.insert(tokens_.begin(), pad_len, pad_token);
      words_idx_.insert(words_idx_.begin(), pad_len, UINT_MAX);
      attention_mask_.insert(attention_mask_.begin(), pad_len, 0);
      special_tokens_mask_.insert(special_tokens_mask_.begin(), pad_len, 1);
      offsets_.insert(offsets_.begin(), pad_len, {0, 0});
    } else {
      ids_.insert(ids_.end(), pad_len, pad_id);
      type_ids_.insert(type_ids_.end(), pad_len, pad_type_id);
      tokens_.insert(tokens_.end(), pad_len, pad_token);
      words_idx_.insert(words_idx_.end(), pad_len, UINT_MAX);
      attention_mask_.insert(attention_mask_.end(), pad_len, 0);
      special_tokens_mask_.insert(special_tokens_mask_.end(), pad_len, 1);
      offsets_.insert(offsets_.end(), pad_len, {0, 0});
    }
  }
}

// Static method
Encoding Encoding::Merge(const std::vector<Encoding>& encodings,
                         bool growing_offsets) {
  Encoding merged_encoding;
  for (auto& encoding : encodings) {
    merged_encoding.MergeWith(encoding, growing_offsets);
  }
  return merged_encoding;
}

std::string Encoding::DebugString() const {
  std::ostringstream oss;
  oss << "The Encoding content: \n";
  oss << "ids: ";
  for (int i = 0; i < ids_.size(); ++i) {
    oss << ids_[i];
    if (i < ids_.size() - 1) {
      oss << ", ";
    }
  }
  oss << "\n";

  oss << "type_ids: ";
  for (int i = 0; i < type_ids_.size(); ++i) {
    oss << type_ids_[i];
    if (i < type_ids_.size() - 1) {
      oss << ", ";
    }
  }
  oss << "\n";

  oss << "tokens: ";
  for (int i = 0; i < tokens_.size(); ++i) {
    oss << tokens_[i];
    if (i < tokens_.size() - 1) {
      oss << ", ";
    }
  }
  oss << "\n";

  oss << "words_idx: ";
  for (int i = 0; i < words_idx_.size(); ++i) {
    if (words_idx_[i] == static_cast<uint32_t>(-1)) {
      // The [CLS], [SEP] word id
      oss << "-";
    } else {
      oss << words_idx_[i];
    }
    if (i < words_idx_.size() - 1) {
      oss << ", ";
    }
  }
  oss << "\n";

  oss << "offsets: ";
  for (int i = 0; i < offsets_.size(); ++i) {
    oss << "(" << offsets_[i].first << ", " << offsets_[i].second << ")";
    if (i < offsets_.size() - 1) {
      oss << ", ";
    }
  }
  oss << "\n";

  oss << "special_tokens_mask: ";
  for (int i = 0; i < special_tokens_mask_.size(); ++i) {
    oss << special_tokens_mask_[i];
    if (i < special_tokens_mask_.size() - 1) {
      oss << ", ";
    }
  }
  oss << "\n";

  oss << "attention_mask: ";
  for (int i = 0; i < attention_mask_.size(); ++i) {
    oss << attention_mask_[i];
    if (i < attention_mask_.size() - 1) {
      oss << ", ";
    }
  }
  oss << "\n";

  oss << "sequence_ranges: ";
  for (auto iter = sequence_ranges_.begin(); iter != sequence_ranges_.end();
       ++iter) {
    oss << "{" << iter->first << " : (" << iter->second.first << ", "
        << iter->second.second << ") }, ";
  }
  oss << "\n";
  return oss.str();
}


bool TruncateEncodings(Encoding* encoding,
                       Encoding* pair_encoding,
                       const TruncMethod& method) {
  if (method.max_len_ == 0) {
    encoding->Truncate(0, method.stride_, method.direction_);
    if (pair_encoding != nullptr) {
      pair_encoding->Truncate(0, method.stride_, method.direction_);
    }
    return true;
  }
  size_t total_length = encoding->GetIds().size();
  if (pair_encoding != nullptr) {
    total_length += pair_encoding->GetIds().size();
  }
  if (total_length <= method.max_len_) {
    return true;
  }
  auto num_of_removed_ids = total_length - method.max_len_;

  if (method.strategy_ == TruncStrategy::LONGEST_FIRST) {
    if (pair_encoding == nullptr) {
      encoding->Truncate(method.max_len_, method.stride_, method.direction_);
    } else {
      auto encoding_len = encoding->GetIds().size();
      auto pair_encoding_len = pair_encoding->GetIds().size();
      bool has_swapped = false;
      if (encoding_len > pair_encoding_len) {
        std::swap(encoding_len, pair_encoding_len);
        has_swapped = true;
      }
      if (encoding_len > method.max_len_) {
        pair_encoding_len = encoding_len;
      } else {
        pair_encoding_len =
            std::max(method.max_len_ - encoding_len, encoding_len);
      }
      if (pair_encoding_len + encoding_len > method.max_len_) {
        // In this case, make sure the encoding_len is larger than
        // pair_encoding_len
        encoding_len = method.max_len_ / 2;
        pair_encoding_len = encoding_len + method.max_len_ % 2;
      }
      if (has_swapped) {
        std::swap(encoding_len, pair_encoding_len);
      }
      encoding->Truncate(encoding_len, method.stride_, method.direction_);
      pair_encoding->Truncate(
          pair_encoding_len, method.stride_, method.direction_);
    }
  } else {
    // TruncStrategy::ONLY_FIRST or TruncStrategy::ONLY_SECOND
    Encoding* result = nullptr;
    if (method.strategy_ == TruncStrategy::ONLY_FIRST) {
      result = encoding;
    } else if (method.strategy_ == TruncStrategy::ONLY_SECOND) {
      if (pair_encoding == nullptr) {
        // Can't truncate the pair text when it doesn't exist
        return false;
      }
      result = pair_encoding;
    }
    if (result->GetIds().size() > num_of_removed_ids) {
      result->Truncate(result->GetIds().size() - num_of_removed_ids,
                       method.stride_,
                       method.direction_);
    } else {
      // Target sequence is too short to be truncated.
      return false;
    }
  }
  return true;
}

void PadEncodings(std::vector<Encoding>* encodings, const PadMethod& method) {
  if (encodings == nullptr || encodings->empty()) {
    return;
  }
  size_t pad_length = 0;
  if (method.strategy_ == PadStrategy::BATCH_LONGEST) {
    for (const auto& encoding : *encodings) {
      pad_length = std::max(encoding.GetIds().size(), pad_length);
    }
  } else {
    pad_length = method.pad_len_;
  }
  if (method.pad_to_multiple_of_ > 0 &&
      pad_length % method.pad_to_multiple_of_) {
    pad_length += pad_length - pad_length % method.pad_to_multiple_of_;
  }
  for (auto& encoding : *encodings) {
    encoding.Pad(pad_length,
                 method.pad_id_,
                 method.pad_token_type_id_,
                 method.pad_token_,
                 method.direction_);
  }
}

}  // namespace core
}  // namespace faster_tokenizer
}  // namespace paddlenlp
