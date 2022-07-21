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

#include <functional>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"
#include "nlohmann/json.hpp"

namespace std {
template <>
struct hash<std::pair<uint32_t, uint32_t>> {
  size_t operator()(const std::pair<uint32_t, uint32_t>& x) const {
    size_t h1 = hash<uint32_t>()(x.first);
    size_t h2 = hash<uint32_t>()(x.second);
    return h1 ^ (h2 << 1);
  }
};
}

namespace paddlenlp {
namespace faster_tokenizer {
namespace core {

enum OffsetType { CHAR, BYTE };
enum Direction { LEFT, RIGHT };
enum TruncStrategy { LONGEST_FIRST, ONLY_FIRST, ONLY_SECOND };
enum PadStrategy { BATCH_LONGEST, FIXED_SIZE };

NLOHMANN_JSON_SERIALIZE_ENUM(OffsetType,
                             {
                                 {CHAR, "CHAR"}, {BYTE, "BYTE"},
                             });

NLOHMANN_JSON_SERIALIZE_ENUM(Direction,
                             {
                                 {LEFT, "LEFT"}, {RIGHT, "RIGHT"},
                             });

NLOHMANN_JSON_SERIALIZE_ENUM(TruncStrategy,
                             {
                                 {LONGEST_FIRST, "LONGEST_FIRST"},
                                 {ONLY_FIRST, "ONLY_FIRST"},
                                 {ONLY_SECOND, "ONLY_SECOND"},
                             });


NLOHMANN_JSON_SERIALIZE_ENUM(PadStrategy,
                             {
                                 {BATCH_LONGEST, "BATCH_LONGEST"},
                                 {FIXED_SIZE, "FIXED_SIZE"},
                             });

struct TruncMethod {
  Direction direction_;
  size_t max_len_;
  TruncStrategy strategy_;
  size_t stride_;
  TruncMethod()
      : max_len_(512),
        stride_(0),
        strategy_(LONGEST_FIRST),
        direction_(RIGHT) {}
};

struct PadMethod {
  PadStrategy strategy_;
  Direction direction_;
  uint32_t pad_id_;
  uint32_t pad_token_type_id_;
  std::string pad_token_;
  uint32_t pad_len_;
  uint32_t pad_to_multiple_of_;

  PadMethod()
      : strategy_(BATCH_LONGEST),
        direction_(RIGHT),
        pad_id_(0),
        pad_token_type_id_(0),
        pad_token_("[PAD]"),
        pad_len_(0),
        pad_to_multiple_of_(0) {}
};

inline void to_json(nlohmann::json& j, const TruncMethod& trunc_method) {
  j = {
      {"strategy", trunc_method.strategy_},
      {"direction", trunc_method.direction_},
      {"max_len", trunc_method.max_len_},
      {"stride", trunc_method.stride_},
  };
}

inline void from_json(const nlohmann::json& j, TruncMethod& trunc_method) {
  j["strategy"].get_to(trunc_method.strategy_);
  j["direction"].get_to(trunc_method.direction_);
  j["max_len"].get_to(trunc_method.max_len_);
  j["stride"].get_to(trunc_method.stride_);
}


inline void to_json(nlohmann::json& j, const PadMethod& pad_method) {
  j = {
      {"strategy", pad_method.strategy_},
      {"direction", pad_method.direction_},
      {"pad_id", pad_method.pad_id_},
      {"pad_token_type_id", pad_method.pad_token_type_id_},
      {"pad_token", pad_method.pad_token_},
      {"pad_len", pad_method.pad_len_},
      {"pad_to_multiple_of", pad_method.pad_to_multiple_of_},
  };
}

inline void from_json(const nlohmann::json& j, PadMethod& pad_method) {
  j["strategy"].get_to(pad_method.strategy_);
  j["direction"].get_to(pad_method.direction_);
  j["pad_id"].get_to(pad_method.pad_id_);
  j["pad_token_type_id"].get_to(pad_method.pad_token_type_id_);
  j["pad_token"].get_to(pad_method.pad_token_);
  j["pad_len"].get_to(pad_method.pad_len_);
  j["pad_to_multiple_of"].get_to(pad_method.pad_to_multiple_of_);
}

using Offset = std::pair<uint32_t, uint32_t>;
using Range = std::pair<uint32_t, uint32_t>;
using Vocab = std::unordered_map<std::string, uint32_t>;
using VocabList = std::vector<std::pair<std::string, float>>;
using VocabReversed = std::unordered_map<uint32_t, std::string>;
using SortedVocabReversed = std::map<uint32_t, std::string>;
using Pair = std::pair<uint32_t, uint32_t>;
using MergeMap = std::unordered_map<Pair, std::pair<uint32_t, uint32_t>>;
using Merges = std::vector<std::pair<std::string, std::string>>;

inline void to_json(nlohmann::json& j,
                    const SortedVocabReversed& sorted_vocab_r) {
  j = nlohmann::ordered_json();
  for (const auto& item : sorted_vocab_r) {
    j[item.second] = item.first;
  }
}

struct Token {
  uint32_t id_;
  std::string value_;
  Offset offset_;
  Token() = default;
  Token(uint32_t id, const std::string& value, const Offset& offset)
      : id_(id), value_(value), offset_(offset) {}
};

struct Merge {
  size_t pos_;
  uint32_t rank_;
  uint32_t new_id_;

  bool operator==(const Merge& other) const {
    return pos_ == other.pos_ && rank_ == other.rank_;
  }
  bool operator<(const Merge& other) const {
    // Used in priority queue
    // The queue will output the Merge value
    // in ascending order of rank_
    if (rank_ != other.rank_) {
      return rank_ > other.rank_;
    }
    return pos_ > other.pos_;
  }
};

struct Symbol {
  uint32_t ch_;  // symbol id
  int prev_;
  int next_;
  size_t len_;

  Symbol() = default;
  Symbol(uint32_t ch, int prev, int next, size_t len)
      : ch_(ch), prev_(prev), next_(next), len_(len) {}
  // Merges the current Symbol with the other one.
  // In order to update prev/next, we consider Self to be the Symbol on the
  // left,
  // and other to be the next one on the right.
  void MergeWith(const Symbol& other, uint32_t ch) {
    ch_ = ch;
    next_ = other.next_;
    len_ += other.len_;
  }
};

struct BPEWord {
  BPEWord() = default;
  BPEWord(size_t capacity) { Reserve(capacity); }
  void Reserve(size_t capacity) { symbols_.reserve(capacity); }
  void Add(uint32_t ch, size_t byte_len) {
    int len = symbols_.size();
    int next = -1;
    int prev = -1;
    if (len >= 1) {
      symbols_.back().next_ = len;
      prev = len - 1;
    }
    symbols_.emplace_back(ch, prev, next, byte_len);
  }

  void Merge(uint32_t c1,
             uint32_t c2,
             uint32_t replacement,
             std::vector<std::pair<Pair, int>>* changes) {
    for (int i = 0; i < symbols_.size(); ++i) {
      // Found a byte pair
      if (symbols_[i].ch_ == c1 && i + 1 < symbols_.size() &&
          symbols_[i + 1].ch_ == c2) {
        auto& first = symbols_[i];
        auto& second = symbols_[i + 1];
        // If there are other characters before the pair
        if (i > 0) {
          changes->push_back({{symbols_[i - 1].ch_, first.ch_}, -1});
          changes->push_back({{symbols_[i - 1].ch_, replacement}, 1});
        }
        Symbol symbols{
            replacement, first.prev_, second.next_, first.len_ + second.len_};
        symbols_.insert(symbols_.begin() + i, symbols);
        symbols_.erase(symbols_.begin() + i + 1, symbols_.begin() + i + 3);
        if (i + 1 < symbols_.size()) {
          changes->push_back({{second.ch_, symbols_[i + 1].ch_}, -1});
          changes->push_back({{replacement, symbols_[i + 1].ch_}, 1});
        }
      }
    }
  }

  void MergeAll(const MergeMap& merges, const std::vector<float>& dropout) {
    std::priority_queue<core::Merge> queue;
    std::vector<core::Merge> skip;
    skip.reserve(symbols_.size());
    for (int i = 0; i < symbols_.size() - 1; ++i) {
      auto& first = symbols_[i];
      auto& second = symbols_[i + 1];
      if (merges.find({first.ch_, second.ch_}) != merges.end()) {
        auto new_merge_info = merges.at({first.ch_, second.ch_});
        core::Merge new_merge{static_cast<size_t>(i),
                              new_merge_info.first,
                              new_merge_info.second};
        queue.push(new_merge);
      }
    }
    std::random_device
        rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with
                             // rd()
    std::uniform_real_distribution<float> distrib(0.0, 1.0);
    bool can_skip = (dropout.size() > 0);
    while (!queue.empty()) {
      // Can't use reference there, because the pop operation will change the
      // top value
      auto top = queue.top();
      queue.pop();
      if (can_skip && distrib(gen) < dropout[0]) {
        // May dropout some merges
        skip.push_back(top);
      } else {
        for (auto& skip_merge : skip) {
          queue.push(skip_merge);
        }
        skip.clear();
        if (symbols_[top.pos_].len_ == 0) {
          continue;
        }
        if (symbols_[top.pos_].next_ == -1) {
          continue;
        }
        size_t next_pos = symbols_[top.pos_].next_;
        auto& right = symbols_[next_pos];
        // Make sure we are not processing an expired queue entry
        auto target_new_pair = Pair{symbols_[top.pos_].ch_, right.ch_};
        if (merges.find(target_new_pair) == merges.end() ||
            merges.at(target_new_pair).second != top.new_id_) {
          continue;
        }
        // Otherwise, let's merge
        symbols_[top.pos_].MergeWith(right, top.new_id_);
        // Tag the right part as removed
        symbols_[next_pos].len_ = 0;
        // Update `prev` on the new `next` to the current pos
        if (right.next_ > -1 && (right.next_ < symbols_.size())) {
          symbols_[right.next_].prev_ = top.pos_;
        }
        // Insert the new pair formed with the previous symbol
        auto& current = symbols_[top.pos_];
        if (current.prev_ >= 0) {
          auto prev = current.prev_;
          auto& prev_symbol = symbols_[prev];
          auto new_pair = Pair{prev_symbol.ch_, current.ch_};
          if (merges.find(new_pair) != merges.end()) {
            auto new_merge = merges.at(new_pair);
            queue.push({static_cast<size_t>(current.prev_),
                        new_merge.first,
                        new_merge.second});
          }
        }

        // Insert the new pair formed with the next symbol
        size_t next = current.next_;
        if (next < symbols_.size()) {
          auto& next_symbol = symbols_[next];
          auto next_pair = Pair{current.ch_, next_symbol.ch_};
          if (merges.find(next_pair) != merges.end()) {
            auto new_merge = merges.at(next_pair);
            queue.push({top.pos_, new_merge.first, new_merge.second});
          }
        }
      }
    }
    symbols_.erase(
        std::remove_if(symbols_.begin(),
                       symbols_.end(),
                       [](const Symbol& symbol) { return symbol.len_ == 0; }),
        symbols_.end());
  }

  void GetChars(std::vector<uint32_t>* result) const {
    result->reserve(symbols_.size());
    for (const auto& symbol : symbols_) {
      result->emplace_back(symbol.ch_);
    }
  }

  void GetOffset(std::vector<Offset>* result) const {
    result->reserve(symbols_.size());
    uint32_t pos = 0;
    for (const auto& symbol : symbols_) {
      result->emplace_back(pos, pos + symbol.len_);
      pos += symbol.len_;
    }
  }

  std::vector<Symbol> symbols_;
};

}  // namespace core
}  // namespace faster_tokenizer
}  // namespace paddlenlp
