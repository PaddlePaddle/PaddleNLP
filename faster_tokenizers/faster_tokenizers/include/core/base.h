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
#include <string>
#include <unordered_map>
#include <vector>
#include "nlohmann/json.hpp"

namespace tokenizers {
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
  uint pad_id_;
  uint pad_token_type_id_;
  std::string pad_token_;
  uint pad_len_;
  uint pad_to_mutiple_of;

  PadMethod()
      : strategy_(BATCH_LONGEST),
        direction_(RIGHT),
        pad_id_(0),
        pad_token_type_id_(0),
        pad_token_("[PAD]"),
        pad_len_(0),
        pad_to_mutiple_of(0) {}
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
      {"pad_to_mutiple_of", pad_method.pad_to_mutiple_of},
  };
}

inline void from_json(const nlohmann::json& j, PadMethod& pad_method) {
  j["strategy"].get_to(pad_method.strategy_);
  j["direction"].get_to(pad_method.direction_);
  j["pad_id"].get_to(pad_method.pad_id_);
  j["pad_token_type_id"].get_to(pad_method.pad_token_type_id_);
  j["pad_token"].get_to(pad_method.pad_token_);
  j["pad_len"].get_to(pad_method.pad_len_);
  j["pad_to_mutiple_of"].get_to(pad_method.pad_to_mutiple_of);
}

using Offset = std::pair<uint, uint>;
using Range = std::pair<uint, uint>;
using Vocab = std::unordered_map<std::string, uint>;
using VocabReversed = std::unordered_map<uint, std::string>;

struct Token {
  uint id;
  std::string value;
  Offset offset;
  Token() = default;
  Token(uint id, const std::string& value, const Offset& offset)
      : id(id), value(value), offset(offset) {}
};

}  // core
}  // tokenizers
