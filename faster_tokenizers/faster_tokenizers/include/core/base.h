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

namespace tokenizers {
namespace core {

enum OffsetType { CHAR, BYTE };
enum Direction { LEFT, RIGHT };
enum TruncStrategy { LONGEST_FIRST, ONLY_FIRST, ONLY_SECOND };
enum PadStrategy { BATCH_LONGEST, FIXED_SIZE };

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
