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
#include "faster_tokenizer/core/base.h"
#include "faster_tokenizer/core/encoding.h"
#include "faster_tokenizer/normalizers/normalizer.h"
#include "faster_tokenizer/utils/utils.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace pretokenizers {

struct FASTERTOKENIZER_DECL StringSplit {
  normalizers::NormalizedString normalized_;
  std::vector<core::Token> tokens_;
  StringSplit(normalizers::NormalizedString&& normalized)
      : normalized_(std::move(normalized)) {}
  StringSplit(const normalizers::NormalizedString& normalized)
      : normalized_(normalized) {}
  StringSplit(const normalizers::NormalizedString& normalized,
              const std::vector<core::Token>& tokens)
      : normalized_(normalized), tokens_(tokens) {}
  StringSplit() = default;
  StringSplit(const StringSplit& other) = default;
  StringSplit(StringSplit&& other)
      : tokens_(std::move(other.tokens_)),
        normalized_(std::move(other.normalized_)) {}

  StringSplit& operator=(const StringSplit& other) = default;
  StringSplit& operator=(StringSplit&& other) {
    tokens_ = std::move(other.tokens_);
    normalized_ = std::move(other.normalized_);
    return *this;
  }
};

class FASTERTOKENIZER_DECL PreTokenizedString {
public:
  PreTokenizedString() = default;
  PreTokenizedString(const std::string& original);
  PreTokenizedString(const normalizers::NormalizedString& normalized);
  PreTokenizedString& operator=(PreTokenizedString&& other);

  void Split(std::function<void(int,
                                normalizers::NormalizedString*,
                                std::vector<StringSplit>*)> split_fn);
  void Normalize(
      std::function<void(normalizers::NormalizedString*)> normalize_fn);
  // For wordpiece, bpe ......
  void Tokenize(std::function<std::vector<core::Token>(
                    normalizers::NormalizedString*)> tokenize_fn);
  bool TransformToEncoding(const std::vector<uint32_t>& word_idx,
                           uint32_t type_id,
                           core::OffsetType offset_type,
                           core::Encoding* encodings) const;
  template <typename Convertor>
  bool TransformToEncodingUseConvertor(const std::vector<uint32_t>& word_idx,
                                       uint32_t type_id,
                                       core::Encoding* encodings) const;
  size_t GetSplitsSize() const;
  StringSplit GetSplit(int idx) const;
  const std::string& GetOriginStr() const;
  void SetOriginalStr(const std::string& original);

private:
  std::string original_;
  std::vector<StringSplit> splits_;
};

struct FASTERTOKENIZER_DECL PreTokenizer {
  virtual void operator()(PreTokenizedString* pretokenized) const = 0;
};

struct FASTERTOKENIZER_DECL OffsetConverter {
  OffsetConverter(const std::string&) {}
  virtual bool convert(const core::Offset&, core::Offset*) const {
    return true;
  }
};

struct FASTERTOKENIZER_DECL BytesToCharOffsetConverter
    : public OffsetConverter {
  std::vector<size_t> offset_map_;
  BytesToCharOffsetConverter(const std::string&);
  virtual bool convert(const core::Offset&, core::Offset*) const;
};

struct FASTERTOKENIZER_DECL CharToBytesOffsetConverter
    : public OffsetConverter {
  std::vector<size_t> offset_map_;
  CharToBytesOffsetConverter(const std::string&);
  virtual bool convert(const core::Offset&, core::Offset*) const;
};

}  // namespace pretokenizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
