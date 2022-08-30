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

#include "normalizers/utils.h"
#include "normalizers/bert.h"
#include "normalizers/precompiled.h"
#include "normalizers/replace.h"
#include "normalizers/strip.h"
#include "normalizers/unicode.h"
#include "unicode/unistr.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace normalizers {

void SequenceNormalizer::AppendNormalizer(Normalizer* normalizer) {
  std::shared_ptr<Normalizer> normalizer_ptr;
  if (typeid(*normalizer) == typeid(SequenceNormalizer)) {
    auto cast_normalizer = dynamic_cast<SequenceNormalizer*>(normalizer);
    normalizer_ptr = std::make_shared<SequenceNormalizer>(*cast_normalizer);
  } else if (typeid(*normalizer) == typeid(LowercaseNormalizer)) {
    auto cast_normalizer = dynamic_cast<LowercaseNormalizer*>(normalizer);
    normalizer_ptr = std::make_shared<LowercaseNormalizer>(*cast_normalizer);
  } else if (typeid(*normalizer) == typeid(StripNormalizer)) {
    auto cast_normalizer = dynamic_cast<StripNormalizer*>(normalizer);
    normalizer_ptr = std::make_shared<StripNormalizer>(*cast_normalizer);
  } else if (typeid(*normalizer) == typeid(StripAccentsNormalizer)) {
    auto cast_normalizer = dynamic_cast<StripAccentsNormalizer*>(normalizer);
    normalizer_ptr = std::make_shared<StripAccentsNormalizer>(*cast_normalizer);
  } else if (typeid(*normalizer) == typeid(NFCNormalizer)) {
    auto cast_normalizer = dynamic_cast<NFCNormalizer*>(normalizer);
    normalizer_ptr = std::make_shared<NFCNormalizer>(*cast_normalizer);
  } else if (typeid(*normalizer) == typeid(NFDNormalizer)) {
    auto cast_normalizer = dynamic_cast<NFDNormalizer*>(normalizer);
    normalizer_ptr = std::make_shared<NFDNormalizer>(*cast_normalizer);
  } else if (typeid(*normalizer) == typeid(NFKCNormalizer)) {
    auto cast_normalizer = dynamic_cast<NFKCNormalizer*>(normalizer);
    normalizer_ptr = std::make_shared<NFKCNormalizer>(*cast_normalizer);
  } else if (typeid(*normalizer) == typeid(NFKDNormalizer)) {
    auto cast_normalizer = dynamic_cast<NFKDNormalizer*>(normalizer);
    normalizer_ptr = std::make_shared<NFKDNormalizer>(*cast_normalizer);
  } else if (typeid(*normalizer) == typeid(NmtNormalizer)) {
    auto cast_normalizer = dynamic_cast<NmtNormalizer*>(normalizer);
    normalizer_ptr = std::make_shared<NmtNormalizer>(*cast_normalizer);
  } else if (typeid(*normalizer) == typeid(ReplaceNormalizer)) {
    auto cast_normalizer = dynamic_cast<ReplaceNormalizer*>(normalizer);
    normalizer_ptr = std::make_shared<ReplaceNormalizer>(*cast_normalizer);
  } else if (typeid(*normalizer) == typeid(BertNormalizer)) {
    auto cast_normalizer = dynamic_cast<BertNormalizer*>(normalizer);
    normalizer_ptr = std::make_shared<BertNormalizer>(*cast_normalizer);
  } else if (typeid(*normalizer) == typeid(PrecompiledNormalizer)) {
    auto cast_normalizer = dynamic_cast<PrecompiledNormalizer*>(normalizer);
    normalizer_ptr = std::make_shared<PrecompiledNormalizer>(*cast_normalizer);
  }
  normalizer_ptrs_.push_back(std::move(normalizer_ptr));
}

SequenceNormalizer::SequenceNormalizer(
    const std::vector<Normalizer*>& normalizers) {
  for (auto& normalizer : normalizers) {
    AppendNormalizer(normalizer);
  }
}

void SequenceNormalizer::operator()(NormalizedString* input) const {
  std::string result;
  for (auto& normalizer : normalizer_ptrs_) {
    normalizer->operator()(input);
  }
}
void LowercaseNormalizer::operator()(NormalizedString* input) const {
  input->Lowercase();
}

void to_json(nlohmann::json& j, const SequenceNormalizer& normalizer) {
  nlohmann::json jlist;
  for (auto& ptr : normalizer.normalizer_ptrs_) {
    nlohmann::json jitem;
    if (typeid(*ptr) == typeid(SequenceNormalizer)) {
      jitem = *dynamic_cast<SequenceNormalizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(LowercaseNormalizer)) {
      jitem = *dynamic_cast<LowercaseNormalizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(StripNormalizer)) {
      jitem = *dynamic_cast<StripNormalizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(StripAccentsNormalizer)) {
      jitem = *dynamic_cast<StripAccentsNormalizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(NFCNormalizer)) {
      jitem = *dynamic_cast<NFCNormalizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(NFDNormalizer)) {
      jitem = *dynamic_cast<NFDNormalizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(NFKCNormalizer)) {
      jitem = *dynamic_cast<NFKCNormalizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(NFKDNormalizer)) {
      jitem = *dynamic_cast<NFKDNormalizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(NmtNormalizer)) {
      jitem = *dynamic_cast<NmtNormalizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(ReplaceNormalizer)) {
      jitem = *dynamic_cast<ReplaceNormalizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(BertNormalizer)) {
      jitem = *dynamic_cast<BertNormalizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(PrecompiledNormalizer)) {
      jitem = *dynamic_cast<PrecompiledNormalizer*>(ptr.get());
    }
    jlist.push_back(jitem);
  }
  j = {{"type", "SequenceNormalizer"}, {"normalizers", jlist}};
}

void from_json(const nlohmann::json& j,
               SequenceNormalizer& sequence_normalizer) {
#define TRY_APPEND_NORMALIZER(NORMALIZER_TYPE)         \
  if (normalizer_type == #NORMALIZER_TYPE) {           \
    NORMALIZER_TYPE normalizer;                        \
    normalizer_json.get_to(normalizer);                \
    sequence_normalizer.AppendNormalizer(&normalizer); \
  }

  for (auto& normalizer_json : j.at("normalizers")) {
    std::string normalizer_type;
    normalizer_json.at("type").get_to(normalizer_type);
    TRY_APPEND_NORMALIZER(BertNormalizer);
    TRY_APPEND_NORMALIZER(PrecompiledNormalizer);
    TRY_APPEND_NORMALIZER(ReplaceNormalizer);
    TRY_APPEND_NORMALIZER(StripAccentsNormalizer);
    TRY_APPEND_NORMALIZER(StripNormalizer);
    TRY_APPEND_NORMALIZER(NFCNormalizer);
    TRY_APPEND_NORMALIZER(NFKCNormalizer);
    TRY_APPEND_NORMALIZER(NFDNormalizer);
    TRY_APPEND_NORMALIZER(NFKDNormalizer);
    TRY_APPEND_NORMALIZER(NmtNormalizer);
    TRY_APPEND_NORMALIZER(LowercaseNormalizer);
    TRY_APPEND_NORMALIZER(SequenceNormalizer);
  }
#undef TRY_APPEND_NORMALIZER
}

void to_json(nlohmann::json& j, const LowercaseNormalizer& normalizer) {
  j = {
      {"type", "LowercaseNormalizer"},
  };
}

void from_json(const nlohmann::json& j, LowercaseNormalizer& normalizer) {}

}  // namespace normalizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
