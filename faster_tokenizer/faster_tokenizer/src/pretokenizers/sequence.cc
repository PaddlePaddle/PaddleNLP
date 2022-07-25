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

#include "pretokenizers/pretokenizers.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace pretokenizers {

SequencePreTokenizer::SequencePreTokenizer(
    const std::vector<PreTokenizer*>& pretokenizers) {
  for (auto& pretokenizer : pretokenizers) {
    AppendPreTokenizer(pretokenizer);
  }
}

void SequencePreTokenizer::AppendPreTokenizer(PreTokenizer* pretokenizer) {
  std::shared_ptr<PreTokenizer> pretokenizer_ptr;
  if (typeid(*pretokenizer) == typeid(SequencePreTokenizer)) {
    auto cast_pretokenizer = dynamic_cast<SequencePreTokenizer*>(pretokenizer);
    pretokenizer_ptr =
        std::make_shared<SequencePreTokenizer>(*cast_pretokenizer);
  } else if (typeid(*pretokenizer) == typeid(BertPreTokenizer)) {
    auto cast_pretokenizer = dynamic_cast<BertPreTokenizer*>(pretokenizer);
    pretokenizer_ptr = std::make_shared<BertPreTokenizer>(*cast_pretokenizer);
  } else if (typeid(*pretokenizer) == typeid(MetaSpacePreTokenizer)) {
    auto cast_pretokenizer = dynamic_cast<MetaSpacePreTokenizer*>(pretokenizer);
    pretokenizer_ptr =
        std::make_shared<MetaSpacePreTokenizer>(*cast_pretokenizer);
  } else if (typeid(*pretokenizer) == typeid(WhitespacePreTokenizer)) {
    auto cast_pretokenizer =
        dynamic_cast<WhitespacePreTokenizer*>(pretokenizer);
    pretokenizer_ptr =
        std::make_shared<WhitespacePreTokenizer>(*cast_pretokenizer);
  }
  pretokenzer_ptrs_.push_back(pretokenizer_ptr);
}

void SequencePreTokenizer::operator()(PreTokenizedString* pretokenized) const {
  for (auto& pretokenizer : pretokenzer_ptrs_) {
    pretokenizer->operator()(pretokenized);
  }
}

void to_json(nlohmann::json& j,
             const SequencePreTokenizer& sequence_pretokenizer) {
  nlohmann::json jlist;
  for (auto& ptr : sequence_pretokenizer.pretokenzer_ptrs_) {
    nlohmann::json jitem;
    if (typeid(*ptr) == typeid(SequencePreTokenizer)) {
      jitem = *dynamic_cast<SequencePreTokenizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(BertPreTokenizer)) {
      jitem = *dynamic_cast<BertPreTokenizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(MetaSpacePreTokenizer)) {
      jitem = *dynamic_cast<MetaSpacePreTokenizer*>(ptr.get());
    } else if (typeid(*ptr) == typeid(WhitespacePreTokenizer)) {
      jitem = *dynamic_cast<WhitespacePreTokenizer*>(ptr.get());
    }
    jlist.push_back(jitem);
  }
  j = {{"type", "SequencePreTokenizer"}, {"pretokenizers", jlist}};
}

void from_json(const nlohmann::json& j,
               SequencePreTokenizer& sequence_pretokenizer) {
#define TRY_APPEND_PRETOKENIZER(PRETOKENIZER_TYPE)           \
  if (pretokenizer_type == #PRETOKENIZER_TYPE) {             \
    PRETOKENIZER_TYPE pretokenizer;                          \
    pretokenizer_json.get_to(pretokenizer);                  \
    sequence_pretokenizer.AppendPreTokenizer(&pretokenizer); \
  }
  for (auto& pretokenizer_json : j.at("pretokenizers")) {
    std::string pretokenizer_type;
    pretokenizer_json.at("type").get_to(pretokenizer_type);
    TRY_APPEND_PRETOKENIZER(SequencePreTokenizer);
    TRY_APPEND_PRETOKENIZER(WhitespacePreTokenizer);
    TRY_APPEND_PRETOKENIZER(MetaSpacePreTokenizer);
    TRY_APPEND_PRETOKENIZER(BertPreTokenizer);
  }
#undef TRY_APPEND_PRETOKENIZER
}

}  // namespace pretokenizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
