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

#include <algorithm>

#include "core/encoding.h"
#include "glog/logging.h"
#include "postprocessors/bert.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace postprocessors {

BertPostProcessor::BertPostProcessor()
    : sep_({"[SEP]", 102}), cls_({"[CLS]", 101}) {}
BertPostProcessor::BertPostProcessor(
    const std::pair<std::string, uint32_t>& sep,
    const std::pair<std::string, uint32_t>& cls)
    : sep_(sep), cls_(cls) {}
size_t BertPostProcessor::AddedTokensNum(bool is_pair) const {
  if (is_pair) {
    // [CLS] A [SEP] B [SEP]
    return 3;
  }
  // [CLS] A [SEP]
  return 2;
}

void BertPostProcessor::operator()(core::Encoding* encoding,
                                   core::Encoding* pair_encoding,
                                   bool add_special_tokens,
                                   core::Encoding* result_encoding) const {
  if (!add_special_tokens) {
    DefaultProcess(encoding, pair_encoding, result_encoding);
    return;
  }
// Construct the sequence as: [CLS] A [SEP]
#define CREATE_PROCESSED_ENCODING_SEQ(                                         \
    encoding_ptr, attr, name, head_value, back_value)                          \
  auto encoding_##name = encoding_ptr->Get##attr();                            \
  decltype(encoding_##name) name(encoding_##name.size() + 2);                  \
  std::copy(encoding_##name.begin(), encoding_##name.end(), name.begin() + 1); \
  name.front() = head_value;                                                   \
  name.back() = back_value
  // ids
  CREATE_PROCESSED_ENCODING_SEQ(encoding, Ids, ids, cls_.second, sep_.second);
  // type_ids
  CREATE_PROCESSED_ENCODING_SEQ(encoding, TypeIds, type_ids, 0, 0);
  // tokens
  CREATE_PROCESSED_ENCODING_SEQ(
      encoding, Tokens, tokens, cls_.first, sep_.first);
  // word_idx
  CREATE_PROCESSED_ENCODING_SEQ(encoding, WordsIdx, word_idx, -1, -1);
  // offsets
  core::Offset empty_offsets = {0, 0};
  CREATE_PROCESSED_ENCODING_SEQ(
      encoding, Offsets, offsets, empty_offsets, empty_offsets);
  // special_tokens_mask
  std::vector<uint32_t> special_tokens_mask(ids.size(), 0);
  special_tokens_mask.front() = special_tokens_mask.back() = 1;
  // attention_mask
  std::vector<uint32_t> attention_mask(ids.size(), 1);
  // sequence_ranges
  std::unordered_map<uint32_t, core::Range> sequence_ranges;
  sequence_ranges[0] = {1, ids.size() - 1};
  // overflowing
  auto& overflowings = encoding->GetMutableOverflowing();
  for (auto& overflow_encoding : overflowings) {
    CREATE_PROCESSED_ENCODING_SEQ(
        (&overflow_encoding), Ids, ids, cls_.second, sep_.second);
    CREATE_PROCESSED_ENCODING_SEQ(
        (&overflow_encoding), TypeIds, type_ids, 0, 0);
    CREATE_PROCESSED_ENCODING_SEQ(
        (&overflow_encoding), Tokens, tokens, cls_.first, sep_.first);
    CREATE_PROCESSED_ENCODING_SEQ(
        (&overflow_encoding), WordsIdx, word_idx, -1, -1);
    CREATE_PROCESSED_ENCODING_SEQ(
        (&overflow_encoding), Offsets, offsets, empty_offsets, empty_offsets);

    std::vector<uint32_t> special_tokens_mask(ids.size(), 0);
    special_tokens_mask.front() = special_tokens_mask.back() = 1;

    std::vector<uint32_t> attention_mask(ids.size(), 1);

    std::unordered_map<uint32_t, core::Range> sequence_ranges;
    sequence_ranges[0] = {1, ids.size() - 1};

    overflow_encoding = std::move(
        core::Encoding(std::move(ids),
                       std::move(type_ids),
                       std::move(tokens),
                       std::move(word_idx),
                       std::move(offsets),
                       std::move(special_tokens_mask),
                       std::move(attention_mask),
                       std::vector<core::Encoding>(),  // No overflowing
                       std::move(sequence_ranges)));
  }

  core::Encoding new_encoding(std::move(ids),
                              std::move(type_ids),
                              std::move(tokens),
                              std::move(word_idx),
                              std::move(offsets),
                              std::move(special_tokens_mask),
                              std::move(attention_mask),
                              std::move(overflowings),
                              std::move(sequence_ranges));
  if (pair_encoding != nullptr) {
#define CREATE_PROCESSED_PARI_ENCODING_SEQ(                                \
    encoding_ptr, attr, name, back_value)                                  \
  auto encoding_##name = encoding_ptr->Get##attr();                        \
  decltype(encoding_##name) name(encoding_##name.size() + 1);              \
  std::copy(encoding_##name.begin(), encoding_##name.end(), name.begin()); \
  name.back() = back_value

    CREATE_PROCESSED_PARI_ENCODING_SEQ(pair_encoding, Ids, ids, sep_.second);
    CREATE_PROCESSED_PARI_ENCODING_SEQ(pair_encoding, TypeIds, type_ids, 1);
    CREATE_PROCESSED_PARI_ENCODING_SEQ(
        pair_encoding, Tokens, tokens, sep_.first);
    CREATE_PROCESSED_PARI_ENCODING_SEQ(pair_encoding, WordsIdx, word_idx, -1);
    core::Offset empty_offsets = {0, 0};
    CREATE_PROCESSED_PARI_ENCODING_SEQ(
        pair_encoding, Offsets, offsets, empty_offsets);

    std::vector<uint32_t> special_tokens_mask(ids.size(), 0);
    special_tokens_mask.back() = 1;

    std::vector<uint32_t> attention_mask(ids.size(), 1);
    std::unordered_map<uint32_t, core::Range> sequence_ranges;
    sequence_ranges[1] = {0, ids.size() - 1};
    // overflowing
    auto& overflowings = pair_encoding->GetMutableOverflowing();
    for (auto& overflow_pair_encoding : overflowings) {
      CREATE_PROCESSED_PARI_ENCODING_SEQ(
          (&overflow_pair_encoding), Ids, ids, sep_.second);
      CREATE_PROCESSED_PARI_ENCODING_SEQ(
          (&overflow_pair_encoding), TypeIds, type_ids, 1);
      CREATE_PROCESSED_PARI_ENCODING_SEQ(
          (&overflow_pair_encoding), Tokens, tokens, sep_.first);
      CREATE_PROCESSED_PARI_ENCODING_SEQ(
          (&overflow_pair_encoding), WordsIdx, word_idx, -1);
      core::Offset empty_offsets = {0, 0};
      CREATE_PROCESSED_PARI_ENCODING_SEQ(
          (&overflow_pair_encoding), Offsets, offsets, empty_offsets);

      std::vector<uint32_t> special_tokens_mask(ids.size(), 0);
      special_tokens_mask.back() = 1;

      std::vector<uint32_t> attention_mask(ids.size(), 1);
      std::unordered_map<uint32_t, core::Range> sequence_ranges;
      sequence_ranges[0] = {1, ids.size() - 1};

      overflow_pair_encoding = std::move(
          core::Encoding(std::move(ids),
                         std::move(type_ids),
                         std::move(tokens),
                         std::move(word_idx),
                         std::move(offsets),
                         std::move(special_tokens_mask),
                         std::move(attention_mask),
                         std::vector<core::Encoding>(),  // No overflowing
                         std::move(sequence_ranges)));
    }

    core::Encoding new_pair_encoding(std::move(ids),
                                     std::move(type_ids),
                                     std::move(tokens),
                                     std::move(word_idx),
                                     std::move(offsets),
                                     std::move(special_tokens_mask),
                                     std::move(attention_mask),
                                     std::move(overflowings),
                                     std::move(sequence_ranges));
    new_encoding.MergeWith(new_pair_encoding, false);
  }
#undef CREATE_PROCESSED_ENCODING_SEQ
#undef CREATE_PROCESSED_PARI_ENCODING_SEQ
  *result_encoding = std::move(new_encoding);
}

void to_json(nlohmann::json& j, const BertPostProcessor& bert_postprocessor) {
  j = {
      {"type", "BertPostProcessor"},
      {"sep", bert_postprocessor.sep_},
      {"cls", bert_postprocessor.cls_},
  };
}

void from_json(const nlohmann::json& j, BertPostProcessor& bert_postprocessor) {
  j["cls"].get_to(bert_postprocessor.cls_);
  j["sep"].get_to(bert_postprocessor.sep_);
}

}  // namespace postprocessors
}  // namespace faster_tokenizer
}  // namespace paddlenlp
