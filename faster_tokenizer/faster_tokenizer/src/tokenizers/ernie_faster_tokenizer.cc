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

#include "tokenizers/ernie_faster_tokenizer.h"
#include "core/encoding.h"
#include "models/models.h"
#include "normalizers/normalizers.h"
#include "postprocessors/postprocessors.h"
#include "pretokenizers/pretokenizers.h"
#include "utils/utils.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace tokenizers_impl {

ErnieFasterTokenizer::ErnieFasterTokenizer(const std::string& vocab_path,
                                           const std::string& unk_token,
                                           const std::string& sep_token,
                                           const std::string& cls_token,
                                           const std::string& pad_token,
                                           const std::string& mask_token,
                                           bool clean_text,
                                           bool handle_chinese_chars,
                                           bool strip_accents,
                                           bool lowercase,
                                           const std::string& wordpieces_prefix,
                                           uint32_t max_sequence_len) {
  core::Vocab vocab;
  utils::GetVocabFromFiles(vocab_path, &vocab);
  VLOG(6) << "The vocab size of ErnieFasterTokenizer is " << vocab.size();
  Init(vocab,
       unk_token,
       sep_token,
       cls_token,
       pad_token,
       mask_token,
       clean_text,
       handle_chinese_chars,
       strip_accents,
       lowercase,
       wordpieces_prefix,
       max_sequence_len);
}


ErnieFasterTokenizer::ErnieFasterTokenizer(const core::Vocab& vocab,
                                           const std::string& unk_token,
                                           const std::string& sep_token,
                                           const std::string& cls_token,
                                           const std::string& pad_token,
                                           const std::string& mask_token,
                                           bool clean_text,
                                           bool handle_chinese_chars,
                                           bool strip_accents,
                                           bool lowercase,
                                           const std::string& wordpieces_prefix,
                                           uint32_t max_sequence_len) {
  Init(vocab,
       unk_token,
       sep_token,
       cls_token,
       pad_token,
       mask_token,
       clean_text,
       handle_chinese_chars,
       strip_accents,
       lowercase,
       wordpieces_prefix,
       max_sequence_len);
}


void ErnieFasterTokenizer::Init(const core::Vocab& vocab,
                                const std::string& unk_token,
                                const std::string& sep_token,
                                const std::string& cls_token,
                                const std::string& pad_token,
                                const std::string& mask_token,
                                bool clean_text,
                                bool handle_chinese_chars,
                                bool strip_accents,
                                bool lowercase,
                                const std::string& wordpieces_prefix,
                                uint32_t max_sequence_len) {
  models::WordPiece wordpiece(
      vocab, unk_token, 100 /* max_input_chars_per_word */, wordpieces_prefix);
  this->SetModel(wordpiece);

  std::vector<core::AddedToken> added_tokens;
  uint32_t id;
  if (!this->TokenToId(unk_token, &id)) {
    added_tokens.emplace_back(unk_token, true);
  }
  if (!this->TokenToId(sep_token, &id)) {
    added_tokens.emplace_back(sep_token, true);
  }
  if (!this->TokenToId(cls_token, &id)) {
    added_tokens.emplace_back(cls_token, true);
  }
  if (!this->TokenToId(pad_token, &id)) {
    added_tokens.emplace_back(pad_token, true);
  }
  if (!this->TokenToId(mask_token, &id)) {
    added_tokens.emplace_back(mask_token, true);
  }
  this->AddSpecialTokens(added_tokens);


  normalizers::BertNormalizer bert_normalizer(
      clean_text, handle_chinese_chars, strip_accents, lowercase);
  this->SetNormalizer(bert_normalizer);

  pretokenizers::BertPreTokenizer bert_pretokenizer;
  this->SetPreTokenizer(bert_pretokenizer);

  if (vocab.size() > 0) {
    uint32_t sep_id, cls_id;
    if (!this->TokenToId(sep_token, &sep_id)) {
      throw std::invalid_argument("sep_token not found in the vocabulary");
    }
    if (!this->TokenToId(cls_token, &cls_id)) {
      throw std::invalid_argument("cls_token not found in the vocabulary");
    }
    postprocessors::BertPostProcessor bert_postprocessor({sep_token, sep_id},
                                                         {cls_token, cls_id});
    this->SetPostProcessor(bert_postprocessor);
  }
  if (max_sequence_len == 0) {
    this->DisableTruncMethod();
  } else {
    this->EnableTruncMethod(max_sequence_len,
                            0,
                            core::Direction::RIGHT,
                            core::TruncStrategy::LONGEST_FIRST);
  }
}

}  // namespace tokenizers_impl
}  // namespace faster_tokenizer
}  // namespace paddlenlp
