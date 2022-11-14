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

#include "fast_tokenizer/tokenizers/clip_fast_tokenizer.h"
#include "fast_tokenizer/models/models.h"
#include "fast_tokenizer/normalizers/normalizers.h"
#include "fast_tokenizer/postprocessors/postprocessors.h"
#include "fast_tokenizer/pretokenizers/pretokenizers.h"
#include "glog/logging.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace tokenizers_impl {

ClipFastTokenizer::ClipFastTokenizer(
    const std::string& vocab_path,
    const std::string& merges_path,
    uint32_t max_length,
    const std::string& unk_token,
    const std::string& pad_token,
    const std::string& bos_token,
    const std::string& eos_token,
    bool add_prefix_space,
    const std::string& continuing_subword_prefix,
    const std::string& end_of_word_suffix,
    bool trim_offsets) {
  core::Vocab vocab;
  core::Merges merges;
  models::BPE::GetVocabAndMergesFromFile(
      vocab_path, merges_path, &vocab, &merges);
  VLOG(6) << "The vocab size of ClipFastTokenizer is " << vocab.size();
  VLOG(6) << "The merges size of ClipFastTokenizer is " << merges.size();

  models::BPE bpe(vocab,
                  merges,
                  10000,
                  {},
                  {unk_token},
                  {continuing_subword_prefix},
                  {end_of_word_suffix},
                  false);
  // Set tokenizer model
  this->SetModel(bpe);

  // Set added tokens
  std::vector<core::AddedToken> added_tokens;
  uint32_t id;
  unk_token_ = unk_token;
  if (this->TokenToId(unk_token, &id)) {
    added_tokens.emplace_back(unk_token, true);
  }
  pad_token_ = pad_token;
  if (this->TokenToId(pad_token, &id)) {
    added_tokens.emplace_back(pad_token, true);
    pad_token_id_ = id;
  }
  bos_token_ = bos_token;
  if (this->TokenToId(bos_token, &id)) {
    added_tokens.emplace_back(bos_token, true);
    bos_token_id_ = id;
  }
  eos_token_ = eos_token;
  if (this->TokenToId(eos_token, &id)) {
    added_tokens.emplace_back(eos_token, true);
    eos_token_id_ = id;
  }
  this->AddSpecialTokens(added_tokens);

  // Set normalizers
  normalizers::NFCNormalizer nfc_normalizer;
  normalizers::ReplaceNormalizer replace_normalizer(R"(\s+)", " ");
  normalizers::LowercaseNormalizer lower_normalizer;
  normalizers::SequenceNormalizer seq_normalizer;
  seq_normalizer.AppendNormalizer(&nfc_normalizer);
  seq_normalizer.AppendNormalizer(&replace_normalizer);
  seq_normalizer.AppendNormalizer(&lower_normalizer);
  this->SetNormalizer(seq_normalizer);

  // Set pretokenizers
  pretokenizers::ByteLevelPreTokenizer byte_level_pretokenizer(add_prefix_space,
                                                               true);
  pretokenizers::SplitPreTokenizer split_pretokenizer(
      R"('s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+)",
      core::SplitMode::REMOVED,
      true);
  pretokenizers::SequencePreTokenizer seq_pretokenizer;
  seq_pretokenizer.AppendPreTokenizer(&split_pretokenizer);
  seq_pretokenizer.AppendPreTokenizer(&byte_level_pretokenizer);
  this->SetPreTokenizer(seq_pretokenizer);

  // Set postprocessors
  postprocessors::RobertaPostProcessor roberta_postprocessor(
      {eos_token, eos_token_id_},
      {bos_token, bos_token_id_},
      /* trim_offsets= */ false,
      add_prefix_space);
  this->SetPostProcessor(roberta_postprocessor);

  if (max_length == 0) {
    this->DisableTruncMethod();
  } else {
    this->EnableTruncMethod(max_length,
                            0,
                            core::Direction::RIGHT,
                            core::TruncStrategy::LONGEST_FIRST);
  }
}

std::string ClipFastTokenizer::GetPadToken() const { return pad_token_; }

uint32_t ClipFastTokenizer::GetPadTokenId() const { return pad_token_id_; }

std::string ClipFastTokenizer::GetUNKToken() const { return unk_token_; }

uint32_t ClipFastTokenizer::GetUNKTokenId() const { return unk_token_id_; }

std::string ClipFastTokenizer::GetBOSToken() const { return bos_token_; }

uint32_t ClipFastTokenizer::GetBOSTokenId() const { return bos_token_id_; }

std::string ClipFastTokenizer::GetEOSToken() const { return eos_token_; }

uint32_t ClipFastTokenizer::GetEOSTokenId() const { return eos_token_id_; }

}  // namespace tokenizers_impl
}  // namespace fast_tokenizer
}  // namespace paddlenlp
