// Copyright 2022 TF.Text Authors.
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <codecvt>
#include <fstream>
#include <locale>
#include <map>

#include "glog/logging.h"
#include "unicode/uchar.h"

#include "models/faster_wordpiece.h"
#include "models/wordpiece.h"
#include "utils/path.h"
#include "utils/utf8.h"
#include "utils/utils.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace models {

const std::string WHITESPACE = " \n\r\t\f\v";

void FasterWordPiece::InitFailureAndTrie() {
  unk_token_id_ = vocab_.at(unk_token_);
  trie_.SetWithPretokenization(with_pretokenization_);
  trie_.SetUNKToken(unk_token_);
  trie_.SetContinuingSubwordPrefix(continuing_subword_prefix_);
  failure_array_.SetWithPretokenization(with_pretokenization_);
  failure_array_.InitFromVocabAndTrie(
      vocab_, &trie_, unk_token_, continuing_subword_prefix_);
  PrecomputeEncodeValueForSubwordPrefix();
}

FasterWordPiece::FasterWordPiece()
    : WordPiece(), with_pretokenization_(false) {}

FasterWordPiece::FasterWordPiece(const core::Vocab& vocab,
                                 const std::string& unk_token,
                                 size_t max_input_chars_per_word,
                                 const std::string& continuing_subword_prefix,
                                 bool with_pretokenization)
    : WordPiece(vocab,
                unk_token,
                max_input_chars_per_word,
                continuing_subword_prefix),
      trie_(continuing_subword_prefix, unk_token, with_pretokenization),
      with_pretokenization_(with_pretokenization),
      failure_array_(with_pretokenization) {
  InitFailureAndTrie();
}

void FasterWordPiece::PrecomputeEncodeValueForSubwordPrefix() {
  auto subword_prefix_tokens = WordPiece::Tokenize(continuing_subword_prefix_);
  encoded_value_for_subword_prefix_.reserve(subword_prefix_tokens.size());

  for (auto& token : subword_prefix_tokens) {
    utils::FailureVocabToken failure_vocab_token(
        token.value_, token.id_, continuing_subword_prefix_);
    int encoded_value = utils::EncodeToken(
        failure_vocab_token.TokenId(),
        failure_vocab_token.TokenLengthWithoutContinuingSubwordPrefix(),
        failure_vocab_token.IsSuffixToken());
    encoded_value_for_subword_prefix_.push_back(encoded_value);
  }
}

bool FasterWordPiece::TryFollowFailureLinkAndCollectTokens(
    const std::string& sequence,
    int sequence_offset_in_text,
    int* curr_offset_in_sequence,
    utils::Trie::TraversalCursor* node,
    std::vector<core::Token>* tokens) const {
  int curr_node_value = 0;
  if (trie_.TryGetData(*node, &curr_node_value)) {
    AppendTokensToOutput(sequence,
                         sequence_offset_in_text,
                         curr_offset_in_sequence,
                         curr_node_value,
                         tokens);
    trie_.SetTraversalCursor(
        node, failure_array_.GetFailure(node->node_id_)->failure_link_);
    return true;
  }
  const auto& node_aux = failure_array_.GetFailure(node->node_id_);

  if (node_aux->failure_link_ == utils::kNullNode) {
    // No failure_link can be followed.
    return false;
  }
  int offset = 0, length = 0;
  utils::GetFailurePopsOffsetAndLength(
      node_aux->failure_pops_offset_length_, &offset, &length);
  for (int i = offset; i < offset + length; ++i) {
    AppendTokensToOutput(sequence,
                         sequence_offset_in_text,
                         curr_offset_in_sequence,
                         failure_array_.GetFailurePop(i),
                         tokens);
  }
  trie_.SetTraversalCursor(node, node_aux->failure_link_);
  return true;
}

void FasterWordPiece::AppendTokensToOutput(
    const std::string& sequence,
    int sequence_offset_in_text,
    int* curr_offset_in_sequence,
    int curr_node_value,
    std::vector<core::Token>* tokens) const {
  uint32_t id = utils::GetTokenIdFromEncodedValue(curr_node_value);
  std::string value;
  int token_substr_length =
      utils::GetTokenLengthFromEncodedValue(curr_node_value);
  if (*curr_offset_in_sequence == 0 &&
      utils::IsSuffixTokenFromEncodedValue(curr_node_value)) {
    token_substr_length += continuing_subword_prefix_.size();
  }

  if (id == unk_token_id_) {
    value = unk_token_;
  } else {
    value = sequence.substr(*curr_offset_in_sequence, token_substr_length);
  }

  if (*curr_offset_in_sequence > 0) {
    value = continuing_subword_prefix_ + value;
  }
  core::Offset offset = {
      sequence_offset_in_text + *curr_offset_in_sequence,
      sequence_offset_in_text + *curr_offset_in_sequence + token_substr_length};
  tokens->emplace_back(id, value, offset);

  *curr_offset_in_sequence += token_substr_length;
}

void FasterWordPiece::ResetOutputAppendUNK(
    int sequence_offset_in_text,
    int sequence_size,
    int* original_num_tokens,
    std::vector<core::Token>* tokens) const {
  tokens->resize(*original_num_tokens + 1);
  tokens->back() = {
      unk_token_id_,
      unk_token_,
      {sequence_offset_in_text, sequence_offset_in_text + sequence_size}};
  (*original_num_tokens)++;
}

bool FasterWordPiece::TryHandleContinuingSubWordPrefix(
    const std::string& sequence,
    int sequence_offset_in_text,
    const utils::Trie::TraversalCursor& curr_node,
    int* original_num_tokens,
    int* curr_offset_in_sequence,
    std::vector<core::Token>* tokens) const {
  if (curr_node.node_id_ != trie_.GetSuffixRoot()) {
    return false;
  }
  int cur_num_tokens = tokens->size();
  if (cur_num_tokens != *original_num_tokens) {
    return false;
  }
  if (encoded_value_for_subword_prefix_.size() == 1 &&
      utils::GetTokenIdFromEncodedValue(encoded_value_for_subword_prefix_[0]) ==
          unk_token_id_) {
    ResetOutputAppendUNK(
        sequence_offset_in_text, sequence.size(), original_num_tokens, tokens);
    return true;
  }
  for (int encoded_token_value : encoded_value_for_subword_prefix_) {
    AppendTokensToOutput(sequence,
                         sequence_offset_in_text,
                         curr_offset_in_sequence,
                         encoded_token_value,
                         tokens);
  }
  return true;
}

void FasterWordPiece::HandleTheRemainingStringOnTriePath(
    const std::string& sequence,
    int sequence_offset_in_text,
    utils::Trie::TraversalCursor* curr_node,
    int* original_num_tokens,
    int* curr_offset_in_sequence,
    std::vector<core::Token>* tokens) const {
  if (curr_node->node_id_ == utils::Trie::kRootNodeId) {
    return;
  }
  if (TryHandleContinuingSubWordPrefix(sequence,
                                       sequence_offset_in_text,
                                       *curr_node,
                                       original_num_tokens,
                                       curr_offset_in_sequence,
                                       tokens)) {
    *original_num_tokens = tokens->size();
    return;
  }
  while (curr_node->node_id_ != trie_.GetSuffixRoot() &&
         curr_node->node_id_ != trie_.GetPuncFailureNode()) {
    if (!TryFollowFailureLinkAndCollectTokens(sequence,
                                              sequence_offset_in_text,
                                              curr_offset_in_sequence,
                                              curr_node,
                                              tokens)) {
      ResetOutputAppendUNK(sequence_offset_in_text,
                           sequence.size(),
                           original_num_tokens,
                           tokens);
      return;
    }
  }
  *original_num_tokens = tokens->size();
}

int FasterWordPiece::SkipRemainingOfWordAndTrailingWhiteSpaces(
    const std::string& sequence, int* curr_idx) const {
  int seq_len = sequence.length();
  uint32_t curr_unicode_char;
  int end_of_word = *curr_idx;
  while (*curr_idx < seq_len) {
    auto chwidth =
        utils::UTF8ToUInt32(sequence.data() + *curr_idx, &curr_unicode_char);
    curr_unicode_char = utils::UTF8ToUnicode(curr_unicode_char);
    if (u_isUWhiteSpace(curr_unicode_char)) {
      *curr_idx += chwidth;
      break;
    }
    if (utils::IsPunctuationOrChineseChar(curr_unicode_char)) {
      break;
    }
    *curr_idx += chwidth;
    end_of_word = *curr_idx;
  }
  return end_of_word;
}

std::vector<core::Token> FasterWordPiece::TokenizeWithoutPreTokenize(
    const std::string& sequence) const {
  VLOG(6) << "Using FasterWordPiece::TokenizeWithoutPreTokenize to tokenize "
             "sequence";
  if (sequence.empty()) {
    return {};
  }
  std::vector<core::Token> all_tokens;
  size_t unicode_len =
      utils::GetUnicodeLenFromUTF8(sequence.data(), sequence.length());
  int original_num_tokens = 0;
  if (unicode_len > max_input_chars_per_word_) {
    ResetOutputAppendUNK(0, sequence.size(), &original_num_tokens, &all_tokens);
  } else {
    int curr_offset_in_sequence = 0;
    auto curr_node = trie_.CreateRootTraversalCursor();
    for (auto ch : sequence) {
      while (!trie_.TryTraverseOneStep(&curr_node, ch)) {
        if (!TryFollowFailureLinkAndCollectTokens(sequence,
                                                  0,
                                                  &curr_offset_in_sequence,
                                                  &curr_node,
                                                  &all_tokens)) {
          ResetOutputAppendUNK(
              0, sequence.size(), &original_num_tokens, &all_tokens);
          return all_tokens;
        }
      }
    }
    HandleTheRemainingStringOnTriePath(sequence,
                                       0,
                                       &curr_node,
                                       &original_num_tokens,
                                       &curr_offset_in_sequence,
                                       &all_tokens);
  }
  VLOG(6) << "All tokens num from TokenizeWithoutPreTokenize: "
          << all_tokens.size();
  return all_tokens;
}

std::vector<core::Token> FasterWordPiece::TokenizeWithPreTokenize(
    const std::string& sequence) const {
  VLOG(6)
      << "Using FasterWordPiece::TokenizeWithPreTokenize to tokenize sequence";
  // Need to implement
  if (sequence.empty()) {
    return {};
  }
  std::vector<core::Token> all_tokens;
  int original_num_tokens = 0;
  uint32_t prev_unicode_char, curr_unicode_char;
  int curr_idx = 0;
  int chwidth = 0;
  auto seq_len = sequence.length();
  while (curr_idx < seq_len) {
    int curr_offset_in_word = 0;
    auto curr_node = trie_.CreateRootTraversalCursor();
    int bytes_length = 0;
    int word_offset_in_sequence = curr_idx;
    std::string sequence_substr = sequence.substr(curr_idx);
    bool fail_to_match = false;
    while (curr_idx < seq_len) {
      prev_unicode_char = curr_unicode_char;
      chwidth =
          utils::UTF8ToUInt32(sequence.data() + curr_idx, &curr_unicode_char);
      curr_unicode_char = utils::UTF8ToUnicode(curr_unicode_char);
      if (bytes_length + chwidth > max_input_chars_per_word_) {
        break;
      }
      std::string curr_substr = sequence.substr(curr_idx, chwidth);
      while (!trie_.TryTraverseSeveralSteps(&curr_node, curr_substr)) {
        if (!TryFollowFailureLinkAndCollectTokens(sequence_substr,
                                                  word_offset_in_sequence,
                                                  &curr_offset_in_word,
                                                  &curr_node,
                                                  &all_tokens)) {
          fail_to_match = true;
          break;
        }
      }
      if (fail_to_match) {
        break;
      }
      bytes_length += chwidth;
      curr_idx += chwidth;
    }
    if (curr_idx >= seq_len) {
      HandleTheRemainingStringOnTriePath(sequence_substr,
                                         word_offset_in_sequence,
                                         &curr_node,
                                         &original_num_tokens,
                                         &curr_offset_in_word,
                                         &all_tokens);
      break;
    }
    bool curr_unicode_char_is_space = u_isUWhiteSpace(curr_unicode_char);
    if (curr_unicode_char_is_space ||
        utils::IsPunctuationOrChineseChar(curr_unicode_char) ||
        (curr_idx > 0 &&
         utils::IsPunctuationOrChineseChar(prev_unicode_char))) {
      HandleTheRemainingStringOnTriePath(
          sequence_substr.substr(0, curr_idx - word_offset_in_sequence),
          word_offset_in_sequence,
          &curr_node,
          &original_num_tokens,
          &curr_offset_in_word,
          &all_tokens);
      if (curr_unicode_char_is_space) {
        curr_idx += chwidth;
      }
      continue;
    }
    curr_idx += chwidth;
    int end_of_word =
        SkipRemainingOfWordAndTrailingWhiteSpaces(sequence, &curr_idx);
    ResetOutputAppendUNK(word_offset_in_sequence,
                         end_of_word - word_offset_in_sequence,
                         &original_num_tokens,
                         &all_tokens);
  }
  VLOG(6) << "All tokens num from TokenizeWithPreTokenize: "
          << all_tokens.size();
  return all_tokens;
}

std::vector<core::Token> FasterWordPiece::Tokenize(
    const std::string& sequence) {
  if (!with_pretokenization_) {
    return TokenizeWithoutPreTokenize(sequence);
  }
  return TokenizeWithPreTokenize(sequence);
}

void to_json(nlohmann::json& j, const FasterWordPiece& model) {
  j = {
      {"type", "FasterWordPiece"},
      {"vocab", model.vocab_},
      {"unk_token", model.unk_token_},
      {"max_input_chars_per_word", model.max_input_chars_per_word_},
      {"continuing_subword_prefix", model.continuing_subword_prefix_},
      {"with_pretokenization", model.with_pretokenization_},
  };
}

void from_json(const nlohmann::json& j, FasterWordPiece& model) {
  j["vocab"].get_to(model.vocab_);
  j["unk_token"].get_to(model.unk_token_);
  j["max_input_chars_per_word"].get_to(model.max_input_chars_per_word_);
  j["continuing_subword_prefix"].get_to(model.continuing_subword_prefix_);
  j["with_pretokenization"].get_to(model.with_pretokenization_);
  model.InitFailureAndTrie();
}

}  // namespace models
}  // namespace faster_tokenizer
}  // namespace paddlenlp
