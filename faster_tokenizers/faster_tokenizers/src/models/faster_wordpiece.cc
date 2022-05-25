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

#include <algorithm>
#include <codecvt>
#include <fstream>
#include <locale>
#include <map>

#include "glog/logging.h"
#include "models/faster_wordpiece.h"
#include "models/wordpiece.h"
#include "utils/path.h"
#include "utils/utf8.h"
#include "utils/utils.h"

namespace tokenizers {
namespace models {

const std::string WHITESPACE = " \n\r\t\f\v";

FasterWordPiece::FasterWordPiece() : WordPiece() {
  failure_array_.InitFromVocabAndTrie(vocab_, trie_);
  PrecomputeEncodeValueForSubwordPrefix();
}

FasterWordPiece::FasterWordPiece(const core::Vocab& vocab,
                                 const std::string& unk_token,
                                 size_t max_input_chars_per_word,
                                 const std::string& continuing_subword_prefix)
    : WordPiece(vocab,
                unk_token,
                max_input_chars_per_word,
                continuing_subword_prefix),
      trie_(vocab, continuing_subword_prefix, unk_token) {
  failure_array_.InitFromVocabAndTrie(vocab_, trie_);
  PrecomputeEncodeValueForSubwordPrefix();
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

bool FasterWordPiece::TokenToId(const std::string& token, uint* id) const {
  auto curr_cursor = trie_.CreateRootTraversalCursor();
  for (auto& ch : token) {
    if (!trie_.TryTraverseOneStep(&curr_cursor, ch)) {
      return false;
    }
  }
  int encoded_value;
  if (!trie_.TryGetData(curr_cursor, &encoded_value)) {
    return false;
  }
  // Decode the encoded_value
  *id = utils::GetTokenIdFromEncodedValue(encoded_value);
  return true;
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
  uint id = utils::GetTokenIdFromEncodedValue(curr_node_value);
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
    if (!TryFollowFailureLinkAndCollectTokens(
            sequence, 0, curr_offset_in_sequence, curr_node, tokens)) {
      ResetOutputAppendUNK(sequence_offset_in_text,
                           sequence.size(),
                           original_num_tokens,
                           tokens);
      return;
    }
  }
  *original_num_tokens = tokens->size();
}

std::vector<core::Token> FasterWordPiece::Tokenize(
    const std::string& sequence) const {
  if (sequence.empty()) {
    return {};
  }
  std::vector<core::Token> all_tokens;
  size_t unicode_len =
      utils::GetUnicodeLenFromUTF8(sequence.data(), sequence.length());
  int original_num_tokens = 0;
  if (unicode_len > max_input_chars_per_word_) {
    ResetOutputAppendUNK(
              0, sequence.size(), &original_num_tokens, &all_tokens);
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
  return all_tokens;
}

void to_json(nlohmann::json& j, const FasterWordPiece& model) {
  j = {
      {"type", "FasterWordPiece"},
      {"vocab", model.vocab_},
      {"unk_token", model.unk_token_},
      {"max_input_chars_per_word", model.max_input_chars_per_word_},
      {"continuing_subword_prefix", model.continuing_subword_prefix_},
  };
}

void from_json(const nlohmann::json& j, FasterWordPiece& model) {
  j["vocab"].get_to(model.vocab_);
  j["unk_token"].get_to(model.unk_token_);
  j["max_input_chars_per_word"].get_to(model.max_input_chars_per_word_);
  j["continuing_subword_prefix"].get_to(model.continuing_subword_prefix_);
}

}  // models
}  // tokenizers
