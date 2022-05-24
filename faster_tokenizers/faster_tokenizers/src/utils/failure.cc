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
#include <sstream>

#include "glog/logging.h"
#include "utils/failure.h"
#include "utils/trie.h"
#include "utils/utf8.h"
#include "utils/utils.h"

namespace tokenizers {
namespace utils {

Failure::Failure()
    : failure_link_(utils::kNullNode),
      failure_pops_offset_length_(utils::kNullFailurePopsList) {}

FailureVocabToken::FailureVocabToken(
    const std::string& token,
    int token_id,
    const std::string& continuing_subword_prefix)
    : token_(token),
      token_id_(token_id),
      is_suffix_token_(false),
      actual_token_start_offset_(0),
      actual_token_unicode_len_(0),
      contains_punctuation_(false) {
  if (!continuing_subword_prefix.empty() &&
      token_ != continuing_subword_prefix &&
      utils::IsSuffixWord(token_, continuing_subword_prefix)) {
    is_suffix_token_ = true;
    actual_token_start_offset_ = continuing_subword_prefix.size();
  }
  // Iterate over the Unicode chars from the token, to initialize
  // contains_punctuation_ and actual_token_unicode_len_.
  int token_len = token.size();
  int cur_pos = actual_token_start_offset_;
  uint32_t ch;
  const char* pSrc = token.c_str();
  while (cur_pos < token_len) {
    uint32_t count = utils::UTF8ToUInt32(pSrc + cur_pos, &ch);
    cur_pos += count;
    ch = utils::UTF8ToUnicode(ch);
    if (!contains_punctuation_ && utils::IsPunctuationOrChineseChar(ch)) {
      contains_punctuation_ = true;
    }
    ++actual_token_unicode_len_;
  }
}

std::string FailureVocabToken::Token() const { return token_; }

int FailureVocabToken::TokenId() const { return token_id_; }

bool FailureVocabToken::IsSuffixToken() const { return is_suffix_token_; }

bool FailureVocabToken::ContainsPunctuation() const {
  return contains_punctuation_;
}

int FailureVocabToken::TokenUnicodeLengthWithoutContinuingSubwordPrefix()
    const {
  return actual_token_unicode_len_;
}

int FailureVocabToken::TokenLengthWithoutContinuingSubwordPrefix() const {
  return token_.size() - actual_token_start_offset_;
}

void FailureArray::BuildFailureVocab(
    const std::unordered_map<std::string, uint>& vocab, const Trie& trie) {
  if (vocab.size() > utils::kMaxSupportedVocabSize) {
    std::ostringstream oss;
    oss << "Vocab size exceeds the max supported ("
        << utils::kMaxSupportedVocabSize
        << "). Found vocab size: " << vocab.size();
    throw std::invalid_argument(oss.str());
  }
  failure_vocab_tokens_.reserve(vocab.size());
  auto continuing_subword_prefix = trie.GetContinuingSubwordPrefix();
  for (auto& item : vocab) {
    if (item.first == continuing_subword_prefix) {
      VLOG(6)
          << "The empty suffix token is found in the vocabulary, which takes "
             "place in token id space but will (almost) never be used in the "
             "result. Consider cleaning it from the vocabulary.";
      continue;
    }
    if (item.first.empty()) {
      VLOG(6)
          << "The empty string is found in the vocabulary, which takes place "
             "in the token id space but will never be used in the result. "
             "Consider cleaning it from the vocabulary.";
      continue;
    }
    FailureVocabToken vocab_token(
        item.first, item.second, continuing_subword_prefix);
    if (vocab_token.TokenLengthWithoutContinuingSubwordPrefix() >
        utils::kMaxVocabTokenLengthInUTF8Bytes) {
      std::ostringstream oss;
      oss << "Vocab token utf8 length (excluding suffix indicator) exceeds the "
             "max supported ("
          << utils::kMaxVocabTokenLengthInUTF8Bytes
          << "). The vocab token is: " << item.first
          << " with utf8 length (excluding suffix indicator): "
          << vocab_token.TokenLengthWithoutContinuingSubwordPrefix();
      throw std::invalid_argument(oss.str());
    }
    failure_vocab_tokens_.emplace_back(vocab_token);
  }
  if (failure_vocab_tokens_.empty()) {
    std::ostringstream oss;
    oss << "No valid vocab tokens were found to build the trie.";
    throw std::invalid_argument(oss.str());
  }
  if (!continuing_subword_prefix.empty()) {
    const bool suffix_token_exists = std::any_of(
        failure_vocab_tokens_.begin(),
        failure_vocab_tokens_.end(),
        [](const FailureVocabToken& token) { return token.IsSuffixToken(); });
    if (!suffix_token_exists) {
      auto new_suffix_token = continuing_subword_prefix +
                              std::string(1, utils::kInvalidControlChar);
      failure_vocab_tokens_.emplace_back(FailureVocabToken(
          new_suffix_token, trie.GetUNKTokenID(), continuing_subword_prefix));
    }
  }
}

FailureArray::FailureArray(const std::unordered_map<std::string, uint>& vocab,
                           const Trie& trie) {
  BuildFailureVocab(vocab, trie);
  BuildFailureArray(vocab, trie);
}

void FailureArray::BuildFailureArray(
    const std::unordered_map<std::string, uint>& vocab, const Trie& trie) {}

void FailureArray::BuildOutgoingEdgeLabelsForTrie(
    const std::unordered_map<std::string, uint>& vocab,
    const Trie& trie,
    std::vector<std::unordered_set<char>>* node_outgoing_edge_labels) {
  node_outgoing_edge_labels->resize(trie.Size());
  const std::string dummy_token = std::string(1, utils::kInvalidControlChar);
  for (auto& item : vocab) {
    if (item.first != dummy_token) {
      BuildOutgoingEdgeLabelsFromToken(item, trie, node_outgoing_edge_labels);
    }
  }
}

void FailureArray::BuildOutgoingEdgeLabelsFromToken(
    const std::pair<std::string, uint>& vocab_token,
    const Trie& trie,
    std::vector<std::unordered_set<char>>* node_outgoing_edge_labels) {
  const std::string& token = vocab_token.first;
  Trie::TraversalCursor curr_node;
  int char_pos = 0;
  trie.SetTraversalCursor(&curr_node, Trie::kRootNodeId);
  while (char_pos < token.size()) {
    const char edge_label = token[char_pos];
    (*node_outgoing_edge_labels)[curr_node.node_id_].insert(edge_label);
    if (!trie.TryTraverseOneStep(&curr_node, edge_label)) {
      std::ostringstream oss;
      oss << "Error in traversing to child following edge " << edge_label
          << " from the prefix " << token.substr(0, char_pos)
          << " at parent id " << curr_node.node_id_ << ". The token is "
          << token << ". The char position"
          << " is " << char_pos << ".";

      throw std::runtime_error(oss.str());
    }
    ++char_pos;
  }
  //   node_id_is_punc_map_[cur_node.node_id] =
  //       !vocab_token.IsSuffixToken() && vocab_token.ContainsPunctuation() &&
  //       vocab_token.TokenUnicodeLengthWithoutSuffixIndicator() == 1;
  //   return absl::OkStatus();
  //   node_id_is_punc_map_[curr_node.node_id_] =
  //     !utils::IsSuffixWord(token, trie.GetContinuingSubwordPrefix())
  //   && utils::IsPunctuationOrChineseChar()
}


}  // namespace utils
}  // namespace tokenizers