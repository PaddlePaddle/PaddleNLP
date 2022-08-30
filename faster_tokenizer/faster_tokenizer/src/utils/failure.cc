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
#include <queue>
#include <sstream>

#include "glog/logging.h"
#include "utils/failure.h"
#include "utils/trie.h"
#include "utils/utf8.h"
#include "utils/utils.h"

namespace paddlenlp {
namespace faster_tokenizer {
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

const std::string& FailureVocabToken::Token() const { return token_; }

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
    const std::unordered_map<std::string, uint32_t>& vocab,
    const std::string& unk_token,
    const std::string& continuing_subword_prefix) {
  if (vocab.size() > utils::kMaxSupportedVocabSize) {
    std::ostringstream oss;
    oss << "Vocab size exceeds the max supported ("
        << utils::kMaxSupportedVocabSize
        << "). Found vocab size: " << vocab.size();
    throw std::invalid_argument(oss.str());
  }
  failure_vocab_tokens_.reserve(vocab.size());
  int unk_id = vocab.at(unk_token);
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
    // Skip the word which contains punctuation but not a punctuation.
    if (with_pretokenization_ && vocab_token.ContainsPunctuation() &&
        (vocab_token.IsSuffixToken() ||
         vocab_token.TokenUnicodeLengthWithoutContinuingSubwordPrefix() > 1)) {
      continue;
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
      failure_vocab_tokens_.emplace_back(
          new_suffix_token, unk_id, continuing_subword_prefix);
    }
  }
  if (with_pretokenization_) {
    for (uint32_t cp = 1; cp <= 0x0010FFFF; ++cp) {
      if (!utils::IsUnicodeChar(cp) || !utils::IsPunctuationOrChineseChar(cp)) {
        continue;
      }
      char utf_str[5];
      utils::GetUTF8Str(reinterpret_cast<char32_t*>(&cp), utf_str, 1);
      std::string punc_str(utf_str);
      if (vocab.count(punc_str) == 0) {
        failure_vocab_tokens_.emplace_back(
            punc_str, unk_id, continuing_subword_prefix);
      }
    }
    failure_vocab_tokens_.emplace_back(
        std::string(1, kInvalidControlChar), unk_id, continuing_subword_prefix);
  }
}

void FailureArray::CreateVocabFromFailureVocab(
    const std::vector<FailureVocabToken>& failure_vocab_tokens,
    std::unordered_map<std::string, uint32_t>* vocab) const {
  for (auto&& failure_vocab : failure_vocab_tokens) {
    (*vocab)[failure_vocab.Token()] = failure_vocab.TokenId();
  }
}

void FailureArray::InitFromVocabAndTrie(
    const std::unordered_map<std::string, uint32_t>& vocab,
    Trie* trie,
    const std::string& unk_token,
    const std::string& continuing_subword_prefix) {
  BuildFailureVocab(vocab, unk_token, continuing_subword_prefix);

  // Create Trie
  std::unordered_map<std::string, uint32_t> new_vocab;
  CreateVocabFromFailureVocab(failure_vocab_tokens_, &new_vocab);
  trie->SetVocab(new_vocab);

  // Create failure array
  BuildFailureArray(failure_vocab_tokens_, trie);
}

void FailureArray::RemovePunctuationTrieLink(Trie* trie) const {
  auto continuing_subword_prefix = trie->GetContinuingSubwordPrefix();
  if (with_pretokenization_ && !continuing_subword_prefix.empty()) {
    int cur_idx = 0;
    int next_idx = 0;
    uint32_t curr_char, next_char;
    bool prev_node_is_root = false;
    auto node = trie->CreateRootTraversalCursor();
    while (cur_idx < continuing_subword_prefix.length()) {
      next_idx = cur_idx;
      auto chwidth = utils::UTF8ToUInt32(
          continuing_subword_prefix.data() + next_idx, &curr_char);
      curr_char = utils::UTF8ToUnicode(curr_char);
      next_idx = cur_idx + chwidth;
      prev_node_is_root = (node.node_id_ == trie->kRootNodeId);
      std::string cur_unicode_char(continuing_subword_prefix.data() + cur_idx,
                                   chwidth);
      if (!trie->TryTraverseSeveralSteps(&node, cur_unicode_char)) {
        throw std::runtime_error(
            "Cannot locate a character in suffix_indicator_. It should never "
            "happen.");
      }
      if (IsPunctuationOrChineseChar(curr_char)) {
        if (prev_node_is_root) {
          cur_idx = next_idx;
          auto next_chwidth = utils::UTF8ToUInt32(
              continuing_subword_prefix.data() + next_idx, &next_char);
          next_idx += next_chwidth;
          std::string next_unicode_char(
              continuing_subword_prefix.data() + cur_idx, next_chwidth);
          auto child_node = node;
          if (!trie->TryTraverseSeveralSteps(&child_node, next_unicode_char)) {
            throw std::runtime_error(
                "Cannot locate a character in suffix_indicator_. It should "
                "never happen.");
          }
          trie->DeleteLinkFromParent(child_node.node_id_);
        } else {
          trie->DeleteLinkFromParent(node.node_id_);
        }
        break;
      }
      cur_idx = next_idx;
    }
  }
}

// Algorithm 2 in https://arxiv.org/pdf/2012.15524.pdf
void FailureArray::BuildFailureArray(
    const std::vector<FailureVocabToken>& failure_vocab_tokens, Trie* trie) {
  std::vector<std::unordered_set<char>> node_outgoing_edge_labels;
  BuildOutgoingEdgeLabelsForTrie(
      failure_vocab_tokens, trie, &node_outgoing_edge_labels);
  failure_array_.resize(trie->Size());
  std::queue<uint32_t> trie_node_queue({trie->kRootNodeId});
  if (trie->GetSuffixRoot() != trie->kRootNodeId) {
    trie_node_queue.push(trie->GetSuffixRoot());
  }
  while (!trie_node_queue.empty()) {
    uint32_t parent_id = trie_node_queue.front();
    trie_node_queue.pop();
    std::vector<char> outgoing_labels_sorted(
        node_outgoing_edge_labels[parent_id].begin(),
        node_outgoing_edge_labels[parent_id].end());
    std::sort(outgoing_labels_sorted.begin(), outgoing_labels_sorted.end());
    for (const char edge_label : outgoing_labels_sorted) {
      auto child_node = trie->CreateTraversalCursor(parent_id);
      if (!trie->TryTraverseOneStep(&child_node, edge_label)) {
        std::ostringstream oss;
        oss << "Failed to traverse to child following edge " << edge_label
            << " at parent " << parent_id << ".";
        throw std::runtime_error(oss.str());
      }
      if (child_node.node_id_ == trie->GetSuffixRoot()) {
        continue;
      }
      int child_data_value = -1;
      // Case 1: str(v) in V
      //  * f(v) = trie.GetSuffixRoot()
      //  * F(v) = [str(v)]
      if (trie->TryGetData(child_node, &child_data_value)) {
        uint32_t failure_link = trie->GetSuffixRoot();
        if (node_id_is_punc_map_.count(child_node.node_id_) == 0) {
          throw std::invalid_argument(
              "Failed to find if an end node in the trie is a punctuation char "
              "in node_id_is_punc_map_. It should never happen.");
        }
        if (with_pretokenization_ &&
            node_id_is_punc_map_.at(child_node.node_id_)) {
          failure_link = trie->GetPuncFailureNode();
        }
        AssignFailureLinkAndPops(child_node.node_id_,
                                 failure_link,
                                 {child_data_value},
                                 utils::kNullFailurePopsList);
        trie_node_queue.push(child_node.node_id_);
        continue;
      }

      // Case 2: str(v) is not in V
      const Failure& parent_failure = failure_array_[parent_id];
      if (parent_failure.failure_link_ != utils::kNullNode) {
        std::vector<int> one_step_pops;
        auto curr_node =
            trie->CreateTraversalCursor(parent_failure.failure_link_);
        // Find the failure link util the failure link is root or
        // the node has the outgoing label correspoding to edge_label.
        while (true) {
          if (trie->TryTraverseOneStep(&curr_node, edge_label)) {
            AssignFailureLinkAndPops(
                child_node.node_id_,
                curr_node.node_id_,
                one_step_pops,
                parent_failure.failure_pops_offset_length_);
            break;
          }
          const Failure& curr_node_failure = failure_array_[curr_node.node_id_];
          if (curr_node_failure.failure_link_ == utils::kNullNode) {
            break;
          }
          GetFailurePopsAndAppendToOut(
              curr_node_failure.failure_pops_offset_length_, &one_step_pops);
          trie->SetTraversalCursor(&curr_node, curr_node_failure.failure_link_);
        }
      }
      // If the failure_link of parent is root,
      // * f(v) = none
      // * F(v) = []
      trie_node_queue.push(child_node.node_id_);
    }
  }
  RemovePunctuationTrieLink(trie);
}

void FailureArray::AssignFailureLinkAndPops(
    uint32_t cur_node,
    uint32_t failure_link,
    const std::vector<int>& one_step_pops,
    int parent_failure_pops_offset_length) {
  if (failure_link == utils::kNullNode) {
    return;
  }
  auto& curr_node_failure = failure_array_[cur_node];
  curr_node_failure.failure_link_ = failure_link;
  if (one_step_pops.empty()) {
    curr_node_failure.failure_pops_offset_length_ =
        parent_failure_pops_offset_length;
  } else {
    const int offset = failure_pops_pool_.size();
    if (offset > utils::kMaxSupportedFailurePoolOffset) {
      std::ostringstream oss;
      oss << "Failure pops list offset is " << offset
          << ", which exceeds maximum supported offset "
          << utils::kMaxSupportedFailurePoolOffset
          << ". The vocabulary seems to be too large to be supported.";
      throw std::runtime_error(oss.str());
    }
    GetFailurePopsAndAppendToOut(parent_failure_pops_offset_length,
                                 &failure_pops_pool_);
    failure_pops_pool_.insert(
        failure_pops_pool_.end(), one_step_pops.begin(), one_step_pops.end());
    const int length = failure_pops_pool_.size() - offset;
    if (length > utils::kMaxSupportedFailurePoolOffset) {
      std::ostringstream oss;
      oss << "Failure pops list size is " << length
          << ", which exceeds maximum supported offset "
          << utils::kMaxFailurePopsListSize;
      throw std::runtime_error(oss.str());
    }
    curr_node_failure.failure_pops_offset_length_ =
        utils::EncodeFailurePopList(offset, length);
  }
}

void FailureArray::GetFailurePopsAndAppendToOut(
    uint32_t failure_pops_offset_length, std::vector<int>* out_failure_pops) {
  if (failure_pops_offset_length == utils::kNullFailurePopsList) {
    return;
  }
  int offset = 0, length = 0;
  utils::GetFailurePopsOffsetAndLength(
      failure_pops_offset_length, &offset, &length);
  out_failure_pops->insert(out_failure_pops->end(),
                           failure_pops_pool_.begin() + offset,
                           failure_pops_pool_.begin() + offset + length);
}

void FailureArray::BuildOutgoingEdgeLabelsForTrie(
    const std::vector<FailureVocabToken>& failure_vocab_tokens,
    Trie* trie,
    std::vector<std::unordered_set<char>>* node_outgoing_edge_labels) {
  node_outgoing_edge_labels->resize(trie->Size());
  const std::string dummy_token = std::string(1, utils::kInvalidControlChar);
  for (auto& item : failure_vocab_tokens) {
    if (item.Token() != dummy_token) {
      BuildOutgoingEdgeLabelsFromToken(item, trie, node_outgoing_edge_labels);
    }
  }
}

void FailureArray::BuildOutgoingEdgeLabelsFromToken(
    const FailureVocabToken& vocab_token,
    Trie* trie,
    std::vector<std::unordered_set<char>>* node_outgoing_edge_labels) {
  const std::string& token = vocab_token.Token();
  Trie::TraversalCursor curr_node;
  int char_pos = 0;
  trie->SetTraversalCursor(&curr_node, Trie::kRootNodeId);
  while (char_pos < token.size()) {
    const char edge_label = token[char_pos];
    (*node_outgoing_edge_labels)[curr_node.node_id_].insert(edge_label);
    if (!trie->TryTraverseOneStep(&curr_node, edge_label)) {
      std::ostringstream oss;
      oss << "Error in traversing to child following edge `" << edge_label
          << "` from the prefix `" << token.substr(0, char_pos)
          << "` at parent id " << curr_node.node_id_ << ". The token is `"
          << token << "`. The char position"
          << " is " << char_pos << ".";

      throw std::runtime_error(oss.str());
    }
    ++char_pos;
  }
  node_id_is_punc_map_[curr_node.node_id_] =
      !vocab_token.IsSuffixToken() && vocab_token.ContainsPunctuation() &&
      vocab_token.TokenUnicodeLengthWithoutContinuingSubwordPrefix() == 1;
}


}  // namespace utils
}  // namespace faster_tokenizer
}  // namespace paddlenlp
