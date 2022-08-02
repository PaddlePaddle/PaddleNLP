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

#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace paddlenlp {
namespace faster_tokenizer {
namespace utils {

class Trie;

// Used in Faster WordPiece Model specially
struct Failure {
  uint32_t failure_link_;
  // Indicate the number of failure_pops
  // and the offset in failure_pops_pool
  uint32_t failure_pops_offset_length_;
  Failure();
};

class FailureVocabToken {
public:
  FailureVocabToken(const std::string& token,
                    int token_id,
                    const std::string& continuing_subword_prefix);

  const std::string& Token() const;

  int TokenId() const;
  bool IsSuffixToken() const;
  bool ContainsPunctuation() const;
  int TokenUnicodeLengthWithoutContinuingSubwordPrefix() const;
  int TokenLengthWithoutContinuingSubwordPrefix() const;

private:
  std::string token_;
  int token_id_;
  bool is_suffix_token_;
  int actual_token_start_offset_;
  int actual_token_unicode_len_;
  bool contains_punctuation_;
};

struct FailureArray {
  FailureArray(bool with_pretokenization = false)
      : with_pretokenization_(with_pretokenization) {}
  void BuildFailureArray(
      const std::vector<FailureVocabToken>& failure_vocab_tokens, Trie* trie);
  void BuildFailureVocab(const std::unordered_map<std::string, uint32_t>& vocab,
                         const std::string& unk_token,
                         const std::string& continuing_subword_prefix);
  void InitFromVocabAndTrie(
      const std::unordered_map<std::string, uint32_t>& vocab,
      Trie* trie,
      const std::string& unk_token,
      const std::string& continuing_subword_prefix);
  const Failure* GetFailure(int idx) const { return &(failure_array_.at(idx)); }
  int GetFailurePop(int idx) const { return failure_pops_pool_.at(idx); }
  void SetWithPretokenization(bool with_pretokenization) {
    with_pretokenization_ = with_pretokenization;
  }

private:
  void BuildOutgoingEdgeLabelsForTrie(
      const std::vector<FailureVocabToken>& failure_vocab_tokens,
      Trie* trie,
      std::vector<std::unordered_set<char>>* node_outgoing_edge_labels);
  void BuildOutgoingEdgeLabelsFromToken(
      const FailureVocabToken& vocab_token,
      Trie* trie,
      std::vector<std::unordered_set<char>>* node_outgoing_edge_labels);
  void AssignFailureLinkAndPops(uint32_t cur_node,
                                uint32_t failure_link,
                                const std::vector<int>& one_step_pops,
                                int parent_failure_pops_offset_length);
  void GetFailurePopsAndAppendToOut(uint32_t failure_pops_offset_length,
                                    std::vector<int>* out_failure_pops);
  void RemovePunctuationTrieLink(Trie* trie) const;
  void CreateVocabFromFailureVocab(
      const std::vector<FailureVocabToken>& failure_vocab_tokens,
      std::unordered_map<std::string, uint32_t>* vocab) const;
  std::vector<Failure> failure_array_;
  std::vector<int> failure_pops_pool_;
  std::unordered_map<uint32_t, bool> node_id_is_punc_map_;
  std::vector<FailureVocabToken> failure_vocab_tokens_;
  bool with_pretokenization_;  // The end-to-end version of FailureArray
};

}  // namespace utils
}  // namespace faster_tokenizer
}  // namespace paddlenlp
