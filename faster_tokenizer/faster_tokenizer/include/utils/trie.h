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
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "darts.h"

namespace tokenizers {
namespace utils {

class Trie {
public:
  static constexpr uint32_t kRootNodeId = 0;

  Trie(const std::string& continuing_subword_prefix = "##",
       const std::string& unk_token = "[UNK]",
       bool with_pretokenization = false);
  Trie(const std::unordered_map<std::string, uint>& vocab,
       const std::string& continuing_subword_prefix = "##",
       const std::string& unk_token = "[UNK]",
       bool with_pretokenization = false);
  Trie(const std::vector<std::string>& keys,
       const std::string& continuing_subword_prefix = "##",
       const std::string& unk_token = "[UNK]",
       bool with_pretokenization = false);
  struct TraversalCursor {
    uint32_t node_id_;
    uint32_t unit_;
    TraversalCursor(uint32_t node_id = 0, uint32_t unit = 0)
        : node_id_(node_id), unit_(unit) {}
  };

  TraversalCursor CreateRootTraversalCursor() const;
  TraversalCursor CreateTraversalCursor(uint32_t node_id) const;
  void SetTraversalCursor(TraversalCursor* cursor, uint32_t node_id) const;
  bool TryTraverseOneStep(TraversalCursor* cursor, unsigned char ch) const;
  bool TryTraverseSeveralSteps(TraversalCursor* cursor,
                               const std::string& path) const;
  bool TryGetData(const TraversalCursor& cursor, int* out_data) const;
  void SetVocab(const std::unordered_map<std::string, uint>& vocab);
  void SetVocabList(const std::vector<std::string>& vocab);
  void SetWithPretokenization(bool with_pretokenization_);
  void SetUNKToken(const std::string& unk_token);
  void SetContinuingSubwordPrefix(const std::string& continuing_subword_prefix);

  uint Size() const {
    if (trie_.get() != nullptr) {
      return trie_->size();
    }
    return 0;
  }
  std::string GetContinuingSubwordPrefix() const {
    return continuing_subword_prefix_;
  }
  uint GetSuffixRoot() const { return suffix_root_; }
  uint GetPuncFailureNode() const { return punct_failure_link_node_; }
  void DeleteValueOfNode(uint32_t node_id);
  void DeleteLinkFromParent(uint32_t child_node_id);

private:
  void AddPuncVocab(std::vector<std::string>* punc_vocab,
                    const std::unordered_map<std::string, uint>& vocab) const;
  void InitTrieSuffixRoot();
  void InitTrie(const std::vector<const char*>& keys,
                const std::vector<int>& values);
  void GetSortedVocab(const std::vector<const char*>& keys,
                      const std::vector<int>& values,
                      std::vector<const char*>* sorted_keys,
                      std::vector<int>* sorted_values) const;
  int EncodeTokenId(const std::string& token, uint id) const;
  void CreateTrie(const std::vector<const char*>& keys,
                  const std::vector<int>& values);

  bool TryTraverseSeveralSteps(TraversalCursor* cursor,
                               const char* ptr,
                               int size) const;

  static uint32_t Offset(uint32_t unit) {
    return (unit >> 10) << ((unit & 0x200) >> 6);
  }

  // Returns a label associated with a node.
  // A leaf node will have the MSB set and thus return an invalid label.
  static uint32_t Label(uint32_t unit) { return unit & 0x800000ff; }

  // Returns whether a node has a leaf as a child.
  static bool HasLeaf(uint32_t unit) { return unit & 0x100; }

  // Returns a value associated with a node. Available when a node is a leaf.
  static int Value(uint32_t unit) {
    return static_cast<int>(unit & 0x7fffffff);
  }

  std::shared_ptr<Darts::DoubleArray> trie_;
  std::vector<uint> trie_array_;
  std::string continuing_subword_prefix_;
  std::string unk_token_;
  uint suffix_root_;
  uint punct_failure_link_node_;
  bool with_pretokenization_;
};

}  // namespace utils
}  // namespace tokenizers