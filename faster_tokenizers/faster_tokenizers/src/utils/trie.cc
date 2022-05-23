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

#include "utils/trie.h"
#include "utils/utils.h"

namespace tokenizers {
namespace utils {

void Trie::CreateTrie(const std::vector<std::string>& keys,
                      const std::vector<int>& values) {
  std::vector<const char*> char_keys(keys.size());
  for (int i = 0; i < keys.size(); ++i) {
    char_keys[i] = keys[i].c_str();
  }
  trie_ = std::make_shared<Darts::DoubleArray>();
  trie_->build(keys.size(),
               const_cast<char**>(&char_keys[0]),
               nullptr,
               const_cast<int*>(&values[0]));
  trie_array_ = static_cast<const uint*>(trie_->array());
}

void Trie::SetVocab(const std::unordered_map<std::string, uint>& vocab) {
  std::vector<std::string> keys;
  std::vector<int> values;
  for (auto&& item : vocab) {
    keys.push_back(item.first);
    bool is_suffix_token = (item.first.rfind(continuing_subword_prefix_) == 0);
    uint token_length = item.first.length();
    if (is_suffix_token) {
      token_length -= continuing_subword_prefix_.length();
    }
    values.push_back(EncodeToken(item.second, token_length, is_suffix_token));
  }
  CreateTrie(keys, values);
}


Trie::Trie(const std::unordered_map<std::string, uint>& vocab,
           const std::string& continuing_subword_prefix)
    : continuing_subword_prefix_(continuing_subword_prefix) {
  SetVocab(vocab);
}

Trie::Trie(const std::vector<std::string>& keys,
           const std::string& continuing_subword_prefix)
    : continuing_subword_prefix_(continuing_subword_prefix) {
  std::vector<int> values(keys.size());
  std::iota(values.begin(), values.end(), 0);
  CreateTrie(keys, values);
}

Trie::TraversalCursor Trie::CreateRootTraversalCursor() {
  return CreateTraversalCursor(kRootNodeId);
}

Trie::TraversalCursor Trie::CreateTraversalCursor(uint32_t node_id) {
  return Trie::TraversalCursor(node_id, trie_array_[node_id]);
}

void Trie::SetTraversalCursor(Trie::TraversalCursor* cursor, uint32_t node_id) {
  cursor->node_id_ = node_id;
  cursor->unit_ = trie_array_[node_id];
}

bool Trie::TryTraverseOneStep(Trie::TraversalCursor* cursor,
                              unsigned char ch) const {
  const uint32_t next_node_id = cursor->node_id_ ^ Offset(cursor->unit_) ^ ch;
  const uint32_t next_node_unit = trie_array_[next_node_id];
  if (Label(next_node_unit) != ch) {
    return false;
  }
  cursor->node_id_ = next_node_id;
  cursor->unit_ = next_node_unit;
  return true;
}

bool Trie::TryTraverseSeveralSteps(Trie::TraversalCursor* cursor,
                                   const std::string& path) const {
  return TryTraverseSeveralSteps(cursor, path.data(), path.size());
}

bool Trie::TryTraverseSeveralSteps(Trie::TraversalCursor* cursor,
                                   const char* ptr,
                                   int size) const {
  uint32_t cur_id = cursor->node_id_;
  uint32_t cur_unit = cursor->unit_;
  for (; size > 0; --size, ++ptr) {
    const unsigned char ch = static_cast<const unsigned char>(*ptr);
    cur_id ^= Offset(cur_unit) ^ ch;
    cur_unit = trie_array_[cur_id];
    if (Label(cur_unit) != ch) {
      return false;
    }
  }
  cursor->node_id_ = cur_id;
  cursor->unit_ = cur_unit;
  return true;
}

bool Trie::TryGetData(const Trie::TraversalCursor& cursor,
                      int& out_data) const {
  if (!HasLeaf(cursor.unit_)) {
    return false;
  }
  const uint32_t value_unit =
      trie_array_[cursor.node_id_ ^ Offset(cursor.unit_)];
  out_data = Value(value_unit);
  return true;
}

}  // namespace utils
}  // namespace tokenizers