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
#include <cstring>
#include <numeric>

#include "glog/logging.h"
#include "utils/trie.h"
#include "utils/utils.h"

namespace tokenizers {
namespace utils {

void Trie::CreateTrie(const std::vector<const char*>& keys,
                      const std::vector<int>& values) {
  trie_ = std::make_shared<Darts::DoubleArray>();
  trie_->build(keys.size(),
               const_cast<char**>(&keys[0]),
               nullptr,
               const_cast<int*>(&values[0]));
  trie_array_ = static_cast<const uint*>(trie_->array());
}

int Trie::EncodeTokenId(const std::string& token, uint id) const {
  bool is_suffix_token = (token.rfind(continuing_subword_prefix_) == 0);
  uint token_length = token.length();
  if (is_suffix_token) {
    token_length -= continuing_subword_prefix_.length();
  }
  return EncodeToken(id, token_length, is_suffix_token);
}

void Trie::GetSortedVocab(const std::vector<const char*>& keys,
                          const std::vector<int>& values,
                          std::vector<const char*>* sorted_keys,
                          std::vector<int>* sorted_values) const {
  // Sort the vocab
  std::vector<int> sorted_vocab_index(keys.size());
  std::iota(sorted_vocab_index.begin(), sorted_vocab_index.end(), 0);
  std::sort(sorted_vocab_index.begin(),
            sorted_vocab_index.end(),
            [&keys](const int a, const int b) {
              return std::strcmp(keys[a], keys[b]) < 0;
            });

  sorted_keys->resize(keys.size());
  sorted_values->resize(keys.size());
  for (int i = 0; i < sorted_vocab_index.size(); ++i) {
    auto idx = sorted_vocab_index[i];
    (*sorted_keys)[i] = keys[idx];
    (*sorted_values)[i] = values[idx];
  }
}
void Trie::InitTrieSuffixRoot() {
  auto node = CreateRootTraversalCursor();
  if (!TryTraverseSeveralSteps(&node, continuing_subword_prefix_)) {
    throw std::runtime_error(
        "Cannot locate suffix_root_. This should never happen.");
  }
  suffix_root_ = node.node_id_;
}

void Trie::InitTrie(const std::vector<const char*>& keys,
                    const std::vector<int>& values) {
  std::vector<const char*> sorted_keys;
  std::vector<int> sorted_values;
  GetSortedVocab(keys, values, &sorted_keys, &sorted_values);
  CreateTrie(sorted_keys, sorted_values);
  InitTrieSuffixRoot();
}

void Trie::SetVocab(const std::unordered_map<std::string, uint>& vocab) {
  std::vector<const char*> keys;
  std::vector<int> values;
  for (auto&& item : vocab) {
    keys.push_back(item.first.c_str());
    values.push_back(EncodeTokenId(item.first, item.second));
  }
  InitTrie(keys, values);
}

Trie::Trie(const std::string& continuing_subword_prefix, const std::string& unk_token)
      : trie_(nullptr),
        trie_array_(nullptr),
        continuing_subword_prefix_(continuing_subword_prefix),
        suffix_root_(utils::kNullNode),
        punct_failure_link_node_(utils::kNullNode),
        unk_token_(unk_token) {}

Trie::Trie(const std::unordered_map<std::string, uint>& vocab,
           const std::string& continuing_subword_prefix,
           const std::string& unk_token)
    : continuing_subword_prefix_(continuing_subword_prefix),
      unk_token_(unk_token),
      suffix_root_(utils::kNullNode),
      punct_failure_link_node_(utils::kNullNode) {
  unk_token_id_ = vocab.at(unk_token);
  SetVocab(vocab);
}

Trie::Trie(const std::vector<std::string>& keys,
           const std::string& continuing_subword_prefix,
           const std::string& unk_token)
    : continuing_subword_prefix_(continuing_subword_prefix),
      unk_token_(unk_token),
      suffix_root_(utils::kNullNode),
      punct_failure_link_node_(utils::kNullNode) {
  std::vector<const char*> char_keys(keys.size());
  for (int i = 0; i < keys.size(); ++i) {
    char_keys[i] = keys[i].c_str();
    if (keys[i] == unk_token_) {
      unk_token_id_ = i;
    }
  }
  std::vector<int> values(keys.size());
  std::iota(values.begin(), values.end(), 0);
  // Encode value
  for (int i = 0; i < keys.size(); ++i) {
    values[i] = EncodeTokenId(keys[i], values[i]);
  }
  InitTrie(char_keys, values);
}

Trie::TraversalCursor Trie::CreateRootTraversalCursor() const {
  return CreateTraversalCursor(kRootNodeId);
}

Trie::TraversalCursor Trie::CreateTraversalCursor(uint32_t node_id) const {
  return Trie::TraversalCursor(node_id, trie_array_[node_id]);
}

void Trie::SetTraversalCursor(Trie::TraversalCursor* cursor,
                              uint32_t node_id) const {
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
                      int* out_data) const {
  if (!HasLeaf(cursor.unit_)) {
    return false;
  }
  const uint32_t value_unit =
      trie_array_[cursor.node_id_ ^ Offset(cursor.unit_)];
  *out_data = Value(value_unit);
  return true;
}

}  // namespace utils
}  // namespace tokenizers