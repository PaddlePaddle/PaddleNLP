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
#include <cstring>
#include <numeric>

#include "glog/logging.h"
#include "utils/trie.h"
#include "utils/utf8.h"
#include "utils/utils.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace utils {

void Trie::CreateTrie(const std::vector<const char*>& keys,
                      const std::vector<int>& values) {
  trie_ = std::make_shared<Darts::DoubleArray>();
  trie_->build(keys.size(),
               const_cast<char**>(&keys[0]),
               nullptr,
               const_cast<int*>(&values[0]));
  const uint32_t* trie_ptr = reinterpret_cast<const uint32_t*>(trie_->array());
  trie_array_ = std::vector<uint32_t>(trie_ptr, trie_ptr + trie_->size());
}

int Trie::EncodeTokenId(const std::string& token, uint32_t id) const {
  bool is_suffix_token = (token.rfind(continuing_subword_prefix_) == 0);
  uint32_t token_length = token.length();
  if (is_suffix_token) {
    token_length -= continuing_subword_prefix_.length();
  }
  return EncodeToken(id, token_length, is_suffix_token);
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
  if (with_pretokenization_ && keys.size() > 0) {
    auto node = CreateRootTraversalCursor();
    if (!TryTraverseSeveralSteps(&node,
                                 std::string(1, utils::kInvalidControlChar))) {
      throw std::runtime_error(
          "Cannot locate the dummy node for the failure link for punctuation "
          "nodes. This should never happen.");
    }
    punct_failure_link_node_ = node.node_id_;
    DeleteLinkFromParent(punct_failure_link_node_);
    DeleteValueOfNode(punct_failure_link_node_);
  }
}

void Trie::AddPuncVocab(
    std::vector<std::string>* punc_vocab,
    const std::unordered_map<std::string, uint32_t>& vocab) const {
  if (with_pretokenization_) {
    for (uint32_t cp = 1; cp <= 0x0010FFFF; ++cp) {
      if (!utils::IsUnicodeChar(cp) || !utils::IsPunctuationOrChineseChar(cp)) {
        continue;
      }
      char utf_str[5];
      utils::GetUTF8Str(reinterpret_cast<char32_t*>(&cp), utf_str, 1);
      std::string punc_str(utf_str);
      if (vocab.count(punc_str) == 0) {
        punc_vocab->push_back(punc_str);
      }
    }
    punc_vocab->push_back(std::string(1, utils::kInvalidControlChar));
  }
}

void Trie::SetVocab(const std::unordered_map<std::string, uint32_t>& vocab) {
  std::vector<const char*> keys;
  std::vector<int> values;
  for (auto&& item : vocab) {
    keys.push_back(item.first.c_str());
    values.push_back(EncodeTokenId(item.first, item.second));
  }
  InitTrie(keys, values);
}

void Trie::SetVocabList(const std::vector<std::string>& keys) {
  std::unordered_map<std::string, uint32_t> vocab;
  for (int i = 0; i < keys.size(); ++i) {
    vocab[keys[i]] = i;
  }
  SetVocab(vocab);
}

Trie::Trie(const std::string& continuing_subword_prefix,
           const std::string& unk_token,
           bool with_pretokenization)
    : trie_(nullptr),
      continuing_subword_prefix_(continuing_subword_prefix),
      suffix_root_(utils::kNullNode),
      punct_failure_link_node_(utils::kNullNode),
      unk_token_(unk_token),
      with_pretokenization_(with_pretokenization) {}

Trie::Trie(const std::unordered_map<std::string, uint32_t>& vocab,
           const std::string& continuing_subword_prefix,
           const std::string& unk_token,
           bool with_pretokenization)
    : continuing_subword_prefix_(continuing_subword_prefix),
      unk_token_(unk_token),
      suffix_root_(utils::kNullNode),
      punct_failure_link_node_(utils::kNullNode),
      with_pretokenization_(with_pretokenization) {
  SetVocab(vocab);
}

Trie::Trie(const std::vector<std::string>& keys,
           const std::string& continuing_subword_prefix,
           const std::string& unk_token,
           bool with_pretokenization)
    : continuing_subword_prefix_(continuing_subword_prefix),
      unk_token_(unk_token),
      suffix_root_(utils::kNullNode),
      punct_failure_link_node_(utils::kNullNode),
      with_pretokenization_(with_pretokenization) {
  SetVocabList(keys);
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

void Trie::DeleteValueOfNode(uint32_t node_id) {
  trie_array_[node_id] &= 0xFFFFFEFF;
}

void Trie::DeleteLinkFromParent(uint32_t child_node_id) {
  trie_array_[child_node_id] &= 0xFFFFFF00;
}

void Trie::SetWithPretokenization(bool with_pretokenization) {
  with_pretokenization_ = with_pretokenization;
}

void Trie::SetUNKToken(const std::string& unk_token) { unk_token_ = unk_token; }

void Trie::SetContinuingSubwordPrefix(
    const std::string& continuing_subword_prefix) {
  continuing_subword_prefix_ = continuing_subword_prefix;
}

}  // namespace utils
}  // namespace faster_tokenizer
}  // namespace paddlenlp
