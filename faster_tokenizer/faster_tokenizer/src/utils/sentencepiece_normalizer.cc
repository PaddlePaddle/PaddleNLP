// Copyright 2016 Google Inc.
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

#include "utils/sentencepiece_normalizer.h"

namespace tokenizers {
namespace utils {

PrefixMatcher::PrefixMatcher(const std::set<const char*, Cstrless>& dic) {
  if (dic.empty()) return;
  std::vector<const char*> key;
  key.reserve(dic.size());
  for (const auto& it : dic) key.push_back(it);
  trie_ = std::unique_ptr<Darts::DoubleArray>(new Darts::DoubleArray());
  trie_->build(key.size(), const_cast<char**>(&key[0]), nullptr, nullptr);
}

inline size_t OneCharLen(const char* src) {
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

int PrefixMatcher::PrefixMatch(const char* w,
                               size_t w_len,
                               bool* found = nullptr) const {
  if (trie_ == nullptr) {
    if (found) *found = false;
    return std::min<int>(w_len, OneCharLen(w));
  }
  constexpr int kResultSize = 64;
  Darts::DoubleArray::result_pair_type trie_results[kResultSize];
  const int num_nodes =
      trie_->commonPrefixSearch(w, trie_results, kResultSize, w_len);
  if (found) *found = (num_nodes > 0);
  if (num_nodes == 0) {
    return std::min<int>(w_len, OneCharLen(w));
  }

  int mblen = 0;
  for (int i = 0; i < num_nodes; ++i) {
    mblen = std::max<int>(trie_results[i].length, mblen);
  }
  return mblen;
}

std::string PrefixMatcher::GlobalReplace(const char* w,
                                         size_t w_len,
                                         const char* out,
                                         size_t out_len,
                                         const char** result_w) const {
  std::string result;
  if (w_len > 0) {
    bool found = false;
    const int mblen = PrefixMatch(w, w_len, &found);
    if (found) {
      result.append(out, out_len);
    } else {
      result.append(w, mblen);
    }
    *result_w = w + mblen;
  }
  return result;
}

}  // namespace utils
}  // namespace tokenizers