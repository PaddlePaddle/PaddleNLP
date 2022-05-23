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
namespace tokenizers {
namespace utils {

void Trie::CreateTrie(const std::vector<std::string>& keys,
                      const std::vector<uint>& values) {
  std::vector<const char*> char_keys(keys.size());
  for (int i = 0; i < keys.size(); ++i) {
    char_keys[i] = keys[i].c_str();
  }
  trie_ = std::make_shared<DoubleArrayUint>();
  trie_->build(keys.size(),
               const_cast<char**>(&char_keys[0]),
               nullptr,
               const_cast<uint*>(&values[0]));
}

Trie::Trie(const std::unordered_map<std::string, uint>& vocab) {
  std::vector<std::string> keys;
  std::vector<uint> values;
  for (auto&& item : vocab) {
    keys.push_back(item.first);
    values.push_back(item.second);
  }
  CreateTrie(keys, values);
}

Trie::Trie(const std::vector<std::string>& keys) {
  std::vector<uint> values(keys.size());
  std::iota(values.begin(), values.end(), 0);
  CreateTrie(keys, values);
}

}  // namespace utils
}  // namespace tokenizers