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

#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace tokenizers {
namespace utils {

class Trie;

// Used in Faster WordPiece Model specially
struct Failure {
  uint failure_link_;
  uint failure_pops_offset_length_;
  Failure();
};

struct FailureArray {
  FailureArray() = default;
  FailureArray(const std::unordered_map<std::string, uint>& vocab,
               const Trie& trie);
  void BuildFailureArray(const std::unordered_map<std::string, uint>& vocab,
                         const Trie& trie);

  std::vector<Failure> failure_array_;
  std::vector<int> failure_pops_pool_;
  std::unordered_map<uint32_t, bool> node_id_is_punc_map_;
};

}  // namespace utils
}  // namespace tokenizers