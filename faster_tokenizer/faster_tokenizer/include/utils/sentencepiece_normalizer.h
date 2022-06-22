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

#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "darts.h"

namespace tokenizers {
namespace utils {

struct Cstrless {
  bool operator()(const char* a, const char* b) const {
    return std::strcmp(a, b) < 0;
  }
};

class PrefixMatcher {
public:
  // Initializes the PrefixMatcher with `dic`.
  explicit PrefixMatcher(const std::set<const char*, Cstrless>& dic);

  int PrefixMatch(const char* w, size_t w_len, bool* found = nullptr) const;

  std::string GlobalReplace(const char* w,
                            size_t w_len,
                            const char* out,
                            size_t out_len,
                            const char** result_w) const;

private:
  std::unique_ptr<Darts::DoubleArray> trie_;
};

}  // namespace utils
}  // namespace tokenizers