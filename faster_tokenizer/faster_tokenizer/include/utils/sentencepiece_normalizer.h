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

#include "utils/string_view.h"

#include "darts.h"

namespace paddlenlp {
namespace faster_tokenizer {
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

class Normalizer {
public:
  // Instantiates Normalizer with |spec|.
  // |spec| should not be deleted until Normalizer is destroyed.
  explicit Normalizer(const std::string& precompiled_charsmap);
  Normalizer(const Normalizer& other);
  virtual ~Normalizer();

  virtual void SetPrefixMatcher(const PrefixMatcher* matcher) {
    matcher_ = matcher;
  }

  virtual bool Normalize(const char* input,
                         size_t input_len,
                         std::string* normalized,
                         std::vector<int>* norm_to_orig,
                         std::u32string* u32content = nullptr) const;
  std::string GetPrecompiledCharsmap() const;

private:
  void Init();
  void Replace(const simple_string_view& new_part,
               const simple_string_view& old_part,
               std::vector<int>* changes,
               std::u32string* u32content = nullptr) const;
  std::pair<simple_string_view, int> NormalizePrefix(const char* input,
                                                     size_t input_len) const;


  // // Encodes trie_blob and normalized string and return compiled blob.
  static std::string EncodePrecompiledCharsMap(const std::string& trie_blob,
                                               const std::string& normalized);

  // Decodes blob into trie_blob and normalized string.
  static void DecodePrecompiledCharsMap(const char* blob,
                                        size_t blob_size,
                                        std::string* trie_blob,
                                        std::string* normalized,
                                        std::string* buffer = nullptr);

  static constexpr int kMaxTrieResultsSize = 32;

  std::unique_ptr<Darts::DoubleArray> trie_;

  const char* normalized_ = nullptr;
  std::string normalized_blob_;
  std::string trie_blob_;

  // Prefix matcher;
  const PrefixMatcher* matcher_ = nullptr;

  // Split hello world into "hello_" and "world_" instead of
  // "_hello" and "_world".
  const bool treat_whitespace_as_suffix_ = false;
  std::string precompiled_charsmap_buffer_;
  std::string precompiled_charsmap_;
};

}  // namespace utils
}  // namespace faster_tokenizer
}  // namespace paddlenlp
