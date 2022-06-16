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
#include <codecvt>
#include <fstream>
#include <locale>
#include <map>

#include "glog/logging.h"
#include "models/bpe.h"

namespace tokenizers {
namespace models {
const std::string WHITESPACE = " \n\r\t\f\v";

void BPE::Init(const core::Merges& merges) {
  if (dropout_.size() > 0) {
    if (dropout_[0] > 1.0 || dropout_[0] <= 0.0) {
      std::ostringstream oss;
      oss << "The range of dropout rate should be (0,1], but receive "
          << dropout_[0];
      throw std::runtime_error(oss.str());
    }
  }
  // construct vocab_r
  for (auto&& item : vocab_) {
    vocab_reversed_[item.second] = item.first;
  }
  int prefix_len = 0;
  if (continuing_subword_prefix_.size() > 0) {
    prefix_len += continuing_subword_prefix_[0].length();
  }

  // construct merge_map
  for (int i = 0; i < merges.size(); i++) {
    auto&& merge = merges[i];
    try {
      auto a_id = vocab_.at(merge.first);
      auto b_id = vocab_.at(merge.second);
      auto new_token = merge.first + merge.second.substr(prefix_len);
      auto new_id = vocab_.at(new_token);
      merges_.insert({core::Pair(a_id, b_id), {i, new_id}});
    } catch (...) {
      std::ostringstream oss;
      oss << "Can't merge token out of the vocabulary";
      throw std::runtime_error(oss.str());
    }
  }
}

BPE::BPE() : fuse_unk_(false), cache_(utils::DEFAULT_CACHE_CAPACITY) {}

BPE::BPE(const core::Vocab& vocab,
         const core::Merges& merges,
         size_t cache_capacity,
         const std::vector<float>& dropout,
         const std::vector<std::string>& unk_token,
         const std::vector<std::string>& continuing_subword_prefix,
         const std::vector<std::string>& end_of_word_suffix,
         bool fuse_unk)
    : vocab_(vocab),
      fuse_unk_(fuse_unk),
      dropout_(dropout),
      unk_token_(unk_token),
      continuing_subword_prefix_(continuing_subword_prefix),
      end_of_word_suffix_(end_of_word_suffix),
      cache_(utils::DEFAULT_CACHE_CAPACITY) {
  Init(merges);
}

void BPE::ClearCache() { cache_.Clear(); }

core::Vocab BPE::GetVocabFromFile(const std::string& vocab_json_path) {
  std::ifstream fin(vocab_json_path);
  core::Vocab vocab;
  nlohmann::json j;
  fin >> j;
  for (nlohmann::json::iterator it = j.begin(); it != j.end(); ++it) {
    vocab[it.key()] = it.value();
  }
  return vocab;
}

core::Merges BPE::GetMergesFromFile(const std::string& merge_path) {
  std::ifstream fin(merge_path);
  core::Merges merges;
  constexpr int MAX_BUFFER_SIZE = 256;
  char word[MAX_BUFFER_SIZE];
  while (fin.getline(word, MAX_BUFFER_SIZE)) {
    std::string word_str = word;
    auto pair_a_begin = word_str.find_first_not_of(WHITESPACE);
    auto pair_a_end = word_str.find_first_of(WHITESPACE, pair_a_begin);
    auto pair_b_begin = word_str.find_first_not_of(WHITESPACE, pair_a_end);
    auto pair_b_end = word_str.find_first_of(WHITESPACE, pair_b_begin);
    merges.emplace_back(std::pair<std::string, std::string>{
        word_str.substr(pair_a_begin, pair_a_end - pair_a_begin),
        word_str.substr(pair_b_begin, pair_b_end - pair_b_begin)});
  }
  return merges;
}

void BPE::GetVocabAndMergesFromFile(const std::string& vocab_json_path,
                                    const std::string& merge_path,
                                    core::Vocab* vocab,
                                    core::Merges* merges) {
  *vocab = BPE::GetVocabFromFile(vocab_json_path);
  *merges = BPE::GetMergesFromFile(merge_path);
}

void BPE::MergeWord(const std::string& word, core::BPEWord* bpe_word) {}

std::vector<core::Token> BPE::WordToTokens(const core::BPEWord& bpe_word) {
  return {};
}

std::vector<core::Token> BPE::TokenizeWithCache(const std::string& sequence) {
  return {};
}
std::vector<core::Token> BPE::Tokenize(const std::string& tokens) const {
  return {};
}

bool BPE::TokenToId(const std::string& token, uint* id) const {
  if (vocab_.find(token) == vocab_.end()) {
    return false;
  }
  *id = vocab_.at(token);
  return true;
}

bool BPE::IdToToken(uint id, std::string* token) const {
  if (vocab_reversed_.find(id) == vocab_reversed_.end()) {
    return false;
  }
  *token = vocab_reversed_.at(id);
  return true;
}

core::Vocab BPE::GetVocab() const { return vocab_; }

size_t BPE::GetVocabSize() const { return vocab_.size(); }
// Return the saved voacb path
std::string BPE::Save(const std::string& folder,
                      const std::string& filename_prefix) const {
  return "";
}

}  // model
}  // tokenizers