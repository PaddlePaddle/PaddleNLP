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
#include "utils/path.h"
#include "utils/utf8.h"

namespace paddlenlp {
namespace faster_tokenizer {
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

  // construct unk
  if (unk_token_.size() > 0) {
    try {
      unk_token_id_.emplace_back(vocab_.at(unk_token_.front()));
    } catch (...) {
      std::ostringstream oss;
      oss << "Unk token `" << unk_token_.front()
          << "` not found in the vocabulary";
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

void BPE::ConstructMergesPair(const std::string word_line,
                              std::pair<std::string, std::string>* result) {
  auto pair_a_begin = word_line.find_first_not_of(WHITESPACE);
  auto pair_a_end = word_line.find_first_of(WHITESPACE, pair_a_begin);
  auto pair_b_begin = word_line.find_first_not_of(WHITESPACE, pair_a_end);
  auto pair_b_end = word_line.find_first_of(WHITESPACE, pair_b_begin);
  *result = {word_line.substr(pair_a_begin, pair_a_end - pair_a_begin),
             word_line.substr(pair_b_begin, pair_b_end - pair_b_begin)};
}

core::Merges BPE::GetMergesFromFile(const std::string& merge_path) {
  std::ifstream fin(merge_path);
  core::Merges merges;
  std::string word_str;
  while (std::getline(fin, word_str)) {
    if (word_str.find("#version") == 0) {
      continue;
    }
    std::pair<std::string, std::string> result;
    ConstructMergesPair(word_str, &result);
    merges.emplace_back(result);
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

void BPE::MergeWord(const std::string& word, core::BPEWord* bpe_word) {
  std::vector<std::pair<uint32_t, size_t>> unk;
  bpe_word->Reserve(word.length());
  uint32_t start = 0;
  while (start < word.length()) {
    uint32_t content_char;
    uint32_t content_char_width =
        utils::UTF8ToUInt32(word.data() + start, &content_char);
    content_char = utils::UTF8ToUnicode(content_char);
    uint32_t end = start + content_char_width;
    bool is_first = (start == 0);
    bool is_last = (end >= word.length());
    std::string curr_str = word.substr(start, content_char_width);
    // Add the `continuing_subword_prefix` if relevant
    if (!is_first) {
      if (continuing_subword_prefix_.size() > 0) {
        curr_str = continuing_subword_prefix_.front() + curr_str;
      }
    }
    // Add the `end_of_word_suffix` if relevant
    if (is_last) {
      if (end_of_word_suffix_.size() > 0) {
        curr_str = curr_str + end_of_word_suffix_.front();
      }
    }
    if (vocab_.find(curr_str) != vocab_.end()) {
      if (unk.size() > 0) {
        bpe_word->Add(unk.front().first, unk.front().second);
        unk.clear();
      }
      auto id = vocab_.at(curr_str);
      bpe_word->Add(id, content_char_width);
    } else {
      if (unk_token_id_.size() > 0) {
        if (unk.size() == 0) {
          unk.push_back({unk_token_id_.front(), content_char_width});
        } else {
          if (fuse_unk_) {
            unk[0] = {unk[0].first, unk[0].second + content_char_width};
          } else {
            bpe_word->Add(unk[0].first, unk[0].second);
            unk[0] = {unk_token_id_.front(), content_char_width};
          }
        }
      }
    }
    start = end;
  }

  if (unk.size() > 0) {
    bpe_word->Add(unk.front().first, unk.front().second);
  }
  bpe_word->MergeAll(merges_, dropout_);
}

void BPE::WordToTokens(const core::BPEWord& bpe_word,
                       std::vector<core::Token>* tokens) {
  std::vector<uint32_t> chars;
  bpe_word.GetChars(&chars);

  std::vector<core::Offset> offsets;
  bpe_word.GetOffset(&offsets);

  tokens->reserve(offsets.size());
  for (int i = 0; i < offsets.size(); ++i) {
    tokens->emplace_back(chars[i], vocab_reversed_[chars[i]], offsets[i]);
  }
}

void BPE::TokenizeWithCache(const std::string& sequence,
                            std::vector<core::Token>* tokens) {
  core::BPEWord bpe_word;
  if (cache_.GetValue(sequence, &bpe_word)) {
    WordToTokens(bpe_word, tokens);
  } else {
    MergeWord(sequence, &bpe_word);
    WordToTokens(bpe_word, tokens);
    cache_.SetValue(sequence, bpe_word);
  }
}

std::vector<core::Token> BPE::Tokenize(const std::string& sequence) {
  std::vector<core::Token> tokens;
  if (sequence.empty()) {
    return tokens;
  }
  if (dropout_.size() == 0) {
    TokenizeWithCache(sequence, &tokens);
    return tokens;
  }
  core::BPEWord bpe_word;
  MergeWord(sequence, &bpe_word);
  WordToTokens(bpe_word, &tokens);
  return tokens;
}

bool BPE::TokenToId(const std::string& token, uint32_t* id) const {
  if (vocab_.find(token) == vocab_.end()) {
    return false;
  }
  *id = vocab_.at(token);
  return true;
}

bool BPE::IdToToken(uint32_t id, std::string* token) const {
  if (vocab_reversed_.find(id) == vocab_reversed_.end()) {
    return false;
  }
  *token = vocab_reversed_.at(id);
  return true;
}

core::Vocab BPE::GetVocab() const { return vocab_; }

size_t BPE::GetVocabSize() const { return vocab_.size(); }

// Return the saved voacb path and merges.txt
std::vector<std::string> BPE::Save(const std::string& folder,
                                   const std::string& filename_prefix) const {
  // write vocab json
  std::string vocab_path;
  if (filename_prefix == "") {
    vocab_path = utils::PathJoin(folder, "vocab.json");
  } else {
    vocab_path = utils::PathJoin({folder, filename_prefix, "-vocab.json"});
  }
  VLOG(6) << "Vocab path" << vocab_path;
  core::SortedVocabReversed sorted_vocab_r(vocab_reversed_.begin(),
                                           vocab_reversed_.end());
  nlohmann::json j = sorted_vocab_r;
  std::ofstream fout(vocab_path);
  fout << j.dump();
  fout.close();

  // write merges.txt
  std::string merges_path;
  if (filename_prefix == "") {
    merges_path = utils::PathJoin(folder, "merges.txt");
  } else {
    merges_path = utils::PathJoin({folder, filename_prefix, "-merges.txt"});
  }
  VLOG(6) << "Merges path" << merges_path;
  std::ofstream merge_fout(merges_path);
  merge_fout << "#version: 0.2\n";
  for (auto&& merge : merges_) {
    merge_fout << vocab_reversed_.at(merge.first.first) << " "
               << vocab_reversed_.at(merge.first.second) << "\n";
  }
  merge_fout.close();
  return {vocab_path, merges_path};
}

void to_json(nlohmann::json& j, const BPE& model) {
  std::vector<std::pair<core::Pair, uint32_t>> merges;
  for (auto& merge : model.merges_) {
    merges.push_back({merge.first, merge.second.first});
  }
  std::sort(merges.begin(),
            merges.end(),
            [](const std::pair<core::Pair, uint32_t>& a,
               const std::pair<core::Pair, uint32_t>& b) {
              return a.second < b.second;
            });
  std::vector<std::string> merge_strs;
  for (auto& merge : merges) {
    std::string s = model.vocab_reversed_.at(merge.first.first) + " " +
                    model.vocab_reversed_.at(merge.first.second);
    merge_strs.push_back(s);
  }

  core::SortedVocabReversed sorted_vocab_r(model.vocab_reversed_.begin(),
                                           model.vocab_reversed_.end());

  j = {{"type", "BPE"},
       {"unk_token", model.unk_token_},
       {"continuing_subword_prefix", model.continuing_subword_prefix_},
       {"end_of_word_suffix", model.end_of_word_suffix_},
       {"fuse_unk", model.fuse_unk_},
       {"dropout", model.dropout_},
       {"vocab", sorted_vocab_r},
       {"merges", merge_strs}};
}

void from_json(const nlohmann::json& j, BPE& model) {
  j["vocab"].get_to(model.vocab_);
  j["unk_token"].get_to(model.unk_token_);
  j["continuing_subword_prefix"].get_to(model.continuing_subword_prefix_);
  j["end_of_word_suffix"].get_to(model.end_of_word_suffix_);
  j["fuse_unk"].get_to(model.fuse_unk_);
  j["dropout"].get_to(model.dropout_);

  std::vector<std::string> merge_strs;
  j["merges"].get_to(merge_strs);

  core::Merges merges;
  std::pair<std::string, std::string> result;
  for (auto& word_line : merge_strs) {
    BPE::ConstructMergesPair(word_line, &result);
    merges.push_back(result);
  }
  model.Init(merges);
}

}  // namespace model
}  // namespace faster_tokenizer
}  // namespace paddlenlp
