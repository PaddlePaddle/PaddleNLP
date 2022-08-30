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

#include "models/unigram.h"
#include <iomanip>
#include <limits>
#include <sstream>

#include "glog/logging.h"
#include "utils/path.h"
#include "utils/unique_ptr.h"
#include "utils/utils.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace models {

constexpr float kUnkPenalty = 10.0;

Unigram::Unigram() {
  core::VocabList vocab = {{"<unk>", 0.0}};
  std::vector<size_t> unk_id = {0};
  Init(vocab, unk_id);
}

Unigram::Unigram(const core::VocabList& vocab,
                 const std::vector<size_t>& unk_id) {
  Init(vocab, unk_id);
}

Unigram::Unigram(const Unigram& other) { Init(other.vocab_, other.unk_id_); }

void Unigram::Init(const core::VocabList& vocab,
                   const std::vector<size_t>& unk_id) {
  size_t n = vocab.size();
  if (unk_id.size() > 0) {
    if (n == 0) {
      std::ostringstream oss;
      oss << "EmptyVocabulary error occurs when init unigram with unk token.";
      throw std::runtime_error(oss.str());
    } else if (unk_id[0] >= n) {
      std::ostringstream oss;
      oss << "Unk token id is not in vocab when init unigram with unk token.";
      throw std::runtime_error(oss.str());
    }
  }

  vocab_ = vocab;
  unk_id_ = unk_id;

  bos_id_ = n + 1;
  eos_id_ = n + 2;
  min_score_ = std::numeric_limits<double>::max();

  std::vector<const char*> keys;
  std::vector<int> values;
  // id = 0 is unk_id_
  for (size_t id = 0; id < n; ++id) {
    size_t actual_id = id;
    token_to_ids_.insert({vocab[id].first, actual_id});
    keys.push_back(vocab[id].first.c_str());
    values.push_back(actual_id);
    if (vocab[id].second < min_score_) {
      min_score_ = vocab[id].second;
    }
  }

  std::vector<const char*> sorted_keys;
  std::vector<int> sorted_values;
  utils::GetSortedVocab(keys, values, &sorted_keys, &sorted_values);
  trie_ = utils::make_unique<Darts::DoubleArray>();
  if (trie_->build(sorted_keys.size(),
                   const_cast<char**>(&sorted_keys[0]),
                   nullptr,
                   &sorted_values[0]) != 0) {
    std::ostringstream oss;
    oss << "Cannot build double-array.";
    throw std::runtime_error(oss.str());
    return;
  }
  // Computes the maximum number of shared prefixes in the trie.
  const int kMaxTrieResultsSize = 1024;
  std::vector<Darts::DoubleArray::result_pair_type> results(
      kMaxTrieResultsSize);
  trie_results_size_ = 0;
  for (size_t id = 0; id < n; ++id) {
    const int num_nodes = trie_->commonPrefixSearch(vocab[id].first.data(),
                                                    results.data(),
                                                    results.size(),
                                                    vocab[id].first.size());
    trie_results_size_ = std::max(trie_results_size_, num_nodes);
  }
  fuse_unk_ = true;
  is_optimized_ = true;
  if (trie_results_size_ == 0) {
    std::ostringstream oss;
    oss << "No entry is found in the trie.";
    throw std::runtime_error(oss.str());
  }
}

float Unigram::GetVocabScore(uint32_t id) const { return vocab_.at(id).second; }

bool Unigram::TokenToId(const std::string& token, uint32_t* id) const {
  if (token_to_ids_.find(token) == token_to_ids_.end()) {
    return false;
  }
  *id = token_to_ids_.at(token);
  return true;
}

bool Unigram::IdToToken(uint32_t id, std::string* token) const {
  if (id >= vocab_.size()) {
    return false;
  }
  *token = vocab_[id].first;
  return true;
}

core::Vocab Unigram::GetVocab() const { return token_to_ids_; }

size_t Unigram::GetVocabSize() const { return vocab_.size(); }

std::vector<core::Token> Unigram::Tokenize(const std::string& sequence) {
  std::vector<std::string> encode_result;
  Encode(sequence, &encode_result);
  size_t offset = 0;
  std::vector<core::Token> tokens;
  tokens.reserve(encode_result.size());
  auto UpdateTokens = [&](const std::string& str) {
    uint32_t id = 0;
    if (token_to_ids_.find(str) != token_to_ids_.end()) {
      id = token_to_ids_.at(str);
    } else {
      if (unk_id_.size() > 0) {
        id = unk_id_[0];
      }
    }
    auto len = str.length();
    tokens.emplace_back(id, str, core::Offset{offset, offset + len});
    offset += len;
  };

  for (auto&& str : encode_result) {
    // Avoid to append the filtered_token_ to encoded_result
    if (str == filtered_token_) {
      offset += filtered_token_.length();
      continue;
    }
    // Split the tokenized tokens following some regex rule
    if (split_rule_ != nullptr) {
      re2::StringPiece result;
      int start = 0;
      int end = str.length();
      while (split_rule_->Match(str, start, end, RE2::UNANCHORED, &result, 1)) {
        int curr_start = result.data() - str.data();
        int res_len = result.length();
        start = curr_start + res_len;
        std::string result_str(result.data(), res_len);
        if (result_str == filtered_token_) {
          offset += filtered_token_.length();
          continue;
        }
        UpdateTokens(result_str);
      }
      if (start == 0) {
        // Hasn't been splitted
        UpdateTokens(str);
      }
    } else {
      UpdateTokens(str);
    }
  }
  return tokens;
}

std::vector<std::string> Unigram::Save(
    const std::string& folder, const std::string& filename_prefix) const {
  std::string vocab_path;
  if (filename_prefix == "") {
    vocab_path = utils::PathJoin(folder, "unigram.json");
  } else {
    vocab_path = utils::PathJoin({folder, filename_prefix, "-unigram.json"});
  }
  VLOG(6) << "Vocab path" << vocab_path;
  std::ofstream fout(vocab_path);
  nlohmann::json j = *this;
  fout << j.dump();
  fout.close();
  return {vocab_path};
}

void Unigram::PopulateNodes(utils::Lattice* lattice) const {
  auto get_chars_length = [&lattice](int begin_pos, const char* end) {
    int pos = begin_pos;
    while (lattice->surface(pos) < end) ++pos;
    return pos - begin_pos;
  };

  const float unk_score = min_score_ - kUnkPenalty;

  const int len = lattice->size();
  const char* end = lattice->sentence() + lattice->utf8_size();

  // +1 just in case.
  std::vector<Darts::DoubleArray::result_pair_type> trie_results(
      trie_results_size_ + 1);

  for (int begin_pos = 0; begin_pos < len; ++begin_pos) {
    const char* begin = lattice->surface(begin_pos);

    // Finds all pieces which are prefix of surface(begin_pos).
    const size_t num_nodes =
        trie_->commonPrefixSearch(begin,
                                  trie_results.data(),
                                  trie_results.size(),
                                  static_cast<int>(end - begin));
    CHECK_LT(num_nodes, trie_results.size());

    bool has_single_node = false;

    // Inserts pieces to the lattice.
    for (size_t k = 0; k < num_nodes; ++k) {
      const int length =
          get_chars_length(begin_pos, begin + trie_results[k].length);
      const int id = trie_results[k].value;
      utils::Lattice::Node* node = lattice->Insert(begin_pos, length);
      node->id = id;  // the value of Trie stores vocab_id.
      // User defined symbol receives extra bonus to always be selected.
      node->score = vocab_[id].second;

      if (!has_single_node && node->length == 1) {
        has_single_node = true;
      }
    }

    if (!has_single_node) {
      if (unk_id_.size() > 0) {
        utils::Lattice::Node* node = lattice->Insert(begin_pos, 1);
        node->id = unk_id_[0];  // add UNK node.
        node->score = unk_score;
      }
    }
  }
}

void Unigram::Encode(const std::string& normalized,
                     std::vector<std::string>* encode_result) {
  encode_result->clear();
  if (normalized.empty()) {
    return;
  }
  if (!cache_.GetValue(normalized, encode_result)) {
    if (is_optimized_) {
      EncodeOptimized(normalized, encode_result);
    } else {
      EncodeUnoptimized(normalized, encode_result);
    }
    cache_.SetValue(normalized, *encode_result);
  }
}

void Unigram::EncodeOptimized(const std::string& normalized,
                              std::vector<std::string>* encode_result) {
  // Represents the last node of the best path.
  struct BestPathNode {
    int id = -1;  // The vocab id. (maybe -1 for UNK)
    float best_path_score =
        0;  // The total score of the best path ending at this node.
    int starts_at =
        -1;  // The starting position (in utf-8) of this node. The entire best
             // path can be constructed by backtracking along this link.
  };
  const int size = normalized.size();
  const float unk_score = min_score_ - kUnkPenalty;
  // The ends are exclusive.
  std::vector<BestPathNode> best_path_ends_at(size + 1);
  // Generate lattice on-the-fly (not stored) and update best_path_ends_at.
  int starts_at = 0;
  while (starts_at < size) {
    std::size_t node_pos = 0;
    std::size_t key_pos = starts_at;
    const auto best_path_score_till_here =
        best_path_ends_at[starts_at].best_path_score;
    bool has_single_node = false;
    const int mblen = std::min<int>(
        utils::OneCharLen(normalized.data() + starts_at), size - starts_at);
    while (key_pos < size) {
      const int ret =
          trie_->traverse(normalized.data(), node_pos, key_pos, key_pos + 1);
      if (ret == -2) break;
      if (ret >= 0) {
        // Update the best path node.
        auto& target_node = best_path_ends_at[key_pos];
        const auto length = (key_pos - starts_at);
        const auto score = GetVocabScore(ret);
        const auto candidate_best_path_score =
            score + best_path_score_till_here;
        VLOG(4) << "key_pos: " << key_pos;
        VLOG(4) << "score: " << score;
        VLOG(4) << "best_path_score_till_here: " << best_path_score_till_here;
        VLOG(4) << "starts_at: " << starts_at;
        VLOG(4) << "token: " << vocab_.at(ret).first;
        if (target_node.starts_at == -1 ||
            candidate_best_path_score > target_node.best_path_score) {
          target_node.best_path_score = candidate_best_path_score;
          target_node.starts_at = starts_at;
          target_node.id = ret;
        }
        if (!has_single_node && length == mblen) {
          has_single_node = true;
        }
      }
    }
    if (!has_single_node) {
      auto& target_node = best_path_ends_at[starts_at + mblen];
      const auto candidate_best_path_score =
          unk_score + best_path_score_till_here;
      if (target_node.starts_at == -1 ||
          candidate_best_path_score > target_node.best_path_score) {
        target_node.best_path_score = candidate_best_path_score;
        target_node.starts_at = starts_at;
        target_node.id = -1;
        if (unk_id_.size() > 0) {
          target_node.id = unk_id_[0];
        }
      }
    }
    // Move by one unicode character.
    starts_at += mblen;
  }
  int ends_at = size;
  std::vector<std::string> token;
  while (ends_at > 0) {
    const auto& node = best_path_ends_at[ends_at];
    auto starts_at = node.starts_at;
    if (fuse_unk_ && unk_id_.size() > 0 && node.id == unk_id_[0]) {
      token.push_back(normalized.substr(starts_at, ends_at - starts_at));
    } else {
      if (!token.empty()) {
        encode_result->push_back("");
        auto& back = encode_result->back();
        for (int i = token.size() - 1; i >= 0; --i) {
          back.append(token[i]);
        }
        token.clear();
      }
      encode_result->push_back(
          normalized.substr(starts_at, ends_at - starts_at));
    }
    ends_at = node.starts_at;
  }
  if (!token.empty()) {
    encode_result->push_back("");
    auto& back = encode_result->back();
    for (int i = token.size() - 1; i >= 0; --i) {
      back.append(token[i]);
    }
  }
  std::reverse(encode_result->begin(), encode_result->end());
}

void Unigram::EncodeUnoptimized(const std::string& normalized,
                                std::vector<std::string>* encode_result) {
  utils::Lattice lattice;
  lattice.SetSentence(
      utils::simple_string_view(normalized.data(), normalized.size()));
  PopulateNodes(&lattice);
  if (fuse_unk_) {
    std::string token;
    for (const auto* node : lattice.Viterbi().first) {
      if (unk_id_.size() > 0 && node->id == unk_id_[0]) {
        token.append(node->piece.data(), node->piece.size());
      } else {
        if (!token.empty()) {
          encode_result->push_back(token);
          token.clear();
        }
        encode_result->push_back(std::string(node->piece.data()));
      }
      if (!token.empty()) {
        encode_result->push_back(token);
      }
    }
  } else {
    for (const auto* node : lattice.Viterbi().first) {
      encode_result->push_back(std::string(node->piece.data()));
    }
  }
}

void Unigram::SetFilterToken(const std::string& filtered_token) {
  filtered_token_ = filtered_token;
}

void Unigram::SetSplitRule(const std::string& split_rule) {
  split_rule_ = utils::make_unique<re2::RE2>(split_rule);
}

void to_json(nlohmann::json& j, const Unigram& model) {
  std::string split_rule = "";
  if (model.split_rule_ != nullptr) {
    split_rule = model.split_rule_->pattern();
  }
  j = {{"type", "Unigram"},
       {"unk_id", model.unk_id_},
       {"vocab", model.vocab_},
       {"filter_token", model.filtered_token_},
       {"split_rule", split_rule}};
}

void from_json(const nlohmann::json& j, Unigram& model) {
  std::string filter_token = j.at("filter_token").get<std::string>();
  std::string split_rule = j.at("split_rule").get<std::string>();
  model.Init(j.at("vocab").get<core::VocabList>(),
             j.at("unk_id").get<std::vector<size_t>>());
  if (!split_rule.empty()) {
    model.SetSplitRule(split_rule);
  }
  model.SetFilterToken(filter_token);
}

}  // namespace model
}  // namespace faster_tokenizer
}  // namespace paddlenlp
