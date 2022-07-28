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

#include "core/base.h"
#include "models/model.h"
#include "utils/cache.h"
#include "utils/lattice.h"
#include "utils/trie.h"

#include "darts.h"
#include "nlohmann/json.hpp"
#include "re2/re2.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace models {

struct Unigram : public Model {
  Unigram();
  Unigram(const core::VocabList& vocab, const std::vector<size_t>& unk_id);
  Unigram(const Unigram& other);
  virtual bool TokenToId(const std::string& token, uint32_t* id) const override;
  virtual bool IdToToken(uint32_t id, std::string* token) const override;
  virtual core::Vocab GetVocab() const override;
  virtual size_t GetVocabSize() const override;
  virtual std::vector<core::Token> Tokenize(
      const std::string& sequence) override;
  virtual std::vector<std::string> Save(
      const std::string& folder,
      const std::string& filename_prefix) const override;
  // Set the filter token for unigram.
  void SetFilterToken(const std::string& filtered_token);
  // Set the special spliting rule for unigram.
  void SetSplitRule(const std::string& split_rule);

private:
  float GetVocabScore(uint32_t id) const;
  void Init(const core::VocabList& vocab, const std::vector<size_t>& unk_id);
  void PopulateNodes(utils::Lattice* lattice) const;
  void Encode(const std::string& normalized,
              std::vector<std::string>* encode_result);
  void EncodeOptimized(const std::string& normalized,
                       std::vector<std::string>* encode_result);
  void EncodeUnoptimized(const std::string& normalized,
                         std::vector<std::string>* encode_result);

  core::Vocab token_to_ids_;
  core::VocabList vocab_;
  utils::Cache<std::string, std::vector<std::string>> cache_;
  std::unique_ptr<Darts::DoubleArray> trie_;
  double min_score_;
  std::vector<size_t> unk_id_;
  size_t bos_id_;
  size_t eos_id_;
  bool fuse_unk_;
  bool is_optimized_;
  int trie_results_size_;
  // Some tokenizer, such as ernie-m, may avoid to append some special
  // token to final result, the unigram model doesn't filter any tokens
  // by default.
  std::string filtered_token_;
  // For special rule of token spliting after tokenization,
  // the unigram model has no spliting rule by default.
  // It's useful for some cases, such as ernie-m tokenizer.
  std::unique_ptr<re2::RE2> split_rule_;

  friend void to_json(nlohmann::json& j, const Unigram& model);
  friend void from_json(const nlohmann::json& j, Unigram& model);
};

}  // namespace models
}  // namespace faster_tokenizer
}  // namespace paddlenlp
