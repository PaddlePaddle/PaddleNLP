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

#include "models/model.h"
#include "nlohmann/json.hpp"
#include "utils/cache.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace models {

struct BPE : public Model {
  BPE();
  BPE(const core::Vocab& vocab,
      const core::Merges& merges,
      size_t cache_capacity = utils::DEFAULT_CACHE_CAPACITY,
      const std::vector<float>& dropout = {},
      const std::vector<std::string>& unk_token = {},
      const std::vector<std::string>& continuing_subword_prefix = {},
      const std::vector<std::string>& end_of_word_suffix = {},
      bool fuse_unk = false);
  virtual std::vector<core::Token> Tokenize(
      const std::string& sequence) override;
  virtual bool TokenToId(const std::string& token, uint32_t* id) const override;
  virtual bool IdToToken(uint32_t id, std::string* token) const override;
  virtual core::Vocab GetVocab() const override;
  virtual size_t GetVocabSize() const override;
  // Return the saved voacb path
  virtual std::vector<std::string> Save(
      const std::string& folder,
      const std::string& filename_prefix) const override;

  void ClearCache();
  static core::Vocab GetVocabFromFile(const std::string& vocab_json_path);
  static core::Merges GetMergesFromFile(const std::string& merge_path);
  static void GetVocabAndMergesFromFile(const std::string& vocab_json_path,
                                        const std::string& merge_path,
                                        core::Vocab* vocab,
                                        core::Merges* merges);
  static void ConstructMergesPair(const std::string word_line,
                                  std::pair<std::string, std::string>* result);

private:
  void Init(const core::Merges& merges);
  void MergeWord(const std::string& word, core::BPEWord* bpe_word);
  void WordToTokens(const core::BPEWord& bpe_word,
                    std::vector<core::Token>* tokens);
  void TokenizeWithCache(const std::string& sequence,
                         std::vector<core::Token>* tokens);
  core::Vocab vocab_;
  core::VocabReversed vocab_reversed_;
  core::MergeMap merges_;

  // The following vector may contain 0 or 1 element
  utils::Cache<std::string, core::BPEWord> cache_;
  std::vector<float> dropout_;
  std::vector<std::string> unk_token_;
  std::vector<uint32_t> unk_token_id_;
  std::vector<std::string> continuing_subword_prefix_;
  std::vector<std::string> end_of_word_suffix_;
  bool fuse_unk_;
  friend void to_json(nlohmann::json& j, const BPE& model);
  friend void from_json(const nlohmann::json& j, BPE& model);
};

}  // namespace models
}  // namespace faster_tokenizer
}  // namespace paddlenlp
