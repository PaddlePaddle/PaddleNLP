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

namespace tokenizers {
namespace models {

struct BPE : public Model {
  virtual std::vector<core::Token> Tokenize(
      const std::string& tokens) const override;
  virtual bool TokenToId(const std::string& token, uint* id) const override;
  virtual bool IdToToken(uint id, std::string* token) const override;
  virtual core::Vocab GetVocab() const override;
  virtual size_t GetVocabSize() const override;
  // Return the saved voacb path
  virtual std::string Save(const std::string& folder,
                           const std::string& filename_prefix) const override;

private:
  core::Vocab vocab_;
  core::VocabReversed vocab_reversed_;
  core::MergeMap merges_;

  // The following vector may contain 0 or 1 element
  std::vector<utils::Cache<std::string, core::BPEWord>> cache_;
  std::vector<float> dropout_;
  std::vector<std::string> unk_token_;
  std::vector<uint> unk_token_id_;
  std::vector<std::string> continuing_subword_prefix_;
  std::vector<std::string> end_of_word_suffix_;
  bool fuse_unk_;
};

}  // models
}  // tokenizers