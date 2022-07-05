// Copyright 2022 TF.Text Authors.
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

#include "models/model.h"
#include "models/wordpiece.h"
#include "nlohmann/json.hpp"
#include "utils/failure.h"
#include "utils/trie.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace models {

struct FasterWordPiece : public WordPiece {
  FasterWordPiece();
  FasterWordPiece(const core::Vocab& vocab,
                  const std::string& unk_token = "[UNK]",
                  size_t max_input_chars_per_word = 100,
                  const std::string& continuing_subword_prefix = "##",
                  bool with_pretokenization = false);

  virtual std::vector<core::Token> Tokenize(
      const std::string& sequence) override;

private:
  void InitFailureAndTrie();
  std::vector<core::Token> TokenizeWithoutPreTokenize(
      const std::string& sequence) const;
  std::vector<core::Token> TokenizeWithPreTokenize(
      const std::string& sequence) const;
  bool TryFollowFailureLinkAndCollectTokens(
      const std::string& sequence,
      int sequence_offset_in_text,
      int* curr_offset_in_sequence,
      utils::Trie::TraversalCursor* node,
      std::vector<core::Token>* tokens) const;

  void AppendTokensToOutput(const std::string& sequence,
                            int sequence_offset_in_text,
                            int* curr_offset_in_sequence,
                            int curr_node_value,
                            std::vector<core::Token>* tokens) const;
  void HandleTheRemainingStringOnTriePath(
      const std::string& sequence,
      int sequence_offset_in_text,
      utils::Trie::TraversalCursor* node,
      int* original_num_tokens,
      int* curr_offset_in_sequence,
      std::vector<core::Token>* tokens) const;
  bool TryHandleContinuingSubWordPrefix(
      const std::string& sequence,
      int sequence_offset_in_text,
      const utils::Trie::TraversalCursor& node,
      int* original_num_tokens,
      int* curr_offset_in_sequence,
      std::vector<core::Token>* tokens) const;
  void ResetOutputAppendUNK(int sequence_offset_in_text,
                            int sequence_size,
                            int* original_num_tokens,
                            std::vector<core::Token>* tokens) const;
  int SkipRemainingOfWordAndTrailingWhiteSpaces(const std::string& sequence,
                                                int* curr_idx) const;
  void PrecomputeEncodeValueForSubwordPrefix();
  utils::Trie trie_;
  utils::FailureArray failure_array_;
  std::vector<int> encoded_value_for_subword_prefix_;
  friend void to_json(nlohmann::json& j, const FasterWordPiece& model);
  friend void from_json(const nlohmann::json& j, FasterWordPiece& model);
  bool with_pretokenization_;  // The end-to-end version of FasterWordPiece
};

}  // namespace models
}  // namespace faster_tokenizer
}  // namespace paddlenlp
