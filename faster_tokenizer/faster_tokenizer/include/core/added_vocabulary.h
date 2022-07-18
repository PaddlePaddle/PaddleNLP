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

#include <memory>  // For shared_ptr
#include <string>
#include <unordered_set>

#include "core/base.h"
#include "nlohmann/json.hpp"

namespace re2 {
class RE2;
}  // namespace re2

namespace paddlenlp {
namespace faster_tokenizer {

namespace normalizers {
class Normalizer;
class NormalizedString;
}  // namespace normalizers

namespace models {
class Model;
}  // namespace models

namespace pretokenizers {
class PreTokenizedString;
struct StringSplit;
}  // namespace pretokenizers

namespace core {

using MatchSet = std::pair<std::shared_ptr<re2::RE2>, Vocab>;
using MatchResult = std::tuple<uint32_t, bool /* UNK Flag */, core::Offset>;

bool StartWithWord(const std::string& sequence);
bool EndWithWord(const std::string& sequence);
bool StartWithSpace(const std::string& sequence);
bool EndWithSpace(const std::string& sequence);

class AddedToken {
public:
  AddedToken();
  AddedToken(const std::string& content,
             bool is_special = false,
             bool single_word = false,
             bool lstrip = false,
             bool rstrip = false);
  void SetIsSingleWord(bool is_single_word);
  void SetUseLStrip(bool use_lstrip);
  void SetUseRStrip(bool use_rstrip);
  void SetUseNormalized(bool use_normalized);
  void SetContent(const std::string& content);
  void SetIsSpecial(bool is_special);
  std::string GetContent() const;
  bool GetIsSpecial() const;
  bool GetUseNormalized() const;
  bool GetUseLStrip() const;
  bool GetUseRStrip() const;
  bool GetIsSingleWord() const;
  bool operator==(const AddedToken& other) const;

private:
  std::string content_;
  bool is_single_word_;
  bool use_lstrip_;
  bool use_rstrip_;
  bool use_normalized_;
  bool is_special_;
  friend struct AddedTokenWithId;
};

struct AddedTokenWithId {
  AddedToken added_token_;
  uint32_t id_;
  friend void to_json(nlohmann::json& j, const AddedTokenWithId& added_token);
  friend void from_json(const nlohmann::json& j, AddedTokenWithId& added_token);
};

class AddedVocabulary {
public:
  AddedVocabulary();
  size_t GetLen() const;
  core::Vocab& GetMutableVocab();
  core::Vocab GetVocab() const;
  bool TokenToId(const std::string& token,
                 const models::Model& model,
                 uint32_t* id) const;
  bool IdToToken(uint32_t id,
                 const models::Model& model,
                 std::string* token) const;
  bool IsSpecialToken(const std::string& token) const;
  size_t AddSpecialTokens(const std::vector<AddedToken>& tokens,
                          const models::Model& model,
                          const normalizers::Normalizer* normalizers);
  size_t AddTokens(const std::vector<AddedToken>& tokens,
                   const models::Model& model,
                   const normalizers::Normalizer* normalizers);
  void RefreshAddedTokens(const models::Model& model,
                          const normalizers::Normalizer* normalizers);
  bool FindMatch(const std::string& sequence,
                 const MatchSet& pattern,
                 std::vector<MatchResult>* results) const;
  bool SplitWithIndices(
      const normalizers::NormalizedString& normalized,
      const MatchSet& pattern,
      std::vector<pretokenizers::StringSplit>* split_results) const;
  void ExtractAndNormalize(
      const normalizers::Normalizer* normalizers,
      const std::string& sequence,
      pretokenizers::PreTokenizedString* pretokenized) const;
  const std::unordered_map<uint32_t, AddedToken>& GetAddedTokenVocabReversed()
      const;

private:
  core::Vocab vocab_;
  std::unordered_map<uint32_t, AddedToken> vocab_reversed_;
  std::vector<AddedToken> added_tokens_;
  std::vector<AddedToken> special_tokens_;
  std::unordered_set<std::string> special_tokens_set_;
  MatchSet split_trie_;
  MatchSet split_normalized_trie_;
  friend void to_json(nlohmann::json& j, const AddedVocabulary& added_vocab);
  friend void from_json(const nlohmann::json& j, AddedVocabulary& added_vocab);
};

}  // namespace core
}  // namespace faster_tokenizer
}  // namespace paddlenlp

namespace std {
template <>
class hash<paddlenlp::faster_tokenizer::core::AddedToken> {
public:
  size_t operator()(
      const paddlenlp::faster_tokenizer::core::AddedToken& added_token) const {
    return std::hash<std::string>()(added_token.GetContent());
  }
};
}
