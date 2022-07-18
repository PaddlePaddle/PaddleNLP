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

#include "core/added_vocabulary.h"
#include "glog/logging.h"
#include "models/model.h"
#include "normalizers/normalizer.h"
#include "pretokenizers/pretokenizer.h"
#include "re2/re2.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace core {

inline bool StartWithWord(const std::string& sequence) {
  static re2::RE2 pattern("^\\w");
  return re2::RE2::FullMatch(sequence, pattern);
}

inline bool EndWithWord(const std::string& sequence) {
  static re2::RE2 pattern("\\w$");
  return re2::RE2::FullMatch(sequence, pattern);
}

inline bool StartWithSpace(const std::string& sequence) {
  static re2::RE2 pattern("^\\s*");
  return re2::RE2::FullMatch(sequence, pattern);
}

inline bool EndWithSpace(const std::string& sequence) {
  static re2::RE2 pattern("\\s*$");
  return re2::RE2::FullMatch(sequence, pattern);
}

inline size_t GetEndSpaceIdx(const std::string& sequence) {
  static re2::RE2 pattern("\\s*$");
  re2::StringPiece result_str;
  pattern.Match(
      sequence, 0, sequence.length(), RE2::UNANCHORED, &result_str, 1);
  return result_str.data() - sequence.data();
}

inline size_t GetStartSpaceIdx(const std::string& sequence) {
  static re2::RE2 pattern("^\\s*");
  re2::StringPiece result_str;
  pattern.Match(
      sequence, 0, sequence.length(), RE2::UNANCHORED, &result_str, 1);
  return result_str.data() + result_str.length() - sequence.data();
}

inline size_t GetLeftMostSpaceFromEnd(const std::string& sequence) {
  if (EndWithSpace(sequence)) {
    return GetEndSpaceIdx(sequence);
  }
  return sequence.length();
}

inline size_t GetRightMostSpaceFromStart(const std::string& sequence) {
  if (StartWithSpace(sequence)) {
    return GetStartSpaceIdx(sequence);
  }
  return 0;
}

AddedToken::AddedToken()
    : content_(""),
      is_single_word_(false),
      use_lstrip_(false),
      use_rstrip_(false),
      use_normalized_(true),
      is_special_(false) {}

AddedToken::AddedToken(const std::string& content,
                       bool is_special,
                       bool single_word,
                       bool lstrip,
                       bool rstrip)
    : content_(content),
      is_special_(is_special),
      use_normalized_(!is_special),
      is_single_word_(single_word),
      use_lstrip_(lstrip),
      use_rstrip_(rstrip) {}

std::string AddedToken::GetContent() const { return content_; }

bool AddedToken::GetIsSpecial() const { return is_special_; }

bool AddedToken::GetUseNormalized() const { return use_normalized_; }

void AddedToken::SetIsSingleWord(bool is_single_word) {
  is_single_word_ = is_single_word;
}
bool AddedToken::GetUseLStrip() const { return use_lstrip_; }

bool AddedToken::GetUseRStrip() const { return use_rstrip_; }

bool AddedToken::GetIsSingleWord() const { return is_single_word_; }

void AddedToken::SetContent(const std::string& content) { content_ = content; }

void AddedToken::SetUseLStrip(bool use_lstrip) { use_lstrip_ = use_lstrip; }

void AddedToken::SetUseRStrip(bool use_rstrip) { use_rstrip_ = use_rstrip; }

void AddedToken::SetUseNormalized(bool use_normalized) {
  use_normalized_ = use_normalized;
}

void AddedToken::SetIsSpecial(bool is_special) { is_special_ = is_special; }

bool AddedToken::operator==(const AddedToken& other) const {
  return content_ == other.content_;
}

AddedVocabulary::AddedVocabulary()
    : split_trie_({std::make_shared<re2::RE2>(""), Vocab()}),
      split_normalized_trie_({std::make_shared<re2::RE2>(""), Vocab()}) {}

size_t AddedVocabulary::GetLen() const { return vocab_.size(); }

core::Vocab AddedVocabulary::GetVocab() const { return vocab_; }
core::Vocab& AddedVocabulary::GetMutableVocab() { return vocab_; }

bool AddedVocabulary::TokenToId(const std::string& token,
                                const models::Model& model,
                                uint32_t* id) const {
  if (vocab_.find(token) != vocab_.end()) {
    *id = vocab_.at(token);
    return true;
  }
  return model.TokenToId(token, id);
}

bool AddedVocabulary::IdToToken(uint32_t id,
                                const models::Model& model,
                                std::string* token) const {
  if (vocab_reversed_.find(id) != vocab_reversed_.end()) {
    *token = vocab_reversed_.at(id).GetContent();
    return true;
  }
  return model.IdToToken(id, token);
}

bool AddedVocabulary::IsSpecialToken(const std::string& token) const {
  return special_tokens_set_.find(token) != special_tokens_set_.end();
}

size_t AddedVocabulary::AddSpecialTokens(
    const std::vector<AddedToken>& tokens,
    const models::Model& model,
    const normalizers::Normalizer* normalizers) {
  return AddTokens(tokens, model, normalizers);
}

size_t AddedVocabulary::AddTokens(const std::vector<AddedToken>& tokens,
                                  const models::Model& model,
                                  const normalizers::Normalizer* normalizers) {
  for (const auto& token : tokens) {
    if (token.GetIsSpecial() && !token.GetContent().empty() &&
        !IsSpecialToken(token.GetContent())) {
      special_tokens_.push_back(token);
      special_tokens_set_.insert(token.GetContent());
    }
  }
  int ignored_tokens_num = 0;
  for (const auto& token : tokens) {
    if (token.GetContent().empty()) {
      ignored_tokens_num += 1;
      continue;
    }
    uint32_t id;
    if (TokenToId(token.GetContent(), model, &id)) {
      ignored_tokens_num += 1;
    } else {
      uint32_t new_id = model.GetVocabSize() + GetLen();
      vocab_[token.GetContent()] = new_id;
      if (special_tokens_set_.count(token.GetContent()) == 0) {
        added_tokens_.push_back(token);
      }
      id = new_id;
    }
    vocab_reversed_[id] = token;
  }
  RefreshAddedTokens(model, normalizers);
  return tokens.size() - ignored_tokens_num;
}
void AddedVocabulary::RefreshAddedTokens(
    const models::Model& model, const normalizers::Normalizer* normalizers) {
  using TokenAndId = std::pair<AddedToken, uint32_t>;
  std::vector<TokenAndId> normalized, non_normalized;
  for (const auto& tokens : {special_tokens_, added_tokens_}) {
    for (const auto& token : tokens) {
      uint32_t id;
      if (TokenToId(token.GetContent(), model, &id)) {
        if (token.GetUseNormalized()) {
          normalized.push_back({token, id});
        } else {
          non_normalized.push_back({token, id});
        }
      }
    }
  }
  Vocab ids;
  std::vector<AddedToken> tokens;
  for (const auto& token_ids : non_normalized) {
    tokens.push_back(token_ids.first);
    ids[token_ids.first.GetContent()] = token_ids.second;
  }
  // Create a regex pattern
  std::string pattern("");
  for (int i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      pattern += "|";
    }
    std::string pattern_str = "";
    for (const auto& ch : tokens[i].GetContent()) {
      if (ch == '[' || ch == ']') {
        pattern_str.append(1, '\\');
      }
      pattern_str.append(1, ch);
    }
    pattern += "\(" + pattern_str + "\)";
  }
  // Update split_trie_
  split_trie_.first = std::make_shared<re2::RE2>(pattern);
  split_trie_.second = std::move(ids);
  Vocab normalized_ids;
  std::vector<AddedToken> normalized_tokens;
  for (const auto& token_ids : normalized) {
    normalized_tokens.push_back(token_ids.first);
    normalized_ids[token_ids.first.GetContent()] = token_ids.second;
  }

  std::string normalized_pattern("");
  for (int i = 0; i < normalized_tokens.size(); ++i) {
    normalizers::NormalizedString normalized_content(
        normalized_tokens[i].GetContent());
    if (normalizers != nullptr) {
      (*normalizers)(&normalized_content);
    }
    if (i > 0) {
      normalized_pattern += "|";
    }
    std::string pattern_str = "";
    for (const auto& ch : normalized_content.GetStr()) {
      if (ch == '[' || ch == ']') {
        pattern_str.append(1, '\\');
      }
      pattern_str.append(1, ch);
    }
    normalized_pattern += "\(" + pattern_str + "\)";
  }
  split_normalized_trie_.first = std::make_shared<re2::RE2>(normalized_pattern);
  split_normalized_trie_.second = std::move(normalized_ids);
}

bool AddedVocabulary::FindMatch(const std::string& sequence,
                                const MatchSet& pattern,
                                std::vector<MatchResult>* results) const {
  if (sequence.empty()) {
    return false;
  }
  std::vector<MatchResult> splits;
  size_t start = 0;
  size_t start_offset = 0;
  size_t end = sequence.length();
  re2::StringPiece result_str;
  VLOG(6) << "start = " << start << ", end = " << end
          << ", sequence = " << sequence
          << ", pattern: " << pattern.first->pattern();
  while (pattern.first->Match(
             sequence, start, end, RE2::UNANCHORED, &result_str, 1) &&
         result_str != "") {
    VLOG(6) << "result_str: " << result_str << ", " << pattern.first->pattern();
    size_t curr_start = result_str.data() - sequence.data();
    size_t curr_end = curr_start + result_str.length();
    uint32_t id = pattern.second.at(result_str.ToString());
    AddedToken added_tokens = vocab_reversed_.at(id);
    VLOG(6) << "start = " << start << ", end = " << end
            << ", curr_start = " << curr_start << ", curr_end = " << curr_end;
    if (added_tokens.GetIsSingleWord()) {
      bool start_space =
          (curr_start == 0) || !EndWithWord(sequence.substr(0, curr_start));
      bool stop_space = (curr_end == sequence.length()) ||
                        !StartWithWord(sequence.substr(curr_end));
      if (!start_space || !stop_space) {
        // Discard not single word
        start = curr_end;
        continue;
      }
    }
    if (added_tokens.GetUseLStrip()) {
      auto new_start = GetEndSpaceIdx(sequence.substr(0, curr_start));
      curr_start = std::max(new_start, start_offset);
    }
    if (added_tokens.GetUseRStrip()) {
      curr_end += GetStartSpaceIdx(sequence.substr(curr_end));
    }
    if (curr_start > start_offset) {
      splits.push_back({0, false, {start_offset, curr_start}});
    }
    splits.push_back({id, true, {curr_start, curr_end}});
    start = curr_end;
    start_offset = curr_end;
  }
  if (start != sequence.length()) {
    splits.push_back({0, false, {start, sequence.length()}});
  }
  *results = std::move(splits);
  return true;
}

bool AddedVocabulary::SplitWithIndices(
    const normalizers::NormalizedString& normalized,
    const MatchSet& pattern,
    std::vector<pretokenizers::StringSplit>* split_results) const {
  std::vector<MatchResult> match_results;
  bool status = FindMatch(normalized.GetStr(), pattern, &match_results);
  for (auto& match_result : match_results) {
    normalizers::NormalizedString slice;
    auto id = std::get<0>(match_result);
    auto is_not_unk = std::get<1>(match_result);
    auto offsets = std::get<2>(match_result);
    normalized.Slice(offsets, &slice, false);
    std::vector<core::Token> tokens;
    if (is_not_unk) {
      tokens.emplace_back(core::Token{id, slice.GetStr(), {0, slice.GetLen()}});
    }
    // use push_back({slice, tokens}) will raise error in windows platform.
    split_results->emplace_back(slice, tokens);
  }
  return status;
}

void AddedVocabulary::ExtractAndNormalize(
    const normalizers::Normalizer* normalizers,
    const std::string& sequence,
    pretokenizers::PreTokenizedString* pretokenized) const {
  pretokenized->SetOriginalStr(sequence);
  pretokenized->Split(
      [&](int idx,
          normalizers::NormalizedString* normalized,
          std::vector<pretokenizers::StringSplit>* string_splits) {
        this->SplitWithIndices(*normalized, this->split_trie_, string_splits);
      });
  pretokenized->Split(
      [&](int idx,
          normalizers::NormalizedString* normalized,
          std::vector<pretokenizers::StringSplit>* string_splits) {
        if (normalizers != nullptr) {
          (*normalizers)(normalized);
          VLOG(6) << "After normalized: " << normalized->GetStr();
          this->SplitWithIndices(
              *normalized, this->split_normalized_trie_, string_splits);
        }
      });
}

const std::unordered_map<uint32_t, AddedToken>&
AddedVocabulary::GetAddedTokenVocabReversed() const {
  return vocab_reversed_;
}


void to_json(nlohmann::json& j, const AddedTokenWithId& added_token) {
  j = {
      {"id", added_token.id_},
      {"content", added_token.added_token_.GetContent()},
      {"single_word", added_token.added_token_.GetIsSingleWord()},
      {"lstrip", added_token.added_token_.GetUseLStrip()},
      {"rstrip", added_token.added_token_.GetUseRStrip()},
      {"normalized", added_token.added_token_.GetUseNormalized()},
      {"special", added_token.added_token_.GetIsSpecial()},
  };
}

void from_json(const nlohmann::json& j, AddedTokenWithId& added_token) {
  j.at("id").get_to(added_token.id_);
  std::string content = j.at("content").get<std::string>();
  added_token.added_token_.SetContent(content);

  bool single_word = j.at("single_word").get<bool>();
  added_token.added_token_.SetIsSingleWord(single_word);

  bool lstrip = j.at("lstrip").get<bool>();
  added_token.added_token_.SetUseLStrip(lstrip);

  bool rstrip = j.at("rstrip").get<bool>();
  added_token.added_token_.SetUseRStrip(rstrip);

  bool normalized = j.at("normalized").get<bool>();
  added_token.added_token_.SetUseNormalized(normalized);

  bool special = j.at("special").get<bool>();
  added_token.added_token_.SetIsSpecial(special);
}

void to_json(nlohmann::json& j, const AddedVocabulary& added_vocab) {
  nlohmann::json jarray = nlohmann::json::array();
  for (const auto& vocab_item : added_vocab.vocab_reversed_) {
    AddedTokenWithId added_token_with_id;
    added_token_with_id.id_ = vocab_item.first;
    added_token_with_id.added_token_ = vocab_item.second;
    nlohmann::json jo = added_token_with_id;
    jarray.emplace_back(jo);
  }
  j = jarray;
}

}  // namespace core
}  // namespace faster_tokenizer
}  // namespace paddlenlp
