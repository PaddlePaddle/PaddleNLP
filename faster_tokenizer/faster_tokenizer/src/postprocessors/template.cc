// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <string>

#include "core/encoding.h"
#include "glog/logging.h"
#include "postprocessors/template.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace postprocessors {

void ParseIdFromString(const std::string& template_id_string,
                       TemplatePiece* template_piece) {
  if (template_id_string.find_first_of("$") == 0) {
    *template_piece = TemplateSequence();
    auto& seq = boost::get<TemplateSequence>(*template_piece);
    std::string rest =
        template_id_string.substr(template_id_string.find_first_not_of("$"));
    if (rest == "" || rest == "A" || rest == "a") {
      seq = TemplateSequence{SequenceType::SEQ_A, 0};
    } else if (rest == "B" || rest == "b") {
      seq = TemplateSequence{SequenceType::SEQ_B, 0};
    } else {
      std::string::size_type sz;
      uint32_t type_id = std::stoul(rest, &sz);
      if (sz = rest.length()) {
        seq = TemplateSequence{SequenceType::SEQ_A, type_id};
      } else {
        throw std::runtime_error(
            "ParseIdFromString error! The format of template piece id should "
            "be "
            "$A, $a, $B, $b or ${type_id}");
      }
    }
  } else {
    *template_piece = TemplateSpecialToken();
    boost::get<TemplateSpecialToken>(*template_piece) = {template_id_string, 0};
  }
}

void SetTypeId(uint32_t type_id, TemplatePiece* template_piece) {
  if (boost::get<TemplateSequence>(template_piece) != nullptr) {
    boost::get<TemplateSequence>(*template_piece).second = type_id;
  } else {
    boost::get<TemplateSpecialToken>(*template_piece).second = type_id;
  }
}

void GetTemplatePieceFromString(const std::string& template_string,
                                TemplatePiece* template_piece) {
  auto spliter_idx = template_string.find_first_of(":");
  if (spliter_idx == std::string::npos) {
    ParseIdFromString(template_string, template_piece);
  } else {
    std::string template_id_string = template_string.substr(0, spliter_idx);
    std::string template_type_id_string =
        template_string.substr(spliter_idx + 1);
    ParseIdFromString(template_id_string, template_piece);

    std::string::size_type sz;
    uint32_t type_id = std::stoul(template_type_id_string, &sz);
    if (sz == template_type_id_string.length()) {
      SetTypeId(type_id, template_piece);
    } else {
      throw std::runtime_error(
          "ParseTypeIdFromString error! The type id should be unsigned "
          "integer.");
    }
  }
}

void to_json(nlohmann::json& j, const TemplatePiece& template_piece) {
  if (boost::get<TemplateSequence>(&template_piece) != nullptr) {
    auto& template_sequence = boost::get<TemplateSequence>(template_piece);
    j = {
        {"Sequence",
         {
             {"id", template_sequence.first},
             {"type_id", template_sequence.second},
         }},
    };
  } else {
    auto& template_special_token =
        boost::get<TemplateSpecialToken>(template_piece);
    j = {
        {"SpecialToken",
         {
             {"id", template_special_token.first},
             {"type_id", template_special_token.second},
         }},
    };
  }
}

void from_json(const nlohmann::json& j, TemplatePiece& template_piece) {
  if (j.find("Sequence") != j.end()) {
    template_piece =
        TemplateSequence(j["Sequence"]["id"], j["Sequence"]["type_id"]);
  } else {
    template_piece = TemplateSpecialToken(j["SpecialToken"]["id"],
                                          j["SpecialToken"]["type_id"]);
  }
}

void to_json(nlohmann::json& j, const SpecialToken& special_token) {
  j = {
      {"id", special_token.id_},
      {"ids", special_token.ids_},
      {"tokens", special_token.tokens_},
  };
}

void from_json(const nlohmann::json& j, SpecialToken& special_token) {
  j["id"].get_to(special_token.id_);
  j["ids"].get_to(special_token.ids_);
  j["tokens"].get_to(special_token.tokens_);
}

size_t TemplatePostProcessor::CountAdded(
    Template* template_, const SpecialTokensMap& special_tokens_map) {
  size_t count = 0;
  for (auto& piece : template_->pieces_) {
    TemplateSpecialToken* special_token =
        boost::get<TemplateSpecialToken>(&piece);
    if (special_token != nullptr) {
      auto token_iter =
          special_tokens_map.tokens_map_.find(special_token->first);
      if (token_iter != special_tokens_map.tokens_map_.end()) {
        count += token_iter->second.ids_.size();
      }
    }
  }
  return count;
}

void to_json(nlohmann::json& j, const Template& template_) {
  for (auto& piece : template_.pieces_) {
    j.push_back(piece);
  }
}

void from_json(const nlohmann::json& j, Template& template_) {
  template_.pieces_.resize(j.size());
  for (int i = 0; i < j.size(); ++i) {
    j[i].get_to(template_.pieces_[i]);
  }
}

void to_json(nlohmann::json& j, const SpecialTokensMap& tokens_map) {
  for (auto it = tokens_map.tokens_map_.begin();
       it != tokens_map.tokens_map_.end();
       ++it) {
    j[it->first] = it->second;
  }
}

void from_json(const nlohmann::json& j, SpecialTokensMap& tokens_map) {
  SpecialToken special_token;
  for (auto it = j.begin(); it != j.end(); ++it) {
    tokens_map.tokens_map_[it.key()] = it.value().get_to(special_token);
  }
}

size_t TemplatePostProcessor::DefaultAdded(bool is_single) {
  Template* target = nullptr;
  if (is_single) {
    target = &single_;
  } else {
    target = &pair_;
  }
  return CountAdded(target, special_tokens_map_);
}

void TemplatePostProcessor::UpdateAddedTokensNum() {
  added_single_ = DefaultAdded(true);
  added_pair_ = DefaultAdded(false);
}

void TemplatePostProcessor::UpdateSinglePieces(
    const std::string& template_str) {
  single_.GetPiecesFromStr(template_str);
  added_single_ = DefaultAdded(true);
}

void TemplatePostProcessor::UpdateSinglePieces(
    const std::vector<std::string>& pieces) {
  single_.GetPiecesFromVec(pieces);
  added_single_ = DefaultAdded(true);
}

void TemplatePostProcessor::UpdatePairPieces(const std::string& template_str) {
  pair_.GetPiecesFromStr(template_str);
  added_pair_ = DefaultAdded(false);
}

void TemplatePostProcessor::UpdatePairPieces(
    const std::vector<std::string>& pieces) {
  pair_.GetPiecesFromVec(pieces);
  added_pair_ = DefaultAdded(false);
}

TemplatePostProcessor::TemplatePostProcessor() { UpdateAddedTokensNum(); }

TemplatePostProcessor::TemplatePostProcessor(
    const Template& single,
    const Template& pair,
    const std::vector<SpecialToken>& special_tokens_map)
    : single_(single), pair_(pair), special_tokens_map_(special_tokens_map) {
  UpdateAddedTokensNum();
}

size_t TemplatePostProcessor::AddedTokensNum(bool is_pair) const {
  if (is_pair) {
    return added_pair_;
  }
  return added_single_;
}

void TemplatePostProcessor::SetTokensMap(
    const std::vector<SpecialToken>& special_tokens) {
  special_tokens_map_.SetTokensMap(special_tokens);
  UpdateAddedTokensNum();
}

void TemplatePostProcessor::ApplyTemplate(
    const Template& pieces,
    core::Encoding* encoding,
    core::Encoding* pair_encoding,
    bool add_special_tokens,
    core::Encoding* result_encoding) const {
  size_t new_size = 0;
  for (auto&& piece : pieces.pieces_) {
    if (boost::get<TemplateSequence>(&piece) != nullptr) {
      auto seq_type = boost::get<TemplateSequence>(piece).first;
      if (seq_type == SequenceType::SEQ_A) {
        new_size += encoding->GetLen();
      } else {
        if (pair_encoding == nullptr) {
          throw std::runtime_error(
              "Template expected a pair sequence, but none provided");
        }
        new_size += pair_encoding->GetLen();
      }
    } else {
      if (add_special_tokens) {
        auto&& special_token = boost::get<TemplateSpecialToken>(piece).first;
        if (special_tokens_map_.tokens_map_.find(special_token) !=
            special_tokens_map_.tokens_map_.end()) {
          new_size +=
              special_tokens_map_.tokens_map_.at(special_token).ids_.size();
        }
      }
    }
  }
  std::vector<uint32_t> ids;
  ids.reserve(new_size);
  std::vector<uint32_t> type_ids;
  type_ids.reserve(new_size);
  std::vector<std::string> tokens;
  tokens.reserve(new_size);
  std::vector<uint32_t> words_idx;
  words_idx.reserve(new_size);
  std::vector<core::Offset> offsets;
  offsets.reserve(new_size);
  std::vector<uint32_t> special_tokens_mask;
  special_tokens_mask.reserve(new_size);
  std::vector<uint32_t> attention_mask;
  attention_mask.reserve(new_size);
  std::unordered_map<uint32_t, core::Range> sequence_ranges;
  std::vector<core::Encoding> result_overflowings;
  auto& overflowings = encoding->GetMutableOverflowing();

  core::Encoding result_overflowing_encoding;
  for (auto& overflow_encoding : overflowings) {
    core::Encoding encoding_copy = overflow_encoding;
    core::Encoding pair_encoding_copy;
    if (pair_encoding != nullptr) {
      pair_encoding_copy = *pair_encoding;
      ApplyTemplate(pieces,
                    &encoding_copy,
                    &pair_encoding_copy,
                    add_special_tokens,
                    &result_overflowing_encoding);
      result_overflowings.push_back(result_overflowing_encoding);
      for (auto& pair_overflow_encoding :
           pair_encoding->GetMutableOverflowing()) {
        core::Encoding pair_encoding_copy = pair_overflow_encoding;
        ApplyTemplate(pieces,
                      &encoding_copy,
                      &pair_encoding_copy,
                      add_special_tokens,
                      &result_overflowing_encoding);
        result_overflowings.push_back(result_overflowing_encoding);
      }
    } else {
      ApplyTemplate(pieces,
                    &encoding_copy,
                    pair_encoding,
                    add_special_tokens,
                    &result_overflowing_encoding);
      result_overflowings.push_back(result_overflowing_encoding);
    }
  }
  if (pair_encoding != nullptr) {
    for (auto& pair_overflow_encoding :
         pair_encoding->GetMutableOverflowing()) {
      core::Encoding encoding_copy = *encoding;
      core::Encoding pair_encoding_copy = pair_overflow_encoding;
      ApplyTemplate(pieces,
                    &encoding_copy,
                    &pair_encoding_copy,
                    add_special_tokens,
                    &result_overflowing_encoding);
      result_overflowings.push_back(result_overflowing_encoding);
    }
  }
  VLOG(6) << "Template pieces num: " << pieces.pieces_.size();
  for (auto& piece : pieces.pieces_) {
    if (boost::get<TemplateSequence>(&piece) != nullptr) {
      auto& template_sequence = boost::get<TemplateSequence>(piece);
      if (template_sequence.first == SequenceType::SEQ_A) {
        auto seq_start = ids.size();
        auto seq_end = seq_start + encoding->GetLen();
        sequence_ranges[0] = {seq_start, seq_end};
        ids.insert(
            ids.end(), encoding->GetIds().begin(), encoding->GetIds().end());
        type_ids.insert(
            type_ids.end(), encoding->GetLen(), template_sequence.second);
        tokens.insert(tokens.end(),
                      encoding->GetTokens().begin(),
                      encoding->GetTokens().end());
        words_idx.insert(words_idx.end(),
                         encoding->GetWordsIdx().begin(),
                         encoding->GetWordsIdx().end());
        offsets.insert(offsets.end(),
                       encoding->GetOffsets().begin(),
                       encoding->GetOffsets().end());
        special_tokens_mask.insert(special_tokens_mask.end(),
                                   encoding->GetSpecialTokensMask().begin(),
                                   encoding->GetSpecialTokensMask().end());
        attention_mask.insert(attention_mask.end(),
                              encoding->GetAttentionMask().begin(),
                              encoding->GetAttentionMask().end());
      } else if (template_sequence.first == SequenceType::SEQ_B) {
        if (pair_encoding == nullptr) {
          throw std::runtime_error("Missing pair sequence, checked above");
        }
        auto seq_start = ids.size();
        auto seq_end = seq_start + pair_encoding->GetLen();
        sequence_ranges[0] = {seq_start, seq_end};
        ids.insert(ids.end(),
                   pair_encoding->GetIds().begin(),
                   pair_encoding->GetIds().end());
        type_ids.insert(
            type_ids.end(), pair_encoding->GetLen(), template_sequence.second);
        tokens.insert(tokens.end(),
                      pair_encoding->GetTokens().begin(),
                      pair_encoding->GetTokens().end());
        words_idx.insert(words_idx.end(),
                         pair_encoding->GetWordsIdx().begin(),
                         pair_encoding->GetWordsIdx().end());
        offsets.insert(offsets.end(),
                       pair_encoding->GetOffsets().begin(),
                       pair_encoding->GetOffsets().end());
        special_tokens_mask.insert(
            special_tokens_mask.end(),
            pair_encoding->GetSpecialTokensMask().begin(),
            pair_encoding->GetSpecialTokensMask().end());
        attention_mask.insert(attention_mask.end(),
                              pair_encoding->GetAttentionMask().begin(),
                              pair_encoding->GetAttentionMask().end());
      }
    } else {
      auto& special_token = boost::get<TemplateSpecialToken>(piece);
      if (add_special_tokens) {
        const std::string& id = special_token.first;
        uint32_t type_id = special_token.second;
        auto& tok = special_tokens_map_.tokens_map_.at(
            id);  // We already checked existance above
        auto size = tok.ids_.size();
        ids.insert(ids.end(), tok.ids_.begin(), tok.ids_.end());
        type_ids.insert(type_ids.end(), size, type_id);
        tokens.insert(tokens.end(), tok.tokens_.begin(), tok.tokens_.end());
        words_idx.insert(words_idx.end(), size, -1 /* 2^32 */);
        offsets.insert(offsets.end(), size, {0, 0});
        special_tokens_mask.insert(special_tokens_mask.end(), size, 1);
        attention_mask.insert(attention_mask.end(), size, 1);
      }
    }
  }
  *result_encoding = core::Encoding(std::move(ids),
                                    std::move(type_ids),
                                    std::move(tokens),
                                    std::move(words_idx),
                                    std::move(offsets),
                                    std::move(special_tokens_mask),
                                    std::move(attention_mask),
                                    std::move(result_overflowings),
                                    std::move(sequence_ranges));
}

void TemplatePostProcessor::operator()(core::Encoding* encoding,
                                       core::Encoding* pair_encoding,
                                       bool add_special_tokens,
                                       core::Encoding* result_encoding) const {
  if (pair_encoding != nullptr) {
    ApplyTemplate(
        pair_, encoding, pair_encoding, add_special_tokens, result_encoding);
  } else {
    ApplyTemplate(
        single_, encoding, pair_encoding, add_special_tokens, result_encoding);
  }
}

void to_json(nlohmann::json& j,
             const TemplatePostProcessor& template_postprocessor) {
  j = {
      {"type", "TemplateProcessing"},
      {"single", template_postprocessor.single_},
      {"pair", template_postprocessor.pair_},
      {"special_tokens", template_postprocessor.special_tokens_map_},
  };
}

void from_json(const nlohmann::json& j,
               TemplatePostProcessor& template_postprocessor) {
  j["single"].get_to(template_postprocessor.single_);
  j["pair"].get_to(template_postprocessor.pair_);
  j["special_tokens"].get_to(template_postprocessor.special_tokens_map_);
  template_postprocessor.added_single_ =
      template_postprocessor.DefaultAdded(true);
  template_postprocessor.added_pair_ =
      template_postprocessor.DefaultAdded(false);
}

}  // namespace postprocessors
}  // namespace faster_tokenizer
}  // namespace paddlenlp
