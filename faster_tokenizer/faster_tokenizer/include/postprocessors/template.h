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

#include <string>
#include <unordered_map>
#include <vector>

#include "boost/variant.hpp"
#include "glog/logging.h"
#include "nlohmann/json.hpp"
#include "postprocessors/postprocessor.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace postprocessors {

enum SequenceType { SEQ_A, SEQ_B };
NLOHMANN_JSON_SERIALIZE_ENUM(SequenceType,
                             {
                                 {SEQ_A, "A"}, {SEQ_B, "B"},
                             });
// The template indicate `${Id} : ${TypeId}`
using TemplateSequence = std::pair<SequenceType, uint32_t>;
using TemplateSpecialToken = std::pair<std::string, uint32_t>;

using TemplatePiece = boost::variant<TemplateSequence, TemplateSpecialToken>;
void to_json(nlohmann::json& j, const TemplatePiece& template_piece);
void from_json(const nlohmann::json& j, TemplatePiece& template_piece);

void ParseIdFromString(const std::string& template_id_string,
                       TemplatePiece* template_piece);
void SetTypeId(uint32_t type_id, TemplatePiece* template_piece);
void GetTemplatePieceFromString(const std::string& template_string,
                                TemplatePiece* template_piece);

struct SpecialToken {
  std::string id_;
  std::vector<uint32_t> ids_;
  std::vector<std::string> tokens_;
  SpecialToken() = default;
  SpecialToken(const std::string& id,
               const std::vector<uint32_t>& ids,
               const std::vector<std::string>& tokens)
      : id_(id), ids_(ids), tokens_(tokens) {}
  SpecialToken(const std::string& token, uint32_t id) {
    id_ = token;
    ids_.push_back(id);
    tokens_.push_back(token);
  }
  friend void to_json(nlohmann::json& j, const SpecialToken& special_token);
  friend void from_json(const nlohmann::json& j, SpecialToken& special_token);
};

struct Template {
  std::vector<TemplatePiece> pieces_;
  Template() = default;
  explicit Template(const std::string& template_str) {
    std::vector<std::string> pieces;

    // Parse the pieces
    size_t start = template_str.find_first_not_of(" ");
    size_t pos;
    while ((pos = template_str.find_first_of(" ", start)) !=
           std::string::npos) {
      pieces.push_back(template_str.substr(start, pos - start));
      start = template_str.find_first_not_of(" ", pos);
    }
    if (start != std::string::npos) {
      pieces.push_back(template_str.substr(start));
    }
    AddStringPiece(pieces);
  }

  explicit Template(const std::vector<TemplatePiece>& pieces)
      : pieces_(pieces) {}
  explicit Template(const std::vector<std::string>& pieces) {
    AddStringPiece(pieces);
  }

  void GetPiecesFromVec(const std::vector<std::string>& pieces) {
    AddStringPiece(pieces);
  }

  void GetPiecesFromStr(const std::string& template_str) {
    std::vector<std::string> pieces;

    // Parse the pieces
    size_t start = template_str.find_first_not_of(" ");
    size_t pos;
    while ((pos = template_str.find_first_of(" ", start)) !=
           std::string::npos) {
      pieces.push_back(template_str.substr(start, pos - start));
      start = template_str.find_first_not_of(" ", pos);
    }
    if (start != std::string::npos) {
      pieces.push_back(template_str.substr(start));
    }
    AddStringPiece(pieces);
  }

  void Clean() { pieces_.clear(); }

private:
  void AddStringPiece(const std::vector<std::string>& pieces) {
    for (auto&& piece : pieces) {
      TemplatePiece template_piece;
      GetTemplatePieceFromString(piece, &template_piece);
      if (boost::get<TemplateSequence>(&template_piece)) {
        pieces_.push_back(boost::get<TemplateSequence>(template_piece));
      } else {
        pieces_.push_back(boost::get<TemplateSpecialToken>(template_piece));
      }
    }
  }

  friend void to_json(nlohmann::json& j, const Template& template_);
  friend void from_json(const nlohmann::json& j, Template& template_);
};

struct SpecialTokensMap {
  std::unordered_map<std::string, SpecialToken> tokens_map_;
  SpecialTokensMap() = default;
  explicit SpecialTokensMap(const std::vector<SpecialToken>& special_tokens) {
    SetTokensMap(special_tokens);
  }
  void SetTokensMap(const std::vector<SpecialToken>& special_tokens) {
    tokens_map_.clear();
    for (const auto& special_token : special_tokens) {
      tokens_map_.insert({special_token.id_, special_token});
    }
  }
  friend void to_json(nlohmann::json& j, const SpecialTokensMap& tokens_map);
  friend void from_json(const nlohmann::json& j, SpecialTokensMap& tokens_map);
};

struct TemplatePostProcessor : public PostProcessor {
  TemplatePostProcessor();
  TemplatePostProcessor(const Template&,
                        const Template&,
                        const std::vector<SpecialToken>&);

  virtual void operator()(core::Encoding* encoding,
                          core::Encoding* pair_encoding,
                          bool add_special_tokens,
                          core::Encoding* result_encoding) const override;
  virtual size_t AddedTokensNum(bool is_pair) const override;

  void UpdateSinglePieces(const std::string& template_str);
  void UpdateSinglePieces(const std::vector<std::string>& pieces);
  void UpdatePairPieces(const std::string& template_str);
  void UpdatePairPieces(const std::vector<std::string>& pieces);
  void UpdateAddedTokensNum();
  void SetTokensMap(const std::vector<SpecialToken>& special_tokens);
  size_t CountAdded(Template* template_,
                    const SpecialTokensMap& special_tokens_map);
  size_t DefaultAdded(bool is_single = true);
  void ApplyTemplate(const Template& pieces,
                     core::Encoding* encoding,
                     core::Encoding* pair_encoding,
                     bool add_special_tokens,
                     core::Encoding* result_encoding) const;

  friend void to_json(nlohmann::json& j,
                      const TemplatePostProcessor& template_postprocessor);
  friend void from_json(const nlohmann::json& j,
                        TemplatePostProcessor& template_postprocessor);

  Template single_;
  Template pair_;
  size_t added_single_;
  size_t added_pair_;
  SpecialTokensMap special_tokens_map_;
};

}  // namespace postprocessors
}  // namespace faster_tokenizer
}  // namespace paddlenlp
