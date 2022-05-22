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
#include "nlohmann/json.hpp"
#include "postprocessors/postprocessor.h"

namespace tokenizers {
namespace postprocessors {

enum SequenceType { SEQ_A, SEQ_B };
NLOHMANN_JSON_SERIALIZE_ENUM(SequenceType,
                             {
                                 {SEQ_A, "A"}, {SEQ_B, "B"},
                             });
// The template indicate `${Id} : ${TypeId}`
using TemplateSequence = std::pair<SequenceType, uint>;
using TemplateSpecialToken = std::pair<std::string, uint>;

struct TemplatePiece : boost::variant<TemplateSequence, TemplateSpecialToken> {
  void ParseIdFromString(const std::string& template_id_string);
  void SetTypeId(uint type_id);
  static TemplatePiece CreateTemplatePiece(const std::string& template_string);
  friend void to_json(nlohmann::json& j, const TemplatePiece& template_piece);
  friend void from_json(const nlohmann::json& j, TemplatePiece& template_piece);
};

struct SpecialToken {
  std::string id_;
  std::vector<uint> ids_;
  std::vector<std::string> tokens_;
  SpecialToken() = default;
  SpecialToken(const std::string& id,
               const std::vector<uint>& ids,
               const std::vector<std::string>& tokens)
      : id_(id), ids_(ids), tokens_(tokens) {}
  SpecialToken(const std::string& token, uint id) {
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
  Template(const std::string& template_str) {
    TemplatePiece template_piece;
    std::vector<std::string> pieces;

    // Parse the pieces
    size_t start = template_str.find_first_not_of(" ");
    size_t pos;
    while ((pos = template_str.find_first_of(" ", start)) !=
           std::string::npos) {
      pieces.push_back(template_str.substr(start, pos - start));
      start = template_str.find_first_not_of(" ", pos);
    }
    AddStringPiece(pieces);
  }

  Template(const std::vector<TemplatePiece>& pieces) : pieces_(pieces) {}
  Template(const std::vector<std::string>& pieces) { AddStringPiece(pieces); }

private:
  void AddStringPiece(const std::vector<std::string>& pieces) {
    TemplatePiece template_piece;
    for (auto&& piece : pieces) {
      template_piece.ParseIdFromString(piece);
      pieces_.push_back(template_piece);
    }
  }

  friend void to_json(nlohmann::json& j, const Template& template_);
  friend void from_json(const nlohmann::json& j, Template& template_);
};

struct SpecialTokensMap {
  std::unordered_map<std::string, SpecialToken> tokens_map_;
  SpecialTokensMap() = default;
  SpecialTokensMap(const std::vector<SpecialToken>& special_tokens) {
    for (auto&& special_token : special_tokens) {
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

}  // postprocessors
}  // tokenizers
