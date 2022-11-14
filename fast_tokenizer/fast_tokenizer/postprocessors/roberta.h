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

#pragma once

#include "fast_tokenizer/postprocessors/postprocessor.h"
#include "fast_tokenizer/utils/utils.h"
#include "nlohmann/json.hpp"

namespace paddlenlp {
namespace fast_tokenizer {
namespace postprocessors {

struct FASTTOKENIZER_DECL RobertaPostProcessor : public PostProcessor {
  RobertaPostProcessor(const std::pair<std::string, uint32_t>& sep = {"</s>",
                                                                      2},
                       const std::pair<std::string, uint32_t>& cls = {"<s>", 0},
                       bool trim_offsets = true,
                       bool add_prefix_space = true);
  virtual size_t AddedTokensNum(bool is_pair) const override;
  virtual void operator()(core::Encoding* encoding,
                          core::Encoding* pair_encoding,
                          bool add_special_tokens,
                          core::Encoding* result_encoding) const override;
  std::pair<std::string, uint32_t> sep_;
  std::pair<std::string, uint32_t> cls_;
  bool trim_offsets_;
  bool add_prefix_space_;
  friend void to_json(nlohmann::json& j,
                      const RobertaPostProcessor& roberta_postprocessor);
  friend void from_json(const nlohmann::json& j,
                        RobertaPostProcessor& roberta_postprocessor);
};
}  // namespace postprocessors
}  // namespace fast_tokenizer
}  // namespace paddlenlp