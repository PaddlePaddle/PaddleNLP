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

#include "nlohmann/json.hpp"
#include "postprocessors/postprocessor.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace postprocessors {

struct BertPostProcessor : public PostProcessor {
  BertPostProcessor(const std::pair<std::string, uint32_t>& sep,
                    const std::pair<std::string, uint32_t>& cls);
  BertPostProcessor();
  virtual size_t AddedTokensNum(bool is_pair) const override;
  virtual void operator()(core::Encoding* encoding,
                          core::Encoding* pair_encoding,
                          bool add_special_tokens,
                          core::Encoding* result_encoding) const override;
  std::pair<std::string, uint32_t> sep_;
  std::pair<std::string, uint32_t> cls_;
  friend void to_json(nlohmann::json& j,
                      const BertPostProcessor& bert_postprocessor);
  friend void from_json(const nlohmann::json& j,
                        BertPostProcessor& bert_postprocessor);
};
}  // namespace postprocessors
}  // namespace faster_tokenizer
}  // namespace paddlenlp
