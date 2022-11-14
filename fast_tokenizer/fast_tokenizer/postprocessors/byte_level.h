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

struct FASTTOKENIZER_DECL ByteLevelPostProcessor : public PostProcessor {
  ByteLevelPostProcessor(bool add_prefix_space = true,
                         bool trim_offsets = true,
                         bool use_regex = true);
  virtual size_t AddedTokensNum(bool is_pair) const override;
  virtual void operator()(core::Encoding* encoding,
                          core::Encoding* pair_encoding,
                          bool add_special_tokens,
                          core::Encoding* result_encoding) const override;

  friend void to_json(nlohmann::json& j,
                      const ByteLevelPostProcessor& byte_level_postprocessor);
  friend void from_json(const nlohmann::json& j,
                        ByteLevelPostProcessor& byte_level_postprocessor);
  bool add_prefix_space_;
  bool trim_offsets_;
  bool use_regex_;
};

}  // namespace postprocessors
}  // namespace fast_tokenizer
}  // namespace paddlenlp
