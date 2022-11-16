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

#include "fast_tokenizer/pretokenizers/pretokenizer.h"
#include "fast_tokenizer/utils/utils.h"
#include "nlohmann/json.hpp"


namespace paddlenlp {
namespace fast_tokenizer {
namespace pretokenizers {

struct FASTTOKENIZER_DECL ByteLevelPreTokenizer : public PreTokenizer {
  ByteLevelPreTokenizer(bool add_prefix_space = true, bool use_regex = true);
  virtual void operator()(PreTokenizedString* pretokenized) const override;
  friend void to_json(nlohmann::json& j,
                      const ByteLevelPreTokenizer& byte_pre_tokenizer);
  friend void from_json(const nlohmann::json& j,
                        ByteLevelPreTokenizer& byte_pre_tokenizer);

private:
  bool add_prefix_space_;
  bool use_regex_;
};

void FASTTOKENIZER_DECL ProcessOffsets(core::Encoding* encoding,
                                       bool add_prefix_space);

}  // namespace pretokenizers
}  // namespace fast_tokenizer
}  // namespace paddlenlp
