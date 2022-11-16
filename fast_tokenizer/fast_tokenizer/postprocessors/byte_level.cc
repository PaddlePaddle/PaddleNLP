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

#include "fast_tokenizer/postprocessors/byte_level.h"
#include "fast_tokenizer/pretokenizers/byte_level.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace postprocessors {

ByteLevelPostProcessor::ByteLevelPostProcessor(bool add_prefix_space,
                                               bool trim_offsets,
                                               bool use_regex)
    : add_prefix_space_(add_prefix_space),
      trim_offsets_(trim_offsets),
      use_regex_(use_regex) {}


size_t ByteLevelPostProcessor::AddedTokensNum(bool is_pair) const { return 0; }

void ByteLevelPostProcessor::operator()(core::Encoding* encoding,
                                        core::Encoding* pair_encoding,
                                        bool add_special_tokens,
                                        core::Encoding* result_encoding) const {
  if (trim_offsets_) {
    pretokenizers::ProcessOffsets(encoding, add_special_tokens);
    for (auto& overflowing : encoding->GetMutableOverflowing()) {
      pretokenizers::ProcessOffsets(&overflowing, add_special_tokens);
    }
    if (pair_encoding != nullptr) {
      pretokenizers::ProcessOffsets(pair_encoding, add_special_tokens);
      for (auto& overflowing : pair_encoding->GetMutableOverflowing()) {
        pretokenizers::ProcessOffsets(&overflowing, add_special_tokens);
      }
    }
  }

  encoding->SetSequenceIds(0);
  if (pair_encoding != nullptr) {
    pair_encoding->SetSequenceIds(1);
  }
}

void to_json(nlohmann::json& j,
             const ByteLevelPostProcessor& byte_level_postprocessor) {
  j = {
      {"type", "ByteLevelPostProcessor"},
      {"add_prefix_space", byte_level_postprocessor.add_prefix_space_},
      {"trim_offsets", byte_level_postprocessor.trim_offsets_},
      {"use_regex", byte_level_postprocessor.use_regex_},
  };
}

void from_json(const nlohmann::json& j,
               ByteLevelPostProcessor& byte_level_postprocessor) {
  j["add_prefix_space"].get_to(byte_level_postprocessor.add_prefix_space_);
  j["trim_offsets"].get_to(byte_level_postprocessor.trim_offsets_);
  j["use_regex"].get_to(byte_level_postprocessor.use_regex_);
}

}  // namespace postprocessors
}  // namespace fast_tokenizer
}  // namespace paddlenlp