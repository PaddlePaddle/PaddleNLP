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

#include "fast_tokenizer/core/encoding.h"
#include "fast_tokenizer/postprocessors/roberta.h"
#include "fast_tokenizer/pretokenizers/byte_level.h"
#include "glog/logging.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace postprocessors {

RobertaPostProcessor::RobertaPostProcessor(
    const std::pair<std::string, uint32_t>& sep,
    const std::pair<std::string, uint32_t>& cls,
    bool trim_offsets,
    bool add_prefix_space)
    : sep_(sep),
      cls_(cls),
      trim_offsets_(trim_offsets),
      add_prefix_space_(add_prefix_space) {}

size_t RobertaPostProcessor::AddedTokensNum(bool is_pair) const {
  if (is_pair) {
    return 4;
  }
  return 2;
}

void RobertaPostProcessor::operator()(core::Encoding* encoding,
                                      core::Encoding* pair_encoding,
                                      bool add_special_tokens,
                                      core::Encoding* result_encoding) const {
  if (trim_offsets_) {
  }
}

void to_json(nlohmann::json& j,
             const RobertaPostProcessor& roberta_postprocessor) {
  j = {
      {"type", "RobertaPostProcessor"},
      {"sep", roberta_postprocessor.sep_},
      {"cls", roberta_postprocessor.cls_},
      {"trim_offsets", roberta_postprocessor.trim_offsets_},
      {"add_prefix_space", roberta_postprocessor.add_prefix_space_},
  };
}

void from_json(const nlohmann::json& j,
               RobertaPostProcessor& roberta_postprocessor) {
  j["cls"].get_to(roberta_postprocessor.cls_);
  j["sep"].get_to(roberta_postprocessor.sep_);
  j["trim_offsets"].get_to(roberta_postprocessor.trim_offsets_);
  j["add_prefix_space"].get_to(roberta_postprocessor.add_prefix_space_);
}

}  // namespace postprocessors
}  // namespace fast_tokenizer
}  // namespace paddlenlp