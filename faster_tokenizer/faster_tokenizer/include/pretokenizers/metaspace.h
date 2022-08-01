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
#include "pretokenizers/pretokenizer.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace pretokenizers {

struct MetaSpacePreTokenizer : public PreTokenizer {
  // Replaces white space with U+2581 (LOWER ONE EIGHT BLOCK)
  MetaSpacePreTokenizer(const std::string& replacement = "\xe2\x96\x81",
                        bool add_prefix_space = true);
  MetaSpacePreTokenizer(const MetaSpacePreTokenizer&) = default;
  virtual void operator()(PreTokenizedString* pretokenized) const override;
  std::string GetReplacement() const;
  void SetReplacement(const std::string&);

private:
  void UpdateReplacementChar();
  std::string replacement_;
  bool add_prefix_space_;
  char32_t replacement_char_;

  friend void to_json(nlohmann::json& j,
                      const MetaSpacePreTokenizer& meta_pretokenizer);
  friend void from_json(const nlohmann::json& j,
                        MetaSpacePreTokenizer& meta_pretokenizer);
};

}  // namespace pretokenizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
