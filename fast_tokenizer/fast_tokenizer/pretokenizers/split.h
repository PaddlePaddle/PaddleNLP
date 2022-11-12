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

#include "fast_tokenizer/pretokenizers/pretokenizer.h"
#include "fast_tokenizer/utils/utils.h"

namespace re2 {
class RE2;
}  // namespace re2

namespace paddlenlp {
namespace fast_tokenizer {
namespace pretokenizers {

struct FASTTOKENIZER_DECL SplitPreTokenizer : public PreTokenizer {
  SplitPreTokenizer() = default;
  SplitPreTokenizer(const std::string& pattern,
                    core::SplitMode split_mode,
                    bool invert);
  SplitPreTokenizer(const SplitPreTokenizer& split_pretokenizer);
  virtual void operator()(PreTokenizedString* pretokenized) const override;
  friend void to_json(nlohmann::json& j,
                      const SplitPreTokenizer& split_pretokenizer);
  friend void from_json(const nlohmann::json& j,
                        SplitPreTokenizer& split_pretokenizer);

private:
  bool invert_;
  core::SplitMode split_mode_;
  std::unique_ptr<re2::RE2> pattern_;
};

}  // namespace pretokenizers
}  // namespace fast_tokenizer
}  // namespace paddlenlp