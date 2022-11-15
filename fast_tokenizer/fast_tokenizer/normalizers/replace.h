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

#include <memory>
#include <string>
#include "nlohmann/json.hpp"
#include "fast_tokenizer/normalizers/normalizer.h"
#include "re2/re2.h"
#include "fast_tokenizer/utils/utils.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace normalizers {

struct FASTTOKENIZER_DECL ReplaceNormalizer : public Normalizer {
  ReplaceNormalizer() = default;
  ReplaceNormalizer(const std::string& pattern, const std::string& content);
  ReplaceNormalizer(const ReplaceNormalizer& replace_normalizer);
  virtual void operator()(NormalizedString* mut_str) const override;
  friend void to_json(nlohmann::json& j,
                      const ReplaceNormalizer& replace_normalizer);
  friend void from_json(const nlohmann::json& j,
                        ReplaceNormalizer& replace_normalizer);

private:
  std::unique_ptr<re2::RE2> pattern_;
  std::string content_;
};

}  // namespace normalizers
}  // namespace fast_tokenizer
}  // namespace paddlenlp
