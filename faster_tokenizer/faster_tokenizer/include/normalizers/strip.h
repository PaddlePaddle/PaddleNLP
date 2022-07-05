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

#include "normalizers/normalizer.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace normalizers {

struct StripNormalizer : public Normalizer {
  StripNormalizer(bool left = true, bool right = true);
  virtual void operator()(NormalizedString* input) const override;
  StripNormalizer(StripNormalizer&&) = default;
  StripNormalizer(const StripNormalizer&) = default;

private:
  bool left_;
  bool right_;
  friend void to_json(nlohmann::json& j,
                      const StripNormalizer& strip_normalizer);
  friend void from_json(const nlohmann::json& j,
                        StripNormalizer& strip_normalizer);
};

struct StripAccentsNormalizer : public Normalizer {
  virtual void operator()(NormalizedString* input) const override;
  friend void to_json(nlohmann::json& j,
                      const StripAccentsNormalizer& strip_normalizer);
  friend void from_json(const nlohmann::json& j,
                        StripAccentsNormalizer& strip_normalizer);
};

}  // namespace normalizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
