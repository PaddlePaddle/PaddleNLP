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
#include <vector>
#include "normalizers/normalizer.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace normalizers {

struct SequenceNormalizer : public Normalizer {
  SequenceNormalizer() = default;
  SequenceNormalizer(const SequenceNormalizer&) = default;
  SequenceNormalizer(const std::vector<Normalizer*>& normalizers);
  virtual void operator()(NormalizedString* input) const override;
  void AppendNormalizer(Normalizer* normalizer);

private:
  std::vector<std::shared_ptr<Normalizer>> normalizer_ptrs_;
  friend void to_json(nlohmann::json& j, const SequenceNormalizer& normalizer);
  friend void from_json(const nlohmann::json& j,
                        SequenceNormalizer& normalizer);
};

struct LowercaseNormalizer : public Normalizer {
  virtual void operator()(NormalizedString* input) const override;
  friend void to_json(nlohmann::json& j, const LowercaseNormalizer& normalizer);
  friend void from_json(const nlohmann::json& j,
                        LowercaseNormalizer& normalizer);
};

}  // namespace normalizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
