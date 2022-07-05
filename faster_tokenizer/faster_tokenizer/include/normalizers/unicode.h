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

struct NFCNormalizer : public Normalizer {
  virtual void operator()(NormalizedString* input) const override;
  friend void to_json(nlohmann::json& j, const NFCNormalizer& normalizer);
  friend void from_json(const nlohmann::json& j, NFCNormalizer& normalizer);
};

struct NFDNormalizer : public Normalizer {
  virtual void operator()(NormalizedString* input) const override;
  friend void to_json(nlohmann::json& j, const NFDNormalizer& normalizer);
  friend void from_json(const nlohmann::json& j, NFDNormalizer& normalizer);
};

struct NFKCNormalizer : public Normalizer {
  virtual void operator()(NormalizedString* input) const override;
  friend void to_json(nlohmann::json& j, const NFKCNormalizer& normalizer);
  friend void from_json(const nlohmann::json& j, NFKCNormalizer& normalizer);
};

struct NFKDNormalizer : public Normalizer {
  virtual void operator()(NormalizedString* input) const override;
  friend void to_json(nlohmann::json& j, const NFKDNormalizer& normalizer);
  friend void from_json(const nlohmann::json& j, NFKDNormalizer& normalizer);
};

struct NmtNormalizer : public Normalizer {
  virtual void operator()(NormalizedString* input) const override;
  friend void to_json(nlohmann::json& j, const NmtNormalizer& normalizer);
  friend void from_json(const nlohmann::json& j, NmtNormalizer& normalizer);
};

}  // namespace normalizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
