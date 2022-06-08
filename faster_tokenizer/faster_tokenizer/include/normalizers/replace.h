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
#include "re2/re2.h"

namespace tokenizers {
namespace normalizers {

struct ReplaceNormalizer : public Normalizer {
  ReplaceNormalizer(const std::string& pattern, const std::string& content);
  ReplaceNormalizer(const ReplaceNormalizer& replace_normalizer);
  virtual void operator()(NormalizedString* mut_str) const override;

private:
  re2::RE2 pattern_;
  std::string content_;
};

}  // normalizers
}  // tokenizers
