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

namespace tokenizers {
namespace normalizers {

struct StripNormalizer : public Normalizer {
  StripNormalizer(bool left = true, bool right = true);
  virtual void operator()(NormalizedString* input) const override;

private:
  bool left_;
  bool right_;
};

struct StripAccentsNormalizer : public Normalizer {
  virtual void operator()(NormalizedString* input) const override;
};

}  // normalizers
}  // tokenizers
