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

#include <string>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "normalizers/bert.h"
#include "normalizers/replace.h"
#include "normalizers/strip.h"
#include "normalizers/unicode.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace tests {

TEST(normalizers, utils) {
  std::string input = "ÓÓßSSCHLOË";
  std::string expected_output = "óóßsschloë";
  normalizers::NormalizedString lower_input(input);
  lower_input.Lowercase();
  ASSERT_EQ(expected_output, lower_input.GetStr());
}

}  // namespace tests
}  // namespace faster_tokenizer
}  // namespace paddlenlp