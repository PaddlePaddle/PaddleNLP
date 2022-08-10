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

TEST(normalizers, strip) {
  std::string input = " \t我爱中国\n\f\v";
  std::string expected_lstrip_output = "我爱中国\n\f\v";
  std::string expected_rstrip_output = " \t我爱中国";
  std::string expected_lrstrip_output = "我爱中国";

  normalizers::NormalizedString lrstrip_input(input);
  normalizers::NormalizedString lstrip_input(input);
  normalizers::NormalizedString rstrip_input(input);

  normalizers::StripNormalizer lrstrip(true, true);
  lrstrip(&lrstrip_input);
  std::string lrstrip_output = lrstrip_input.GetStr();
  ASSERT_EQ(expected_lrstrip_output, lrstrip_output);

  normalizers::StripNormalizer lstrip(true, false);
  lstrip(&lstrip_input);
  std::string lstrip_output = lstrip_input.GetStr();
  ASSERT_EQ(expected_lstrip_output, lstrip_output);

  normalizers::StripNormalizer rstrip(false, true);
  rstrip(&rstrip_input);
  std::string rstrip_output = rstrip_input.GetStr();
  ASSERT_EQ(expected_rstrip_output, rstrip_output);
}

}  // namespace tests
}  // namespace faster_tokenizer
}  // namespace paddlenlp