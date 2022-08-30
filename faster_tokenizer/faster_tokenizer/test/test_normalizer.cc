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
#include "re2/re2.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace tests {

TEST(normalizers, split) {
  re2::RE2 pattern("-");
  std::string input = "The-final--countdown";
  normalizers::NormalizedString split_input(input);
  auto test_split = [&pattern, &split_input](
      normalizers::SplitMode mode,
      const std::vector<std::string> expected_strings) {
    std::vector<normalizers::NormalizedString> normalizes;
    split_input.Split(pattern, mode, &normalizes);
    ASSERT_EQ(expected_strings.size(), normalizes.size());
    for (int i = 0; i < expected_strings.size(); ++i) {
      ASSERT_EQ(expected_strings[i], normalizes[i].GetStr());
    }
  };

  test_split(normalizers::REMOVED, {"The", "final", "countdown"});
  test_split(normalizers::ISOLATED,
             {"The", "-", "final", "-", "-", "countdown"});
  test_split(normalizers::CONTIGUOUS, {"The", "-", "final", "--", "countdown"});
  test_split(normalizers::MERGED_WITH_PREVIOUS,
             {"The-", "final-", "-", "countdown"});
  test_split(normalizers::MERGED_WITH_NEXT,
             {"The", "-final", "-", "-countdown"});
}

}  // namespace tests
}  // namespace faster_tokenizer
}  // namespace paddlenlp