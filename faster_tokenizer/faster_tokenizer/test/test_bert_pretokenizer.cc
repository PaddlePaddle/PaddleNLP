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
#include <vector>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "pretokenizers/bert.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace tests {
TEST(pretokenizers, whitespace) {
  std::string input =
      "I \t am good\r   at \nsport. I like\tfootball especially!!!";
  std::vector<std::string> expected_outputs = {"I",
                                               "am",
                                               "good",
                                               "at",
                                               "sport",
                                               ".",
                                               "I",
                                               "like",
                                               "football",
                                               "especially",
                                               "!",
                                               "!",
                                               "!"};
  pretokenizers::PreTokenizedString bert_input(input);
  pretokenizers::BertPreTokenizer()(&bert_input);
  ASSERT_EQ(expected_outputs.size(), bert_input.GetSplitsSize());
  for (int i = 0; i < expected_outputs.size(); ++i) {
    ASSERT_EQ(bert_input.GetSplit(i).normalized_.GetStr(), expected_outputs[i]);
  }
}
}  // namespace tests
}  // namespace faster_tokenizer
}  // namespace paddlenlp