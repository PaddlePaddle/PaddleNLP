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
#include "fast_tokenizer/pretokenizers/split.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "re2/re2.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace tests {

TEST(pretokenizers, split_basic) {
  std::string input = "How are you doing?";
  // All tokens' id are set to 0.
  std::vector<std::pair<core::SplitMode, std::vector<core::Token>>> test_cases =
      {{
           core::SplitMode::REMOVED,
           std::vector<core::Token>{{0, "How", {0, 3}},
                                    {0, "are", {4, 7}},
                                    {0, "you", {8, 11}},
                                    {0, "doing", {12, 17}},
                                    {0, "?", {17, 18}}},
       },
       {
           core::SplitMode::ISOLATED,
           std::vector<core::Token>{{0, "How", {0, 3}},
                                    {0, " ", {3, 4}},
                                    {0, "are", {4, 7}},
                                    {0, " ", {7, 8}},
                                    {0, "you", {8, 11}},
                                    {0, " ", {11, 12}},
                                    {0, "doing", {12, 17}},
                                    {0, "?", {17, 18}}},
       },
       {
           core::SplitMode::MERGED_WITH_PREVIOUS,
           std::vector<core::Token>{{0, "How ", {0, 4}},
                                    {0, "are ", {4, 8}},
                                    {0, "you ", {8, 12}},
                                    {0, "doing", {12, 17}},
                                    {0, "?", {17, 18}}},
       },
       {
           core::SplitMode::MERGED_WITH_NEXT,
           std::vector<core::Token>{{0, "How", {0, 3}},
                                    {0, " are", {3, 7}},
                                    {0, " you", {7, 11}},
                                    {0, " doing", {11, 17}},
                                    {0, "?", {17, 18}}},
       },
       {
           core::SplitMode::CONTIGUOUS,
           std::vector<core::Token>{{0, "How", {0, 3}},
                                    {0, " ", {3, 4}},
                                    {0, "are", {4, 7}},
                                    {0, " ", {7, 8}},
                                    {0, "you", {8, 11}},
                                    {0, " ", {11, 12}},
                                    {0, "doing?", {12, 18}}},
       }};
  std::string pattern = R"(\w+|[^\w\s]+)";
  for (auto&& test : test_cases) {
    pretokenizers::PreTokenizedString pretokenized(input);
    pretokenizers::SplitPreTokenizer pretok(pattern, test.first, true);
    pretok(&pretokenized);
    ASSERT_EQ(test.second.size(), pretokenized.GetSplitsSize());
    for (int i = 0; i < test.second.size(); ++i) {
      auto&& curr_split = pretokenized.GetSplit(i);
      ASSERT_EQ(test.second[i].value_, curr_split.normalized_.GetStr());
      auto original_offset = curr_split.normalized_.GetOrginalOffset();
      ASSERT_EQ(test.second[i].offset_, original_offset);
    }
  }
}

TEST(pretokenizers, split_invert) {
  std::string input = "Hello Hello Hello";
  pretokenizers::PreTokenizedString pretok_str(input),
      pretok_str_for_invert(input);
  pretokenizers::SplitPreTokenizer pretok(" ", core::SplitMode::REMOVED, false);
  pretokenizers::SplitPreTokenizer pretok_invert(
      "Hello", core::SplitMode::REMOVED, true);

  pretok(&pretok_str);
  pretok_invert(&pretok_str_for_invert);

  ASSERT_EQ(pretok_str.GetSplitsSize(), pretok_str_for_invert.GetSplitsSize());
  for (int i = 0; i < pretok_str.GetSplitsSize(); ++i) {
    ASSERT_EQ(pretok_str.GetSplit(i).normalized_.GetStr(),
              pretok_str_for_invert.GetSplit(i).normalized_.GetStr());
  }
}

}  // namespace tests
}  // namespace fast_tokenizer
}  // namespace paddlenlp