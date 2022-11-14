// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "fast_tokenizer/core/encoding.h"
#include "fast_tokenizer/postprocessors/roberta.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace tests {

TEST(postprocessors, roberta) {
  postprocessors::RobertaPostProcessor postprocessor;
  core::Encoding encoding(
      {core::Token(12, "Hello", {0, 5}), core::Token(14, "there", {6, 11})}, 0);
  core::Encoding pair_encoding({core::Token(15, "pair", {0, 4})}, 0);
  core::Encoding result_encoding;

  core::Encoding encoding_copy = encoding;
  core::Encoding pair_encoding_copy = pair_encoding;

  postprocessor(&encoding_copy, nullptr, true, &result_encoding);
  uint32_t special_word_idx = std::numeric_limits<uint32_t>::max();
  ASSERT_EQ(result_encoding,
            core::Encoding({0, 12, 14, 2},
                           {0, 0, 0, 0},
                           {"<s>", "Hello", "there", "</s>"},
                           std::vector<uint32_t>(4, special_word_idx),
                           {{0, 0}, {0, 5}, {6, 11}, {0, 0}},
                           {1, 0, 0, 1},
                           {1, 1, 1, 1},
                           {},
                           {{0, {1, 3}}}));
  ASSERT_EQ(result_encoding.TokenIdxToSequenceIds(2),
            std::vector<uint32_t>(1, 0));
  ASSERT_EQ(result_encoding.TokenIdxToSequenceIds(3).size(), 0);

  encoding_copy = encoding;
  postprocessor(&encoding_copy, &pair_encoding_copy, true, &result_encoding);
  ASSERT_EQ(
      result_encoding,
      core::Encoding({0, 12, 14, 2, 2, 15, 2},
                     {0, 0, 0, 0, 0, 0, 0},
                     {"<s>", "Hello", "there", "</s>", "</s>", "pair", "</s>"},
                     std::vector<uint32_t>(7, special_word_idx),
                     {{0, 0}, {0, 5}, {6, 11}, {0, 0}, {0, 0}, {0, 4}, {0, 0}},
                     {1, 0, 0, 1, 1, 0, 1},
                     {1, 1, 1, 1, 1, 1, 1},
                     {},
                     {{0, {1, 3}}, {1, {5, 6}}}));

  ASSERT_EQ(result_encoding.TokenIdxToSequenceIds(2),
            std::vector<uint32_t>(1, 0));
  ASSERT_EQ(result_encoding.TokenIdxToSequenceIds(3), std::vector<uint32_t>{});
  ASSERT_EQ(result_encoding.TokenIdxToSequenceIds(4), std::vector<uint32_t>{});
  ASSERT_EQ(result_encoding.TokenIdxToSequenceIds(5), std::vector<uint32_t>{1});
  ASSERT_EQ(result_encoding.TokenIdxToSequenceIds(6), std::vector<uint32_t>{});

  encoding_copy = encoding;
  pair_encoding_copy = pair_encoding;
  postprocessor(&encoding_copy, &pair_encoding_copy, false, &result_encoding);
  ASSERT_EQ(result_encoding,
            core::Encoding({12, 14, 15},
                           {0, 0, 0},
                           {"Hello", "there", "pair"},
                           std::vector<uint32_t>(3, special_word_idx),
                           {{0, 5}, {6, 11}, {0, 4}},
                           {0, 0, 0},
                           {1, 1, 1},
                           {},
                           {{0, {0, 2}}, {1, {2, 3}}}));

  ASSERT_EQ(result_encoding.TokenIdxToSequenceIds(0), std::vector<uint32_t>{0});
  ASSERT_EQ(result_encoding.TokenIdxToSequenceIds(1), std::vector<uint32_t>{0});
  ASSERT_EQ(result_encoding.TokenIdxToSequenceIds(2), std::vector<uint32_t>{1});
}

}  // namespace tests
}  // namespace fast_tokenizer
}  // namespace paddlenlp