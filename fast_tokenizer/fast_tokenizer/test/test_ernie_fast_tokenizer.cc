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

#include <array>
#include <string>
#include <vector>
#include "fast_tokenizer/core/added_vocabulary.h"
#include "fast_tokenizer/core/base.h"
#include "fast_tokenizer/core/encoding.h"
#include "fast_tokenizer/core/tokenizer.h"
#include "fast_tokenizer/models/wordpiece.h"
#include "fast_tokenizer/normalizers/bert.h"
#include "fast_tokenizer/postprocessors/bert.h"
#include "fast_tokenizer/pretokenizers/bert.h"
#include "fast_tokenizer/test/utils.h"
#include "fast_tokenizer/tokenizers/ernie_fast_tokenizer.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace tests {

TEST(tokenizer, ernie_fast_tokenizer) {
  std::string vocab_file = "ernie_vocab.txt";
  tokenizers_impl::ErnieFastTokenizer ernie_fast_tokenizer(vocab_file);
  std::vector<core::Encoding> encodings(2);
  ernie_fast_tokenizer.EncodePairStrings("今天天气真好", &encodings[0]);
  ernie_fast_tokenizer.EncodePairStrings(
      "don't know how this missed award nominations.", &encodings[1]);
  std::vector<std::vector<std::string>> expected_tokens = {
      {"[CLS]", "今", "天", "天", "气", "真", "好", "[SEP]"},
      {"[CLS]",
       "don",
       "[UNK]",
       "t",
       "know",
       "how",
       "this",
       "miss",
       "##ed",
       "award",
       "no",
       "##min",
       "##ations",
       ".",
       "[SEP]"}};
  std::vector<std::vector<uint32_t>> expected_ids = {
      {1, 508, 125, 125, 266, 384, 170, 2},
      {1,
       3362,
       17963,
       2052,
       3821,
       5071,
       3730,
       7574,
       9530,
       6301,
       3825,
       10189,
       11005,
       42,
       2}};
  std::vector<std::vector<uint32_t>> expected_type_ids = {
      {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  for (int i = 0; i < encodings.size(); ++i) {
    CheckVectorEqual(expected_tokens[i], encodings[i].GetTokens());
    CheckVectorEqual(expected_ids[i], encodings[i].GetIds());
    CheckVectorEqual(expected_type_ids[i], encodings[i].GetTypeIds());
  }
}

}  // namespace tests
}  // namespace fast_tokenizer
}  // namespace paddlenlp