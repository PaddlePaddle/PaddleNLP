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
#include "core/added_vocabulary.h"
#include "core/base.h"
#include "core/encoding.h"
#include "core/tokenizer.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "models/wordpiece.h"
#include "normalizers/bert.h"
#include "postprocessors/bert.h"
#include "pretokenizers/bert.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace tests {

template <typename T>
void CheckVectorEqual(const std::vector<T>& a, const std::vector<T>& b) {
  ASSERT_EQ(a.size(), b.size());
  auto size = a.size();
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(a[i], b[i]);
  }
}

TEST(tokenizer, bert_tokenizer) {
  models::WordPieceFactory factory;
  std::string vocab_file = "ernie_vocab.txt";
  factory.SetFiles(vocab_file);
  // Declare the components of tokenizer
  auto word_piece = factory.CreateWordPieceModel();
  auto normalizer = normalizers::BertNormalizer();
  auto pretokenizer = pretokenizers::BertPreTokenizer();
  auto postprocessor =
      postprocessors::BertPostProcessor({"[SEP]", 2}, {"[CLS]", 1});
  core::PadMethod pad_method;
  core::TruncMethod trunc_method;

  // Initialize tokenizer
  core::Tokenizer tokenizer(word_piece);
  tokenizer.SetNormalizer(normalizer);
  tokenizer.SetPreTokenizer(pretokenizer);
  tokenizer.SetPostProcessor(postprocessor);
  tokenizer.SetPadMethod(pad_method);
  tokenizer.SetTruncMethod(trunc_method);
  std::vector<core::AddedToken> special_added_tokens = {
      {"[PAD]", true},
      {"[CLS]", true},
      {"[SEP]", true},
      {"[MASK]", true},
      {"[UNK]", true},
  };
  auto special_tokens_num = tokenizer.AddSpecialTokens(special_added_tokens);

  // Tokenize the sample strings
  std::vector<core::Encoding> encodings(2);
  tokenizer.EncodePairStrings("今天天气真好", &encodings[0]);
  tokenizer.EncodePairStrings("don't know how this missed award nominations.",
                              &encodings[1]);
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
}  // namespace faster_tokenizer
}  // namespace paddlenlp