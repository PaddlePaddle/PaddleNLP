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
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "models/wordpiece.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace tests {

TEST(model, wordpiece_factory) {
  models::WordPieceFactory factory;
  auto check_config = [&](const std::string& filename,
                          size_t vocab_size,
                          const std::string& unk_token,
                          size_t max_input_chars_per_word,
                          const std::string& continuing_subword_prefix) {
    ASSERT_EQ(filename, factory.config_.files_);
    ASSERT_EQ(vocab_size, factory.config_.vocab_.size());
    ASSERT_EQ(unk_token, factory.config_.unk_token_);
    ASSERT_EQ(max_input_chars_per_word,
              factory.config_.max_input_chars_per_word_);
    ASSERT_EQ(continuing_subword_prefix,
              factory.config_.continuing_subword_prefix_);
  };
  check_config("", 0, "[UNK]", 100, "##");

  std::string vocab_file = "ernie_vocab.txt";
  factory.SetFiles(vocab_file);
  factory.GetVocabFromFiles(vocab_file);
  check_config(vocab_file, 17964UL, "[UNK]", 100, "##");
}

TEST(model, wordpiece_model) {
  models::WordPieceFactory factory;
  factory.SetFiles("ernie_vocab.txt");

  auto wordpiece_model = factory.CreateWordPieceModel();
  auto check_token_id = [&](const std::string& expected_token,
                            uint32_t expected_id) {
    std::string token;
    uint32_t id;
    wordpiece_model.TokenToId(expected_token, &id);
    wordpiece_model.IdToToken(expected_id, &token);
    ASSERT_EQ(id, expected_id);
    ASSERT_EQ(token, expected_token);
  };
  std::array<std::string, 10> tokens = {
      "[PAD]", "[CLS]", "[SEP]", "[MASK]", "，", "的", "、", "一", "人", "有"};
  for (int i = 0; i < tokens.size(); i++) {
    check_token_id(tokens[i], i);
  }
  // check non-exist token
  uint32_t id;
  ASSERT_FALSE(wordpiece_model.TokenToId("xxsada", &id));
  // check non-exist id
  std::string token;
  ASSERT_FALSE(
      wordpiece_model.IdToToken(wordpiece_model.GetVocabSize(), &token));

  // Check Tokenize Chinese
  auto chinese_tokens = wordpiece_model.Tokenize("今天天气真好");
  auto check_token = [](const core::Token& token,
                        const std::string& expected_string,
                        uint32_t id,
                        core::Offset offset) {
    ASSERT_EQ(token.value_, expected_string);
    ASSERT_EQ(token.id_, id);
    ASSERT_EQ(token.offset_, offset);
  };
  check_token(chinese_tokens[0], "今", 508, {0, 3});
  check_token(chinese_tokens[1], "##天", 12172, {3, 6});
  check_token(chinese_tokens[2], "##天", 12172, {6, 9});
  check_token(chinese_tokens[3], "##气", 12311, {9, 12});
  check_token(chinese_tokens[4], "##真", 12427, {12, 15});
  check_token(chinese_tokens[5], "##好", 12217, {15, 18});
}

}  // namespace tests
}  // namespace faster_tokenizer
}  // namespace paddlenlp