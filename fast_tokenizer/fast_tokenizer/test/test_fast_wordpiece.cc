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
#include "fast_tokenizer/models/fast_wordpiece.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace tests {

TEST(model, fast_wordpiece_token_to_id) {
  auto vocab = models::FastWordPiece::GetVocabFromFile("ernie_vocab.txt");
  models::FastWordPiece fast_wordpiece_model(vocab);
  // Test tokens in vocab
  for (const auto& item : vocab) {
    uint32_t id;
    fast_wordpiece_model.TokenToId(item.first, &id);
    ASSERT_EQ(item.second, id);
  }
  // Test [UNK] token
  uint32_t fast_wordpiece_id;
  ASSERT_FALSE(fast_wordpiece_model.TokenToId("dasd", &fast_wordpiece_id));
}

}  // namespace tests
}  // namespace fast_tokenizer
}  // namespace paddlenlp