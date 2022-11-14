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

#include "fast_tokenizer/core/encoding.h"
#include "fast_tokenizer/tokenizers/clip_fast_tokenizer.h"

#include "fast_tokenizer/test/utils.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace tests {

TEST(tokenizer, clip_full) {
  std::string vocab_path = "clip_vocab.json";
  std::string merges_path = "clip_merges.txt";
  tokenizers_impl::ClipFastTokenizer clip_tokenizer(vocab_path, merges_path);

  core::Encoding encoding;
  std::string input_text = "A\n'll 11p223RF☆ho!!to?'d'd''d of a cat";
  std::vector<uint32_t> expected_ids = {
      49406, 320, 1342,  272, 272,  335,  273, 273, 274, 16368, 13439, 2971,
      748,   531, 13610, 323, 1896, 8445, 323, 539, 320, 2368,  49407};
  std::vector<std::string> expected_tokens = {
      "<|startoftext|>", "a</w>",   "'ll</w>",      "1</w>",  "1</w>",
      "p</w>",           "2</w>",   "2</w>",        "3</w>",  "rf</w>",
      "âĺĨ</w>",         "ho</w>",  "!!</w>",       "to</w>", "?'</w>",
      "d</w>",           "'d</w>",  "''</w>",       "d</w>",  "of</w>",
      "a</w>",           "cat</w>", "<|endoftext|>"};
  clip_tokenizer.EncodePairStrings(input_text, &encoding);
  CheckVectorEqual(expected_ids, encoding.GetIds());
  CheckVectorEqual(expected_tokens, encoding.GetTokens());
}

}  // namespace tests
}  // namespace fast_tokenizer
}  // namespace paddlenlp