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

#include <iostream>
#include <vector>
#include "fast_tokenizer/tokenizers/clip_fast_tokenizer.h"
using namespace paddlenlp;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> vec) {
  os << "[";
  for (int i = 0; i < vec.size(); ++i) {
    if (i == 0) {
      os << vec[i];
    } else {
      os << ", " << vec[i];
    }
  }
  os << "]";
  return os;
}

fast_tokenizer::tokenizers_impl::ClipFastTokenizer CreateClipFastTokenizer(
    const std::string& vocab_path,
    const std::string& merge_path,
    uint32_t max_length,
    bool pad_to_max_length = true) {
  fast_tokenizer::tokenizers_impl::ClipFastTokenizer tokenizer(
      vocab_path, merge_path, max_length);
  if (pad_to_max_length) {
    tokenizer.EnablePadMethod(fast_tokenizer::core::RIGHT,
                              tokenizer.GetPadTokenId(),
                              0,
                              tokenizer.GetPadToken(),
                              &max_length,
                              nullptr);
  }
  return tokenizer;
}

int main() {
  // 1. Define a clip fast tokenizer
  auto tokenizer = CreateClipFastTokenizer("clip_vocab.json",
                                           "clip_merges.txt",
                                           /*max_length = */ 77,
                                           /* pad_to_max_length = */ true);
  // 2. Tokenize the input strings
  std::vector<fast_tokenizer::core::Encoding> encodings;
  std::vector<std::string> texts = {
      "a photo of an astronaut riding a horse on mars"};
  tokenizer.EncodeBatchStrings(texts, &encodings);

  for (int i = 0; i < texts.size(); ++i) {
    std::cout << "text = \"" << texts[i] << "\"" << std::endl;
    std::cout << "ids = " << encodings[i].GetIds() << std::endl;
  }

  return 0;
}
