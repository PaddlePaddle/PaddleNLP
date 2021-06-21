// Copyright (c) 2021 PaddlePaddle Authors & liustung. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>

#include "src/tokenizer.h"

int main() {
  FullTokenizer* pTokenizer = nullptr;

  try {
    pTokenizer = new FullTokenizer("bert-base-chinese-vocab.txt");
  }
  catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    auto tokens = pTokenizer->tokenize(line);
    auto ids = pTokenizer->convertTokensToIds(tokens);

    std::cout << "#" << convertFromUnicode(boost::join(tokens, L" ")) << "#"
              << "\t";
    for (size_t i = 0; i < ids.size(); i++) {
      if (i != 0) std::cout << " ";
      std::cout << ids[i];
    }
    std::cout << std::endl;
  }

  return 0;
}
