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

#include "utils/utils.h"

#include "unicode/uchar.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace utils {

void GetVocabFromFiles(const std::string& files,
                       std::unordered_map<std::string, uint32_t>* vocab) {
  const static std::string WHITESPACE = " \n\r\t\f\v";
  std::ifstream fin(files);
  if (!fin.good()) {
    std::cerr << "The vocab file " << files
              << " seems to be unable to access"
                 " or non-exists, please check again. "
              << std::endl;
    return;
  }
  vocab->clear();
  int i = 0;
  constexpr int MAX_BUFFER_SIZE = 256;
  char word[MAX_BUFFER_SIZE];
  while (fin.getline(word, MAX_BUFFER_SIZE)) {
    std::string word_str = word;
    auto leading_spaces = word_str.find_first_not_of(WHITESPACE);
    if (leading_spaces != std::string::npos) {
      word_str = word_str.substr(leading_spaces);
    }
    auto trailing_spaces = word_str.find_last_not_of(WHITESPACE);
    if (trailing_spaces != std::string::npos) {
      word_str = word_str.substr(0, trailing_spaces + 1);
    }
    if (word_str != "") {
      (*vocab)[word_str] = i++;
    }
  }
}

bool IsChineseChar(int ch) {
  return (
      (ch >= 0x4E00 && ch <= 0x9FFF) || (ch >= 0x3400 && ch <= 0x4DBF) ||
      (ch >= 0x20000 && ch <= 0x2A6DF) || (ch >= 0x2A700 && ch <= 0x2B73F) ||
      (ch >= 0x2B740 && ch <= 0x2B81F) || (ch >= 0x2B820 && ch <= 0x2CEAF) ||
      (ch >= 0xF900 && ch <= 0xFAFF) || (ch >= 0x2F800 && ch <= 0x2FA1F));
}

bool IsPunctuation(int ch) {
  return (ch >= 33 && ch <= 47) || (ch >= 58 && ch <= 64) ||
         (ch >= 91 && ch <= 96) || (ch >= 123 && ch <= 126) || u_ispunct(ch);
}

bool IsPunctuationOrChineseChar(int ch) {
  return IsChineseChar(ch) || IsPunctuation(ch);
}

bool StringReplace(std::string* str,
                   const std::string& from,
                   const std::string& to) {
  size_t start_pos = str->find(from);
  if (start_pos == std::string::npos) return false;
  str->replace(start_pos, from.length(), to);
  return true;
}

void StringReplaceAll(std::string* str,
                      const std::string& from,
                      const std::string& to) {
  if (from.empty()) return;
  size_t start_pos = 0;
  while ((start_pos = str->find(from, start_pos)) != std::string::npos) {
    str->replace(start_pos, from.length(), to);
    start_pos += to.length();  // In case 'to' contains 'from', like replacing
                               // 'x' with 'yx'
  }
}

void GetSortedVocab(const std::vector<const char*>& keys,
                    const std::vector<int>& values,
                    std::vector<const char*>* sorted_keys,
                    std::vector<int>* sorted_values) {
  // Sort the vocab
  std::vector<int> sorted_vocab_index(keys.size());
  std::iota(sorted_vocab_index.begin(), sorted_vocab_index.end(), 0);
  std::sort(sorted_vocab_index.begin(),
            sorted_vocab_index.end(),
            [&keys](const int a, const int b) {
              return std::strcmp(keys[a], keys[b]) < 0;
            });

  sorted_keys->resize(keys.size());
  sorted_values->resize(keys.size());
  for (int i = 0; i < sorted_vocab_index.size(); ++i) {
    auto idx = sorted_vocab_index[i];
    (*sorted_keys)[i] = keys[idx];
    (*sorted_values)[i] = values[idx];
  }
}

}  // namespace utils
}  // namespace faster_tokenizer
}  // namespace paddlenlp
