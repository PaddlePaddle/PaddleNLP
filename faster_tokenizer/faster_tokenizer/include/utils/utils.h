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

#pragma once
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>

#include "unicode/uchar.h"

#if defined(_FREEBSD)
#include <sys/endian.h>
#endif
#if !defined(__APPLE__) && !defined(_WIN32) && !defined(_FREEBSD)
#include <endian.h>
#if BYTE_ORDER == __BIG_ENDIAN
#define IS_BIG_ENDIAN
#endif
#endif

namespace paddlenlp {
namespace faster_tokenizer {
namespace utils {

inline void GetVocabFromFiles(const std::string& files,
                              std::unordered_map<std::string, uint>* vocab) {
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

inline bool IsChineseChar(int ch) {
  return (
      (ch >= 0x4E00 && ch <= 0x9FFF) || (ch >= 0x3400 && ch <= 0x4DBF) ||
      (ch >= 0x20000 && ch <= 0x2A6DF) || (ch >= 0x2A700 && ch <= 0x2B73F) ||
      (ch >= 0x2B740 && ch <= 0x2B81F) || (ch >= 0x2B820 && ch <= 0x2CEAF) ||
      (ch >= 0xF900 && ch <= 0xFAFF) || (ch >= 0x2F800 && ch <= 0x2FA1F));
}

inline bool IsPunctuation(int ch) {
  return (ch >= 33 && ch <= 47) || (ch >= 58 && ch <= 64) ||
         (ch >= 91 && ch <= 96) || (ch >= 123 && ch <= 126) || u_ispunct(ch);
}

inline bool IsPunctuationOrChineseChar(int ch) {
  return IsChineseChar(ch) || IsPunctuation(ch);
}

inline bool StringReplace(std::string* str,
                          const std::string& from,
                          const std::string& to) {
  size_t start_pos = str->find(from);
  if (start_pos == std::string::npos) return false;
  str->replace(start_pos, from.length(), to);
  return true;
}

inline void StringReplaceAll(std::string* str,
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


// Used in faster wordpiece model

static constexpr uint32_t kBitToIndicateSuffixToken = 30;

static constexpr uint32_t kBitsToEncodeVocabTokenLength = 8;

static constexpr uint32_t kMaskToEncodeVocabTokenLength =
    (1 << kBitsToEncodeVocabTokenLength) - 1;

static constexpr uint32_t kMaxVocabTokenLengthInUTF8Bytes =
    (1 << kBitsToEncodeVocabTokenLength);

static constexpr uint32_t kMaxSupportedVocabSize =
    (1 << (32 - 1 - 1 - kBitsToEncodeVocabTokenLength));

static constexpr uint32_t kMaskToEncodeVocabTokenId =
    ((1 << kBitToIndicateSuffixToken) - 1) ^ kMaskToEncodeVocabTokenLength;

inline int EncodeToken(uint token_id, uint token_length, bool is_suffix_token) {
  int encoded_value = (is_suffix_token << kBitToIndicateSuffixToken) |
                      (token_id << kBitsToEncodeVocabTokenLength) |
                      (token_length - 1);
  return encoded_value;
}

inline bool IsSuffixTokenFromEncodedValue(int token_encoded_value) {
  return static_cast<bool>(token_encoded_value >> kBitToIndicateSuffixToken);
}

// Gets the token id from the encoded value.
inline int GetTokenIdFromEncodedValue(int token_encoded_value) {
  return (token_encoded_value & kMaskToEncodeVocabTokenId) >>
         kBitsToEncodeVocabTokenLength;
}

// Gets the token length (without the suffix indicator) from the encoded value.
inline int GetTokenLengthFromEncodedValue(int token_encoded_value) {
  return (token_encoded_value & kMaskToEncodeVocabTokenLength) + 1;
}

static constexpr uint32_t kBitsToEncodeFailurePopsListSize =
    kBitsToEncodeVocabTokenLength;

static constexpr uint32_t kMaskToEncodeFailurePopsListSize =
    (1 << kBitsToEncodeFailurePopsListSize) - 1;

static constexpr uint32_t kMaxFailurePopsListSize =
    (1 << kBitsToEncodeFailurePopsListSize);

static constexpr uint32_t kMaxSupportedFailurePoolOffset =
    (1 << (32 - kBitsToEncodeFailurePopsListSize)) - 1 - 1;

static constexpr uint32_t kNullFailurePopsList =
    std::numeric_limits<uint32_t>::max();

inline uint32_t EncodeFailurePopList(int offset, int length) {
  return (offset << kBitsToEncodeFailurePopsListSize) | (length - 1);
}

inline void GetFailurePopsOffsetAndLength(uint32_t offset_and_length,
                                          int* out_offset,
                                          int* out_length) {
  *out_offset = offset_and_length >> kBitsToEncodeFailurePopsListSize;
  *out_length = (offset_and_length & kMaskToEncodeFailurePopsListSize) + 1;
}

static constexpr uint32_t kNullNode = std::numeric_limits<uint32_t>::max();

static constexpr uint32_t kMaxSupportedTrieSize =
    std::numeric_limits<uint32_t>::max();

// A Unicode control char that never appears in the input as it is filtered
// during text normalization. It is used to build dummy nodes in the trie.
static constexpr char kInvalidControlChar = 0x11;

inline bool IsSuffixWord(const std::string& word,
                         const std::string& continuing_subword_prefix) {
  return word.rfind(continuing_subword_prefix) == 0;
}

template <typename T>
inline bool DecodePOD(const char* str, size_t str_len, T* result) {
  if (sizeof(*result) != str_len) {
    return false;
  }
  memcpy(result, str, sizeof(T));
  return true;
}


template <typename T>
inline std::string EncodePOD(const T& value) {
  std::string s;
  s.resize(sizeof(T));
  memcpy(const_cast<char*>(s.data()), &value, sizeof(T));
  return s;
}

inline size_t OneCharLen(const char* src) {
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

#ifdef IS_BIG_ENDIAN
inline uint32 Swap32(uint32 x) { return __builtin_bswap32(x); }
#endif

inline void GetSortedVocab(const std::vector<const char*>& keys,
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
