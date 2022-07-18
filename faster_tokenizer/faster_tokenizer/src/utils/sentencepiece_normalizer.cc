// Copyright 2016 Google Inc.
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils/sentencepiece_normalizer.h"
#include <algorithm>
#include "utils/unique_ptr.h"
#include "utils/utf8.h"
#include "utils/utils.h"

#include "glog/logging.h"
#include "unicode/brkiter.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace utils {

PrefixMatcher::PrefixMatcher(const std::set<const char*, Cstrless>& dic) {
  if (dic.empty()) return;
  std::vector<const char*> key;
  key.reserve(dic.size());
  for (const auto& it : dic) key.push_back(it);
  trie_ = utils::make_unique<Darts::DoubleArray>();
  trie_->build(key.size(), const_cast<char**>(&key[0]), nullptr, nullptr);
}

int PrefixMatcher::PrefixMatch(const char* w, size_t w_len, bool* found) const {
  if (trie_ == nullptr) {
    if (found) *found = false;
    return std::min<int>(w_len, OneCharLen(w));
  }
  constexpr int kResultSize = 64;
  Darts::DoubleArray::result_pair_type trie_results[kResultSize];
  const int num_nodes =
      trie_->commonPrefixSearch(w, trie_results, kResultSize, w_len);
  if (found) *found = (num_nodes > 0);
  if (num_nodes == 0) {
    return std::min<int>(w_len, OneCharLen(w));
  }

  int mblen = 0;
  for (int i = 0; i < num_nodes; ++i) {
    mblen = std::max<int>(trie_results[i].length, mblen);
  }
  return mblen;
}

std::string PrefixMatcher::GlobalReplace(const char* w,
                                         size_t w_len,
                                         const char* out,
                                         size_t out_len,
                                         const char** result_w) const {
  std::string result;
  if (w_len > 0) {
    bool found = false;
    const int mblen = PrefixMatch(w, w_len, &found);
    if (found) {
      result.append(out, out_len);
    } else {
      result.append(w, mblen);
    }
    *result_w = w + mblen;
  }
  return result;
}

Normalizer::Normalizer(const std::string& precompiled_charsmap)
    : precompiled_charsmap_(precompiled_charsmap) {
  Init();
}

Normalizer::Normalizer(const Normalizer& other)
    : precompiled_charsmap_(other.precompiled_charsmap_) {
  Init();
}


Normalizer::~Normalizer() {}

std::string Normalizer::GetPrecompiledCharsmap() const {
  return precompiled_charsmap_;
}

void Normalizer::Init() {
  if (!precompiled_charsmap_.empty()) {
#ifdef IS_BIG_ENDIAN
    DecodePrecompiledCharsMap(precompiled_charsmap_.data(),
                              precompiled_charsmap_.length(),
                              &trie_blob_,
                              &normalized_blob_,
                              &precompiled_charsmap_buffer_);
#else
    DecodePrecompiledCharsMap(precompiled_charsmap_.data(),
                              precompiled_charsmap_.length(),
                              &trie_blob_,
                              &normalized_blob_);
#endif
    // Reads the body of double array.
    trie_ = utils::make_unique<Darts::DoubleArray>();
    // The second arg of set_array is not the size of blob,
    // but the number of double array units.
    trie_->set_array(const_cast<char*>(trie_blob_.data()),
                     trie_blob_.size() / trie_->unit_size());
    normalized_ = normalized_blob_.data();
  }
}

void Normalizer::DecodePrecompiledCharsMap(const char* blob,
                                           size_t blob_size,
                                           std::string* trie_blob,
                                           std::string* normalized,
                                           std::string* buffer) {
  uint32_t trie_blob_size = 0;
  uint32_t offset = 0;
  if (blob_size <= sizeof(trie_blob_size) ||
      !DecodePOD<uint32_t>(blob, sizeof(trie_blob_size), &trie_blob_size) ||
      trie_blob_size >= blob_size) {
    throw std::runtime_error("Blob for normalization rule is broken.");
  }
#ifdef IS_BIG_ENDIAN
  trie_blob_size = util::Swap32(trie_blob_size);
#endif
  if (trie_blob_size >= blob_size) {
    throw std::runtime_error("Trie data size exceeds the input blob size.");
  }
  offset += sizeof(trie_blob_size);
#ifdef IS_BIG_ENDIAN
  buffer->assign(blob + offset, trie_blob_size);
  uint32* data = reinterpret_cast<uint32_t*>(const_cast<char*>(buffer->data()));
  for (int i = 0; i < trie_blob_size / 4; ++i) data[i] = util::Swap32(data[i]);
  *trie_blob = std::string(buffer->data(), trie_blob_size);
#else
  *trie_blob = std::string(blob + offset, trie_blob_size);
#endif
  offset += trie_blob_size;
  *normalized = std::string(blob + offset, blob_size - offset);
}

std::string Normalizer::EncodePrecompiledCharsMap(
    const std::string& trie_blob, const std::string& normalized) {
  // <trie size(4byte)><double array trie><normalized string>
  std::string blob;
  blob.append(EncodePOD<uint32_t>(trie_blob.size()));
  blob.append(trie_blob.data(), trie_blob.size());
  blob.append(normalized.data(), normalized.size());

#ifdef IS_BIG_ENDIAN
  uint32* data = reinterpret_cast<uint32_t*>(const_cast<char*>(blob.data()));
  for (int i = 0; i <= trie_blob.size() / 4; ++i) {
    data[i] = util::Swap32(data[i]);
  }
#endif
  return blob;
}

std::pair<simple_string_view, int> Normalizer::NormalizePrefix(
    const char* input, size_t input_len) const {
  std::pair<simple_string_view, int> result;
  if (input_len == 0) {
    return result;
  }
  if (matcher_ != nullptr) {
    bool found = false;
    const int mblen = matcher_->PrefixMatch(input, input_len, &found);
    if (found) {
      return std::make_pair(simple_string_view(input, input_len), mblen);
    }
  }
  size_t longest_length = 0;
  int longest_value = 0;
  if (trie_ != nullptr) {
    // Allocates trie_results in stack, which makes the encoding speed 36%
    // faster. (38k sentences/sec => 60k sentences/sec). Builder checks that the
    // result size never exceeds kMaxTrieResultsSize. This array consumes
    // 0.5kByte in stack, which is less than default stack frames (16kByte).
    Darts::DoubleArray::result_pair_type
        trie_results[Normalizer::kMaxTrieResultsSize];
    const size_t num_nodes = trie_->commonPrefixSearch(
        input, trie_results, Normalizer::kMaxTrieResultsSize, input_len);

    // Finds the longest rule.
    for (size_t k = 0; k < num_nodes; ++k) {
      if (longest_length == 0 || trie_results[k].length > longest_length) {
        longest_length = trie_results[k].length;  // length of prefix
        longest_value = trie_results[k].value;    // pointer to |normalized_|.
      }
    }
  }

  if (longest_length == 0) {
    size_t length = 0;
    if (!IsValidDecodeUTF8(input, input + input_len, &length)) {
      // Found a malformed utf8.
      // The rune is set to be 0xFFFD (REPLACEMENT CHARACTER),
      // which is a valid Unicode of three bytes in utf8,
      // but here we only consume one byte.
      result.second = 1;
      static const char kReplacementChar[] = "\xEF\xBF\xBD";
      result.first = simple_string_view(kReplacementChar);
    } else {
      result.second = length;
      result.first = simple_string_view(input, length);
    }
  } else {
    result.second = longest_length;
    // No need to pass the size of normalized sentence,
    // since |normalized| is delimitered by "\0".
    result.first = simple_string_view(&(normalized_[longest_value]));
  }
  return result;
}

bool Normalizer::Normalize(const char* input,
                           size_t input_len,
                           std::string* normalized,
                           std::vector<int>* norm_to_orig,
                           std::u32string* u32content) const {
  bool modified = false;
  norm_to_orig->clear();
  normalized->clear();
  if (input_len == 0) {
    return modified;
  }

  // Reserves the output buffer to avoid re-allocations.
  const size_t kReservedSize = input_len * 3;
  normalized->reserve(kReservedSize);
  norm_to_orig->reserve(kReservedSize);
  if (u32content != nullptr) {
    u32content->reserve(kReservedSize);
  }
  UErrorCode err = U_ZERO_ERROR;
  std::unique_ptr<icu::BreakIterator> iter(
      icu::BreakIterator::createCharacterInstance(icu::Locale::getDefault(),
                                                  err));
  UText utext = UTEXT_INITIALIZER;
  utext_openUTF8(&utext, input, input_len, &err);
  iter->setText(&utext, err);
  int curr_pos = iter->current();
  while (iter->next() != icu::BreakIterator::DONE) {
    int next_pos = iter->current();
    int curr_len = next_pos - curr_pos;
    std::pair<simple_string_view, int> p;
    if (curr_len < 6) {
      p = NormalizePrefix(input + curr_pos, curr_len);
      simple_string_view sp = p.first;
      if (sp.data() != input + curr_pos) {
        if (!sp.empty()) {
          for (size_t n = 0; n < sp.size(); ++n) {
            *normalized += sp.data()[n];
          }
        }
        Replace(sp,
                simple_string_view(input + curr_pos, curr_len),
                norm_to_orig,
                u32content);
        modified = true;
        curr_pos = next_pos;
        continue;
      }
    }
    int curr_grapheme_pos = curr_pos;
    while (curr_grapheme_pos < next_pos) {
      uint32_t content_char;
      auto content_char_width =
          utils::UTF8ToUInt32(input + curr_grapheme_pos, &content_char);
      content_char = utils::UTF8ToUnicode(content_char);
      p = NormalizePrefix(input + curr_grapheme_pos, content_char_width);
      simple_string_view sp = p.first;
      if (sp.data() != input + curr_grapheme_pos) {
        if (!sp.empty()) {
          for (size_t n = 0; n < sp.size(); ++n) {
            *normalized += sp.data()[n];
          }
        }
        Replace(
            sp,
            simple_string_view(input + curr_grapheme_pos, content_char_width),
            norm_to_orig,
            u32content);
        modified = true;
      } else {
        for (int i = 0; i < sp.size(); ++i) {
          *normalized += sp.data()[i];
        }
        if (u32content != nullptr) {
          u32content->push_back(content_char);
        }
        norm_to_orig->push_back(0);
      }
      curr_grapheme_pos += content_char_width;
    }
    curr_pos = next_pos;
  }
  utext_close(&utext);
  return modified;
}

void Normalizer::Replace(const simple_string_view& new_part,
                         const simple_string_view& old_part,
                         std::vector<int>* changes,
                         std::u32string* u32content) const {
  auto new_unicode_len =
      GetUnicodeLenFromUTF8(new_part.data(), new_part.size());
  auto old_unicode_len =
      GetUnicodeLenFromUTF8(old_part.data(), old_part.size());
  if (u32content != nullptr) {
    size_t utf8_len = 0;
    while (utf8_len < new_part.size()) {
      uint32_t content_char;
      auto content_char_width =
          utils::UTF8ToUInt32(new_part.data() + utf8_len, &content_char);
      content_char = utils::UTF8ToUnicode(content_char);
      u32content->push_back(content_char);
      utf8_len += content_char_width;
    }
  }
  changes->insert(changes->end(), new_unicode_len, 0);
  if (new_unicode_len > old_unicode_len) {
    auto diff = new_unicode_len - old_unicode_len;
    for (auto i = changes->size() - 1; i >= changes->size() - diff; --i) {
      (*changes)[i] = 1;
    }
  } else {
    changes->back() -= old_unicode_len - new_unicode_len;
  }
}

}  // namespace utils
}  // namespace faster_tokenizer
}  // namespace paddlenlp
