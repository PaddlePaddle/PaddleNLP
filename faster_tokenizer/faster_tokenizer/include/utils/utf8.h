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
#include <cstring>

namespace paddlenlp {
namespace faster_tokenizer {
namespace utils {

static constexpr uint32_t kUnicodeError = 0xFFFD;

inline bool IsUnicodeNonChar(uint32_t c) {
  return ((c) >= 0xfdd0 && ((c) <= 0xfdef || ((c)&0xfffe) == 0xfffe) &&
          (c) <= 0x10ffff);
}

inline bool IsUnicodeChar(uint32_t c) {
  return ((c) < 0xd800 ||
          (0xdfff < (c) && (c) <= 0x10ffff && !IsUnicodeNonChar(c)));
}

inline uint32_t BytesInUTF8Char(uint8_t byte) {
  unsigned int count = 1;
  // no if-statements means no divergence
  count += static_cast<int>((byte & 0xF0) == 0xF0);
  count += static_cast<int>((byte & 0xE0) == 0xE0);
  count += static_cast<int>((byte & 0xC0) == 0xC0);
  count -= static_cast<int>((byte & 0xC0) == 0x80);
  return count;
}

inline uint32_t GetUnicodeLenFromUTF8(const char* pSrc, size_t length) {
  size_t unicode_len = 0;
  size_t start = 0;
  while (start < length && pSrc[start] != '\0') {
    size_t chwidth = BytesInUTF8Char(pSrc[start]);
    start += chwidth;
    ++unicode_len;
  }
  return unicode_len;
}

inline uint32_t UTF8ToUInt32(const char* pSrc, uint32_t* chr) {
  uint32_t chwidth = BytesInUTF8Char(static_cast<uint32_t>(*pSrc));
  *chr = static_cast<uint32_t>(*pSrc++) & 0xFF;
  if (chwidth > 1) {
    *chr = (*chr) << 8;
    *chr |= (static_cast<uint32_t>(*pSrc++) & 0xFF);  // << 8;
    if (chwidth > 2) {
      *chr = (*chr) << 8;
      *chr |= (static_cast<uint32_t>(*pSrc++) & 0xFF);  // << 16;
      if (chwidth > 3) {
        *chr = (*chr) << 8;
        *chr |= (static_cast<uint32_t>(*pSrc++) & 0xFF);  // << 24;
      }
    }
  }
  return chwidth;
}

inline uint32_t UTF8ToUnicode(uint32_t utf8) {
  uint32_t unchr = 0;
  if (utf8 < 0x00000080) {
    unchr = utf8;
  } else if (utf8 < 0x0000E000) {
    unchr = (utf8 & 0x1F00) >> 2;
    unchr |= (utf8 & 0x003F);
  } else if (utf8 < 0x00F00000) {
    unchr = (utf8 & 0x0F0000) >> 4;
    unchr |= (utf8 & 0x003F00) >> 2;
    unchr |= (utf8 & 0x00003F);
  } else if (utf8 <= static_cast<uint32_t>(0xF8000000)) {
    unchr = (utf8 & 0x03000000) >> 6;
    unchr |= (utf8 & 0x003F0000) >> 4;
    unchr |= (utf8 & 0x00003F00) >> 2;
    unchr |= (utf8 & 0x0000003F);
  }
  return unchr;
}

inline bool IsCharBeginBoundary(char ch) {
  return ((~ch) >> 7) || ((ch & 0xC0) == 0xC0);
}

inline bool IsCharBoundary(const char* ch) {
  return IsCharBeginBoundary(*ch) || IsCharBeginBoundary(*(ch + 1));
}

inline uint32_t UnicodeToUTF8(uint32_t unchr) {
  uint32_t utf8 = 0;
  if (unchr < 0x00000080) {
    utf8 = unchr;
  } else if (unchr < 0x00000800) {
    utf8 = (unchr << 2) & 0x1F00;
    utf8 |= (unchr & 0x3F);
    utf8 |= 0x0000C080;
  } else if (unchr < 0x00010000) {
    utf8 = (unchr << 4) & 0x0F0000;   // upper 4 bits
    utf8 |= (unchr << 2) & 0x003F00;  // next 6 bits
    utf8 |= (unchr & 0x3F);           // last 6 bits
    utf8 |= 0x00E08080;
  } else if (unchr < 0x00110000) {      // 3-byte unicode
    utf8 = (unchr << 6) & 0x07000000;   // upper 3 bits
    utf8 |= (unchr << 4) & 0x003F0000;  // next 6 bits
    utf8 |= (unchr << 2) & 0x00003F00;  // next 6 bits
    utf8 |= (unchr & 0x3F);             // last 6 bits
    utf8 |= static_cast<uint32_t>(0xF0808080);
  }
  return utf8;
}

inline uint32_t BytesInUnicodeChar(uint32_t chr) {
  uint32_t count = 1;
  // no if-statements means no divergence
  count += static_cast<int>((chr & static_cast<uint32_t>(0x0000FF00)) > 0);
  count += static_cast<int>((chr & static_cast<uint32_t>(0x00FF0000)) > 0);
  count += static_cast<int>((chr & static_cast<uint32_t>(0xFF000000)) > 0);
  return count;
}

inline uint32_t UnicodeToUTF8Char(uint32_t chr, char* dst) {
  uint32_t chwidth = BytesInUnicodeChar(chr);
  for (uint32_t idx = 0; idx < chwidth; ++idx) {
    dst[chwidth - idx - 1] = static_cast<char>(chr & 0xFF);
    chr = chr >> 8;
  }
  return chwidth;
}

inline uint32_t GetUTF8CharLen(uint32_t u32chr) {
  return BytesInUnicodeChar(UnicodeToUTF8(u32chr));
}

inline void GetUTF8Str(const char32_t* unicode_str,
                       char* utf8_str,
                       size_t unicode_len) {
  char dst_char[5] = {0};
  for (size_t i = 0; i < unicode_len; ++i) {
    uint32_t utf8_uint32 = UnicodeToUTF8(unicode_str[i]);
    uint32_t utf8_char_count = UnicodeToUTF8Char(utf8_uint32, dst_char);
    dst_char[utf8_char_count] = '\0';
    memcpy(utf8_str, dst_char, utf8_char_count);
    utf8_str += utf8_char_count;
  }
  *utf8_str = '\0';
}

inline void GetUnicodeStr(const char* pSrc,
                          char32_t* unicode_str,
                          size_t unicode_len) {
  uint32_t curr_unicode_char;
  uint32_t count = UTF8ToUInt32(pSrc, &curr_unicode_char);
  curr_unicode_char = UTF8ToUnicode(curr_unicode_char);
  for (size_t i = 0; i < unicode_len; ++i) {
    unicode_str[i] = curr_unicode_char;
    pSrc += count;
    count = UTF8ToUInt32(pSrc, &curr_unicode_char);
    curr_unicode_char = UTF8ToUnicode(curr_unicode_char);
  }
}

inline bool IsTrailByte(char x) { return static_cast<signed char>(x) < -0x40; }

inline bool IsValidCodepoint(char32_t c) {
  return (static_cast<uint32_t>(c) < 0xD800) || (c >= 0xE000 && c <= 0x10FFFF);
}

// mblen sotres the number of bytes consumed after decoding.
inline uint32_t DecodeUTF8(const char* begin, const char* end, size_t* mblen) {
  const size_t len = end - begin;

  if (static_cast<unsigned char>(begin[0]) < 0x80) {
    *mblen = 1;
    return static_cast<unsigned char>(begin[0]);
  } else if (len >= 2 && (begin[0] & 0xE0) == 0xC0) {
    const uint32_t cp = (((begin[0] & 0x1F) << 6) | ((begin[1] & 0x3F)));
    if (IsTrailByte(begin[1]) && cp >= 0x0080 && IsValidCodepoint(cp)) {
      *mblen = 2;
      return cp;
    }
  } else if (len >= 3 && (begin[0] & 0xF0) == 0xE0) {
    const uint32_t cp = (((begin[0] & 0x0F) << 12) | ((begin[1] & 0x3F) << 6) |
                         ((begin[2] & 0x3F)));
    if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) && cp >= 0x0800 &&
        IsValidCodepoint(cp)) {
      *mblen = 3;
      return cp;
    }
  } else if (len >= 4 && (begin[0] & 0xf8) == 0xF0) {
    const uint32_t cp = (((begin[0] & 0x07) << 18) | ((begin[1] & 0x3F) << 12) |
                         ((begin[2] & 0x3F) << 6) | ((begin[3] & 0x3F)));
    if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) &&
        IsTrailByte(begin[3]) && cp >= 0x10000 && IsValidCodepoint(cp)) {
      *mblen = 4;
      return cp;
    }
  }

  // Invalid UTF-8.
  *mblen = 1;
  return kUnicodeError;
}

inline bool IsValidDecodeUTF8(const char* begin,
                              const char* end,
                              size_t* mblen) {
  const uint32_t c = DecodeUTF8(begin, end, mblen);
  return c != kUnicodeError || *mblen == 3;
}

}  // namespace utils
}  // namespace faster_tokenizer
}  // namespace paddlenlp
