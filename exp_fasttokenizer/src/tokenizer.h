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

#ifndef EXP_FASTTOKENIZER_SRC_TOKENIZER_H_
#define EXP_FASTTOKENIZER_SRC_TOKENIZER_H_

#include <utf8proc.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

#include <boost/algorithm/string.hpp>

const std::wstring stripChar = L" \t\n\r\v\f";
using Vocab = std::unordered_map<std::wstring, size_t>;
using InvVocab = std::unordered_map<size_t, std::wstring>;


bool isControl(const wchar_t& ch) {
  if (ch == L'\t' || ch == L'\n' || ch == L'\r') return false;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_CC || cat == UTF8PROC_CATEGORY_CF) return true;
  return false;
}

bool isWhitespace(const wchar_t& ch) {
  if (ch == L' ' || ch == L'\t' || ch == L'\n' || ch == L'\r') return true;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_ZS) return true;
  return false;
}

bool isPunctuation(const wchar_t& ch) {
  if ((ch >= 33 && ch <= 47) || (ch >= 58 && ch <= 64) ||
      (ch >= 91 && ch <= 96) || (ch >= 123 && ch <= 126))
    return true;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_PD || cat == UTF8PROC_CATEGORY_PS ||
      cat == UTF8PROC_CATEGORY_PE || cat == UTF8PROC_CATEGORY_PC ||
      cat == UTF8PROC_CATEGORY_PO  // sometimes ¶ belong SO
      || cat == UTF8PROC_CATEGORY_PI || cat == UTF8PROC_CATEGORY_PF)
    return true;
  return false;
}

class BasicTokenizer {
 public:
  explicit BasicTokenizer(bool doLowerCase = true);
  std::vector<std::wstring> tokenize(const std::string& text) const;

 private:
  std::wstring cleanText(const std::wstring& text) const;
  bool isChineseChar(const wchar_t& ch) const;
  std::wstring tokenizeChineseChars(const std::wstring& text) const;
  std::wstring runStripAccents(const std::wstring& text) const;
  std::vector<std::wstring> runSplitOnPunc(const std::wstring& text) const;

  bool mDoLowerCase;
};

class WordpieceTokenizer {
 public:
  WordpieceTokenizer(std::shared_ptr<Vocab> vocab,
                     const std::wstring& unkToken = L"[UNK]",
                     std::size_t maxInputCharsPerWord = 100);
  std::vector<std::wstring> tokenize(const std::wstring& text) const;

 private:
  std::shared_ptr<Vocab> mVocab;
  std::wstring mUnkToken;
  size_t mMaxInputCharsPerWord;
};

class FullTokenizer {
 public:
  explicit FullTokenizer(const std::string& vocabFile, bool doLowerCase = true);
  std::vector<std::wstring> tokenize(const std::string& text) const;
  std::vector<size_t> convertTokensToIds(const std::vector<std::wstring>& text)
      const;

 private:
  std::shared_ptr<Vocab> mVocab;
  InvVocab mInvVocab;
  std::string mVocabFile;
  bool mDoLowerCase;
  BasicTokenizer mBasicTokenizer;
  WordpieceTokenizer mWordpieceTokenizer;
};

std::string normalize_nfd(const std::string& s) {
  std::string ret;
  char* result = reinterpret_cast<char*>(
    utf8proc_NFD(reinterpret_cast<const unsigned char*>(s.c_str())));
  if (result) {
    ret = std::string(result);
    free(result);
    result = nullptr;
  }

  return ret;
}

bool isStripChar(const wchar_t& ch) {
  return stripChar.find(ch) != std::wstring::npos;
}

std::wstring strip(const std::wstring& text) {
  std::wstring ret = text;
  if (ret.empty()) return ret;
  size_t pos = 0;
  while (pos < ret.size() && isStripChar(ret[pos])) pos++;
  if (pos != 0) ret = ret.substr(pos, ret.size() - pos);
  pos = ret.size() - 1;
  while (isStripChar(ret[pos])) pos--;
  return ret.substr(0, pos + 1);
}

std::vector<std::wstring> split(const std::wstring& text) {
  std::vector<std::wstring> result;
  boost::split(result, text, boost::is_any_of(stripChar));
  return result;
}

std::vector<std::wstring> whitespaceTokenize(const std::wstring& text) {
  std::wstring rtext = strip(text);
  if (rtext.empty()) return std::vector<std::wstring>();
  return split(text);
}

std::wstring convertToUnicode(const std::string& text) {
  size_t i = 0;
  std::wstring ret;
  while (i < text.size()) {
    wchar_t codepoint;
    utf8proc_ssize_t forward =
        utf8proc_iterate(
          reinterpret_cast<const utf8proc_uint8_t*>(&text[i]),
          text.size() - i,
          reinterpret_cast<utf8proc_int32_t*>(&codepoint));
    if (forward < 0) return L"";
    ret += codepoint;
    i += forward;
  }
  return ret;
}

std::string convertFromUnicode(const std::wstring& wText) {
  char dst[64];
  std::string ret;
  for (auto ch : wText) {
    utf8proc_ssize_t num = utf8proc_encode_char(
      ch, reinterpret_cast<utf8proc_uint8_t*>(dst));
    if (num <= 0) return "";
    ret += std::string(dst, dst + num);
  }
  return ret;
}

std::wstring toLower(const std::wstring& s) {
  std::wstring ret(s.size(), L' ');
  for (size_t i = 0; i < s.size(); i++) {
    ret[i] = utf8proc_tolower(s[i]);
  }
  return ret;
}

std::shared_ptr<Vocab> loadVocab(const std::string& vocabFile) {
  std::shared_ptr<Vocab> vocab(new Vocab);
  std::ifstream ifs(vocabFile, std::ifstream::in);
  if (!ifs) {
    throw std::runtime_error(
      "Open the vocab file failly, please check the file " + vocabFile + ".");
  }

  std::string line;
  size_t index = 0;
  while (getline(ifs, line)) {
    std::wstring token = convertToUnicode(line);
    // The input line cann't be converted to unicode.
    // The drop it.
    if (token.empty()) continue;
    token = strip(token);
    (*vocab)[token] = index;
    index++;
  }
  ifs.close();
  return vocab;
}

BasicTokenizer::BasicTokenizer(bool doLowerCase) : mDoLowerCase(doLowerCase) {}

std::wstring BasicTokenizer::cleanText(const std::wstring& text) const {
  std::wstring output;
  for (const wchar_t& cp : text) {
    if (cp == 0 || cp == 0xfffd || isControl(cp)) continue;
    if (isWhitespace(cp))
      output += L" ";
    else
      output += cp;
  }
  return output;
}

bool BasicTokenizer::isChineseChar(const wchar_t& ch) const {
  if ((ch >= 0x4E00 && ch <= 0x9FFF) || (ch >= 0x3400 && ch <= 0x4DBF) ||
      (ch >= 0x20000 && ch <= 0x2A6DF) || (ch >= 0x2A700 && ch <= 0x2B73F) ||
      (ch >= 0x2B740 && ch <= 0x2B81F) || (ch >= 0x2B820 && ch <= 0x2CEAF) ||
      (ch >= 0xF900 && ch <= 0xFAFF) || (ch >= 0x2F800 && ch <= 0x2FA1F))
    return true;
  return false;
}

std::wstring BasicTokenizer::tokenizeChineseChars(const std::wstring& text)
    const {
  std::wstring output;
  for (auto& ch : text) {
    if (isChineseChar(ch)) {
      output += L' ';
      output += ch;
      output += L' ';
    } else {
      output += ch;
    }
  }
  return output;
}

std::wstring BasicTokenizer::runStripAccents(const std::wstring& text) const {
  // Strips accents from a piece of text.
  std::wstring nText;
  try {
    nText = convertToUnicode(normalize_nfd(convertFromUnicode(text)));
  }
  catch (std::bad_cast& e) {
    std::cerr << "bad_cast" << std::endl;
    return L"";
  }

  std::wstring output;
  for (auto& ch : nText) {
    auto cat = utf8proc_category(ch);
    if (cat == UTF8PROC_CATEGORY_MN) continue;
    output += ch;
  }
  return output;
}

std::vector<std::wstring> BasicTokenizer::runSplitOnPunc(
    const std::wstring& text) const {
  size_t i = 0;
  bool startNewWord = true;
  std::vector<std::wstring> output;
  while (i < text.size()) {
    wchar_t ch = text[i];
    if (isPunctuation(ch)) {
      output.push_back(std::wstring(&ch, 1));
      startNewWord = true;
    } else {
      if (startNewWord) output.push_back(std::wstring());
      startNewWord = false;
      output[output.size() - 1] += ch;
    }
    i++;
  }
  return output;
}

std::vector<std::wstring> BasicTokenizer::tokenize(const std::string& text)
    const {
  std::wstring nText = convertToUnicode(text);
  nText = cleanText(nText);

  nText = tokenizeChineseChars(nText);

  const std::vector<std::wstring>& origTokens = whitespaceTokenize(nText);
  std::vector<std::wstring> splitTokens;
  for (std::wstring token : origTokens) {
    if (mDoLowerCase) {
      token = toLower(token);
      token = runStripAccents(token);
    }
    const auto& tokens = runSplitOnPunc(token);
    splitTokens.insert(splitTokens.end(), tokens.begin(), tokens.end());
  }
  return whitespaceTokenize(boost::join(splitTokens, L" "));
}

WordpieceTokenizer::WordpieceTokenizer(const std::shared_ptr<Vocab> vocab,
                                       const std::wstring& unkToken,
                                       size_t maxInputCharsPerWord)
    : mVocab(vocab),
      mUnkToken(unkToken),
      mMaxInputCharsPerWord(maxInputCharsPerWord) {}

std::vector<std::wstring> WordpieceTokenizer::tokenize(const std::wstring& text)
    const {
  std::vector<std::wstring> outputTokens;
  for (auto& token : whitespaceTokenize(text)) {
    if (token.size() > mMaxInputCharsPerWord) {
      outputTokens.push_back(mUnkToken);
    }
    bool isBad = false;
    size_t start = 0;
    std::vector<std::wstring> subTokens;
    while (start < token.size()) {
      size_t end = token.size();
      std::wstring curSubstr;
      bool hasCurSubstr = false;
      while (start < end) {
        std::wstring substr = token.substr(start, end - start);
        if (start > 0) substr = L"##" + substr;
        if (mVocab->find(substr) != mVocab->end()) {
          curSubstr = substr;
          hasCurSubstr = true;
          break;
        }
        end--;
      }
      if (!hasCurSubstr) {
        isBad = true;
        break;
      }
      subTokens.push_back(curSubstr);
      start = end;
    }
    if (isBad)
      outputTokens.push_back(mUnkToken);
    else
      outputTokens.insert(outputTokens.end(), subTokens.begin(),
                          subTokens.end());
  }
  return outputTokens;
}

FullTokenizer::FullTokenizer(const std::string& vocabFile, bool doLowerCase)
    : mVocab(loadVocab(vocabFile)),
      mBasicTokenizer(BasicTokenizer(doLowerCase)),
      mWordpieceTokenizer(WordpieceTokenizer(mVocab)) {
  for (auto& v : *mVocab) mInvVocab[v.second] = v.first;
}

std::vector<std::wstring> FullTokenizer::tokenize(const std::string& text)
    const {
  std::vector<std::wstring> splitTokens;
  for (auto& token : mBasicTokenizer.tokenize(text))
    for (auto& subToken : mWordpieceTokenizer.tokenize(token))
      splitTokens.push_back(subToken);
  return splitTokens;
}

std::vector<size_t> FullTokenizer::convertTokensToIds(
    const std::vector<std::wstring>& text) const {
  std::vector<size_t> ret(text.size());
  for (size_t i = 0; i < text.size(); i++) {
    ret[i] = (*mVocab)[text[i]];
  }
  return ret;
}


class BertTokenizer {
 public:
    explicit BertTokenizer(
      const std::string& vocabFile,
      bool doLowerCase = true,
      const std::wstring& unkToken = L"[UNK]",
      const std::wstring& padToken = L"[UNK]",
      const std::wstring& clsToken = L"[UNK]",
      const std::wstring& maskToken = L"[MASK]");

    std::vector<std::wstring> tokenize(const std::wstring& text) const;
    std::wstring convertTokensToString(
      const std::vector<std::wstring>& tokens) const;
    size_t numSpecialTokensToAdd(bool pair = false) const;
    std::vector<size_t> buildInputsWithSpecialTokens(
      const std::vector<size_t>& token_ids_0,
      const std::vector<size_t>& token_ids_1 = std::vector<size_t>()) const;
    std::vector<size_t> createTokenTypeIdsFromSequences(
      const std::vector<size_t>& token_ids_0,
      const std::vector<size_t>& token_ids_1 = std::vector<size_t>()) const;
    std::vector<size_t> convertTokensToIds(
      std::vector<std::wstring> tokens) const;
    std::wstring convertTokensToString(
      std::vector<size_t> tokens) const;
    std::vector<std::wstring> convertIdsToTokens(
      std::vector<size_t> token_ids) const;
    std::unordered_map<std::string, std::vector<int>> truncateSequence(
      std::vector<size_t> ids,
      std::vector<size_t> pair_ids = std::vector<size_t>(),
      size_t num_tokens_to_remove = 0,
      const std::string&  truncation_strategy = "longest_first",
      size_t stride = 0);

 private:
    std::vector<std::wstring> mAllSpeicialTokens;
    std::vector<size_t> mAllSpecialTokenIds;
};

#endif  // EXP_FASTTOKENIZER_SRC_TOKENIZER_H_
