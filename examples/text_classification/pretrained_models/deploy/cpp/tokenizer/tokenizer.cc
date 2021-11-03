/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <utf8proc.h>

#include <algorithm>
#include <codecvt>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/algorithm/string.hpp>

#include "tokenizer.h"

namespace tokenizer {

std::wstring_convert<std::codecvt_utf8<wchar_t>> kConverter;
const std::wstring kStripChars = L" \t\n\r\v\f";

// Convert the std::string type to the std::wstring type.
bool ConvertStrToWstr(const std::string& src, std::wstring* res) {
  try {
    *res = kConverter.from_bytes(src);
  } catch (std::range_error& e) {
    std::cout << "The string " << src << " was converted to unicode failedly! "
              << std::endl;
    return false;
  }
  return true;
}

// Convert the std::wstring type to the std::string type.
void ConvertWstrToStr(const std::wstring& src, std::string* res) {
  *res = kConverter.to_bytes(src);
}

inline bool IsControl(const wchar_t& ch) {
  if (ch == L'\t' || ch == L'\n' || ch == L'\r') return false;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_CC || cat == UTF8PROC_CATEGORY_CF) return true;
  return false;
}

inline bool IsChineseChar(const wchar_t& ch) {
  if ((ch >= 0x4E00 && ch <= 0x9FFF) || (ch >= 0x3400 && ch <= 0x4DBF) ||
      (ch >= 0x20000 && ch <= 0x2A6DF) || (ch >= 0x2A700 && ch <= 0x2B73F) ||
      (ch >= 0x2B740 && ch <= 0x2B81F) || (ch >= 0x2B820 && ch <= 0x2CEAF) ||
      (ch >= 0xF900 && ch <= 0xFAFF) || (ch >= 0x2F800 && ch <= 0x2FA1F))
    return true;
  return false;
}

inline bool IsWhiteSpace(const wchar_t& ch) {
  if (ch == L' ' || ch == L'\t' || ch == L'\n' || ch == L'\r') return true;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_ZS) return true;
  return false;
}

inline bool IsPunctuation(const wchar_t& ch) {
  if ((ch >= 33 && ch <= 47) || (ch >= 58 && ch <= 64) ||
      (ch >= 91 && ch <= 96) || (ch >= 123 && ch <= 126))
    return true;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_PD || cat == UTF8PROC_CATEGORY_PS ||
      cat == UTF8PROC_CATEGORY_PE || cat == UTF8PROC_CATEGORY_PC ||
      cat == UTF8PROC_CATEGORY_PO  // sometimes ¶ belong SO
      ||
      cat == UTF8PROC_CATEGORY_PI || cat == UTF8PROC_CATEGORY_PF)
    return true;
  return false;
}

void LoadVocab(const std::string& vocabFile, tokenizer::Vocab* vocab) {
  size_t index = 0;
  std::ifstream ifs(vocabFile, std::ifstream::in);
  if (!ifs) {
    throw std::runtime_error("open file failed");
  }
  std::string line;
  while (getline(ifs, line)) {
    std::wstring token;
    auto status = ConvertStrToWstr(line, &token);
    if (!status) {
      std::cout << "Convert the token " << line << " to wstring failedly."
                << std::endl;
      continue;
    }
    if (token.empty()) break;
    (*vocab)[token] = index;
    index++;
  }
}

BasicTokenizer::BasicTokenizer(bool do_lower_case /* = true */)
    : do_lower_case_(do_lower_case) {}

wchar_t BasicTokenizer::do_lower_case(wchar_t ch) const {
  wchar_t new_ch = utf8proc_tolower(ch);
  return new_ch;
}

void BasicTokenizer::Tokenize(const std::string& text,
                              std::vector<std::wstring>* res) const {
  std::wstring unicode_text;
  bool status = ConvertStrToWstr(text, &unicode_text);
  if (!status) {
    // String is converted into wstring failedly.
    return;
  }

  std::wstring dest_text;
  for (auto ch : unicode_text) {
    if (ch == 0 || ch == 0xfffd || IsControl(ch)) {
      continue;
    }
    if (do_lower_case_) {
      ch = do_lower_case(ch);
    }
    if (IsChineseChar(ch) || IsPunctuation(ch)) {
      dest_text += ' ';
      dest_text += ch;
      dest_text += ' ';
    } else if (IsWhiteSpace(ch)) {
      dest_text += ' ';
    } else {
      dest_text += ch;
    }
  }
  boost::split(*res, dest_text, boost::is_any_of(kStripChars));
}

WordPieceTokenizer::WordPieceTokenizer(
    const tokenizer::Vocab& vocab,
    const std::wstring& unk_token /* = L"[UNK]"*/,
    const size_t max_input_chars_per_word /* = 100 */)
    : vocab_(vocab),
      unk_token_(unk_token),
      max_input_chars_per_word_(max_input_chars_per_word) {
  unk_token_id_ = vocab_[unk_token_];
}

void WordPieceTokenizer::Tokenize(const std::wstring& text,
                                  std::vector<int64_t>* token_ids) const {
  size_t len = text.size();
  if (len > max_input_chars_per_word_) {
    token_ids->emplace_back(unk_token_id_);
    return;
  }

  auto it = vocab_.find(text);
  if (it != vocab_.end()) {
    token_ids->emplace_back(it->second);
    return;
  }

  size_t start = 0;
  std::vector<int64_t> wordpiece_ids;
  while (start < len) {
    size_t end = len;
    std::wstring cur_substr;
    int64_t cur_substr_id;
    while (start < end) {
      std::wstring sub = text.substr(start, end - start);
      if (start > 0) {
        sub = L"##" + sub;
      }
      auto it = vocab_.find(sub);
      if (it != vocab_.end()) {
        cur_substr = sub;
        cur_substr_id = it->second;
        break;
      }
      end -= 1;
    }

    if (cur_substr.empty()) {
      token_ids->emplace_back(unk_token_id_);
      return;
    } else {
      start = end;
      wordpiece_ids.emplace_back(cur_substr_id);
    }
  }
  for (auto& token_id : wordpiece_ids) {
    token_ids->emplace_back(token_id);
  }
}

BertTokenizer::BertTokenizer(const tokenizer::Vocab& vocab,
                             bool do_lower_case /* = false */,
                             const std::wstring& unk_token /* = L"[UNK]" */,
                             const std::wstring& pad_token /* = L"[PAD]" */,
                             const std::wstring& cls_token /* = L"[CLS]" */,
                             const std::wstring& mask_token /* = L"[MASK]" */,
                             const std::wstring& sep_token /* = L"[SEP]" */,
                             const std::string& padding_site /* = "right" */)
    : do_lower_case_(do_lower_case),
      unk_token_(unk_token),
      pad_token_(pad_token),
      cls_token_(cls_token),
      mask_token_(mask_token),
      sep_token_(sep_token),
      padding_site_(padding_site),
      vocab_(vocab),
      basic_tokenizer_(do_lower_case_),
      word_piece_tokenizer_(vocab_, unk_token) {
  unk_token_id_ = vocab_.at(unk_token_);
  pad_token_id_ = vocab_.at(pad_token_);
  cls_token_id_ = vocab_.at(cls_token_);
  mask_token_id_ = vocab_.at(mask_token_);
  sep_token_id_ = vocab_.at(sep_token_);

  all_special_tokens_ = std::vector<std::wstring>(
      {unk_token_, pad_token_, cls_token_, mask_token_, sep_token_});
  all_special_token_ids_ = std::unordered_set<std::int64_t>({unk_token_id_,
                                                             pad_token_id_,
                                                             cls_token_id_,
                                                             mask_token_id_,
                                                             sep_token_id_});
}

void BertTokenizer::Tokenize(const std::string& text,
                             std::vector<int64_t>* split_token_ids) const {
  std::vector<std::wstring> tmp_tokens;
  basic_tokenizer_.Tokenize(text, &tmp_tokens);
  if (tmp_tokens.empty()) return;
  split_token_ids->reserve(tmp_tokens.size());
  for (auto& w_token : tmp_tokens) {
    const auto& vec_size = w_token.size();
    if (vec_size == 1) {
      if (IsChineseChar(w_token[0])) {
        auto vocab_it = vocab_.find(w_token);
        if (vocab_it != vocab_.end()) {
          split_token_ids->emplace_back(vocab_it->second);
        } else {
          split_token_ids->emplace_back(unk_token_id_);
        }
      } else {
        word_piece_tokenizer_.Tokenize(w_token, split_token_ids);
      }
    } else if (vec_size > 1) {
      word_piece_tokenizer_.Tokenize(w_token, split_token_ids);
    } else {
      continue;
    }
  }
}

void BertTokenizer::BuildInputsWithSpecialTokens(
    std::vector<int64_t>* inputs,
    const std::vector<int64_t>& token_ids_0,
    const std::vector<int64_t>& token_ids_1 /* = vector<int64_t>() */) const {
  if (token_ids_1.size() == 0) {
    inputs->clear();
    inputs->resize(token_ids_0.size() + 2);
    inputs->at(0) = cls_token_id_;
    size_t i = 1;
    for (auto& token_id : token_ids_0) {
      inputs->at(i) = token_id;
      ++i;
    }
    inputs->at(i) = sep_token_id_;
  } else {
    inputs->clear();
    inputs->resize(token_ids_0.size() + token_ids_1.size() + 3);
    inputs->at(0) = cls_token_id_;
    size_t i = 1;
    for (auto& token_id : token_ids_0) {
      inputs->at(i) = token_id;
      ++i;
    }
    inputs->at(i) = sep_token_id_;
    ++i;
    for (auto& token_id : token_ids_1) {
      inputs->at(i) = token_id;
      ++i;
    }
    inputs->at(i) = sep_token_id_;
  }
}

int64_t BertTokenizer::GetNumSpecialTokensToAdd(const bool pair) const {
  if (pair) {
    return 3;
  } else {
    return 2;
  }
}

void BertTokenizer::CreateTokenTypeIdsFromSequences(
    std::vector<int64_t>* token_type_ids,
    const std::vector<int64_t>& token_ids_0,
    const std::vector<int64_t>& token_ids_1 /* = vector<int64_t>() */) const {
  if (token_ids_1.size() == 0) {
    std::vector<int64_t> tmp(token_ids_0.size() + 2, 0);
    token_type_ids->swap(tmp);
  } else {
    std::vector<int64_t> tmp(token_ids_0.size() + token_ids_1.size() + 3, 0);
    for (size_t i = token_ids_0.size() + 2; i < tmp.size(); i++) {
      tmp[i] = 1;
    }
    token_type_ids->swap(tmp);
  }
}

void BertTokenizer::TruncateSequence(
    std::vector<int64_t>* ids,
    std::vector<int64_t>* pair_ids,
    const size_t num_tokens_to_remove /* = 0 */,
    const size_t stride /* = 0 */) const {
  for (size_t i = 0; i < num_tokens_to_remove; i++) {
    if ((pair_ids->size() == 0) || (ids->size() > pair_ids->size())) {
      ids->pop_back();
    } else {
      pair_ids->pop_back();
    }
  }
}

int64_t BertTokenizer::GetPadTokenID() const { return pad_token_id_; }

int BertTokenizer::Encode(
    std::unordered_map<std::string, std::vector<int64_t>>* encoded_inputs,
    const std::string& text,
    const std::string& text_pair /* = "" */,
    bool is_split_into_words /* = false */,
    const size_t max_seq_len /* = 0 */,
    bool pad_to_max_seq_len /* = false */) const {
  std::vector<int64_t> ids;
  std::vector<int64_t> pair_ids;
  if (!is_split_into_words) {
    Tokenize(text, &ids);
    if (ids.empty()) return 0;
    if (text_pair != "") {
      Tokenize(text_pair, &pair_ids);
      if (pair_ids.empty()) return 0;
    }
  } else {
    std::wstring unicode_text;
    bool status_a = ConvertStrToWstr(text, &unicode_text);
    if (!status_a) {
      return 0;
    }
    for (size_t i = 0; i < unicode_text.size(); i++) {
      std::wstring token = unicode_text.substr(i, 1);
      auto it = vocab_.find(token);
      if (it != vocab_.end()) {
        ids.emplace_back(it->second);
      } else {
        ids.emplace_back(unk_token_id_);
      }
    }
  }

  bool pair = false;
  if (pair_ids.size() != 0) {
    pair = true;
  }

  size_t len_ids = ids.size();
  size_t len_pair_ids = pair_ids.size();

  // Truncation: Handle max sequence length
  // If max_seq_len == 0, then do nothing and keep the real length.
  // If max_seq_len > 0 and
  // all the input sequence len is over the max_seq_len,
  // then we truncate it.
  size_t total_len = len_ids + len_pair_ids + GetNumSpecialTokensToAdd(pair);
  if (max_seq_len > 0 && total_len > max_seq_len) {
    TruncateSequence(&ids, &pair_ids, total_len - max_seq_len);
  }

  // Add special tokens
  std::vector<int64_t> sequence;
  BuildInputsWithSpecialTokens(&sequence, ids, pair_ids);
  size_t seq_len = sequence.size();
  std::vector<int64_t> token_type_ids;
  CreateTokenTypeIdsFromSequences(&token_type_ids, ids, pair_ids);

  // Build output dictionnary
  encoded_inputs->emplace("input_ids", sequence);
  encoded_inputs->emplace("token_type_ids", token_type_ids);
  // Check lengths
  if (max_seq_len > 0 && seq_len > max_seq_len) {
    std::cout << "There is something wrong with the input sequence length."
                 " Please check it."
              << std::endl;
    // Failed.
    return 0;
  }

  // Padding
  bool needs_to_be_padded = false;
  if (pad_to_max_seq_len && max_seq_len > 0 && (seq_len < max_seq_len)) {
    needs_to_be_padded = true;
  }

  if (needs_to_be_padded) {
    int64_t difference = max_seq_len - seq_len;
    size_t pad_start = max_seq_len - 1 - difference;
    encoded_inputs->at("token_type_ids").resize(max_seq_len);
    for (size_t i = max_seq_len - 1; i > pad_start; i--) {
      encoded_inputs->at("token_type_ids")[i] = pad_token_id_;
    }

    encoded_inputs->at("input_ids").resize(max_seq_len);
    for (size_t i = max_seq_len - 1; i > pad_start; i--) {
      encoded_inputs->at("input_ids")[i] = pad_token_id_;
    }
  }
  return 1;
}

void BertTokenizer::BatchEncode(
    std::vector<std::unordered_map<std::string, std::vector<int64_t>>>*
        batch_encode_inputs,
    const std::vector<std::string>& batch_text,
    const std::vector<std::string>& batch_text_pair /* = vector<string>() */,
    bool is_split_into_words /* = false */,
    const size_t max_seq_len /* = 0 */,
    bool pad_to_max_seq_len /* = false */) const {
  bool has_text_pair = false;
  if (batch_text_pair.size() != 0) {
    has_text_pair = true;
  }

  size_t batch_size = batch_text.size();
  for (size_t i = 0; i < batch_size; i++) {
    std::unordered_map<std::string, std::vector<int64_t>> res;
    if (has_text_pair) {
      auto status = Encode(&res,
                           batch_text[i],
                           batch_text_pair[i],
                           is_split_into_words,
                           max_seq_len,
                           pad_to_max_seq_len);
      if (!status) {
        res["input_ids"] =
            std::vector<int64_t>{cls_token_id_, sep_token_id_, sep_token_id_};
        res["token_type_ids"] = std::vector<int64_t>{0, 0, 1};
      }
    } else {
      auto status = Encode(&res,
                           batch_text[i],
                           {},
                           is_split_into_words,
                           max_seq_len,
                           pad_to_max_seq_len);

      if (!status) {
        res["input_ids"] = std::vector<int64_t>{cls_token_id_, sep_token_id_};
        res["token_type_ids"] = std::vector<int64_t>{0, 0};
      }
    }
    batch_encode_inputs->at(i) = std::move(res);
  }
}
};
