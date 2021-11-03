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

#pragma once

#include <codecvt>
#include <iostream>
#include <locale>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tokenizer {

using Vocab = std::unordered_map<std::wstring, int>;

// Convert the std::string type to the std::wstring type.
inline bool ConvertStrToWstr(const std::string& src, std::wstring* res);
// Convert the std::wstring type to the std::string type.
inline void ConvertWstrToStr(const std::wstring& src, std::string* res);
// Judgment control character.
inline bool IsControl(const wchar_t& ch);
// Judgment chinese character.
inline bool IsChineseChar(const wchar_t& ch);
// Judgment whitespace character.
inline bool IsWhiteSpace(const wchar_t& ch);
// Load vocab from file.
void LoadVocab(const std::string& file, Vocab* vocab);

class BasicTokenizer {
public:
  explicit BasicTokenizer(bool do_lower_case = true);
  void Tokenize(const std::string& text, std::vector<std::wstring>* res) const;

private:
  wchar_t do_lower_case(wchar_t ch) const;

  bool do_lower_case_;
};

class WordPieceTokenizer {
public:
  explicit WordPieceTokenizer(const Vocab& vocab,
                              const std::wstring& unk_token = L"[UNK]",
                              const size_t max_input_chars_per_word = 100);
  void Tokenize(const std::wstring& text, std::vector<int64_t>* output) const;

private:
  Vocab vocab_;
  std::wstring unk_token_{L"[UNK]"};
  int64_t unk_token_id_;
  size_t max_input_chars_per_word_;
};

class BertTokenizer {
public:
  explicit BertTokenizer(const Vocab& vocab,
                         bool do_lower_case = false,
                         const std::wstring& unk_token = L"[UNK]",
                         const std::wstring& pad_token = L"[PAD]",
                         const std::wstring& cls_token = L"[CLS]",
                         const std::wstring& mask_token = L"[MASK]",
                         const std::wstring& sep_token = L"[SEP]",
                         const std::string& padding_site = "right");

  void Tokenize(const std::string& text,
                std::vector<int64_t>* split_tokens) const;
  void BuildInputsWithSpecialTokens(
      std::vector<int64_t>* res,
      const std::vector<int64_t>& token_ids_0,
      const std::vector<int64_t>& token_ids_1 = std::vector<int64_t>()) const;
  void CreateTokenTypeIdsFromSequences(
      std::vector<int64_t>* token_type_ids,
      const std::vector<int64_t>& token_ids_0,
      const std::vector<int64_t>& token_ids_1 = std::vector<int64_t>()) const;
  void TruncateSequence(std::vector<int64_t>* ids,
                        std::vector<int64_t>* pair_ids,
                        const size_t num_tokens_to_remove = 0,
                        const size_t stride = 0) const;
  int64_t GetNumSpecialTokensToAdd(const bool pair = false) const;
  int Encode(
      std::unordered_map<std::string, std::vector<int64_t>>* encoded_inputs,
      const std::string& text,
      const std::string& text_pair = "",
      bool is_split_into_words = false,
      const size_t max_seq_len = 0,
      bool pad_to_max_seq_len = false) const;
  void BatchEncode(
      std::vector<std::unordered_map<std::string, std::vector<int64_t>>>*
          batch_encode_inputs,
      const std::vector<std::string>& batch_text,
      const std::vector<std::string>& batch_text_pair =
          std::vector<std::string>(),
      bool is_split_into_words = false,
      const size_t max_seq_len = 0,
      bool pad_to_max_seq_len = false) const;

  int64_t GetPadTokenID() const;

private:
  bool do_lower_case_;
  std::wstring unk_token_, pad_token_, cls_token_, mask_token_, sep_token_;
  std::string padding_site_;
  Vocab vocab_;
  BasicTokenizer basic_tokenizer_;
  WordPieceTokenizer word_piece_tokenizer_;
  int64_t unk_token_id_, cls_token_id_, mask_token_id_, pad_token_id_,
      sep_token_id_;
  std::vector<std::wstring> all_special_tokens_;
  std::unordered_set<int64_t> all_special_token_ids_;
};
};