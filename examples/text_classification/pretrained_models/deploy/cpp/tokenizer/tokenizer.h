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

using std::string;
using std::shared_ptr;
using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using std::wstring;

using Vocab = unordered_map<wstring, int>;
using InvVocab = unordered_map<int, wstring>;

void LoadVocab(const std::string& vocabFile, tokenizer::Vocab* vocab);

class BasicTokenizer {
public:
  explicit BasicTokenizer(bool do_lower_case = true);
  void Tokenize(const string& text, vector<wstring>* res) const;

private:
  void clean_text(const wstring& text, wstring* output) const;
  bool is_chinese_char(const wchar_t& ch) const;
  void tokenize_chinese_chars(const wstring& text, wstring* output) const;
  void run_strip_accents(const wstring& text, wstring* output) const;
  void run_split_on_punc(const wstring& text, vector<wstring>* res) const;

  bool do_lower_case_{true};
};

class WordPieceTokenizer {
public:
  explicit WordPieceTokenizer(Vocab& vocab,
                              const wstring& unk_token = L"[UNK]",
                              const size_t max_input_chars_per_word = 100);
  void Tokenize(const wstring& text, vector<wstring>* output) const;

private:
  Vocab vocab_;
  wstring unk_token_{L"[UNK]"};
  size_t max_input_chars_per_word_;
};

class BertTokenizer {
public:
  explicit BertTokenizer(Vocab& vocab,
                         const bool& do_lower_case = false,
                         const wstring& unk_token = L"[UNK]",
                         const wstring& pad_token = L"[PAD]",
                         const wstring& cls_token = L"[CLS]",
                         const wstring& mask_token = L"[MASK]",
                         const wstring& sep_token = L"[SEP]",
                         const string& padding_site = "right");

  void Tokenize(const string& text, vector<wstring>* split_tokens) const;
  void BuildInputsWithSpecialTokens(
      vector<int64_t>* res,
      const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>()) const;
  void CreateTokenTypeIdsFromSequences(
      vector<int64_t>* token_type_ids,
      const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>()) const;
  void ConvertTokensToIds(const vector<wstring>& tokens,
                          vector<int64_t>* token_ids) const;
  void ConvertTokensToString(const vector<wstring>& tokens, string* res) const;
  int TruncateSequence(vector<int64_t>* ids,
                       vector<int64_t>* pair_ids,
                       const size_t num_tokens_to_remove = 0,
                       const string& truncation_strategy = "longest_first",
                       const size_t stride = 0) const;

  void GetSpecialTokensMask(
      vector<int64_t>* res,
      const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>(),
      const bool already_has_special_tokens = false) const;
  int64_t GetNumSpecialTokensToAdd(const bool pair = false) const;
  int Encode(unordered_map<string, vector<int64_t>>* encoded_inputs,
             const string& text,
             const string& text_pair = "",
             const size_t max_seq_len = 0,
             bool pad_to_max_seq_len = false,
             bool return_length = false,
             bool return_token_type_ids = true,
             bool return_position_ids = false,
             bool return_attention_mask = false,
             const string& truncation_strategy = "longest_first",
             bool return_overflowing_tokens = false,
             bool return_special_tokens_mask = false) const;
  int BatchEncode(
      vector<unordered_map<string, vector<int64_t>>>* batch_encode_inputs,
      const vector<string>& batch_text,
      const vector<string>& batch_text_pair = vector<string>(),
      bool is_split_into_words = false,
      const size_t max_seq_len = 0,
      bool pad_to_max_seq_len = false,
      bool return_length = false,
      bool return_token_type_ids = true,
      bool return_position_ids = false,
      bool return_attention_mask = false,
      const string& truncation_strategy = "longest_first",
      const size_t stride = 0,
      bool return_overflowing_tokens = false,
      bool return_special_tokens_mask = false) const;

  int64_t GetUnkTokenID() const;
  int64_t GetPadTokenID() const;
  int64_t GetClsTokenID() const;
  int64_t GetMaskTokenID() const;
  int64_t GetSepTokenID() const;

private:
  bool do_lower_case_;
  wstring unk_token_, pad_token_, cls_token_, mask_token_, sep_token_;
  string padding_site_;
  Vocab vocab_;
  BasicTokenizer basic_tokenizer_;
  WordPieceTokenizer word_piece_tokenizer_;
  int64_t unk_token_id_, cls_token_id_, mask_token_id_, pad_token_id_,
      sep_token_id_;
  vector<wstring> all_special_tokens_;
  unordered_set<int64_t> all_special_token_ids_;
  InvVocab inv_vocab_;

  void get_input_ids(const string& text, vector<int64_t>* token_ids) const;
};
};