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

#include <algorithm>
#include <codecvt>
#include <fstream>
#include <locale>
#include <map>

#include "glog/logging.h"
#include "models/wordpiece.h"
#include "utils/path.h"
#include "utils/utf8.h"

namespace paddlenlp {
namespace faster_tokenizer {
namespace models {
const std::string WHITESPACE = " \n\r\t\f\v";

WordPiece::WordPiece()
    : unk_token_("[UNK]"),
      continuing_subword_prefix_("##"),
      max_input_chars_per_word_(100),
      unk_token_id_(0) {}

WordPiece::WordPiece(const core::Vocab& vocab,
                     const std::string& unk_token,
                     size_t max_input_chars_per_word,
                     const std::string& continuing_subword_prefix)
    : vocab_(vocab),
      unk_token_(unk_token),
      max_input_chars_per_word_(max_input_chars_per_word),
      continuing_subword_prefix_(continuing_subword_prefix) {
  for (const auto& vocab_item : vocab) {
    vocab_reversed_[vocab_item.second] = vocab_item.first;
  }
  unk_token_id_ = vocab.at(unk_token);
}

// Move version
WordPiece::WordPiece(core::Vocab&& vocab,
                     std::string&& unk_token,
                     size_t max_input_chars_per_word,
                     std::string&& continuing_subword_prefix)
    : vocab_(std::move(vocab)),
      unk_token_(std::move(unk_token)),
      max_input_chars_per_word_(std::move(max_input_chars_per_word)),
      continuing_subword_prefix_(std::move(continuing_subword_prefix)) {
  for (const auto& vocab_item : vocab) {
    vocab_reversed_[vocab_item.second] = vocab_item.first;
  }
  unk_token_id_ = vocab.at(unk_token);
}

core::Vocab WordPiece::GetVocab() const { return vocab_; }

size_t WordPiece::GetVocabSize() const { return vocab_.size(); }

bool WordPiece::TokenToId(const std::string& token, uint32_t* id) const {
  if (vocab_.find(token) == vocab_.end()) {
    return false;
  }
  *id = vocab_.at(token);
  return true;
}

bool WordPiece::IdToToken(uint32_t id, std::string* token) const {
  if (vocab_reversed_.find(id) == vocab_reversed_.end()) {
    return false;
  }
  *token = vocab_reversed_.at(id);
  return true;
}

std::vector<std::string> WordPiece::Save(
    const std::string& folder, const std::string& filename_prefix) const {
  std::string filepath;
  if (filename_prefix == "") {
    filepath = utils::PathJoin(folder, "vocab.txt");
  } else {
    filepath = utils::PathJoin({folder, filename_prefix, "-vocab.txt"});
  }
  VLOG(6) << "Full path" << filepath;
  std::ofstream fout(filepath);
  std::vector<std::pair<std::string, uint32_t>> vocab(vocab_.begin(),
                                                      vocab_.end());
  std::sort(vocab.begin(),
            vocab.end(),
            [](const std::pair<std::string, uint32_t>& left,
               const std::pair<std::string, uint32_t>& right) -> bool {
              return left.second < right.second;
            });
  for (const auto& vocab_item : vocab) {
    fout << vocab_item.first << "\n";
  }
  fout.close();
  return {filepath};
}

std::vector<core::Token> WordPiece::Tokenize(const std::string& sequence) {
  VLOG(6) << "Using WordPiece::Tokenize to tokenize sequence";
  std::vector<core::Token> all_tokens;
  size_t unicode_len =
      utils::GetUnicodeLenFromUTF8(sequence.data(), sequence.length());
  if (unicode_len > max_input_chars_per_word_) {
    all_tokens.emplace_back(
        vocab_.at(unk_token_), unk_token_, core::Offset{0, sequence.length()});
  } else {
    bool found_token = true;
    uint32_t start = 0;
    while (start < sequence.length()) {
      uint32_t end = sequence.length();
      core::Token cur_token;
      bool match_cur_token = false;
      while (start < end) {
        std::string sub_str = sequence.substr(start, end - start);
        if (start > 0) {
          sub_str = continuing_subword_prefix_ + sub_str;
        }
        const auto& vocab_iter = vocab_.find(sub_str);
        if (vocab_iter != vocab_.end()) {
          cur_token = {vocab_iter->second, sub_str, {start, end}};
          match_cur_token = true;
          break;
        }
        // std::u32string u32sub_str = conv.from_bytes(sub_str);
        // end -= utils::GetUTF8CharLen(u32sub_str.back());
        for (auto it = sub_str.rbegin(); it != sub_str.rend(); ++it) {
          --end;
          if (utils::IsCharBeginBoundary(*it)) {
            break;
          }
        }
      }
      if (!match_cur_token) {
        found_token = false;
        break;
      }
      all_tokens.emplace_back(cur_token);
      start = end;
    }
    if (!found_token) {
      all_tokens.clear();
      all_tokens.emplace_back(vocab_.at(unk_token_),
                              unk_token_,
                              core::Offset{0, sequence.length()});
    }
  }
  return all_tokens;
}


core::Vocab WordPiece::GetVocabFromFile(const std::string& file) {
  std::ifstream fin(file);
  core::Vocab vocab;
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
      vocab[word_str] = i++;
    }
  }
  return vocab;
}

WordPiece WordPiece::GetWordPieceFromFile(
    const std::string& file,
    const std::string& unk_token,
    size_t max_input_chars_per_word,
    const std::string& continuing_subword_prefix) {
  auto vocab = GetVocabFromFile(file);
  return WordPiece(
      vocab, unk_token, max_input_chars_per_word, continuing_subword_prefix);
}

void to_json(nlohmann::json& j, const WordPiece& model) {
  j = {
      {"type", "WordPiece"},
      {"vocab", model.vocab_},
      {"unk_token", model.unk_token_},
      {"max_input_chars_per_word", model.max_input_chars_per_word_},
      {"continuing_subword_prefix", model.continuing_subword_prefix_},
  };
}

void from_json(const nlohmann::json& j, WordPiece& model) {
  j["vocab"].get_to(model.vocab_);
  j["unk_token"].get_to(model.unk_token_);
  j["max_input_chars_per_word"].get_to(model.max_input_chars_per_word_);
  j["continuing_subword_prefix"].get_to(model.continuing_subword_prefix_);
}


WordPieceConfig::WordPieceConfig()
    : unk_token_("[UNK]"),
      max_input_chars_per_word_(100),
      continuing_subword_prefix_("##") {}


void WordPieceFactory::SetFiles(const std::string& files) {
  config_.files_ = files;
}

void WordPieceFactory::SetUNKToken(const std::string& unk_token) {
  config_.unk_token_ = unk_token;
}

void WordPieceFactory::SetMaxInputCharsPerWord(
    size_t max_input_chars_per_word) {
  config_.max_input_chars_per_word_ = max_input_chars_per_word;
}

void WordPieceFactory::SetContinuingSubwordPrefix(
    const std::string& continuing_subword_prefix) {
  config_.continuing_subword_prefix_ = continuing_subword_prefix;
}

WordPiece WordPieceFactory::CreateWordPieceModel() {
  std::ifstream fin(config_.files_);
  if (fin) {
    GetVocabFromFiles(config_.files_);
  } else {
    VLOG(0) << "File " << config_.files_
            << " doesn't exist or can't be accessed.";
    config_.vocab_ = core::Vocab();
  }
  return WordPiece{config_.vocab_,
                   config_.unk_token_,
                   config_.max_input_chars_per_word_,
                   config_.continuing_subword_prefix_};
}

void WordPieceFactory::GetVocabFromFiles(const std::string& files) {
  std::ifstream fin(files);
  config_.vocab_.clear();
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
      config_.vocab_[word_str] = i++;
    }
  }
}

}  // namespace model
}  // namespace faster_tokenizer
}  // namespace paddlenlp
