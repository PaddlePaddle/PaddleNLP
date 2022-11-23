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

#include <string>
#include <unordered_map>
#include "fast_tokenizer/core/encoding.h"
#include "fast_tokenizer/core/tokenizer.h"
#include "fast_tokenizer/utils/utils.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace tokenizers_impl {

struct FASTTOKENIZER_DECL ClipFastTokenizer : public core::Tokenizer {
  ClipFastTokenizer(const std::string& vocab_path,
                    const std::string& merges_path,
                    uint32_t max_length = 0,
                    const std::string& unk_token = "<|endoftext|>",
                    const std::string& pad_token = "<|endoftext|>",
                    const std::string& bos_token = "<|startoftext|>",
                    const std::string& eos_token = "<|endoftext|>",
                    bool add_prefix_space = false,
                    const std::string& continuing_subword_prefix = "",
                    const std::string& end_of_word_suffix = "</w>",
                    bool trim_offsets = false);
  std::string GetPadToken() const;
  uint32_t GetPadTokenId() const;
  std::string GetUNKToken() const;
  uint32_t GetUNKTokenId() const;
  std::string GetBOSToken() const;
  uint32_t GetBOSTokenId() const;
  std::string GetEOSToken() const;
  uint32_t GetEOSTokenId() const;

private:
  std::string pad_token_;
  uint32_t pad_token_id_;
  std::string unk_token_;
  uint32_t unk_token_id_;
  std::string bos_token_;
  uint32_t bos_token_id_;
  std::string eos_token_;
  uint32_t eos_token_id_;
};

}  // namespace fast_tokenizer_impl
}  // namespace fast_tokenizer
}  // namespace paddlenlp
