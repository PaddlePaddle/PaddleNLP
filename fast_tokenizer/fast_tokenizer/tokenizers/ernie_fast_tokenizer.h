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

struct FASTTOKENIZER_DECL ErnieFastTokenizer : public core::Tokenizer {
  ErnieFastTokenizer(const std::string& vocab_path,
                       const std::string& unk_token = "[UNK]",
                       const std::string& sep_token = "[SEP]",
                       const std::string& cls_token = "[CLS]",
                       const std::string& pad_token = "[PAD]",
                       const std::string& mask_token = "[MASK]",
                       bool clean_text = true,
                       bool handle_chinese_chars = true,
                       bool strip_accents = true,
                       bool lowercase = true,
                       const std::string& wordpieces_prefix = "##",
                       uint32_t max_sequence_len = 0);

  ErnieFastTokenizer(const core::Vocab& vocab,
                       const std::string& unk_token = "[UNK]",
                       const std::string& sep_token = "[SEP]",
                       const std::string& cls_token = "[CLS]",
                       const std::string& pad_token = "[PAD]",
                       const std::string& mask_token = "[MASK]",
                       bool clean_text = true,
                       bool handle_chinese_chars = true,
                       bool strip_accents = true,
                       bool lowercase = true,
                       const std::string& wordpieces_prefix = "##",
                       uint32_t max_sequence_len = 0);

private:
  void Init(const core::Vocab& vocab,
            const std::string& unk_token = "[UNK]",
            const std::string& sep_token = "[SEP]",
            const std::string& cls_token = "[CLS]",
            const std::string& pad_token = "[PAD]",
            const std::string& mask_token = "[MASK]",
            bool clean_text = true,
            bool handle_chinese_chars = true,
            bool strip_accents = true,
            bool lowercase = true,
            const std::string& wordpieces_prefix = "##",
            uint32_t max_sequence_len = 0);
};

}  // namespace fast_tokenizer_impl
}  // namespace fast_tokenizer
}  // namespace paddlenlp
