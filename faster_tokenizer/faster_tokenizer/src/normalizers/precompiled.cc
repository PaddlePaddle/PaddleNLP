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

#include "normalizers/precompiled.h"
#include <iomanip>
#include <sstream>

#include "glog/logging.h"
#include "utils/unique_ptr.h"


namespace paddlenlp {
namespace faster_tokenizer {
namespace normalizers {

PrecompiledNormalizer::PrecompiledNormalizer(
    const std::string& precompiled_charsmap) {
  SetPrecompiledCharsMap(precompiled_charsmap);
}

PrecompiledNormalizer::PrecompiledNormalizer(
    const PrecompiledNormalizer& precompiled_normalizer)
    : sentencepiece_normalizer_(new utils::Normalizer(
          *precompiled_normalizer.sentencepiece_normalizer_.get())) {}

static std::string GetByteFromString(const std::string& str) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (int i = 0; i < str.length(); ++i) {
    oss << "\\x" << std::setw(2) << (static_cast<int>(str[i]) & 0xFF);
  }
  return oss.str();
}

void PrecompiledNormalizer::SetPrecompiledCharsMap(
    const std::string& precompiled_charsmap) {
  sentencepiece_normalizer_ =
      utils::make_unique<utils::Normalizer>(precompiled_charsmap);
}

void PrecompiledNormalizer::operator()(NormalizedString* mut_str) const {
  std::string normalized;
  std::vector<int> norm_to_orig;
  std::u32string u32content;
  if (sentencepiece_normalizer_->Normalize(mut_str->GetStr().data(),
                                           mut_str->GetStr().length(),
                                           &normalized,
                                           &norm_to_orig,
                                           &u32content)) {
    mut_str->UpdateNormalized({u32content, norm_to_orig}, 0);
  }
}

void to_json(nlohmann::json& j,
             const PrecompiledNormalizer& precompiled_normalizer) {
  const auto& precompiled_str =
      precompiled_normalizer.sentencepiece_normalizer_
          ->GetPrecompiledCharsmap();
  std::vector<uint8_t> bytes(precompiled_str.begin(), precompiled_str.end());
  j = {{"type", "PrecompiledNormalizer"}, {"precompiled_charsmap", bytes}};
}

void from_json(const nlohmann::json& j,
               PrecompiledNormalizer& precompiled_normalizer) {
  std::vector<uint8_t> bytes;
  j.at("precompiled_charsmap").get_to(bytes);
  std::ostringstream precompiled_charsmap_oss;
  for (int i = 0; i < bytes.size(); ++i) {
    precompiled_charsmap_oss << static_cast<char>(bytes[i]);
  }
  precompiled_normalizer.SetPrecompiledCharsMap(precompiled_charsmap_oss.str());
}

}  // namespace normalizers
}  // namespace faster_tokenizer
}  // namespace paddlenlp
