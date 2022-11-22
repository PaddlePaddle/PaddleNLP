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

#include "fast_tokenizer/decoders/decoder.h"
#include "fast_tokenizer/utils/utils.h"
#include "nlohmann/json.hpp"

namespace paddlenlp {
namespace fast_tokenizer {
namespace decoders {

struct FASTTOKENIZER_DECL WordPiece : public Decoder {
  virtual void operator()(const std::vector<std::string> tokens,
                          std::string* result) const;

  WordPiece(const std::string prefix = "##", bool cleanup = true);

private:
  void CleanUp(std::string* result) const;
  std::string prefix_;
  bool cleanup_;

  friend void to_json(nlohmann::json& j, const WordPiece& decoder);
  friend void from_json(const nlohmann::json& j, WordPiece& decoder);
};

}  // namespace decoders
}  // namespace fast_tokenizer
}  // namespace paddlenlp
