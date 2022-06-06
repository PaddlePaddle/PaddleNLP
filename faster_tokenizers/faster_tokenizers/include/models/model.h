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

#include <unordered_map>
#include <vector>
#include "core/base.h"

namespace tokenizers {
namespace models {

struct Model {
  virtual std::vector<core::Token> Tokenize(
      const std::string& tokens) const = 0;
  virtual bool TokenToId(const std::string& token, uint* id) const = 0;
  virtual bool IdToToken(uint id, std::string* token) const = 0;
  virtual core::Vocab GetVocab() const = 0;
  virtual size_t GetVocabSize() const = 0;
  // Return the saved voacb path
  virtual std::string Save(const std::string& folder,
                           const std::string& filename_prefix) const = 0;
};

}  // model
}  // tokenizers
