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
#include "models/bpe.h"

namespace tokenizers {
namespace models {

std::vector<core::Token> BPE::Tokenize(const std::string& tokens) const {
  return {};
}

bool BPE::TokenToId(const std::string& token, uint* id) const { return true; }

bool BPE::IdToToken(uint id, std::string* token) const { return true; }

core::Vocab BPE::GetVocab() const { return {}; }

size_t BPE::GetVocabSize() const { return 0; }
// Return the saved voacb path
std::string BPE::Save(const std::string& folder,
                      const std::string& filename_prefix) const {
  return "";
}

}  // model
}  // tokenizers