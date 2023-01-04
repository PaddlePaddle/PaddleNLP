// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ernie_tiny_jni/uie_utils_jni.h" // NOLINT

namespace ernie_tiny {
namespace jni {

#ifdef ENABLE_TEXT
std::string UIEResultStr(const fastdeploy::text::UIEResult& result) {
  std::ostringstream oss;
  oss << result;
  return oss.str();
}

std::string UIEResultsStr(
    const std::vector<std::unordered_map<
        std::string, std::vector<fastdeploy::text::UIEResult>>>& results) {
  std::ostringstream oss;
  oss << results;
  return oss.str();
}

std::string UIETextsStr(const std::vector<std::string>& texts) {
  std::string str = "";
  for (const auto& s: texts) {
    str += (s + ";");
  }
  return str;
}

std::string UIESchemasStr(const std::vector<std::string>& schemas) {
  std::string str = "";
  for (const auto& s: schemas) {
    str += (s + ";");
  }
  return str;
}
#endif

}  // namespace jni
}  // namespace ernie_tiny
