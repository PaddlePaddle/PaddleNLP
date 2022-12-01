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

#include <array>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddlenlp {
namespace fast_tokenizer {
namespace tests {

template <typename T>
void CheckVectorEqual(const std::vector<T>& a, const std::vector<T>& b) {
  ASSERT_EQ(a.size(), b.size());
  auto size = a.size();
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(a[i], b[i]);
  }
}

}  // namespace tests
}  // namespace fast_tokenizer
}  // namespace paddlenlp