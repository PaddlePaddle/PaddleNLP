// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cassert>
#include <cstring>

namespace paddlenlp {
namespace faster_tokenizer {
namespace utils {

struct simple_string_view {
  const char* ptr_;
  size_t offset_;
  size_t size_;
  explicit simple_string_view(const char* ptr = nullptr)
      : ptr_(ptr), offset_(0), size_(0) {
    while (ptr_ && ptr_[size_] != '\0') {
      size_++;
    }
  }
  simple_string_view(const char* ptr, size_t size) : ptr_(ptr), size_(size) {}

  const char* data() const {
    if (!ptr_) {
      return ptr_ + offset_;
    }
    return ptr_;
  }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  void remove_prefix(size_t n) {
    assert(n <= size_);
    ptr_ += n;
    size_ -= n;
  }
};

}  // namespace utils
}  // namespace faster_tokenizer
}  // namespace paddlenlp
