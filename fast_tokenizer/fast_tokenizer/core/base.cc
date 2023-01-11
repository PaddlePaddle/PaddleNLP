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

#include "fast_tokenizer/core/base.h"

#include <thread>

namespace paddlenlp {
namespace fast_tokenizer {
namespace core {

static int fast_tokenizer_thread_num = 1;

void SetThreadNum(int thread_num) { fast_tokenizer_thread_num = thread_num; }

int GetThreadNum() { return fast_tokenizer_thread_num; }

void RunMultiThread(std::function<void(size_t, size_t)> func,
                    size_t batch_size) {
  int thread_num = GetThreadNum();
  if (thread_num == 1) {
    // Note(zhoushunjie): No need to create threads when
    // thread_num equals to 1.
    func(0, batch_size);
  } else {
    std::vector<std::thread> vectorOfThread;
    size_t start_index = 0;
    size_t step_index = ceil(batch_size / float(thread_num));

    for (size_t thread_index = 0; thread_index < thread_num; thread_index++) {
      vectorOfThread.emplace_back(std::thread(func, start_index, step_index));
      start_index = start_index + step_index;
    }
    for (size_t thread_index = 0; thread_index < thread_num; thread_index++) {
      vectorOfThread[thread_index].join();
    }
  }
}

}  // namespace core
}  // namespace fast_tokenizer
}  // namespace paddlenlp
