// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T>
void cpu_rand_data(T *c) {
  auto t = *c;

  using ValueType = typename T::value_type;

  int n = size(t);
  for (int i = 0; i < n; ++i) {
    float v = ((rand() % 200) - 100.f) * 0.01f;
    // printf("v = %f\n", v);
    t(i) = ValueType(v);
  }
}