// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "sm.h"


int GetSMVersion() {
  int device{-1};
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  return props.major * 10 + props.minor;
}

SMVersion* SMVersion::GetInstance() {
  static SMVersion* sm_version_ = nullptr;
  if (sm_version_ == nullptr) {
    sm_version_ = new SMVersion();
  }
  return sm_version_;
}
