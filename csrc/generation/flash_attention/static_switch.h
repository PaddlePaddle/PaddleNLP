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

// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#define FWD_HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM <= 32) {                   \
      constexpr static int kHeadDim = 32;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 64) {            \
      constexpr static int kHeadDim = 64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 96) {            \
      constexpr static int kHeadDim = 96;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 128) {           \
      constexpr static int kHeadDim = 128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 160) {           \
      constexpr static int kHeadDim = 160; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 192) {           \
      constexpr static int kHeadDim = 192; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 224) {           \
      constexpr static int kHeadDim = 224; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 256) {           \
      constexpr static int kHeadDim = 256; \
      return __VA_ARGS__();                \
    }                                      \
  }()
