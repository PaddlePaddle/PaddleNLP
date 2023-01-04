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

#pragma once
#include <jni.h>

#ifdef __ANDROID__
#include <android/log.h>  // NOLINT
#endif
#include <fstream>  // NOLINT
#include <string>   // NOLINT
#include <vector>   // NOLINT

#include "fastdeploy/fastdeploy_model.h"

#define TAG "[FastDeploy][ERNIE][JNI]"
#ifdef __ANDROID__
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL, TAG, __VA_ARGS__)
#else
#define LOGD(...) {}
#define LOGI(...) {}
#define LOGW(...) {}
#define LOGE(...) {}
#define LOGF(...) {}
#endif

#define ENABLE_RUNTIME_PERF

namespace ernie_tiny {
namespace jni {

/// Time counter
inline int64_t GetCurrentTime() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * static_cast<int64_t>(time.tv_sec) +
         static_cast<int64_t>(time.tv_usec);
}

inline double GetElapsedTime(int64_t time) {
  return (GetCurrentTime() - time) / 1000.0f;
}

/// Show the performance of Runtime
inline void PerfTimeOfRuntime(
    fastdeploy::FastDeployModel *c_model_ptr, int64_t start = -1) {
#ifdef ENABLE_RUNTIME_PERF
  if (c_model_ptr == nullptr) {
    return;
  }
  if (start > 0) {
    auto tc = GetElapsedTime(start);
    LOGD("Predict from native costs %f ms", tc);
  }
  if (c_model_ptr->EnabledRecordTimeOfRuntime()) {
    auto info_of_runtime = c_model_ptr->PrintStatisInfoOfRuntime();
    const float avg_time = info_of_runtime["avg_time"] * 1000.0f;
    LOGD("Avg runtime costs %f ms", avg_time);
  }
#endif
}

#define INITIALIZED_OR_RETURN(c_model_ptr) \
  if (!(c_model_ptr)->Initialized()) { \
    LOGE("Failed to initialize!"); \
    delete (c_model_ptr); \
    return 0; \
  }

#define PERF_TIME_OF_RUNTIME(c_model_ptr, start) \
  ernie_tiny::jni::PerfTimeOfRuntime((c_model_ptr), (start));

}  // namespace jni
}  // namespace ernie_tiny
