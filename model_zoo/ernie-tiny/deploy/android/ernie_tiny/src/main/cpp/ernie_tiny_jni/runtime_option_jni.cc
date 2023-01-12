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

#include "ernie_tiny_jni/convert_jni.h"         // NOLINT
#include "ernie_tiny_jni/runtime_option_jni.h"  // NOLINT

namespace ernie_tiny {
namespace jni {

fastdeploy::RuntimeOption NewCxxRuntimeOption(
    JNIEnv *env, jobject j_runtime_option_obj) {
  // WARN: Please make sure 'j_runtime_option_obj' param is a
  // ref of Java RuntimeOption.
  // Field signatures of Java RuntimeOption.
  // (1) mCpuThreadNum int:               I
  // (2) mEnableLiteFp16 boolean:         Z
  // (3) mLitePowerMode LitePowerMode:    com/baidu/paddle/paddlenlp/ernie_tiny/LitePowerMode
  // (4) mLiteOptimizedModelDir String:   java/lang/String

  const jclass j_runtime_option_clazz = env->FindClass(
      "com/baidu/paddle/paddlenlp/ernie_tiny/RuntimeOption");
  const jfieldID j_cpu_num_thread_id = env->GetFieldID(
      j_runtime_option_clazz, "mCpuThreadNum", "I");
  const jfieldID j_enable_lite_fp16_id = env->GetFieldID(
      j_runtime_option_clazz, "mEnableLiteFp16", "Z");
  const jfieldID j_enable_lite_int8_id = env->GetFieldID(
      j_runtime_option_clazz, "mEnableLiteInt8", "Z");
  const jfieldID j_lite_power_mode_id = env->GetFieldID(
      j_runtime_option_clazz, "mLitePowerMode",
      "Lcom/baidu/paddle/paddlenlp/ernie_tiny/LitePowerMode;");
  const jfieldID j_lite_optimized_model_dir_id = env->GetFieldID(
      j_runtime_option_clazz, "mLiteOptimizedModelDir", "Ljava/lang/String;");

  // mLitePowerMode is Java Enum.
  const jclass j_lite_power_mode_clazz = env->FindClass(
      "com/baidu/paddle/paddlenlp/ernie_tiny/LitePowerMode");
  const jmethodID j_lite_power_mode_ordinal_id = env->GetMethodID(
      j_lite_power_mode_clazz, "ordinal", "()I");

  fastdeploy::RuntimeOption c_runtime_option;
  c_runtime_option.UseCpu();
  c_runtime_option.UseLiteBackend();

  if (!env->IsInstanceOf(j_runtime_option_obj, j_runtime_option_clazz)) {
    return c_runtime_option;
  }

  // Get values from Java RuntimeOption.
  jint j_cpu_num_thread = env->GetIntField(
      j_runtime_option_obj, j_cpu_num_thread_id);
  jboolean j_enable_lite_fp16 = env->GetBooleanField(
      j_runtime_option_obj, j_enable_lite_fp16_id);
  jboolean j_enable_lite_int8 = env->GetBooleanField(
      j_runtime_option_obj, j_enable_lite_int8_id);
  jstring j_lite_optimized_model_dir = static_cast<jstring>(
      env->GetObjectField(j_runtime_option_obj, j_lite_optimized_model_dir_id));
  jobject j_lite_power_mode_obj = env->GetObjectField(
      j_runtime_option_obj, j_lite_power_mode_id);
  jint j_lite_power_mode = env->CallIntMethod(
      j_lite_power_mode_obj, j_lite_power_mode_ordinal_id);

  int c_cpu_num_thread = static_cast<int>(j_cpu_num_thread);
  bool c_enable_lite_fp16 = static_cast<bool>(j_enable_lite_fp16);
  bool c_enable_lite_int8 = static_cast<bool>(j_enable_lite_int8);
  fastdeploy::LitePowerMode c_lite_power_mode =
      static_cast<fastdeploy::LitePowerMode>(j_lite_power_mode);
  std::string c_lite_optimized_model_dir =
      ConvertTo<std::string>(env, j_lite_optimized_model_dir);

  // Setup Cxx RuntimeOption
  c_runtime_option.SetCpuThreadNum(c_cpu_num_thread);
  c_runtime_option.SetLitePowerMode(c_lite_power_mode);
  c_runtime_option.SetLiteOptimizedModelDir(c_lite_optimized_model_dir);
  if (c_enable_lite_fp16) {
    c_runtime_option.EnableLiteFP16();
  }
  if (c_enable_lite_int8) {
    c_runtime_option.EnableLiteInt8();
  }

  env->DeleteLocalRef(j_runtime_option_clazz);
  env->DeleteLocalRef(j_lite_power_mode_clazz);

  return c_runtime_option;
}

}  // namespace jni
}  // namespace ernie_tiny
