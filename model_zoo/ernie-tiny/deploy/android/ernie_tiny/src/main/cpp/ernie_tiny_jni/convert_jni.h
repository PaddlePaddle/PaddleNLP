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

#include <jni.h>   // NOLINT
#include <string>  // NOLINT
#include <vector>  // NOLINT

namespace ernie_tiny {
namespace jni {

template <typename OutputType, typename InputType>
OutputType ConvertTo(JNIEnv *env, InputType input);

template <typename OutputType, typename InputType>
OutputType ConvertTo(JNIEnv *env, const InputType *input, int64_t len);

/// jstring -> std::string
template <>
inline std::string ConvertTo(JNIEnv *env, jstring jstr) {
  // In java, a unicode char will be encoded using 2 bytes (utf16).
  // so jstring will contain characters utf16. std::string in c++ is
  // essentially a string of bytes, not characters, so if we want to
  // pass jstring from JNI to c++, we have convert utf16 to bytes.
  if (!jstr) {
    return "";
  }
  const jclass jstring_clazz = env->GetObjectClass(jstr);
  const jmethodID getBytesID =
      env->GetMethodID(jstring_clazz, "getBytes", "(Ljava/lang/String;)[B");
  const jbyteArray jstring_bytes = reinterpret_cast<jbyteArray>(
      env->CallObjectMethod(jstr, getBytesID, env->NewStringUTF("UTF-8")));

  size_t length = static_cast<size_t>(env->GetArrayLength(jstring_bytes));
  jbyte *jstring_bytes_ptr = env->GetByteArrayElements(jstring_bytes, NULL);

  std::string res =
      std::string(reinterpret_cast<char *>(jstring_bytes_ptr), length);
  env->ReleaseByteArrayElements(jstring_bytes, jstring_bytes_ptr, JNI_ABORT);

  env->DeleteLocalRef(jstring_bytes);
  env->DeleteLocalRef(jstring_clazz);
  return res;
}

/// jstring s-> std::vector<std::string>
template <>
inline std::vector<std::string> ConvertTo(JNIEnv *env, jobjectArray jstrs) {
  // In java, a unicode char will be encoded using 2 bytes (utf16).
  // so jstring will contain characters utf16. std::string in c++ is
  // essentially a string of bytes, not characters, so if we want to
  // pass jstring from JNI to c++, we have convert utf16 to bytes.
  if (!jstrs) {
    return {};
  }
  std::vector<std::string> res;
  const int len = env->GetArrayLength(jstrs);
  if (len > 0) {
    for (int i = 0; i < len; ++i) {
      auto j_str =
          reinterpret_cast<jstring>(env->GetObjectArrayElement(jstrs, i));
      res.push_back(ernie_tiny::jni::ConvertTo<std::string>(env, j_str));
    }
  }
  return res;
}

/// std::string -> jstring
template <>
inline jstring ConvertTo(JNIEnv *env, std::string str) {
  auto *cstr_data_ptr = str.c_str();
  jclass jstring_clazz = env->FindClass("java/lang/String");
  jmethodID initID =
      env->GetMethodID(jstring_clazz, "<init>", "([BLjava/lang/String;)V");

  jbyteArray jstring_bytes = env->NewByteArray(strlen(cstr_data_ptr));
  env->SetByteArrayRegion(jstring_bytes, 0, strlen(cstr_data_ptr),
                          reinterpret_cast<const jbyte *>(cstr_data_ptr));

  jstring jstring_encoding = env->NewStringUTF("UTF-8");
  jstring res = reinterpret_cast<jstring>(
      env->NewObject(jstring_clazz, initID, jstring_bytes, jstring_encoding));

  env->DeleteLocalRef(jstring_clazz);
  env->DeleteLocalRef(jstring_bytes);
  env->DeleteLocalRef(jstring_encoding);

  return res;
}

/// jlongArray -> std::vector<int64_t>
template <>
inline std::vector<int64_t> ConvertTo(JNIEnv *env, jlongArray jdata) {
  int jdata_size = env->GetArrayLength(jdata);
  jlong *jdata_ptr = env->GetLongArrayElements(jdata, nullptr);
  std::vector<int64_t> res(jdata_ptr, jdata_ptr + jdata_size);
  env->ReleaseLongArrayElements(jdata, jdata_ptr, 0);
  return res;
}

/// jintArray -> std::vector<int>
template <>
inline std::vector<int> ConvertTo(JNIEnv *env, jintArray jdata) {
  int jdata_size = env->GetArrayLength(jdata);
  jint *jdata_ptr = env->GetIntArrayElements(jdata, nullptr);
  std::vector<int> res(jdata_ptr, jdata_ptr + jdata_size);
  env->ReleaseIntArrayElements(jdata, jdata_ptr, 0);
  return res;
}

/// jfloatArray -> std::vector<float>
template <>
inline std::vector<float> ConvertTo(JNIEnv *env, jfloatArray jdata) {
  int jdata_size = env->GetArrayLength(jdata);
  jfloat *jdata_ptr = env->GetFloatArrayElements(jdata, nullptr);
  std::vector<float> res(jdata_ptr, jdata_ptr + jdata_size);
  env->ReleaseFloatArrayElements(jdata, jdata_ptr, 0);
  return res;
}

/// std::vector<int64_t> -> jlongArray
template <>
inline jlongArray ConvertTo(JNIEnv *env, const std::vector<int64_t> &cvec) {
  jlongArray res = env->NewLongArray(cvec.size());
  jlong *jbuf = new jlong[cvec.size()];
  for (size_t i = 0; i < cvec.size(); ++i) {
    jbuf[i] = static_cast<jlong>(cvec[i]);
  }
  env->SetLongArrayRegion(res, 0, cvec.size(), jbuf);
  delete[] jbuf;
  return res;
}

/// cxx float buffer -> jfloatArray
template <>
inline jfloatArray ConvertTo(JNIEnv *env, const float *cbuf, int64_t len) {
  jfloatArray res = env->NewFloatArray(len);
  env->SetFloatArrayRegion(res, 0, len, cbuf);
  return res;
}

/// cxx int buffer -> jintArray
template <>
inline jintArray ConvertTo(JNIEnv *env, const int *cbuf, int64_t len) {
  jintArray res = env->NewIntArray(len);
  env->SetIntArrayRegion(res, 0, len, cbuf);
  return res;
}

/// cxx int8_t buffer -> jbyteArray
template <>
inline jbyteArray ConvertTo(JNIEnv *env, const int8_t *cbuf, int64_t len) {
  jbyteArray res = env->NewByteArray(len);
  env->SetByteArrayRegion(res, 0, len, cbuf);
  return res;
}

}  // namespace jni
}  // namespace ernie_tiny
