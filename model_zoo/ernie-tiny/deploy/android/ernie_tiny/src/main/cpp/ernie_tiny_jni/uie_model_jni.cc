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

#include "ernie_tiny_jni/convert_jni.h"             // NOLINT
#include "ernie_tiny_jni/perf_jni.h"                // NOLINT
#include "ernie_tiny_jni/runtime_option_jni.h"      // NOLINT
#include "ernie_tiny_jni/text_results_jni.h"   // NOLINT
#include "ernie_tiny_jni/uie_utils_jni.h"  // NOLINT
#include <jni.h>                                    // NOLINT
#ifdef ENABLE_TEXT
#include "fastdeploy/text.h"  // NOLINT
#endif

#ifdef ENABLE_TEXT
namespace text = fastdeploy::text;
#endif

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_paddlenlp_ernie_1tiny_UIEModel_bindNative(
    JNIEnv* env, jobject thiz, jstring model_file, jstring params_file,
    jstring vocab_file, jfloat position_prob, jint max_length,
    jobjectArray schema, jint batch_size, jobject runtime_option,
    jint schema_language) {
#ifndef ENABLE_TEXT
  return 0;
#else
  auto c_model_file = ernie_tiny::jni::ConvertTo<std::string>(env, model_file);
  auto c_params_file = ernie_tiny::jni::ConvertTo<std::string>(env, params_file);
  auto c_vocab_file = ernie_tiny::jni::ConvertTo<std::string>(env, vocab_file);
  auto c_position_prob = static_cast<jfloat>(position_prob);
  auto c_max_length = static_cast<size_t>(max_length);
  auto c_schema = ernie_tiny::jni::ConvertTo<std::vector<std::string>>(env, schema);
  auto c_batch_size = static_cast<int>(batch_size);
  auto c_runtime_option = ernie_tiny::jni::NewCxxRuntimeOption(env, runtime_option);
  auto c_schema_language = static_cast<text::SchemaLanguage>(schema_language);
  auto c_paddle_model_format = fastdeploy::ModelFormat::PADDLE;
  auto c_model_ptr = new text::UIEModel(
      c_model_file, c_params_file, c_vocab_file, c_position_prob, c_max_length,
      c_schema, c_batch_size, c_runtime_option, c_paddle_model_format,
      c_schema_language);
  INITIALIZED_OR_RETURN(c_model_ptr)

#ifdef ENABLE_RUNTIME_PERF
  c_model_ptr->EnableRecordTimeOfRuntime();
#endif
  return reinterpret_cast<jlong>(c_model_ptr);
#endif
}

JNIEXPORT jobjectArray JNICALL
Java_com_baidu_paddle_paddlenlp_ernie_1tiny_UIEModel_predictNative(
    JNIEnv* env, jobject thiz, jlong cxx_context, jobjectArray texts) {
#ifndef ENABLE_TEXT
  return NULL;
#else
  if (cxx_context == 0) {
    return NULL;
  }
  auto c_model_ptr = reinterpret_cast<text::UIEModel*>(cxx_context);
  auto c_texts = ernie_tiny::jni::ConvertTo<std::vector<std::string>>(env, texts);
  if (c_texts.empty()) {
    LOGE("c_texts is empty!");
    return NULL;
  }
  LOGD("c_texts: %s", ernie_tiny::jni::UIETextsStr(c_texts).c_str());

  std::vector<std::unordered_map<std::string, std::vector<text::UIEResult>>>
      c_results;

  auto t = ernie_tiny::jni::GetCurrentTime();
  c_model_ptr->Predict(c_texts, &c_results);
  PERF_TIME_OF_RUNTIME(c_model_ptr, t)

  if (c_results.empty()) {
    LOGE("c_results is empty!");
    return NULL;
  }
  LOGD("c_results: %s", ernie_tiny::jni::UIEResultsStr(c_results).c_str());

  // Push results to HashMap array
  const char* j_hashmap_put_signature =
      "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;";
  const jclass j_hashmap_clazz = env->FindClass("java/util/HashMap");
  const jclass j_uie_result_clazz =
      env->FindClass("com/baidu/paddle/paddlenlp/ernie_tiny/UIEResult");
  // Get HashMap method id
  const jmethodID j_hashmap_init =
      env->GetMethodID(j_hashmap_clazz, "<init>", "()V");
  const jmethodID j_hashmap_put =
      env->GetMethodID(j_hashmap_clazz, "put", j_hashmap_put_signature);

  const int c_uie_result_hashmap_size = c_results.size();
  jobjectArray j_hashmap_uie_result_arr =
      env->NewObjectArray(c_uie_result_hashmap_size, j_hashmap_clazz, NULL);

  for (int i = 0; i < c_uie_result_hashmap_size; ++i) {
    auto& curr_c_uie_result_map = c_results[i];

    // Convert unordered_map<string, vector<UIEResult>>
    // -> HashMap<String, UIEResult[]>
    jobject curr_j_uie_result_hashmap =
        env->NewObject(j_hashmap_clazz, j_hashmap_init);

    for (auto&& curr_c_uie_result : curr_c_uie_result_map) {

      const auto& curr_inner_c_uie_key = curr_c_uie_result.first;
      jstring curr_inner_j_uie_key =
          ernie_tiny::jni::ConvertTo<jstring>(env, curr_inner_c_uie_key);  // Key of HashMap

      if (curr_c_uie_result.second.size() > 0) {
        // Value of HashMap: HashMap<String, UIEResult[]>
        jobjectArray curr_inner_j_uie_result_values = env->NewObjectArray(
            curr_c_uie_result.second.size(), j_uie_result_clazz, NULL);

        // Convert vector<UIEResult> -> Java UIEResult[]
        for (int j = 0; j < curr_c_uie_result.second.size(); ++j) {

          text::UIEResult* inner_c_uie_result =
              (&(curr_c_uie_result.second[j]));

          jobject curr_inner_j_uie_result_obj = ernie_tiny::jni::NewUIEJavaResultFromCxx(
              env, reinterpret_cast<void*>(inner_c_uie_result));

          env->SetObjectArrayElement(curr_inner_j_uie_result_values, j,
                                     curr_inner_j_uie_result_obj);
          env->DeleteLocalRef(curr_inner_j_uie_result_obj);
        }

        // Set element of 'curr_j_uie_result_hashmap':
        // HashMap<String, UIEResult[]>
        env->CallObjectMethod(curr_j_uie_result_hashmap, j_hashmap_put,
                              curr_inner_j_uie_key,
                              curr_inner_j_uie_result_values);

        env->DeleteLocalRef(curr_inner_j_uie_key);
        env->DeleteLocalRef(curr_inner_j_uie_result_values);
      }  // end if
    }    // end for

    // Set current HashMap<String, UIEResult[]> to HashMap[i]
    env->SetObjectArrayElement(j_hashmap_uie_result_arr, i,
                               curr_j_uie_result_hashmap);
    env->DeleteLocalRef(curr_j_uie_result_hashmap);
  }

  return j_hashmap_uie_result_arr;
#endif
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_paddlenlp_ernie_1tiny_UIEModel_releaseNative(
    JNIEnv* env, jobject thiz, jlong cxx_context) {
#ifndef ENABLE_TEXT
  return JNI_FALSE;
#else
  if (cxx_context == 0) {
    return JNI_FALSE;
  }
  auto c_model_ptr = reinterpret_cast<text::UIEModel*>(cxx_context);
  PERF_TIME_OF_RUNTIME(c_model_ptr, -1)

  delete c_model_ptr;
  LOGD("[End] Release UIEModel in native !");
  return JNI_TRUE;
#endif
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_paddlenlp_ernie_1tiny_UIEModel_setSchemaStringNative(
    JNIEnv* env, jobject thiz, jlong cxx_context, jobjectArray schema) {
#ifndef ENABLE_TEXT
  return JNI_FALSE;
#else
  if (cxx_context == 0) {
    return JNI_FALSE;
  }
  auto c_model_ptr = reinterpret_cast<text::UIEModel*>(cxx_context);
  auto c_schema = ernie_tiny::jni::ConvertTo<std::vector<std::string>>(env, schema);
  if (c_schema.empty()) {
    LOGE("c_schema is empty!");
    return JNI_FALSE;
  }
  LOGD("c_schema is: %s", ernie_tiny::jni::UIESchemasStr(c_schema).c_str());
  c_model_ptr->SetSchema(c_schema);
  return JNI_TRUE;
#endif
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_paddlenlp_ernie_1tiny_UIEModel_setSchemaNodeNative(
    JNIEnv* env, jobject thiz, jlong cxx_context, jobjectArray schema) {
#ifndef ENABLE_TEXT
  return JNI_FALSE;
#else
  if (schema == NULL) {
    return JNI_FALSE;
  }
  const int j_schema_size = env->GetArrayLength(schema);
  if (j_schema_size == 0) {
    return JNI_FALSE;
  }
  if (cxx_context == 0) {
    return JNI_FALSE;
  }
  auto c_model_ptr = reinterpret_cast<text::UIEModel*>(cxx_context);

  std::vector<text::SchemaNode> c_schema;
  for (int i = 0; i < j_schema_size; ++i) {
    jobject curr_j_schema_node = env->GetObjectArrayElement(schema, i);
    text::SchemaNode curr_c_schema_node;
    if (ernie_tiny::jni::AllocateUIECxxSchemaNodeFromJava(
            env, curr_j_schema_node,
            reinterpret_cast<void*>(&curr_c_schema_node))) {
      c_schema.push_back(curr_c_schema_node);
    }
    env->DeleteLocalRef(curr_j_schema_node);
  }

  if (c_schema.empty()) {
    LOGE("c_schema is empty!");
    return JNI_FALSE;
  }

  c_model_ptr->SetSchema(c_schema);

  return JNI_TRUE;
#endif
}

#ifdef __cplusplus
}
#endif
