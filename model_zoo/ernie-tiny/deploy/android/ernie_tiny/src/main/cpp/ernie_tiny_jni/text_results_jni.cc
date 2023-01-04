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

#include <jni.h>             // NOLINT
#include "ernie_tiny_jni/perf_jni.h"  // NOLINT
#include "ernie_tiny_jni/convert_jni.h"  // NOLINT
#include "ernie_tiny_jni/text_results_jni.h" // NOLINT
#ifdef ENABLE_TEXT
#include "fastdeploy/text.h"  // NOLINT
#endif

namespace ernie_tiny {
namespace jni {

jobject NewUIEJavaResultFromCxx(JNIEnv *env, void *cxx_result) {
  // Field signatures of Java UIEResult:
  // (1) mStart long:                              J
  // (2) mEnd long:                                J
  // (3) mProbability double:                      D
  // (4) mText String:                             Ljava/lang/String;
  // (5) mRelation HashMap<String, UIEResult[]>:   Ljava/util/HashMap;
  // (6) mInitialized boolean:                     Z

  // Return NULL directly if Text API was not enabled. The Text API
  // is not necessarily enabled. Whether or not it is enabled depends
  // on the C++ SDK.
#ifndef ENABLE_TEXT
  return NULL;
#else
  // Allocate Java UIEResult if Text API was enabled.
  if (cxx_result == nullptr) {
    return NULL;
  }
  auto c_result_ptr = reinterpret_cast<fastdeploy::text::UIEResult*>(cxx_result);
  const int len = static_cast<int>(c_result_ptr->text_.size());
  if (len == 0) {
    return NULL;
  }
  const jclass j_uie_result_clazz = env->FindClass(
      "com/baidu/paddle/paddlenlp/ernie_tiny/UIEResult");
  const jfieldID j_uie_start_id = env->GetFieldID(
      j_uie_result_clazz, "mStart", "J");
  const jfieldID j_uie_end_id = env->GetFieldID(
      j_uie_result_clazz, "mEnd", "J");
  const jfieldID j_uie_probability_id = env->GetFieldID(
      j_uie_result_clazz, "mProbability", "D");
  const jfieldID j_uie_text_id = env->GetFieldID(
      j_uie_result_clazz, "mText", "Ljava/lang/String;");
  const jfieldID j_uie_relation_id = env->GetFieldID(
      j_uie_result_clazz, "mRelation", "Ljava/util/HashMap;");
  const jfieldID j_uie_initialized_id = env->GetFieldID(
      j_uie_result_clazz, "mInitialized", "Z");
  // Default UIEResult constructor.
  const jmethodID j_uie_result_init = env->GetMethodID(
      j_uie_result_clazz, "<init>", "()V");

  jobject j_uie_result_obj = env->NewObject(j_uie_result_clazz, j_uie_result_init);

  // Allocate for current UIEResult
  // mStart long: J & mEnd   long: J
  env->SetLongField(j_uie_result_obj, j_uie_start_id,
                    static_cast<jlong>(c_result_ptr->start_));
  env->SetLongField(j_uie_result_obj, j_uie_end_id,
                    static_cast<jlong>(c_result_ptr->end_));
  // mProbability double: D
  env->SetDoubleField(j_uie_result_obj, j_uie_probability_id,
                      static_cast<jdouble>(c_result_ptr->probability_));
  // mText String: Ljava/lang/String;
  env->SetObjectField(j_uie_result_obj, j_uie_text_id,
                      ConvertTo<jstring>(env, c_result_ptr->text_));
  // mInitialized boolean: Z
  env->SetBooleanField(j_uie_result_obj, j_uie_initialized_id, JNI_TRUE);

  // mRelation HashMap<String, UIEResult[]>: Ljava/util/HashMap;
  if (c_result_ptr->relation_.size() > 0) {
    const jclass j_hashmap_clazz = env->FindClass("java/util/HashMap");
    const jmethodID j_hashmap_init = env->GetMethodID(
        j_hashmap_clazz, "<init>", "()V");
    const jmethodID j_hashmap_put = env->GetMethodID(
        j_hashmap_clazz,"put",
        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    // std::unordered_map<std::string, std::vector<UIEResult>> relation_;
    jobject j_uie_relation_hashmap = env->NewObject(j_hashmap_clazz, j_hashmap_init);

    for (auto&& curr_relation : c_result_ptr->relation_) {
      // Processing each key-value cxx uie relation:
      // Key: string, Value: std::vector<UIEResult>
      const auto& curr_c_relation_key = curr_relation.first;
      jstring curr_j_relation_key = ConvertTo<jstring>(env, curr_c_relation_key);
      // Init current relation array (array of UIEResult)
      const int curr_c_uie_result_size = curr_relation.second.size();
      jobjectArray curr_j_uie_result_obj_arr = env->NewObjectArray(
          curr_c_uie_result_size, j_uie_result_clazz, NULL);
      for (int i = 0; i < curr_c_uie_result_size; ++i) {
        fastdeploy::text::UIEResult* child_cxx_result = (&(curr_relation.second[i]));
        // Recursively generates the curr_j_uie_result_obj
        jobject curr_j_uie_result_obj = NewUIEJavaResultFromCxx(
            env, reinterpret_cast<void*>(child_cxx_result));
        env->SetObjectArrayElement(curr_j_uie_result_obj_arr, i, curr_j_uie_result_obj);
        env->DeleteLocalRef(curr_j_uie_result_obj);
      }
      // Put current relation array (array of UIEResult) to HashMap
      env->CallObjectMethod(j_uie_relation_hashmap, j_hashmap_put, curr_j_relation_key,
                            curr_j_uie_result_obj_arr);

      env->DeleteLocalRef(curr_j_relation_key);
      env->DeleteLocalRef(curr_j_uie_result_obj_arr);
    }
    // Set relation HashMap from native
    env->SetObjectField(j_uie_result_obj, j_uie_relation_id, j_uie_relation_hashmap);
    env->DeleteLocalRef(j_hashmap_clazz);
    env->DeleteLocalRef(j_uie_relation_hashmap);
  }

  env->DeleteLocalRef(j_uie_result_clazz);

  return j_uie_result_obj;
#endif
}

bool AllocateUIECxxSchemaNodeFromJava(
    JNIEnv *env, jobject j_schema_node_obj, void *cxx_schema_node) {
#ifndef ENABLE_TEXT
  return false;
#else
  // Allocate cxx SchemaNode from Java SchemaNode
  if (cxx_schema_node == nullptr) {
    return false;
  }
  fastdeploy::text::SchemaNode* c_mutable_schema_node_ptr =
      reinterpret_cast<fastdeploy::text::SchemaNode*>(cxx_schema_node);

  const jclass j_schema_node_clazz = env->FindClass(
      "com/baidu/paddle/paddlenlp/ernie_tiny/SchemaNode");
  if (!env->IsInstanceOf(j_schema_node_obj, j_schema_node_clazz)) {
    return false;
  }

  const jfieldID j_schema_node_name_id = env->GetFieldID(
      j_schema_node_clazz, "mName", "Ljava/lang/String;");
  const jfieldID j_schema_node_child_id = env->GetFieldID(
      j_schema_node_clazz, "mChildren",
      "Ljava/util/ArrayList;");

  // Java ArrayList in JNI
  const jclass j_array_list_clazz = env->FindClass(
      "java/util/ArrayList");
  const jmethodID j_array_list_get = env->GetMethodID(
      j_array_list_clazz,"get", "(I)Ljava/lang/Object;");
  const jmethodID j_array_list_size = env->GetMethodID(
      j_array_list_clazz,"size", "()I");

  // mName String:          Ljava/lang/String;
  c_mutable_schema_node_ptr->name_ = ConvertTo<std::string>(
      env, reinterpret_cast<jstring>(env->GetObjectField(
          j_schema_node_obj, j_schema_node_name_id)));

  // mChildren ArrayList:   Ljava/util/ArrayList;
  jobject j_schema_node_child_array_list = env->GetObjectField(
      j_schema_node_obj, j_schema_node_child_id); // ArrayList
  const int j_schema_node_child_size = static_cast<int>(
      env->CallIntMethod(j_schema_node_child_array_list,
                         j_array_list_size));

  // Recursively add child if child size > 0
  if (j_schema_node_child_size > 0) {
    for (int i = 0; i < j_schema_node_child_size; ++i) {
      fastdeploy::text::SchemaNode curr_c_schema_node_child;
      jobject curr_j_schema_node_child = env->CallObjectMethod(
          j_schema_node_child_array_list, j_array_list_get, i);
      if (AllocateUIECxxSchemaNodeFromJava(
          env, curr_j_schema_node_child, reinterpret_cast<void*>(
              &curr_c_schema_node_child))) {
        c_mutable_schema_node_ptr->AddChild(curr_c_schema_node_child);
      }
    }
  }

  return true;
#endif
}

}  // namespace jni
}  // namespace ernie_tiny_jni
