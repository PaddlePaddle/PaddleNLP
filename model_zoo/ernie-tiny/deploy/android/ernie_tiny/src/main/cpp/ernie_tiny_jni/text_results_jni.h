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

#include <jni.h>  // NOLINT

namespace ernie_tiny {
namespace jni {

/**
 * @param env A Pointer of JNIENV.
 * @param cxx_result A pointer of cxx 'text::UIEResult'
 * @return jobject that stands for Java UIEResult
 */
jobject NewUIEJavaResultFromCxx(JNIEnv *env, void *cxx_result);

/**
 * Allocate one cxx SchemaNode from Java SchemaNode
 * @param env A Pointer of JNIENV.
 * @param j_schema_node_obj jobject that stands for Java SchemaNode
 * @param cxx_schema_node A pointer of cxx 'text::SchemaNode'
 * @return true if success, false if failed.
 */
bool AllocateUIECxxSchemaNodeFromJava(
    JNIEnv *env, jobject j_schema_node_obj, void *cxx_schema_node);

}  // namespace jni
}  // namespace ernie_tiny
