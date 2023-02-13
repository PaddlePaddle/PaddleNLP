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

#include <jni.h>                                                 // NOLINT
#include <iostream>                                              // NOLINT
#include <sstream>                                               // NOLINT
#include <vector>                                                // NOLINT
#include "ernie_tiny_jni/convert_jni.h"                          // NOLINT
#include "ernie_tiny_jni/perf_jni.h"                             // NOLINT
#include "ernie_tiny_jni/runtime_option_jni.h"                   // NOLINT
#include "fast_tokenizer/pretokenizers/pretokenizer.h"           // NOLINT
#include "fast_tokenizer/tokenizers/ernie_fast_tokenizer.h"      // NOLINT
#include "fast_tokenizer/utils/utf8.h"                           // NOLINT
#include "fastdeploy/function/functions.h"                       // NOLINT
#include "fastdeploy/runtime.h"                                  // NOLINT
#include "nlohmann/json.hpp"                                     // NOLINT

using namespace paddlenlp;
using namespace fast_tokenizer::tokenizers_impl;

static bool BatchiFyTexts(const std::vector<std::string>& texts, int batch_size,
                          std::vector<std::vector<std::string>>* batch_texts) {
  for (int idx = 0; idx < texts.size(); idx += batch_size) {
    int rest = texts.size() - idx;
    int curr_size = std::min(batch_size, rest);
    std::vector<std::string> batch_text(curr_size);
    std::copy_n(texts.begin() + idx, curr_size, batch_text.begin());
    batch_texts->emplace_back(std::move(batch_text));
  }
  return true;
}

struct IntentDetAndSlotFillResult {
  struct IntentDetResult {
    std::string intent_label;
    float intent_confidence;
  } intent_result;
  struct SlotFillResult {
    std::string slot_label;
    std::string entity;
    std::pair<int, int> pos;
  };
  std::vector<SlotFillResult> slot_result;

  friend std::ostream& operator<<(std::ostream& os,
                                  const IntentDetAndSlotFillResult& result);
};

std::ostream& operator<<(std::ostream& os,
                         const IntentDetAndSlotFillResult& result) {
  os << "intent result: label = " << result.intent_result.intent_label
     << ", confidence = " << result.intent_result.intent_confidence
     << std::endl;
  os << "slot result: " << std::endl;
  for (auto&& slot : result.slot_result) {
    os << "slot = " << slot.slot_label << ", entity = '" << slot.entity
       << "', pos = [" << slot.pos.first << ", " << slot.pos.second << "]"
       << std::endl;
  }
  return os;
}

static std::string ResultStr(const IntentDetAndSlotFillResult& result) {
  std::ostringstream oss;
  oss << result;
  return oss.str();
}

static std::string TextsStr(const std::vector<std::string>& texts) {
  std::string str = "";
  for (const auto& s: texts) {
    str += (s + ";");
  }
  return str;
}

static bool AddTokensToTokenizer(
    const std::string& path, ErnieFastTokenizer* tokenizer) {
  std::string actual_path = path;
  std::ifstream fin(actual_path);
  nlohmann::json j;
  fin >> j;
  using VOCAB_ITEM = std::pair<std::string, uint32_t>;
  std::vector<VOCAB_ITEM> vocab_list;
  for (nlohmann::json::iterator it = j.begin(); it != j.end(); ++it) {
    vocab_list.emplace_back(it.key(), it.value());
  }
  std::sort(vocab_list.begin(),
            vocab_list.end(),
            [](const VOCAB_ITEM& a, const VOCAB_ITEM& b) {
              return a.second < b.second;
            });
  std::vector<fast_tokenizer::core::AddedToken> added_tokens;
  int increase_idx = 0;
  for (auto&& vocab_item : vocab_list) {
    if (vocab_item.second != tokenizer->GetVocabSize() + increase_idx) {
      fastdeploy::FDERROR << "Non-consecutive added token '" << vocab_item.first
                          << "' found. "
                          << "Should have index " << tokenizer->GetVocabSize()
                          << " but has index " << vocab_item.second
                          << " in saved vocabulary." << std::endl;
      return false;
    }
    added_tokens.emplace_back(vocab_item.first);
    ++increase_idx;
  }
  std::ostringstream oss;
  oss << "Try to add the following tokens to tokenizer vocab. AddedTokens = [";
  for (int i = 0; i < added_tokens.size(); ++i) {
    auto&& token = added_tokens[i];
    oss << token.GetContent();
    if (i < added_tokens.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  fastdeploy::FDINFO << oss.str() << std::endl;
  tokenizer->AddTokens(added_tokens);
  return true;
}

struct Predictor {
  fastdeploy::Runtime runtime_;
  ErnieFastTokenizer tokenizer_;
  std::unordered_map<int, std::string> slot_labels_;
  std::unordered_map<int, std::string> intent_labels_;
  bool runtime_initialed_ = false;

  Predictor(const fastdeploy::RuntimeOption& option,
            const ErnieFastTokenizer& tokenizer,
            const std::unordered_map<int, std::string>& slot_labels,
            const std::unordered_map<int, std::string>& intent_labels)
      : tokenizer_(tokenizer), slot_labels_(slot_labels),
        intent_labels_(intent_labels) {
    runtime_initialed_ = runtime_.Init(option);
  }

  bool Initialized() {
    return runtime_initialed_;
  }

  bool Preprocess(const std::vector<std::string>& texts,
                  std::vector<fastdeploy::FDTensor>* inputs) {
    std::vector<fast_tokenizer::core::Encoding> encodings;
    // 1. Tokenize the text
    tokenizer_.EncodeBatchStrings(texts, &encodings);
    // 2. Construct the input vector tensor
    // 2.1 Allocate input tensor
    int64_t batch_size = texts.size();
    int64_t seq_len = 0;
    if (batch_size > 0) {
      seq_len = encodings[0].GetLen();
    }
    inputs->resize(runtime_.NumInputs());
    for (int i = 0; i < runtime_.NumInputs(); ++i) {
      (*inputs)[i].Allocate({batch_size, seq_len},
                            fastdeploy::FDDataType::INT32,
                            runtime_.GetInputInfo(i).name);
    }
    // 2.2 Set the value of data
    size_t start = 0;
    int* input_ids_ptr = reinterpret_cast<int*>((*inputs)[0].MutableData());
    for (int i = 0; i < encodings.size(); ++i) {
      auto&& curr_input_ids = encodings[i].GetIds();
      std::copy(curr_input_ids.begin(), curr_input_ids.end(),
                input_ids_ptr + start);
      start += seq_len;
    }
    return true;
  }

  bool IntentClsPostprocess(const fastdeploy::FDTensor& intent_logits,
                            std::vector<IntentDetAndSlotFillResult>* results) {
    fastdeploy::FDTensor probs;
    fastdeploy::function::Softmax(intent_logits, &probs);

    fastdeploy::FDTensor labels, confidences;
    fastdeploy::function::Max(probs, &confidences, {-1});
    fastdeploy::function::ArgMax(probs, &labels, -1);
    if (labels.Numel() != confidences.Numel()) {
      return false;
    }
    int64_t* label_ptr = reinterpret_cast<int64_t*>(labels.Data());
    float* confidence_ptr = reinterpret_cast<float*>(confidences.Data());
    for (int i = 0; i < labels.Numel(); ++i) {
      (*results)[i].intent_result.intent_label = intent_labels_[label_ptr[i]];
      (*results)[i].intent_result.intent_confidence = confidence_ptr[i];
    }
    return true;
  }

  bool SlotClsPostprocess(const fastdeploy::FDTensor& slot_logits,
                          const std::vector<std::string>& texts,
                          std::vector<IntentDetAndSlotFillResult>* results) {
    fastdeploy::FDTensor batch_preds;
    fastdeploy::function::ArgMax(slot_logits, &batch_preds, -1);
    for (int i = 0; i < results->size(); ++i) {
      fastdeploy::FDTensor preds;
      fastdeploy::function::Slice(batch_preds, {0}, {i}, &preds);
      int start = -1;
      std::string label_name = "";
      std::vector<IntentDetAndSlotFillResult::SlotFillResult> items;
      fast_tokenizer::pretokenizers::CharToBytesOffsetConverter convertor(
          texts[i]);
      fast_tokenizer::core::Offset curr_offset;
      int unicode_len = fast_tokenizer::utils::GetUnicodeLenFromUTF8(
          texts[i].data(), texts[i].length());
      int seq_len = preds.Shape()[0];
      int64_t* preds_ptr = reinterpret_cast<int64_t*>(preds.Data());
      for (int j = 0; j < seq_len; ++j) {
        fastdeploy::FDTensor pred;
        fastdeploy::function::Slice(preds, {0}, {j}, &pred);
        int64_t slot_label_id = preds_ptr[j];
        const std::string& curr_label = slot_labels_[slot_label_id];
        if ((curr_label == "O" || curr_label.find("B-") != std::string::npos ||
             (j - 1 >= unicode_len)) &&
            start >= 0) {
          // Convert the unicode character offset to byte offset.
          convertor.convert({start, j - 1}, &curr_offset);
          items.emplace_back(IntentDetAndSlotFillResult::SlotFillResult{
              label_name,
              texts[i].substr(curr_offset.first,
                              curr_offset.second - curr_offset.first),
              {start, j - 2}});
          start = -1;
          if (j - 1 >= unicode_len) {
            break;
          }
        }
        if (curr_label.find("B-") != std::string::npos) {
          start = j - 1;
          label_name = curr_label.substr(2);
        }
      }
      (*results)[i].slot_result = std::move(items);
    }
    return true;
  }

  bool Postprocess(const std::vector<fastdeploy::FDTensor>& outputs,
                   const std::vector<std::string>& texts,
                   std::vector<IntentDetAndSlotFillResult>* results) {
    const auto& intent_logits = outputs[0];
    const auto& slot_logits = outputs[1];
    return IntentClsPostprocess(intent_logits, results) &&
           SlotClsPostprocess(slot_logits, texts, results);
  }

  bool Predict(const std::vector<std::string>& texts,
               std::vector<IntentDetAndSlotFillResult>* results) {
    std::vector<fastdeploy::FDTensor> inputs;
    if (!Preprocess(texts, &inputs)) {
      return false;
    }

    std::vector<fastdeploy::FDTensor> outputs(runtime_.NumOutputs());

    auto t = ernie_tiny::jni::GetCurrentTime();
    runtime_.Infer(inputs, &outputs);
    LOGD("Runtime cost %f ms.", ernie_tiny::jni::GetElapsedTime(t));

    results->resize(texts.size());
    if (!Postprocess(outputs, texts, results)) {
      return false;
    }
    return true;
  }
};

static void ReadLabelMapFromTxt(
    const std::string path, std::unordered_map<int, std::string>* label_map) {
  std::fstream fin(path);
  int id = 0;
  std::string label;
  while (fin) {
    fin >> label;
    if (label.size() > 0) {
      label_map->insert({id++, label});
    } else {
      break;
    }
  }
}

static void ReadDatasetFromTxt(
    const std::string path, std::vector<std::string>* dataset) {
  std::fstream fin(path);
  std::string text;
  while (fin >> text) {
    if (text.length() > 0) {
      dataset->push_back(text);
    } else {
      break;
    }
  }
}

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_paddlenlp_ernie_1tiny_Predictor_bindNative(JNIEnv *env,
                                                                 jobject thiz,
                                                                 jstring model_file,
                                                                 jstring params_file,
                                                                 jstring vocab_file,
                                                                 jstring slot_labels_file,
                                                                 jstring intent_labels_file,
                                                                 jstring added_tokens_file,
                                                                 jobject runtime_option,
                                                                 jint max_length) {

  auto c_model_file = ernie_tiny::jni::ConvertTo<std::string>(env, model_file);
  auto c_params_file = ernie_tiny::jni::ConvertTo<std::string>(env, params_file);
  auto c_vocab_file = ernie_tiny::jni::ConvertTo<std::string>(env, vocab_file);
  auto c_slot_labels_file = ernie_tiny::jni::ConvertTo<std::string>(env, slot_labels_file);
  auto c_intent_labels_file = ernie_tiny::jni::ConvertTo<std::string>(env, intent_labels_file);
  auto c_added_tokens_file = ernie_tiny::jni::ConvertTo<std::string>(env, added_tokens_file);
  auto c_runtime_option = ernie_tiny::jni::NewCxxRuntimeOption(env, runtime_option);
  c_runtime_option.SetModelPath(c_model_file, c_params_file);

  uint32_t c_max_length = static_cast<uint32_t>(max_length);
  ErnieFastTokenizer c_tokenizer(c_vocab_file);
  AddTokensToTokenizer(c_added_tokens_file, &c_tokenizer);

  c_tokenizer.EnableTruncMethod(
      c_max_length,
      0,
      fast_tokenizer::core::Direction::RIGHT,
      fast_tokenizer::core::TruncStrategy::LONGEST_FIRST);
  c_tokenizer.EnablePadMethod(
      fast_tokenizer::core::Direction::RIGHT,
      0,
      0,
      "[PAD]",
      &c_max_length,
      nullptr);

  std::unordered_map<int, std::string> c_slot_label_map;
  std::unordered_map<int, std::string> c_intent_label_map;
  ReadLabelMapFromTxt(c_slot_labels_file, &c_slot_label_map);
  ReadLabelMapFromTxt(c_intent_labels_file, &c_intent_label_map);

  auto c_predictor_ptr = new Predictor(
      c_runtime_option, c_tokenizer, c_slot_label_map, c_intent_label_map);

  if (!c_predictor_ptr->Initialized()) {
    LOGE("Predictor initialize failed!");
    return 0;
  }

  return reinterpret_cast<jlong>(c_predictor_ptr);
}

JNIEXPORT jobjectArray JNICALL
Java_com_baidu_paddle_paddlenlp_ernie_1tiny_Predictor_predictNative(JNIEnv *env,
                                                                    jobject thiz,
                                                                    jlong cxx_context,
                                                                    jobjectArray texts) {
  if (cxx_context == 0) {
    return NULL;
  }
  auto c_predictor_ptr = reinterpret_cast<Predictor*>(cxx_context);
  auto c_texts = ernie_tiny::jni::ConvertTo<std::vector<std::string>>(env, texts);
  if (c_texts.empty()) {
    LOGE("c_texts is empty!");
    return NULL;
  }
  LOGD("c_texts: %s", TextsStr(c_texts).c_str());

  std::vector<IntentDetAndSlotFillResult> c_results;

  auto t = ernie_tiny::jni::GetCurrentTime();
  // TODO: may add BatchiFyTexts (not need now)
  if(c_predictor_ptr->Predict(c_texts, &c_results)) {
    LOGD("Predict cost %f ms.", ernie_tiny::jni::GetElapsedTime(t));

    // Show some log info to logcat
    for (int i = 0; i < c_results.size(); ++i) {
      std::string info = "No." + std::to_string(i) + " text = "
          + c_texts[i] + "\n" + ResultStr(c_results[i]);
      LOGD("%s", info.c_str());
    }

    // Assign C++ Results -> Java IntentDetAndSlotFillResult[]
    const jclass j_intent_slot_result_clazz = env->FindClass(
        "com/baidu/paddle/paddlenlp/ernie_tiny/IntentDetAndSlotFillResult");
    const jclass j_intent_result_clazz = env->FindClass(
        "com/baidu/paddle/paddlenlp/ernie_tiny/IntentDetAndSlotFillResult$IntentDetResult");
    const jclass j_slot_result_clazz = env->FindClass(
        "com/baidu/paddle/paddlenlp/ernie_tiny/IntentDetAndSlotFillResult$SlotFillResult");

    // jfieldID of IntentDetAndSlotFillResult
    const jfieldID j_intent_slot_str_id = env->GetFieldID(
        j_intent_slot_result_clazz, "mStr", "Ljava/lang/String;");
    const jfieldID j_intent_slot_initialized_id = env->GetFieldID(
        j_intent_slot_result_clazz, "mInitialized", "Z");
    const jfieldID j_intent_slot_intent_id = env->GetFieldID(
        j_intent_slot_result_clazz, "mIntentResult",
        "Lcom/baidu/paddle/paddlenlp/ernie_tiny/IntentDetAndSlotFillResult$IntentDetResult;");
    const jfieldID j_intent_slot_slot_id = env->GetFieldID(
        j_intent_slot_result_clazz, "mSlotResult",
        "[Lcom/baidu/paddle/paddlenlp/ernie_tiny/IntentDetAndSlotFillResult$SlotFillResult;");

    // jfieldID of IntentDetResult
    const jfieldID j_intent_label_id = env->GetFieldID(
        j_intent_result_clazz, "mIntentLabel", "Ljava/lang/String;");
    const jfieldID j_intent_conf_id = env->GetFieldID(
        j_intent_result_clazz, "mIntentConfidence", "F");

    // jfieldID of SlotFillResult
    const jfieldID j_slot_label_id = env->GetFieldID(
        j_slot_result_clazz, "mSlotLabel", "Ljava/lang/String;");
    const jfieldID j_slot_entity_id = env->GetFieldID(
        j_slot_result_clazz, "mEntity", "Ljava/lang/String;");
    const jfieldID j_slot_pos_id = env->GetFieldID(
        j_slot_result_clazz, "mPos", "[I");

    // jmethodID of constructor method
    const jmethodID j_intent_slot_init_method = env->GetMethodID(
        j_intent_slot_result_clazz, "<init>", "()V");
    const jmethodID j_intent_init_method = env->GetMethodID(
        j_intent_result_clazz, "<init>", "()V");
    const jmethodID j_slot_init_method = env->GetMethodID(
        j_slot_result_clazz, "<init>", "()V");

    // IntentDetAndSlotFillResult[]
    const int c_results_size = c_results.size();
    jobjectArray j_intent_slot_results_arr = env->NewObjectArray(
        c_results_size, j_intent_slot_result_clazz, NULL);

    for (int i = 0; i < c_results_size; ++i) {
      const auto& curr_c_result = c_results.at(i);

      // IntentDetAndSlotFillResult
      jobject j_intent_slot_result_obj = env->NewObject(
          j_intent_slot_result_clazz, j_intent_slot_init_method);

      // 1. Set mStr String
      env->SetObjectField(j_intent_slot_result_obj, j_intent_slot_str_id,
                          ernie_tiny::jni::ConvertTo<jstring>(
                              env,ResultStr(curr_c_result)));

      // 2. Set mInitialized boolean
      env->SetBooleanField(j_intent_slot_result_obj, j_intent_slot_initialized_id,
                           JNI_TRUE);

      // 3. Set mIntentResult IntentDetResult
      jobject j_intent_result_obj = env->NewObject(
          j_intent_result_clazz, j_intent_init_method);
      const auto& curr_c_intent_result = curr_c_result.intent_result;
      // 3.1 Set mIntentLabel String
      env->SetObjectField(j_intent_result_obj, j_intent_label_id,
                          ernie_tiny::jni::ConvertTo<jstring>(
                              env, curr_c_intent_result.intent_label));
      // 3.2 Set mIntentConfidence float
      env->SetFloatField(j_intent_result_obj, j_intent_conf_id,
                         static_cast<jfloat>(curr_c_intent_result.intent_confidence));

      // 3.3 Set mIntentResult -> IntentDetAndSlotFillResult
      env->SetObjectField(j_intent_slot_result_obj, j_intent_slot_intent_id,
                          j_intent_result_obj);

      env->DeleteLocalRef(j_intent_result_obj);

      // 4. Set mSlotResult SlotFillResult[]
      const int curr_c_slot_size = curr_c_result.slot_result.size();
      jobjectArray j_slot_result_obj = env->NewObjectArray(
          curr_c_slot_size, j_slot_result_clazz, NULL);
      for (int j = 0; j < curr_c_slot_size; ++j) {

        jobject j_curr_sub_slot_obj = env->NewObject(j_slot_result_clazz, j_slot_init_method);
        const auto& curr_c_sub_slot_result = curr_c_result.slot_result.at(j);

        // 4.1 Set mSlotLabel String
        env->SetObjectField(j_curr_sub_slot_obj, j_slot_label_id,
                            ernie_tiny::jni::ConvertTo<jstring>(
                                env, curr_c_sub_slot_result.slot_label));

        // 4.2 Set mEntity String
        env->SetObjectField(j_curr_sub_slot_obj, j_slot_entity_id,
                            ernie_tiny::jni::ConvertTo<jstring>(
                                env, curr_c_sub_slot_result.entity));

        // 4.3 Set mPos int[2]
        std::vector<int> curr_c_sub_slot_pos = {
            curr_c_sub_slot_result.pos.first, curr_c_sub_slot_result.pos.second
        };
        jintArray j_curr_sub_slot_pos_arr = env->NewIntArray(2);
        env->SetIntArrayRegion(j_curr_sub_slot_pos_arr, 0, 2,
                               curr_c_sub_slot_pos.data());
        env->SetObjectField(j_curr_sub_slot_obj, j_slot_pos_id,
                            j_curr_sub_slot_pos_arr);

        // 4.4 Set j-th value of SlotFillResult[]
        env->SetObjectArrayElement(j_slot_result_obj, j, j_curr_sub_slot_obj);

        env->DeleteLocalRef(j_curr_sub_slot_pos_arr);
        env->DeleteLocalRef(j_curr_sub_slot_obj);
      }
      // 4.5 Set mSlotResult -> IntentDetAndSlotFillResult
      env->SetObjectField(j_intent_slot_result_obj, j_intent_slot_slot_id,
                          j_slot_result_obj);

      env->DeleteLocalRef(j_slot_result_obj);

      // 5. Set i-th value of IntentDetAndSlotFillResult[]
      env->SetObjectArrayElement(j_intent_slot_results_arr, i, j_intent_slot_result_obj);

      env->DeleteLocalRef(j_intent_slot_result_obj);
    }
    return j_intent_slot_results_arr;
  }

  return NULL;
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_paddlenlp_ernie_1tiny_Predictor_releaseNative(JNIEnv *env,
                                                                    jobject thiz,
                                                                    jlong cxx_context) {
  if (cxx_context == 0) {
    return JNI_FALSE;
  }
  auto c_predictor_ptr = reinterpret_cast<Predictor*>(cxx_context);

  delete c_predictor_ptr;
  LOGD("[End] Release ERNIE Tiny Predictor in native !");
  return JNI_TRUE;
}

#ifdef __cplusplus
}
#endif