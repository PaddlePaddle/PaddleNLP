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

using namespace paddlenlp;
using namespace fast_tokenizer::tokenizers_impl;

static bool BatchFyTexts(const std::vector<std::string>& texts, int batch_size,
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

static std::string Str(const IntentDetAndSlotFillResult& result) {
  std::ostringstream oss;
  oss << result;
  return oss.str();
}

struct Predictor {
  fastdeploy::Runtime runtime_;
  ErnieFastTokenizer tokenizer_;
  std::unordered_map<int, std::string> slot_labels_;
  std::unordered_map<int, std::string> intent_labels_;

  Predictor(const fastdeploy::RuntimeOption& option,
            const ErnieFastTokenizer& tokenizer,
            const std::unordered_map<int, std::string>& slot_labels,
            const std::unordered_map<int, std::string>& intent_labels)
      : tokenizer_(tokenizer), slot_labels_(slot_labels),
        intent_labels_(intent_labels) {
    runtime_.Init(option);
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
    runtime_.Infer(inputs, &outputs);
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
                                                                 jobject runtime_option) {

  return 0;
}

JNIEXPORT jobjectArray JNICALL
Java_com_baidu_paddle_paddlenlp_ernie_1tiny_Predictor_predictNative(JNIEnv *env,
                                                                    jobject thiz,
                                                                    jlong cxx_context,
                                                                    jobjectArray texts) {
  return NULL;
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_paddlenlp_ernie_1tiny_Predictor_releaseNative(JNIEnv *env,
                                                                    jobject thiz,
                                                                    jlong cxx_context) {
  return JNI_FALSE;
}

#ifdef __cplusplus
}
#endif
























