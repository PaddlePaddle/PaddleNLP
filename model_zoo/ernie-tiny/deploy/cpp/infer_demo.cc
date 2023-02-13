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

#include <iostream>
#include <sstream>
#include <vector>

#include "fast_tokenizer/pretokenizers/pretokenizer.h"
#include "fast_tokenizer/tokenizers/ernie_fast_tokenizer.h"
#include "fast_tokenizer/utils/utf8.h"
#include "fastdeploy/function/functions.h"
#include "fastdeploy/runtime.h"
#include "fastdeploy/utils/path.h"
#include "gflags/gflags.h"
#include "nlohmann/json.hpp"

using namespace paddlenlp;
using namespace fast_tokenizer::tokenizers_impl;
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_string(model_prefix, "infer_model", "The model and params file prefix");
DEFINE_string(vocab_path, "", "Path of the vocab file.");
DEFINE_string(added_tokens_path, "", "Path of the added tokens json file.");
DEFINE_string(slot_label_path, "", "Path of the slot label file.");
DEFINE_string(intent_label_path, "", "Path of the intent label file.");
DEFINE_string(device,
              "cpu",
              "Type of inference device, support 'cpu' or 'gpu'.");
DEFINE_string(backend,
              "paddle",
              "The inference runtime backend, support: ['onnx_runtime', "
              "'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']");
DEFINE_int32(batch_size, 1, "The batch size of data.");
DEFINE_int32(max_length, 16, "The batch size of data.");
DEFINE_int32(cpu_num_threads,
             1,
             "The number of threads when inferring on cpu.");
DEFINE_bool(use_trt_fp16, false, "Wheter to use FP16 mode.");

void PrintUsage() {
  fastdeploy::FDINFO
      << "Usage: infer_demo --model_dir dir --device [cpu|gpu] "
         "--backend "
         "[onnx_runtime|paddle|openvino|tensorrt|paddle_tensorrt] "
         "--batch_size size --max_length len --use_trt_fp16 false"
      << std::endl;
  fastdeploy::FDINFO << "Default value of device: cpu" << std::endl;
  fastdeploy::FDINFO << "Default value of backend: onnx_runtime" << std::endl;
  fastdeploy::FDINFO << "Default value of batch_size: 1" << std::endl;
  fastdeploy::FDINFO << "Default value of max_length: 128" << std::endl;
  fastdeploy::FDINFO << "Default value of use_trt_fp16: false" << std::endl;
}

bool CreateRuntimeOption(fastdeploy::RuntimeOption* option) {
  if (FLAGS_device == "gpu") {
    option->UseGpu();
  } else if (FLAGS_device == "cpu") {
    option->UseCpu();
    option->SetCpuThreadNum(FLAGS_cpu_num_threads);
  } else {
    fastdeploy::FDERROR << "The avilable device should be one of the list "
                           "['cpu', 'gpu']. But receive '"
                        << FLAGS_device << "'" << std::endl;
    return false;
  }

  if (FLAGS_backend == "onnx_runtime") {
    option->UseOrtBackend();
  } else if (FLAGS_backend == "paddle") {
    option->UsePaddleInferBackend();
  } else if (FLAGS_backend == "openvino") {
    option->UseOpenVINOBackend();
  } else if (FLAGS_backend == "tensorrt" ||
             FLAGS_backend == "paddle_tensorrt") {
    option->UseTrtBackend();
    if (FLAGS_backend == "paddle_tensorrt") {
      option->EnablePaddleToTrt();
      option->EnablePaddleTrtCollectShape();
    }
    std::string trt_file = FLAGS_model_dir + sep + "infer.trt";
    option->SetTrtInputShape("input_ids",
                             {1, 1},
                             {FLAGS_batch_size, FLAGS_max_length},
                             {FLAGS_batch_size, FLAGS_max_length});
    if (FLAGS_use_trt_fp16) {
      option->EnableTrtFP16();
      trt_file = trt_file + ".fp16";
    }
  } else {
    fastdeploy::FDERROR << "The avilable backend should be one of the list "
                           "['paddle', 'openvino', 'tensorrt', "
                           "'paddle_tensorrt']. But receive '"
                        << FLAGS_backend << "'" << std::endl;
    return false;
  }

  std::string model_path =
      FLAGS_model_dir + sep + FLAGS_model_prefix + ".pdmodel";
  std::string param_path =
      FLAGS_model_dir + sep + FLAGS_model_prefix + ".pdiparams";
  fastdeploy::FDINFO << "model_path = " << model_path
                     << ", param_path = " << param_path << std::endl;
  option->SetModelPath(model_path, param_path);
  return true;
}

bool BatchiFyTexts(const std::vector<std::string>& texts,
                   int batch_size,
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

bool AddTokensToTokenizer(const std::string& path,
                          ErnieFastTokenizer* tokenizer) {
  std::string actual_path = path;
  if (!fastdeploy::CheckFileExists(actual_path)) {
    actual_path = fastdeploy::PathJoin(FLAGS_model_dir, "added_tokens.json");
    if (!fastdeploy::CheckFileExists(actual_path)) {
      fastdeploy::FDINFO << "No added_tokens have been added to tokenizer."
                         << std::endl;
      return false;
    }
  }
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

  Predictor(const fastdeploy::RuntimeOption& option,
            const ErnieFastTokenizer& tokenizer,
            const std::unordered_map<int, std::string>& slot_labels,
            const std::unordered_map<int, std::string>& intent_labels)
      : tokenizer_(tokenizer),
        slot_labels_(slot_labels),
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
      std::copy(
          curr_input_ids.begin(), curr_input_ids.end(), input_ids_ptr + start);
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

void ReadLabelMapFromTxt(const std::string path,
                         std::unordered_map<int, std::string>* label_map) {
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

bool GetFilePath(const std::string& path,
                 const std::string& default_path,
                 std::string* actual_path) {
  *actual_path = path;
  if (!fastdeploy::CheckFileExists(path)) {
    *actual_path = fastdeploy::PathJoin(FLAGS_model_dir, default_path);
    if (!fastdeploy::CheckFileExists(*actual_path)) {
      fastdeploy::FDERROR << "The path `" << *actual_path << "` doesn't exist"
                          << std::endl;
      PrintUsage();
      return false;
    }
  }
  return true;
}


int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option)) {
    PrintUsage();
    return -1;
  }

  std::string vocab_path;
  if (!GetFilePath(FLAGS_vocab_path, "vocab.txt", &vocab_path)) {
    return -1;
  }
  ErnieFastTokenizer tokenizer(vocab_path);
  AddTokensToTokenizer(FLAGS_added_tokens_path, &tokenizer);

  uint32_t max_length = FLAGS_max_length;
  tokenizer.EnableTruncMethod(
      max_length,
      0,
      fast_tokenizer::core::Direction::RIGHT,
      fast_tokenizer::core::TruncStrategy::LONGEST_FIRST);
  tokenizer.EnablePadMethod(fast_tokenizer::core::Direction::RIGHT,
                            0,
                            0,
                            "[PAD]",
                            &max_length,
                            nullptr);
  std::unordered_map<int, std::string> slot_label_map;
  std::unordered_map<int, std::string> intent_label_map;

  std::string slot_label_path;
  if (!GetFilePath(
          FLAGS_slot_label_path, "slots_label.txt", &slot_label_path)) {
    return -1;
  }
  ReadLabelMapFromTxt(slot_label_path, &slot_label_map);

  std::string intent_label_path;
  if (!GetFilePath(
          FLAGS_intent_label_path, "intent_label.txt", &intent_label_path)) {
    return -1;
  }
  ReadLabelMapFromTxt(intent_label_path, &intent_label_map);

  Predictor predictor(option, tokenizer, slot_label_map, intent_label_map);

  std::vector<IntentDetAndSlotFillResult> results;
  std::vector<std::vector<std::string>> batch_texts;
  std::vector<std::string> texts = {
      "来一首周华健的花心", "播放我们都一样", "到信阳市汽车配件城"};
  BatchiFyTexts(texts, FLAGS_batch_size, &batch_texts);
  for (int i = 0; i < batch_texts.size(); ++i) {
    auto& curr_texts = batch_texts[i];
    predictor.Predict(curr_texts, &results);
    for (int k = 0; k < curr_texts.size(); ++k) {
      std::cout << "No." << i * FLAGS_batch_size + k
                << " text = " << curr_texts[k] << std::endl;
      std::cout << results[k] << std::endl;
    }
    results.clear();
  }


  return 0;
}