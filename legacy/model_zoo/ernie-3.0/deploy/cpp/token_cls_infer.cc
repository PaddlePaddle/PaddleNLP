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
#include <array>
#include <iostream>
#include <sstream>
#include <vector>

#include "fast_tokenizer/pretokenizers/pretokenizer.h"
#include "fast_tokenizer/tokenizers/ernie_fast_tokenizer.h"
#include "fastdeploy/function/functions.h"
#include "fastdeploy/runtime.h"
#include "fastdeploy/utils/path.h"
#include "gflags/gflags.h"

using namespace paddlenlp;
using namespace fast_tokenizer::tokenizers_impl;
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_string(vocab_path, "", "Path of the vocab file.");
DEFINE_string(model_prefix, "model", "The model and params file prefix.");
DEFINE_string(device,
              "cpu",
              "Type of inference device, support 'cpu' or 'gpu'.");
DEFINE_string(backend,
              "paddle",
              "The inference runtime backend, support: ['onnx_runtime', "
              "'paddle', 'openvino', 'tensorrt', 'paddle_tensorrt']");
DEFINE_int32(batch_size, 1, "The batch size of data.");
DEFINE_int32(max_length, 128, "The batch size of data.");
DEFINE_bool(use_fp16, false, "Wheter to use FP16 mode.");

void PrintUsage() {
  fastdeploy::FDINFO
      << "Usage: seq_cls_infer_demo --model_dir dir --device [cpu|gpu] "
         "--backend "
         "[onnx_runtime|paddle|openvino|tensorrt|paddle_tensorrt] "
         "--batch_size size --max_length len --use_fp16 false"
      << std::endl;
  fastdeploy::FDINFO << "Default value of device: cpu" << std::endl;
  fastdeploy::FDINFO << "Default value of backend: onnx_runtime" << std::endl;
  fastdeploy::FDINFO << "Default value of batch_size: 1" << std::endl;
  fastdeploy::FDINFO << "Default value of max_length: 128" << std::endl;
  fastdeploy::FDINFO << "Default value of use_fp16: false" << std::endl;
}

bool CreateRuntimeOption(fastdeploy::RuntimeOption* option) {
  std::string model_path =
      FLAGS_model_dir + sep + FLAGS_model_prefix + ".pdmodel";
  std::string param_path =
      FLAGS_model_dir + sep + FLAGS_model_prefix + ".pdiparams";
  fastdeploy::FDINFO << "model_path = " << model_path
                     << ", param_path = " << param_path << std::endl;
  option->SetModelPath(model_path, param_path);

  if (FLAGS_device == "gpu") {
    option->UseGpu();
  } else if (FLAGS_device == "cpu") {
    option->UseCpu();
  } else if (FLAGS_device == "kunlunxin") {
    option->UseKunlunXin();
    return true;
  } else {
    fastdeploy::FDERROR << "The avilable device should be one of the list "
                           "['cpu', 'gpu', 'kunlunxin']. But receive '"
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
    option->SetTrtInputShape("token_type_ids",
                             {1, 1},
                             {FLAGS_batch_size, FLAGS_max_length},
                             {FLAGS_batch_size, FLAGS_max_length});
    if (FLAGS_use_fp16) {
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

  return true;
}

bool BatchFyTexts(const std::vector<std::string>& texts,
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

struct TokenClsResult {
  struct TokenResult {
    std::string token_label;
    std::string entity;
    std::pair<int, int> pos;
    friend std::ostream& operator<<(std::ostream& os,
                                    const TokenResult& result);
  };
  std::vector<TokenResult> token_results;
};

std::ostream& operator<<(std::ostream& os,
                         const typename TokenClsResult::TokenResult& result) {
  os << "entity: " << result.entity << ", label: " << result.token_label
     << ", pos: [" << result.pos.first << ", " << result.pos.second << "]";
  return os;
}

void PrintResult(const std::vector<TokenClsResult>& token_cls_results,
                 const std::vector<std::string>& data) {
  for (int i = 0; i < data.size(); ++i) {
    std::cout << "input data: " << data[i] << std::endl;
    std::cout << "The model detects all entities:" << std::endl;
    auto& curr_results = token_cls_results[i];
    for (auto& token_result : curr_results.token_results) {
      std::cout << token_result << std::endl;
    }
    std::cout << "-----------------------------" << std::endl;
  }
}

struct ErnieForTokenClassificationPredictor {
  fastdeploy::Runtime runtime_;
  ErnieFastTokenizer tokenizer_;
  std::vector<std::string> label_list_;

  ErnieForTokenClassificationPredictor(
      const fastdeploy::RuntimeOption& option,
      const ErnieFastTokenizer& tokenizer,
      const std::vector<std::string>& label_list)
      : tokenizer_(tokenizer), label_list_(label_list) {
    runtime_.Init(option);
  }

  bool Preprocess(const std::vector<std::string>& texts,
                  std::vector<fastdeploy::FDTensor>* inputs) {
    std::vector<fast_tokenizer::core::Encoding> encodings;
    std::vector<fast_tokenizer::core::EncodeInput> text_pair_input;
    // 1. Tokenize the text or (text, text_pair)
    for (int i = 0; i < texts.size(); ++i) {
      text_pair_input.emplace_back(texts[i]);
    }
    tokenizer_.EncodeBatchStrings(text_pair_input, &encodings);
    // 2. Construct the input vector tensor
    // 2.1 Allocate input tensor
    int64_t batch_size = texts.size();
    int64_t seq_len = 0;
    if (batch_size > 0) {
      seq_len = encodings[0].GetIds().size();
    }
    inputs->resize(runtime_.NumInputs());
    for (int i = 0; i < runtime_.NumInputs(); ++i) {
      (*inputs)[i].Allocate({batch_size, seq_len},
                            fastdeploy::FDDataType::INT64,
                            runtime_.GetInputInfo(i).name);
    }
    // 2.2 Set the value of data
    size_t start = 0;
    int64_t* input_ids_ptr =
        reinterpret_cast<int64_t*>((*inputs)[0].MutableData());
    int64_t* type_ids_ptr =
        reinterpret_cast<int64_t*>((*inputs)[1].MutableData());
    for (int i = 0; i < encodings.size(); ++i) {
      auto&& curr_input_ids = encodings[i].GetIds();
      auto&& curr_type_ids = encodings[i].GetTypeIds();
      std::copy(
          curr_input_ids.begin(), curr_input_ids.end(), input_ids_ptr + start);
      std::copy(
          curr_type_ids.begin(), curr_type_ids.end(), type_ids_ptr + start);
      start += seq_len;
    }
    return true;
  }

  bool Postprocess(const std::vector<fastdeploy::FDTensor>& outputs,
                   const std::vector<std::string>& texts,
                   std::vector<TokenClsResult>* results) {
    fastdeploy::FDTensor batch_preds;
    auto& logits = outputs[0];
    fastdeploy::function::ArgMax(logits, &batch_preds, -1);
    for (int i = 0; i < results->size(); ++i) {
      fastdeploy::FDTensor preds;
      fastdeploy::function::Slice(batch_preds, {0}, {i}, &preds);
      int start = -1;
      std::string label_name = "";
      std::vector<typename TokenClsResult::TokenResult> items;

      int seq_len = preds.Shape()[0];

      fast_tokenizer::pretokenizers::CharToBytesOffsetConverter convertor(
          texts[i]);
      fast_tokenizer::core::Offset curr_offset;
      for (int j = 0; j < seq_len; ++j) {
        int64_t label_id = (reinterpret_cast<int64_t*>(preds.Data()))[j];
        const std::string& curr_label = label_list_[label_id];
        if ((curr_label == "O" || curr_label.find("B-") != std::string::npos) &&
            start >= 0) {
          // Convert the unicode character offset to byte offset.
          convertor.convert({start, j - 1}, &curr_offset);
          if (curr_offset.first >= texts[i].length()) {
            break;
          }
          items.emplace_back(typename TokenClsResult::TokenResult{
              label_name,
              texts[i].substr(curr_offset.first,
                              curr_offset.second - curr_offset.first),
              {start, j - 2}});
          start = -1;
        }
        if (curr_label.find("B-") != std::string::npos) {
          start = j - 1;
          label_name = curr_label.substr(2);
        }
      }
      (*results)[i].token_results = std::move(items);
    }
    return true;
  }
  bool Predict(const std::vector<std::string>& texts,
               std::vector<TokenClsResult>* results) {
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

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option)) {
    PrintUsage();
    return -1;
  }

  std::string vocab_path = FLAGS_vocab_path;
  if (!fastdeploy::CheckFileExists(vocab_path)) {
    vocab_path = fastdeploy::PathJoin(FLAGS_model_dir, "vocab.txt");
    if (!fastdeploy::CheckFileExists(vocab_path)) {
      fastdeploy::FDERROR << "The path of vocab " << vocab_path
                          << " doesn't exist" << std::endl;
      PrintUsage();
      return -1;
    }
  }
  uint32_t max_length = FLAGS_max_length;
  ErnieFastTokenizer tokenizer(vocab_path);
  tokenizer.EnableTruncMethod(
      max_length,
      0,
      fast_tokenizer::core::Direction::RIGHT,
      fast_tokenizer::core::TruncStrategy::LONGEST_FIRST);

  std::vector<std::string> label_list = {
      "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"};
  ErnieForTokenClassificationPredictor predictor(option, tokenizer, label_list);
  std::vector<TokenClsResult> token_cls_results;
  std::vector<std::string> texts_ds = {
      "北京的涮肉，重庆的火锅，成都的小吃都是极具特色的美食。",
      "乔丹、科比、詹姆斯和姚明都是篮球界的标志性人物。"};
  std::vector<std::vector<std::string>> batch_texts;
  BatchFyTexts(texts_ds, FLAGS_batch_size, &batch_texts);
  for (int bs = 0; bs < batch_texts.size(); ++bs) {
    predictor.Predict(batch_texts[bs], &token_cls_results);
    PrintResult(token_cls_results, batch_texts[bs]);
  }
  return 0;
}