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

#include "fast_tokenizer/tokenizers/ernie_fast_tokenizer.h"
#include "fastdeploy/function/reduce.h"
#include "fastdeploy/function/softmax.h"
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
              "Type of inference device, support 'cpu', 'kunlunxin' or 'gpu'.");
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
  if (FLAGS_device == "kunlunxin") {
    option->UseKunlunXin();
    return true;
  } else if (FLAGS_device == "gpu") {
    option->UseGpu();
  } else if (FLAGS_device == "cpu") {
    option->UseCpu();
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

struct SeqClsResult {
  int label;
  float confidence;
};

struct ErnieForSequenceClassificationPredictor {
  fastdeploy::Runtime runtime_;
  ErnieFastTokenizer tokenizer_;
  ErnieForSequenceClassificationPredictor(
      const fastdeploy::RuntimeOption& option,
      const ErnieFastTokenizer& tokenizer)
      : tokenizer_(tokenizer) {
    runtime_.Init(option);
  }

  bool Preprocess(const std::vector<std::string>& texts,
                  const std::vector<std::string>& texts_pair,
                  std::vector<fastdeploy::FDTensor>* inputs) {
    std::vector<fast_tokenizer::core::Encoding> encodings;
    std::vector<fast_tokenizer::core::EncodeInput> text_pair_input;
    // 1. Tokenize the text or (text, text_pair)
    if (texts_pair.empty()) {
      for (int i = 0; i < texts.size(); ++i) {
        text_pair_input.emplace_back(texts[i]);
      }
    } else {
      if (texts.size() != texts_pair.size()) {
        return false;
      }
      for (int i = 0; i < texts.size(); ++i) {
        text_pair_input.emplace_back(
            std::pair<std::string, std::string>(texts[i], texts_pair[i]));
      }
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
                   std::vector<SeqClsResult>* seq_cls_results) {
    const auto& logits = outputs[0];
    fastdeploy::FDTensor probs;
    fastdeploy::function::Softmax(logits, &probs);

    fastdeploy::FDTensor labels, confidences;
    fastdeploy::function::Max(probs, &confidences, {-1});
    fastdeploy::function::ArgMax(probs, &labels, -1);
    if (labels.Numel() != confidences.Numel()) {
      return false;
    }

    seq_cls_results->resize(labels.Numel());
    int64_t* label_ptr = reinterpret_cast<int64_t*>(labels.Data());
    float* confidence_ptr = reinterpret_cast<float*>(confidences.Data());
    for (int i = 0; i < labels.Numel(); ++i) {
      (*seq_cls_results)[i].label = label_ptr[i];
      (*seq_cls_results)[i].confidence = confidence_ptr[i];
    }
    return true;
  }

  bool Predict(const std::vector<std::string>& texts,
               const std::vector<std::string>& texts_pair,
               std::vector<SeqClsResult>* seq_cls_results) {
    std::vector<fastdeploy::FDTensor> inputs;
    if (!Preprocess(texts, texts_pair, &inputs)) {
      return false;
    }

    std::vector<fastdeploy::FDTensor> outputs(runtime_.NumOutputs());
    runtime_.Infer(inputs, &outputs);

    if (!Postprocess(outputs, seq_cls_results)) {
      return false;
    }
    return true;
  }
};

void PrintResult(const std::vector<SeqClsResult>& seq_cls_results,
                 const std::vector<std::string>& data,
                 const std::vector<std::string>& data_pair) {
  static std::vector<std::string> label_list{"Similar", "Not similar"};
  for (int i = 0; i < data.size(); ++i) {
    std::cout << "input data: " << data[i] << ", " << data_pair[i] << std::endl;
    std::cout << "seq cls result: " << std::endl;
    std::cout << "label: " << label_list[seq_cls_results[i].label]
              << " confidence: " << seq_cls_results[i].confidence << std::endl;
    std::cout << "-----------------------------" << std::endl;
  }
}

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
  ErnieFastTokenizer tokenizer(vocab_path);
  tokenizer.EnableTruncMethod(
      FLAGS_max_length,
      0,
      fast_tokenizer::core::Direction::RIGHT,
      fast_tokenizer::core::TruncStrategy::LONGEST_FIRST);

  ErnieForSequenceClassificationPredictor predictor(option, tokenizer);

  std::vector<SeqClsResult> seq_cls_results;
  std::vector<std::string> texts_ds = {"花呗收款额度限制",
                                       "花呗支持高铁票支付吗"};
  std::vector<std::string> texts_pair_ds = {"收钱码，对花呗支付的金额有限制吗",
                                            "为什么友付宝不支持花呗付款"};
  std::vector<std::vector<std::string>> batch_texts, batch_texts_pair;
  BatchFyTexts(texts_ds, FLAGS_batch_size, &batch_texts);
  BatchFyTexts(texts_pair_ds, FLAGS_batch_size, &batch_texts_pair);
  for (int bs = 0; bs < batch_texts.size(); ++bs) {
    predictor.Predict(batch_texts[bs], batch_texts_pair[bs], &seq_cls_results);
    PrintResult(seq_cls_results, batch_texts[bs], batch_texts_pair[bs]);
  }
  return 0;
}
