/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Directory of the inference model.");
DEFINE_bool(use_gpu, true, "enable gpu");

std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  if (FLAGS_use_gpu) {
    config.EnableUseGpu(100, 0);
  }
  return CreatePredictor(config);
}

template <typename T>
void GetOutput(Predictor* predictor,
               std::string output_name,
               std::vector<T>* out_data) {
  auto output = predictor->GetOutputHandle(output_name);
  std::vector<int> output_shape = output->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  out_data->resize(out_num);
  output->CopyToCpu(out_data->data());
}

void Run(Predictor* predictor,
         const std::vector<std::string>& input_data,
         std::vector<float>* logits,
         std::vector<int64_t>* predictions) {
  auto input_names = predictor->GetInputNames();

  auto text = predictor->GetInputHandle(input_names[0]);
  text->ReshapeStrings(input_data.size());
  text->CopyStringsFromCpu(&input_data);

  CHECK(predictor->Run());

  auto output_names = predictor->GetOutputNames();
  GetOutput(predictor, output_names[0], logits);
  GetOutput(predictor, output_names[1], predictions);
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();

  std::vector<std::string> data{
      "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般",
      "怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的"
      "动画片",
      "作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上"
      "办理入住手续，节省时间。"};
  std::unordered_map<std::size_t, std::string> label_map = {{0, "negative"},
                                                            {1, "positive"}};
  for (size_t i = 0; i < data.size(); i += FLAGS_batch_size) {
    std::vector<std::string> batch(FLAGS_batch_size);
    batch.assign(data.begin() + i, data.begin() + i + FLAGS_batch_size);
    std::vector<float> logits;
    std::vector<int64_t> predictions;
    Run(predictor.get(), batch, &logits, &predictions);
    for (size_t j = 0; j < FLAGS_batch_size; j++) {
      LOG(INFO) << "The text is " << batch[j] << "; The predition label is "
                << label_map[predictions[j]];
    }
  }

  return 0;
}