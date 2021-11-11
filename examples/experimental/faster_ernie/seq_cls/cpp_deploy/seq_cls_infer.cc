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

#include <iostream>
#include <numeric>

#include "paddle/include/paddle_inference_api.h"

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_bool(use_gpu, true, "enable gpu");

template <typename T>
void GetOutput(paddle_infer::Predictor* predictor,
               std::string output_name,
               std::vector<T>* out_data) {
  auto output = predictor->GetOutputHandle(output_name);
  std::vector<int> output_shape = output->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  out_data->resize(out_num);
  output->CopyToCpu(out_data->data());
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  paddle_infer::Config config;
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  if (FLAGS_use_gpu) {
    config.EnableUseGpu(100, 0);
  }
  auto pass_builder = config.pass_builder();
  // TODO(Steffy-zxf): delete embedding_eltwise_layernorm_fuse_pass to avoid IR
  // optimization.
  pass_builder->DeletePass("embedding_eltwise_layernorm_fuse_pass");
  auto predictor = paddle_infer::CreatePredictor(config);

  std::vector<std::string> data{
      "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般",
      "请问：有些字打错了，我怎么样才可以回去编辑一下啊？",
      "本次入住酒店的网络不是很稳定，断断续续，希望能够改进。"};
  auto input_names = predictor->GetInputNames();
  auto text = predictor->GetInputHandle(input_names[0]);
  text->ReshapeStrings(data.size());
  text->CopyStringsFromCpu(&data);
  predictor->Run();

  std::vector<float> logits;
  std::vector<int64_t> preds;
  auto output_names = predictor->GetOutputNames();
  GetOutput(predictor.get(), output_names[0], &logits);
  GetOutput(predictor.get(), output_names[1], &preds);
  for (size_t i = 0; i < data.size(); i++) {
    std::cout << data[i] << " : " << preds[i] << std::endl;
  }
  return 0;
}