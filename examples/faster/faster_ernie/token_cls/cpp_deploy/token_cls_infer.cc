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

#include <codecvt>
#include <iostream>
#include <locale>
#include <numeric>
#include "paddle/include/paddle_inference_api.h"

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_bool(use_gpu, true, "enable gpu");

template <typename T>
void GetOutput(paddle_infer::Predictor* predictor,
               std::string output_name,
               std::vector<T>* out_data,
               int* max_seq_len) {
  auto output = predictor->GetOutputHandle(output_name);
  std::vector<int> output_shape = output->shape();
  *max_seq_len = output_shape[1];
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
      "本能地感知到民主与其切身利益的关系，因而对投票选举表现出极大的热情和认真"
      "。",
      "记票板上出现了左金文的名字，而且票数扶摇直上，远远超过另外两名候选人。",
      "片中，人们看到在民主选举之前，村民的心态是漠然的。"};
  auto input_name = predictor->GetInputNames()[0];
  auto text = predictor->GetInputHandle(input_name);
  text->ReshapeStrings(data.size());
  text->CopyStringsFromCpu(&data);
  predictor->Run();

  std::vector<float> logits;
  std::vector<int64_t> preds;
  int max_seq_len = 0;
  auto output_names = predictor->GetOutputNames();
  GetOutput(predictor.get(), output_names[0], &logits, &max_seq_len);
  GetOutput(predictor.get(), output_names[1], &preds, &max_seq_len);
  std::string label_map[] = {
      "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"};
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  for (size_t i = 0; i < data.size(); i++) {
    std::wstring wdata = converter.from_bytes(data[i]);
    size_t seq_len = wdata.size();
    // +1 for the concated CLS token
    size_t start = i * max_seq_len + 1;
    size_t end = start + seq_len;
    std::cout << "Text: " << data[i] << "\tLabel: ";
    for (size_t j = start; j < end; j++) {
      std::cout << label_map[preds[j]] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}