

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <numeric>

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

void Run(Predictor* predictor,
         std::vector<std::string>* input_data,
         std::vector<float>* probs) {
  auto input_names = predictor->GetInputNames();

  auto text = predictor->GetInputHandle(input_names[0]);
  text->ReshapeStrings(input_data->size());
  text->CopyStringsFromCpu(input_data);

  predictor->Run();

  auto output_names = predictor->GetOutputNames();
  auto logits = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = logits->shape();
  int logits_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  probs->resize(logits_num);
  logits->CopyToCpu(probs->data());
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

  std::vector<float> probs;
  Run(predictor.get(), &data, &probs);

  return 0;
}