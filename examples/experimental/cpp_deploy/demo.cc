

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <unordered_map>

#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 10, "Directory of the inference model.");
DEFINE_bool(use_gpu, true, "enable gpu");

template <class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last) {
  return std::distance(first, std::max_element(first, last));
}

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
         std::vector<float>* out_data_0) {
  const int batch_size = FLAGS_batch_size;

  auto input_names = predictor->GetInputNames();

  auto text = predictor->GetInputHandle(input_names[0]);
  text->ReshapeStrings(input_data->size());
  text->CopyStringsFromCpu(input_data);

  CHECK(predictor->Run());

  auto output_names = predictor->GetOutputNames();
  auto logits = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = logits->shape();
  int logits_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  out_data_0->resize(logits_num);
  logits->CopyToCpu(out_data_0->data());
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
  std::unordered_map<size_t, std::string> label_map{{0, "negative"},
                                                    {1, "positive"}};

  std::vector<float> out_data_0;
  Run(predictor.get(), &data, &out_data_0);

  std::vector<std::vector<float>> probs;
  size_t i = 0;
  while (i < out_data_0.size()) {
    std::vector<float> tmp;
    tmp.emplace_back(std::move(out_data_0[i]));
    tmp.emplace_back(std::move(out_data_0[i + 1]));
    probs.emplace_back(std::move(tmp));
    i += 2;
  }
  for (size_t j = 0; j < probs.size(); j++) {
    size_t label_idx = argmax(probs[j].begin(), probs[j].end());
    LOG(INFO) << "Text: " << data[j] << " label:" << label_map[label_idx];
  }
  return 0;
}