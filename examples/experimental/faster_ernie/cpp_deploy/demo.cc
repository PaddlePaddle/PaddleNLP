

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

void softmax(const std::vector<float>& src,
             std::vector<float>* res,
             int num_classes = 2) {
  size_t length = src.size();
  assert(length % num_classes == 0);

  res->resize(src.size());
  transform(src.begin(), src.end(), res->begin(), exp);
  for (size_t i = 0; i < length; i += num_classes) {
    float sum =
        accumulate(res->begin() + i, res->begin() + i + num_classes, 0.0);
    for (size_t j = i; j < i + num_classes; j++) {
      res->at(j) /= sum;
    }
  }
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
         std::vector<std::string>* input_data,
         std::vector<float>* logits,
         std::vector<int64_t>* predictions) {
  auto input_names = predictor->GetInputNames();

  auto text = predictor->GetInputHandle(input_names[0]);
  text->ReshapeStrings(input_data->size());
  text->CopyStringsFromCpu(input_data);

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
    Run(predictor.get(), &batch, &logits, &predictions);
    for (size_t j = 0; j < FLAGS_batch_size; j++) {
      LOG(INFO) << "The text is " << batch[j] << "; The predition label is "
                << label_map[predictions[j]];
    }
  }

  return 0;
}