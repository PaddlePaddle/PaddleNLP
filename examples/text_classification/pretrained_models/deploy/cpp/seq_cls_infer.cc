

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include "paddle/include/paddle_inference_api.h"
#include "tokenizer.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

DEFINE_string(model_file, "", "Path to the inference model file.");
DEFINE_string(params_file, "", "Path to the inference parameters file.");
DEFINE_string(vocab_file, "", "Path to the vocab file.");
DEFINE_int32(seq_len, 128, "Sequence length should less than or equal to 512.");
DEFINE_bool(use_gpu, true, "enable gpu");


template <typename T>
void PrepareInput(Predictor* predictor,
                  const std::vector<T>& data,
                  const std::string& name,
                  const std::vector<int>& shape) {
  auto input = predictor->GetInputHandle(name);
  input->Reshape(shape);
  input->CopyFromCpu(data.data());
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

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  tokenizer::Vocab vocab;
  tokenizer::LoadVocab(FLAGS_vocab_file, &vocab);
  tokenizer::BertTokenizer tokenizer(vocab, false);

  std::vector<std::string> data{
      "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般",
      "请问：有些字打错了，我怎么样才可以回去编辑一下啊？",
      "本次入住酒店的网络不是很稳定，断断续续，希望能够改进。"};
  int batch_size = data.size();
  std::vector<std::unordered_map<std::string, std::vector<int64_t>>>
      batch_encode_inputs(batch_size);
  tokenizer.BatchEncode(&batch_encode_inputs,
                        data,
                        std::vector<std::string>(),
                        false,
                        FLAGS_seq_len,
                        true);

  std::vector<int64_t> input_ids(batch_size * FLAGS_seq_len);
  for (size_t i = 0; i < batch_size; i++) {
    std::copy(batch_encode_inputs[i]["input_ids"].begin(),
              batch_encode_inputs[i]["input_ids"].end(),
              input_ids.begin() + i * FLAGS_seq_len);
  }
  std::vector<int64_t> toke_type_ids(batch_size * FLAGS_seq_len);
  for (size_t i = 0; i < batch_size; i++) {
    std::copy(batch_encode_inputs[i]["token_type_ids"].begin(),
              batch_encode_inputs[i]["token_type_ids"].end(),
              toke_type_ids.begin() + i * FLAGS_seq_len);
  }

  Config config;
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  if (FLAGS_use_gpu) {
    config.EnableUseGpu(100, 0);
  }
  auto predictor = CreatePredictor(config);

  auto input_names = predictor->GetInputNames();
  std::vector<int> shape{batch_size, FLAGS_seq_len};
  PrepareInput(predictor.get(), input_ids, input_names[0], shape);
  PrepareInput(predictor.get(), toke_type_ids, input_names[1], shape);

  predictor->Run();
  std::vector<float> logits;
  auto output_names = predictor->GetOutputNames();
  GetOutput(predictor.get(), output_names[0], &logits);
  for (size_t i = 0; i < data.size(); i++) {
    std::string label =
        (logits[i * 2] < logits[i * 2 + 1]) ? "negative" : "positive";
    std::cout << data[i] << " : " << label << std::endl;
  }
  return 0;
}