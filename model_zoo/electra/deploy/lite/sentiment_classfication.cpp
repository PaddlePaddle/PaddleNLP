// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <arm_neon.h>
#include <math.h>
#include <sys/time.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include "paddle_api.h"  // NOLINT

using namespace paddle::lite_api;  // NOLINT
using namespace std;

#undef stderr
FILE *stderr = &__sF[2];

struct RESULT {
  std::string class_name;
  int class_id;
  float logits;
};

std::vector<RESULT> PostProcess(const float *output_data,
                                int predict_num,
                                int predict_class,
                                const std::vector<std::string> &word_labels) {
  int predict_result[predict_num] = {0};
  float predict_logits[predict_num] = {0};
  for (int i = 0; i < predict_num; i++) {
    int index = -1;
    float max_score = -100.0;
    for (int j = 0; j < predict_class; j++) {
      float score = output_data[i * predict_class + j];
      if (score > max_score) {
        max_score = score;
        index = j;
      }
    }
    predict_result[i] = index;
    predict_logits[i] = max_score;
  }

  std::vector<RESULT> results(predict_num);
  for (int i = 0; i < results.size(); i++) {
    results[i].class_name = "Unknown";
    if (predict_result[i] >= 0 && predict_result[i] < word_labels.size()) {
      results[i].class_name = word_labels[predict_result[i]];
    }
    results[i].class_id = predict_result[i];
    results[i].logits = predict_logits[i];
  }
  return results;
}

std::shared_ptr<PaddlePredictor> LoadModel(std::string model_file) {
  MobileConfig config;
  config.set_model_from_file(model_file);

  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);
  return predictor;
}

std::vector<std::string> split(const std::string &str,
                               const std::string &delim) {
  std::vector<std::string> res;
  if ("" == str) return res;
  char *strs = new char[str.length() + 1];
  std::strcpy(strs, str.c_str());

  char *d = new char[delim.length() + 1];
  std::strcpy(d, delim.c_str());

  char *p = std::strtok(strs, d);
  while (p) {
    string s = p;
    res.push_back(s);
    p = std::strtok(NULL, d);
  }

  return res;
}

std::vector<std::string> ReadDict(const std::string &path) {
  std::ifstream in(path);
  std::string filename;
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such file" << std::endl;
  }
  return m_vec;
}

std::map<std::string, std::string> LoadConfigTxt(
    const std::string &config_path) {
  auto config = ReadDict(config_path);

  std::map<std::string, std::string> dict;
  for (int i = 0; i < config.size(); i++) {
    std::vector<std::string> res = split(config[i], " ");
    dict[res[0]] = res[1];
  }
  return dict;
}

void PrintConfig(const std::map<std::string, std::string> &config) {
  std::cout << "=======PaddleClas lite demo config======" << std::endl;
  for (auto iter = config.begin(); iter != config.end(); iter++) {
    std::cout << iter->first << " : " << iter->second << std::endl;
  }
  std::cout << "=======End of PaddleClas lite demo config======" << std::endl;
}

std::vector<std::string> LoadLabels(const std::string &path) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(path);
  while (file) {
    std::string line;
    std::getline(file, line);
    std::string::size_type pos = line.find(" ");
    if (pos != std::string::npos) {
      line = line.substr(pos + 1);
    }
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

std::vector<RESULT> RunModel(std::shared_ptr<PaddlePredictor> predictor,
                             const std::map<std::string, std::string> &config,
                             double &cost_time) {
  // read config
  std::string label_path = config.at("label_file");
  std::string predict_file_bin = config.at("predict_file_bin");
  int predict_num = stoi(config.at("predict_num"));
  int predict_length = stoi(config.at("predict_length"));

  // Load Labels
  std::vector<std::string> word_labels = LoadLabels(label_path);

  // Read predict data
  int64_t predict_data[predict_num][predict_length] = {0};
  ifstream in(predict_file_bin, ios::in | ios::binary);
  in.read((char *)&predict_data, sizeof predict_data);
  in.close();

  // Fill input tensor
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({predict_num, predict_length});
  auto *data = input_tensor->mutable_data<int64_t>();
  for (int i = 0; i < predict_num; i++) {
    for (int j = 0; j < predict_length; j++) {
      data[i * predict_length + j] = predict_data[i][j];
    }
  }

  auto start = std::chrono::system_clock::now();
  // Run predictor
  predictor->Run();

  // Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto *output_data = output_tensor->data<float>();
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  cost_time = double(duration.count()) *
              std::chrono::microseconds::period::num /
              std::chrono::microseconds::period::den;

  if (output_tensor->shape().size() != 2) {
    std::cerr << "[ERROR] the size of output tensor shape must equal to 2\n";
    exit(1);
  }
  int predict_class = int(output_tensor->shape()[1]);

  auto results =
      PostProcess(output_data, predict_num, predict_class, word_labels);

  return results;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "[ERROR] usage: " << argv[0] << " config_path\n";
    exit(1);
  }

  // load config
  std::string config_path = argv[1];
  auto config = LoadConfigTxt(config_path);
  PrintConfig(config);

  // init predictor
  std::string lite_model_file = config.at("lite_model_file");
  auto electra_predictor = LoadModel(lite_model_file);

  double elapsed_time = 0.0;
  double run_time = 0;

  // run lite inference
  std::vector<RESULT> results = RunModel(electra_predictor, config, run_time);

  // print result
  std::string predict_file_txt = config.at("predict_file_txt");
  auto sentences = ReadDict(predict_file_txt);
  std::cout << "=== electra predict result: " << predict_file_txt
            << "===" << std::endl;
  for (int i = 0; i < results.size(); i++) {
    std::cout << "sentence: " << sentences[i]
              << ", class_id: " << results[i].class_id << "("
              << results[i].class_name << ")"
              << ", logits: " << results[i].logits << std::endl;
  }
  std::cout << "total time : " << run_time << " s." << std::endl;

  return 0;
}
