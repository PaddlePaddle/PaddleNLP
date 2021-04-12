#include <pthread.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>

#include "helper.h"

#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

using namespace paddle_infer;


std::string model_dir = "";
std::string dict_dir = "";
std::string datapath = "";

const int eos_idx = 1;
const int pad_idx = 0;
const int beam_size = 5;
const int max_length = 256;
const int n_best = 1;

int batch_size = 1;
int gpu_id = 0;

namespace paddle {
namespace inference {

struct DataInput {
  std::vector<int64_t> src_data;
};

struct DataResult {
  std::string result_q;
};

bool get_result_tensor(const std::unique_ptr<paddle_infer::Tensor>& seq_ids,
                       std::vector<DataResult>& dataresultvec,
                       std::unordered_map<int, std::string>& num2word_dict) {
  std::vector<int> output_shape = seq_ids->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  std::vector<int> seq_ids_out;
  seq_ids_out.resize(out_num);
  seq_ids->CopyToCpu(seq_ids_out.data());

  dataresultvec.resize(batch_size * n_best);
  auto max_output_length = output_shape[0];

  for (int bsz = 0; bsz < output_shape[1]; ++bsz) {
    for (int k = 0; k < n_best; ++k) {
      dataresultvec[bsz * n_best + k].result_q = "";
      for (int len = 0; len < max_output_length; ++len) {
        if (seq_ids_out[len * batch_size * beam_size + bsz * beam_size + k] ==
            eos_idx)
          break;
        dataresultvec[bsz * n_best + k].result_q =
            dataresultvec[bsz * n_best + k].result_q +
            num2word_dict[seq_ids_out[len * batch_size * beam_size +
                                      bsz * beam_size + k]] +
            " ";
      }
    }
  }
  return true;
}

class DataReader {
public:
  explicit DataReader(const std::string& path)
      : file(new std::ifstream(path)) {}

  bool chg_tensor_more_batch(
      std::shared_ptr<paddle_infer::Predictor>& predictor,
      std::vector<DataInput>& data_input_vec,
      int max_len,
      int batch_size) {
    auto src_word_t = predictor->GetInputHandle("src_word");
    std::vector<int64_t> src_word_vec;
    src_word_vec.resize(max_len * batch_size);
    for (int i = 0; i < batch_size; ++i) {
      for (int k = 0; k < max_len; ++k) {
        if (k < data_input_vec[i].src_data.size()) {
          src_word_vec[i * max_len + k] = data_input_vec[i].src_data[k];
        } else {
          src_word_vec[i * max_len + k] = pad_idx;
        }
      }
    }
    src_word_t->Reshape({batch_size, max_len});
    src_word_t->CopyFromCpu(src_word_vec.data());

    return true;
  }

  bool NextBatch(std::shared_ptr<paddle_infer::Predictor>& predictor,
                 const int& batch_size,
                 std::vector<std::string>& source_query_vec) {
    std::string line;
    std::vector<std::string> word_data;
    std::vector<DataInput> data_input_vec;
    int max_len = 0;
    for (int i = 0; i < batch_size; i++) {
      if (!std::getline(*file, line)) return false;
      DataInput data_input;
      split(line, ' ', &word_data);
      std::string query_str = "";
      for (int j = 0; j < word_data.size(); ++j) {
        if (j >= max_length) {
          break;
        }
        query_str += word_data[j];
        if (word2num_dict.find(word_data[j]) == word2num_dict.end()) {
          data_input.src_data.push_back(word2num_dict["<unk>"]);
        } else {
          data_input.src_data.push_back(word2num_dict[word_data[j]]);
        }
      }
      source_query_vec.push_back(query_str);
      data_input.src_data.push_back(eos_idx);
      max_len = std::max(max_len, static_cast<int>(data_input.src_data.size()));
      max_len = std::min(max_len, max_length);
      data_input_vec.push_back(data_input);
    }
    return chg_tensor_more_batch(
        predictor, data_input_vec, max_len, batch_size);
  }

  bool get_word_dict() {
    std::ifstream fin(dict_dir);
    std::string line;
    int k = 0;
    while (std::getline(fin, line)) {
      word2num_dict[line] = k;
      num2word_dict[k] = line;
      k += 1;
    }

    fin.close();

    return true;
  }

public:
  std::unordered_map<std::string, int> word2num_dict;
  std::unordered_map<int, std::string> num2word_dict;
  std::unique_ptr<std::ifstream> file;
};


template <typename... Args>
void SummaryConfig(const paddle_infer::Config& config,
                   double infer_time,
                   int num_batches) {
  LOG(INFO) << "----------------------- Data info -----------------------";
  LOG(INFO) << "batch_size: " << batch_size;
  LOG(INFO) << "num_of_samples: " << num_batches * batch_size;
  LOG(INFO) << "----------------------- Conf info -----------------------";
  LOG(INFO) << "runtime_device: " << (config.use_gpu() ? "gpu" : "cpu");
  LOG(INFO) << "ir_optim: " << (config.ir_optim() ? "true" : "false");
  LOG(INFO) << "----------------------- Perf info -----------------------";
  LOG(INFO) << "average_latency(ms): "
            << infer_time / (num_batches * batch_size) << ", "
            << "QPS: " << (num_batches * batch_size) / (infer_time / 1000.0);
}


void Main(int batch_size, int gpu_id) {
  Config config;
  config.SetModel(model_dir + "/transformer.pdmodel",
                  model_dir + "/transformer.pdiparams");

  config.EnableUseGpu(100, gpu_id);

  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  auto predictor = CreatePredictor(config);
  DataReader reader(datapath);
  reader.get_word_dict();

  double whole_time = 0;
  Timer timer;
  int num_batches = 0;
  std::vector<std::string> source_query_vec;
  std::ofstream out("predict.txt");

  while (reader.NextBatch(predictor, batch_size, source_query_vec)) {
    timer.tic();
    predictor->Run();
    std::vector<DataResult> dataresultvec;
    auto output_names = predictor->GetOutputNames();
    get_result_tensor(predictor->GetOutputHandle(output_names[0]),
                      dataresultvec,
                      reader.num2word_dict);

    whole_time += timer.toc();
    num_batches++;

    if (out.is_open()) {
      for (int i = 0; i < dataresultvec.size(); ++i) {
        out << dataresultvec[i].result_q << "\n";
      }
    }

    source_query_vec.clear();
  }
  SummaryConfig(config, whole_time, num_batches);
  out.close();
}
}  // namespace inference
}  // namespace paddle

int main(int argc, char** argv) {
  batch_size = std::stoi(std::string(argv[1]));
  gpu_id = std::stoi(std::string(argv[2]));

  model_dir = std::string(argv[3]);
  dict_dir = std::string(argv[4]);
  datapath = std::string(argv[5]);

  paddle::inference::Main(batch_size, gpu_id);

  return 0;
}
