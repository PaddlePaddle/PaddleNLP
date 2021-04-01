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

const std::string model_dir = "/paddle/origin/transformer/models";
const std::string dict_dir =
    "/paddle/origin/transformer/wmt16_ende_data_bpe_clean/"
    "train.tok.clean.bpe.32000.en-de";
const std::string datapath = "./seq.data";
const int batch_size = 1;
const int num_threads = 1;
const int eos_idx = 1;
const int beam_search = 4;
const int max_query_len = 32;

int gpu_id = 0;

namespace paddle {
namespace inference {

struct DataInput {
  std::vector<int64_t> src_data;
  std::vector<int64_t> src_pos;
};

struct DataResult {
  std::string reslult_q;
  float score;
};

bool get_result_tensor(
    const std::unique_ptr<paddle::ZeroCopyTensor> &seq_ids,
    const std::unique_ptr<paddle::ZeroCopyTensor> &seq_scores,
    std::vector<DataResult> &dataresultvec,
    std::unordered_map<int, std::string> &num2word_dict) {
  auto lod_data = seq_ids->lod();
  std::vector<int> output_shape = seq_ids->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  std::vector<int64_t> seq_ids_out;
  std::vector<float> seq_scores_out;
  seq_ids_out.resize(out_num);
  seq_scores_out.resize(out_num);
  seq_ids->copy_to_cpu(seq_ids_out.data());
  seq_scores->copy_to_cpu(seq_scores_out.data());
  for (int i = 0; i < lod_data[1].size() - 1; i++) {
    int j = lod_data[1][i] + 1;
    int k = lod_data[1][i + 1] - 1;
    DataResult dataresult;
    dataresult.reslult_q = "";
    while (j < k) {
      dataresult.reslult_q += num2word_dict[seq_ids_out[j]];
      j += 1;
    }
    dataresult.score = seq_scores_out[j];
    dataresultvec.push_back(dataresult);
  }
  return true;
}

class DataReader {
public:
  explicit DataReader(const std::string &path)
      : file(new std::ifstream(path)) {}

  bool chg_tensor_more_batch(
      std::unique_ptr<paddle::PaddlePredictor> &predictor,
      std::vector<DataInput> &data_input_vec,
      int max_len,
      int batch_size) {
    auto src_word_t = predictor->GetInputTensor("src_word");
    int64_t src_word_vec[max_len * batch_size];
    for (int i = 0; i < max_len * batch_size; ++i) {
      src_word_vec[i] = 1;
    }

    return true;
  }

  bool NextBatch(std::unique_ptr<paddle::PaddlePredictor> &predictor,
                 int batch_size,
                 std::vector<std::string> &source_query_vec) {
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
        if (j >= max_query_len) {
          break;
        }
        query_str += word_data[j];
        if (word2num_dict.find(word_data[j]) == word2num_dict.end()) {
          data_input.src_data.push_back(word2num_dict["<unk>"]);
        } else {
          data_input.src_data.push_back(word2num_dict[word_data[j]]);
        }
        data_input.src_pos.push_back(j);
      }
      source_query_vec.push_back(query_str);
      data_input.src_data.push_back(eos_idx);
      data_input.src_pos.push_back(data_input.src_pos.size());
      max_len = std::max(max_len, static_cast<int>(data_input.src_data.size()));
      max_len = std::min(max_len, max_query_len);
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
      // LOG(INFO) << k;
      // LOG(INFO) << line;
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

  float *src_slf_attn_bias_vec;  //[32*8*255*255];
  float *src_slf_attn_bias_vec_;
};

bool get_result(const std::unique_ptr<paddle::ZeroCopyTensor> &seq_ids,
                const std::unique_ptr<paddle::ZeroCopyTensor> &seq_scores,
                std::vector<DataResult> &dataresultvec,
                std::unordered_map<int, std::string> &num2word_dict) {
  auto lod_data = seq_ids->lod();
  // LOG(INFO) << lod_data.size() << lod_data[0].size() << " " <<
  // lod_data[1].size();
  std::vector<int> output_shape = seq_ids->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  std::vector<int64_t> seq_ids_out;
  std::vector<float> seq_scores_out;
  seq_ids_out.resize(out_num);
  seq_scores_out.resize(out_num);
  seq_ids->copy_to_cpu(seq_ids_out.data());
  seq_scores->copy_to_cpu(seq_scores_out.data());
  LOG(INFO) << "IDS:";
  for (int i = 0; i < out_num; ++i) {
    std::cout << seq_ids_out[i] << " ";
  }
  std::cout << std::endl;

  LOG(INFO) << "SCORES:";
  for (int i = 0; i < out_num; ++i) {
    std::cout << seq_scores_out[i] << " ";
  }
  std::cout << std::endl;
  return true;
}


void Main1(int batch_size) {
  AnalysisConfig config;
  config.SetModel(model_dir);
  // gpu
  config.EnableUseGpu(2000, gpu_id);

  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);

  auto predictor = CreatePaddlePredictor(config);
  DataReader reader(datapath);
  reader.get_word_dict();

  double whole_time = 0;
  int num_batches = 0;
  std::vector<std::string> source_query_vec;

  std::vector<float> print_f;
  std::vector<int64_t> print_int;
  LOG(INFO) << "run case";

  while (reader.NextBatch(predictor, batch_size, source_query_vec)) {
    CHECK(predictor->ZeroCopyRun());
    /*
          for (int sour_idx = 0; sour_idx < source_query_vec.size(); sour_idx++)
       {
              std::string out_str = source_query_vec[sour_idx]+"\t";
              if (dataresultvec[0].reslult_q == source_query_vec[sour_idx]) {
                  //std::cout << out_str << "0\n";
                  continue;
              }
              for (int i = 0; i < beam_search; ++i) {
                  out_str += dataresultvec[i].reslult_q + "\001" +
       to_string(dataresultvec[i].score);
                  if (i != (beam_search - 1)) {
                      out_str += "\002";
                  }
              }
              //std::cout << out_str << "\n";
          }
    */
    num_batches++;
    source_query_vec.clear();
  }

  // std::cout << "total number of samples: " << num_batches * batch_size <<
  // std::endl;
  // std::cout << "batch_size:" << batch_size <<", time: " << whole_time <<
  // std::endl;
  // std::cout << "average latency of each sample: " << whole_time / num_batches
  // / batch_size << std::endl;
  // std::cout << num_batches / (whole_time / 1000) << std::endl;
}
}  // namespace inference
}  // namespace paddle


int main(int argc, char **argv) {
  gpu_id = std::stoi(std::string(argv[1]));
  paddle::inference::Main1(batch_size);
  return 0;
}
