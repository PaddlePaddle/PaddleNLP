// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <mutex>

#include "helper.h"
#include "cutlass/half.h"
#include "cutlass/bfloat16.h"
#include "paddle/extension.h"

template <paddle::DataType D>
class CutlassDtypeTraits;

template <>
class CutlassDtypeTraits<paddle::DataType::FLOAT32> {
public:
  typedef float DataType;
  typedef float data_t;
};

template <>
class CutlassDtypeTraits<paddle::DataType::FLOAT16> {
public:
  typedef cutlass::half_t DataType;
  typedef paddle::float16 data_t;
};

template <>
class CutlassDtypeTraits<paddle::DataType::BFLOAT16> {
public:
  typedef cutlass::bfloat16_t DataType;
  typedef paddle::bfloat16 data_t;
};

class CutlassGemmConfigMannager {
public:
    static CutlassGemmConfigMannager& getInstance() {
        static CutlassGemmConfigMannager instance;
        return instance;
    }

    CutlassGemmConfigMannager(const CutlassGemmConfigMannager&) = delete;
    CutlassGemmConfigMannager& operator=(const CutlassGemmConfigMannager&) = delete;

    void up_date_configs(const nlohmann::json& j){
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = j.begin(); it != j.end(); ++it) {
          json_[it.key()] = it.value();
      }
    }

    nlohmann::json* get_gemm_best_configs(const std::string & config_file_path) {
      if (!load_initialized_) {
        std::ifstream file(config_file_path);
        if(!file.good()){
            throw std::runtime_error("cutlass gemm_best_config can not be found, please set gemm_best_config'path as FLAGS_use_cutlass_device_best_config_path, or unset FLAGS_use_cutlass_device_best_config_path to tune gemm_best_config");
        }
        json_ = ReadJsonFromFile(config_file_path);
        load_initialized_ = true;
        save_initialized_ = false;
      } 
      return &json_;
    }

private:
    void save_gemm_best_configs_(const std::string & config_file_path) {
      std::ifstream file(config_file_path);
      if(!file.good()){
        std::ofstream new_file(config_file_path);
        new_file << json_.dump(4);
        new_file.close();
      } else {
        nlohmann::json old_json = ReadJsonFromFile(config_file_path);
        for (auto it = json_.begin(); it != json_.end(); ++it) {
            old_json[it.key()] = it.value();
        }
        json_ = old_json;
        std::ofstream new_file(config_file_path, std::ios::out | std::ios::trunc);
        new_file << json_.dump(4);
        new_file.close();
        file.close();
      }
      return;
    }

    CutlassGemmConfigMannager() : json_(nullptr), load_initialized_(false) , save_initialized_(true){}
    ~CutlassGemmConfigMannager() {
      std::lock_guard<std::mutex> lock(mutex_);
      if(save_initialized_){
        std::string config_file_path = "fp8_fuse_gemm_config.json";
        save_gemm_best_configs_(config_file_path);
      }
      save_initialized_=true;
      load_initialized_=false;
      json_.clear();
    }
    mutable std::mutex mutex_;
    nlohmann::json json_;
    bool load_initialized_;
    bool save_initialized_;
};