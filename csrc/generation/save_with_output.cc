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

#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "stdlib.h"
#include <stdio.h>
#include <dlfcn.h>  // dladdr
#include <sys/time.h>
#include <sys/stat.h>
#include "paddle/extension.h"

constexpr char kSEP = '/';

std::string DirName(const std::string &filepath) {
  auto pos = filepath.rfind(kSEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

bool FileExists(const std::string &filepath) {
  struct stat buffer;
  return (stat(filepath.c_str(), &buffer) == 0);
}

void MkDir(const char *path) {
  std::string path_error(path);
  path_error += " mkdir failed!";
  if (mkdir(path, 0755)) {
    if (errno != EEXIST) {
      throw std::runtime_error(path_error);
    }
  }
}

void MkDirRecursively(const char *fullpath) {
  if (*fullpath == '\0') return;  // empty string
  if (FileExists(fullpath)) return;
  MkDirRecursively(DirName(fullpath).c_str());
  MkDir(fullpath);
}


template<typename data_t>
void saveToFile(std::ostream & os, const void* x_data, std::vector<int64_t> shape, int64_t x_numel, const char type_id) {
  // 1.type
  os.write(reinterpret_cast<const char *>(&type_id),sizeof(type_id));
  // 2.data
  uint64_t size = x_numel * sizeof(data_t);
  os.write(static_cast<const char*>(x_data),static_cast<std::streamsize>(size));

}

template<typename data_t>
void save_with_output_kernel(const paddle::Tensor& x,
                             const paddle::Tensor& batch_idx,
                             const paddle::Tensor& step_idx,
                             std::string file_path,
                             int64_t rank_id,
                             char type_id) {
  std::vector<int64_t> x_shape = x.shape();

  if(rank_id >= 0) {
      file_path += "_rank_" + std::to_string(rank_id);
  }

  int batch_idx_data = -1, step_idx_data = -1;

  if(batch_idx.is_gpu()) {
    paddle::Tensor batch_idx_cpu = batch_idx.copy_to<int32_t>(paddle::CPUPlace());
    batch_idx_data = batch_idx_cpu.data<int32_t>()[0];
  } else {
    batch_idx_data = batch_idx.data<int32_t>()[0];
  }
  if(step_idx.is_gpu()) {
    paddle::Tensor step_idx_cpu = step_idx.copy_to<int64_t>(paddle::CPUPlace());
    step_idx_data = step_idx_cpu.data<int64_t>()[0];
  } else {
    step_idx_data = step_idx.data<int64_t>()[0];
  }
  auto x_data = x.data<data_t>();

  if(batch_idx_data >= 0) {
    file_path += "_batch_" + std::to_string(batch_idx_data);
  }
  if(step_idx_data >= 0) {
    file_path += "_step_" + std::to_string(step_idx_data);
  }
  MkDirRecursively(DirName(file_path).c_str());
  std::ofstream fout(file_path, std::ios::binary);
  fout.write("0",1);
  saveToFile<data_t>(fout, x_data, x_shape, x.numel(),type_id);
  fout.seekp(std::ios::beg);
  fout.write("1",1);
  fout.close();

}

void print_shape(const paddle::Tensor& tmp, char *tmp_str){
    std::vector<int64_t> shape = tmp.shape();
    printf("%s's shape: \n", tmp_str);
    for(int i=0; i < shape.size(); i++) {
        printf("%d ", (int)shape[i]);
    }
    printf("\n");
}

std::vector<paddle::Tensor> SaveWithOutputForward(const paddle::Tensor& x,
                                                  const paddle::Tensor& batch_idx,
                                                  const paddle::Tensor& step_idx,
                                                  std::string file_path,
                                                  int64_t rank_id) {
    auto out = x.copy_to(paddle::CPUPlace(), false);
    switch(x.type()) {
      case paddle::DataType::FLOAT32:
         save_with_output_kernel<float>(out, batch_idx, step_idx, file_path, rank_id, '0');
         break;
      case paddle::DataType::INT64:
        save_with_output_kernel<int64_t>(out, batch_idx, step_idx, file_path, rank_id,'1');
         break;
      case paddle::DataType::INT32:
        save_with_output_kernel<int32_t>(out, batch_idx, step_idx, file_path, rank_id, '2');
         break;
      default:
        PD_THROW("function SaveWithOutputForward is not implemented for data type");
    }
   return {out};
}

std::vector<std::vector<int64_t>> SaveWithOutputInferShape(const std::vector<int64_t>& x_shape,
                                                           const std::vector<int64_t>& batch_idx_shape,
                                                           const std::vector<int64_t>& step_idx_shape) {
    return {x_shape};
}

std::vector<paddle::DataType> SaveWithOutputInferDtype(const paddle::DataType& x_dtype,
                                                       const paddle::DataType& batch_idx_dtype,
                                                       const paddle::DataType& step_idx_dtype) {
    return {x_dtype};
}

PD_BUILD_OP(save_with_output)
    .Inputs({"x", "batch_idx", "step_idx"})
    .Attrs({"file_path: std::string",
            "rank_id: int64_t"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(SaveWithOutputForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SaveWithOutputInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SaveWithOutputInferDtype));
