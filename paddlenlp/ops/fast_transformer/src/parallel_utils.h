/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <unordered_map>
#include <memory>
#include <mutex>
#include <random>
#include <assert.h>

#include "fastertransformer/utils/nccl_utils.h"


void MPIExit();

void InitMPIOnce();

void InitNCCLComm(ncclUniqueId& tensor_para_nccl_uid,
                  ncclUniqueId& layer_para_nccl_uid,
                  ncclComm_t& tensor_para_nccl_comm,
                  ncclComm_t& layer_para_nccl_comm,
                  int rank,
                  int tensor_para_size,
                  int layer_para_size,
                  int tensor_para_rank,
                  int layer_para_rank);

struct ModelParaDesc {
  TensorParallelParam tensor_parallel_param;
  LayerParallelParam layer_parallel_param;
  ncclComm_t tensor_para_nccl_comm, layer_para_nccl_comm;
  std::mt19937_64 gen;
  std::uniform_int_distribution<> dist{0, std::numeric_limits<int>::max()};

  ModelParaDesc(int head_num,
                int size_per_head,
                int layer_num,
                int tensor_para_size,
                int layer_para_size,
                int layer_para_batch_size) {
    int rank;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    const int local_head_num = head_num / tensor_para_size;
    const int local_hidden_units = local_head_num * size_per_head;
    const int layers_per_group = layer_num / layer_para_size;
    assert(layer_num % layer_para_size == 0);
    const int tensor_para_rank = rank % tensor_para_size;
    const int layer_para_rank = rank / tensor_para_size;
    ncclUniqueId tensor_para_nccl_uid, layer_para_nccl_uid;
    InitNCCLComm(tensor_para_nccl_uid,
                 layer_para_nccl_uid,
                 tensor_para_nccl_comm,
                 layer_para_nccl_comm,
                 rank,
                 tensor_para_size,
                 layer_para_size,
                 tensor_para_rank,
                 layer_para_rank);
    tensor_parallel_param.rank = tensor_para_rank;
    tensor_parallel_param.world_size = tensor_para_size;
    tensor_parallel_param.local_head_num_ = local_head_num;
    tensor_parallel_param.local_hidden_units_ = local_hidden_units;
    tensor_parallel_param.nccl_comm = tensor_para_nccl_comm;
    layer_parallel_param.rank = layer_para_rank;
    layer_parallel_param.world_size = layer_para_size;
    layer_parallel_param.layers_per_group = layers_per_group;
    layer_parallel_param.local_batch_size = layer_para_batch_size;
    layer_parallel_param.nccl_comm = layer_para_nccl_comm;
    // fix the seed to prevent the seed of different gpu are differnet in Tensor
    // Parallel
    size_t meta_seed =
        *(reinterpret_cast<size_t*>(tensor_para_nccl_uid.internal));
    gen = std::mt19937_64(meta_seed);
  }

  ~ModelParaDesc() {
    if (tensor_para_nccl_comm) ncclCommDestroy(tensor_para_nccl_comm);
    if (layer_para_nccl_comm) ncclCommDestroy(layer_para_nccl_comm);
  }
};

struct ModelParaDescFactory {
  static ModelParaDesc* CreateModelParaDesc(int head_num,
                                            int size_per_head,
                                            int layer_num,
                                            int tensor_para_size,
                                            int layer_para_size,
                                            int layer_para_batch_size,
                                            void* param_ptr);
};
