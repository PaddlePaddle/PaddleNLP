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

#include "parallel_utils.h"

static std::mutex mpi_global_mutex;
static std::once_flag once_flag_init_mpi;

void MPIExit() {
  std::unique_lock<std::mutex> global_lock(mpi_global_mutex);
  MPICHECK(MPI_Finalize());
}

void InitMPIOnce() {
  // Initialize MPI environment
  std::call_once(once_flag_init_mpi, []() {
    MPICHECK(MPI_Init(nullptr, nullptr));
    if (std::atexit(MPIExit)) {
      throw std::runtime_error("Fail to register the MPI exit handler");
    }
  });
}

void InitNCCLComm(ncclUniqueId& tensor_para_nccl_uid,
                  ncclUniqueId& layer_para_nccl_uid,
                  ncclComm_t& tensor_para_nccl_comm,
                  ncclComm_t& layer_para_nccl_comm,
                  int rank,
                  int tensor_para_size,
                  int layer_para_size,
                  int tensor_para_rank,
                  int layer_para_rank) {
  // assume gpu_num = n * k,
  // tensor parallelism group size is n
  // layer parallelism group size is k

  if (tensor_para_rank == 0) {
    // get the uid of each tensor parallelism group
    // here, 0, 1, ..., n-1 are in group 0,
    //       n, ..., 2n - 1 are in group 1.
    NCCLCHECK(ncclGetUniqueId(&tensor_para_nccl_uid));
    for (int i = 1; i < tensor_para_size; i++) {
      printf("[INFO] rank %d sends tensor_para_nccl_uid to rank %d \n",
             rank,
             rank + i);
      MPICHECK(MPI_Send(&tensor_para_nccl_uid,
                        sizeof(tensor_para_nccl_uid),
                        MPI_BYTE,
                        rank + i,
                        0,
                        MPI_COMM_WORLD));
    }
  } else {
    MPI_Status status;
    printf("[INFO] rank %d receives tensor_para_nccl_uid from rank %d \n",
           rank,
           rank - tensor_para_rank);
    MPICHECK(MPI_Recv(&tensor_para_nccl_uid,
                      sizeof(tensor_para_nccl_uid),
                      MPI_BYTE,
                      rank - tensor_para_rank,
                      0,
                      MPI_COMM_WORLD,
                      &status));
  }

  if (layer_para_rank == 0) {
    // get the uid of each layer parallelism group
    // 0, k, 2k, are in group 0
    // 1, k+1, 2k+1 are in group 1
    NCCLCHECK(ncclGetUniqueId(&layer_para_nccl_uid));
    for (int i = 1; i < layer_para_size; i++) {
      printf("[INFO] rank %d sends layer_para_nccl_uid to rank %d \n",
             rank,
             rank + i * tensor_para_size);
      MPICHECK(MPI_Send(&layer_para_nccl_uid,
                        sizeof(layer_para_nccl_uid),
                        MPI_BYTE,
                        rank + i * tensor_para_size,
                        0,
                        MPI_COMM_WORLD));
    }
  } else {
    MPI_Status status;
    printf("[INFO] rank %d receives layer_para_nccl_uid from rank %d \n",
           rank,
           rank % tensor_para_size);
    MPICHECK(MPI_Recv(&layer_para_nccl_uid,
                      sizeof(layer_para_nccl_uid),
                      MPI_BYTE,
                      rank % tensor_para_size,
                      0,
                      MPI_COMM_WORLD,
                      &status));
  }

  NCCLCHECK(ncclCommInitRank(&tensor_para_nccl_comm,
                             tensor_para_size,
                             tensor_para_nccl_uid,
                             tensor_para_rank));
  NCCLCHECK(ncclCommInitRank(&layer_para_nccl_comm,
                             layer_para_size,
                             layer_para_nccl_uid,
                             layer_para_rank));
}

// Make model parallel settings init only once for one model by using a global
// dict mapping parameters representing different models to corresponding
// settings. Note: `paddle::Tensor` for custom_op is re-created every step and
// we use pointers as keys. Maybe using weakref as keys is better.
static std::unordered_map<void*, std::unique_ptr<ModelParaDesc>>
    model_para_infos;

ModelParaDesc* ModelParaDescFactory::CreateModelParaDesc(
    int head_num,
    int size_per_head,
    int layer_num,
    int tensor_para_size,
    int layer_para_size,
    int layer_para_batch_size,
    void* param_ptr = nullptr) {
  InitMPIOnce();
  auto it = model_para_infos.find(param_ptr);
  if (it != model_para_infos.end()) {
    return it->second.get();
  } else {
    model_para_infos.emplace(param_ptr,
                             std::unique_ptr<ModelParaDesc>(
                                 new ModelParaDesc(head_num,
                                                   size_per_head,
                                                   layer_num,
                                                   tensor_para_size,
                                                   layer_para_size,
                                                   layer_para_batch_size)));
    return model_para_infos[param_ptr].get();
  }
}
