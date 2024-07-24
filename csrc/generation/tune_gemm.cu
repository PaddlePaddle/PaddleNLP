/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "helper.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <list>
#include <sys/time.h>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#include <cuda_fp8.h>
#endif

typedef struct {
  cublasLtMatmulAlgo_t algo;
  cublasStatus_t status;
  float time;
  size_t workspaceSize;
  cublasMath_t mathMode;
  cublasLtReductionScheme_t reductionScheme;
  int customOption;
  float wavesCount;
} customMatmulPerf_t;

typedef struct {
  cublasLtMatmulAlgo_t algo;
  int m;
  int n;
  int k;
  int algoId;
  int swizzle;
  int customOption;
  int tile;
  int splitK_val;
  int reductionScheme;
  int stages;
  size_t workspaceSize;
  float time;
} algoSelect_t;

inline double diffTime(timeval start, timeval end) {
  return (end.tv_sec - start.tv_sec) * 1000 +
         (end.tv_usec - start.tv_usec) * 0.001;
}

const int splitKSequenceA[] = {1, 2, 3, 4, 5, 6, 8, 12, 16, 32};

static inline bool time_compare_perf(const customMatmulPerf_t &perf_a,
                                     const customMatmulPerf_t &perf_b) {
  return ((perf_a.status == CUBLAS_STATUS_SUCCESS) &&
          (perf_a.time < perf_b.time));
}

static inline bool time_compare_algo_para(const algoSelect_t &algo_para_a,
                                          const algoSelect_t &algo_para_b) {
  return (algo_para_a.time < algo_para_b.time);
}

template <typename InT, typename OutT, typename ScaleT = OutT>
static cublasStatus_t TestMatmulRun(cublasLtHandle_t ltHandle,
                                    cublasLtMatmulDesc_t matmulDesc,
                                    cublasLtMatrixLayout_t A_desc,
                                    cublasLtMatrixLayout_t B_desc,
                                    cublasLtMatrixLayout_t C_desc,
                                    const InT *A,
                                    const InT *B,
                                    OutT *C,
                                    const cublasLtMatmulAlgo_t &algo,
                                    customMatmulPerf_t &perfResults,
                                    cudaEvent_t &startEvent,
                                    cudaEvent_t &stopEvent) {
  cudaStream_t stream = 0;
  cublasLtMatmulHeuristicResult_t heurResult;
  cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(
      ltHandle, matmulDesc, A_desc, B_desc, C_desc, C_desc, &algo, &heurResult);
  if (algoStatus == CUBLAS_STATUS_SUCCESS) {
    cudaError_t err;
    ScaleT alpha = static_cast<ScaleT>(1), beta = static_cast<ScaleT>(0);
    void *workSpace;
    cudaMalloc(&workSpace, heurResult.workspaceSize);
    err = cudaEventRecord(startEvent, stream);
    int repeats = 100;
    for (int loop = 0; loop < repeats; loop++) {
      cublasStatus_t currStatus = cublasLtMatmul(ltHandle,
                                                 matmulDesc,
                                                 &alpha,
                                                 A,
                                                 A_desc,
                                                 B,
                                                 B_desc,
                                                 &beta,
                                                 C,
                                                 C_desc,
                                                 C,
                                                 C_desc,
                                                 &algo,
                                                 workSpace,
                                                 heurResult.workspaceSize,
                                                 stream);
      if (currStatus != CUBLAS_STATUS_SUCCESS) {
        algoStatus = currStatus;
        break;
      }
      cudaDeviceSynchronize();
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      algoStatus = CUBLAS_STATUS_INTERNAL_ERROR;
    }

    err = cudaEventRecord(stopEvent, stream);
    err = cudaEventSynchronize(stopEvent);

    float time;
    err = cudaEventElapsedTime(&time, startEvent, stopEvent);
    if (err != cudaSuccess) {
      algoStatus = CUBLAS_STATUS_INTERNAL_ERROR;
    }
    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
      perfResults.algo = algo;
      perfResults.time = time / repeats;
      perfResults.workspaceSize = heurResult.workspaceSize;
      perfResults.wavesCount = heurResult.wavesCount;
    }
    cudaFree(workSpace);
  } else {
    std::cerr << "not enough workspace! current workspace is "
              << heurResult.workspaceSize;
    algoStatus = CUBLAS_STATUS_NOT_SUPPORTED;  // Not enough workspace
  }
  return algoStatus;
}

template <typename InT, typename OutT, typename ScaleT = OutT>
std::vector<customMatmulPerf_t> FindAlgo(cublasLtHandle_t ltHandle,
                                         int m,
                                         int n,
                                         int k,
                                         const InT *A,
                                         const InT *B,
                                         OutT *C,
                                         cublasLtMatmulDesc_t matmulDesc,
                                         cublasLtMatrixLayout_t A_desc,
                                         cublasLtMatrixLayout_t B_desc,
                                         cublasLtMatrixLayout_t C_desc,
                                         cublasComputeType_t computeType,
                                         cudaDataType_t scaleType,
                                         cudaDataType_t Atype,
                                         cudaDataType_t Btype,
                                         cudaDataType_t Ctype,
                                         std::vector<algoSelect_t> &algos,
                                         std::string path) {
  // Get Ids
  // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoGetIds
  // Input
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  int AlgoCount = 0;
  int AlgoCombinations = 20000;
  std::vector<customMatmulPerf_t> perfResultsTmp;

  // Output
  int algoIdA[100];
  int nbAlgoIds;
  status = cublasLtMatmulAlgoGetIds(ltHandle,
                                    computeType,
                                    scaleType,
                                    Atype,
                                    Btype,
                                    Ctype,
                                    Ctype,
                                    100,
                                    algoIdA,
                                    &nbAlgoIds);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasLtMatmulAlgoGetIds error with status " << status;
    return perfResultsTmp;
  }

  std::clog << "get " << nbAlgoIds << " algoIds";

  for (int idx = 0; idx < nbAlgoIds; idx++) {
    cublasLtMatmulAlgo_t algo;
    std::clog << "Process algo " << algoIdA[idx];

    /* Initialize algo structure with given Algp ID */
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoInit
    status = cublasLtMatmulAlgoInit(ltHandle,
                                    computeType,
                                    scaleType,
                                    Atype,
                                    Btype,
                                    Ctype,
                                    Ctype,
                                    algoIdA[idx],
                                    &algo);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << " cublasLtMatmulAlgoInit error with status " << status;
      continue;
    }

    // Query the tiles enums supported by that algo which is used to alloc
    // enough space to store it
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoCapGetAttribute
    cublasLtMatmulTile_t tileA[CUBLASLT_MATMUL_TILE_END];
    size_t nbTiles, sizeWritten;
    if (cublasLtMatmulAlgoCapGetAttribute(&algo,
                                          CUBLASLT_ALGO_CAP_TILE_IDS,
                                          tileA,
                                          sizeof(tileA),
                                          &sizeWritten) ==
        CUBLAS_STATUS_SUCCESS) {
      nbTiles = sizeWritten / sizeof(tileA[0]);
    }
    // Query the stages enums supported by that algo (cuda must >= 11.0)
    cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL, 0, &sizeWritten);
    int nbStages = int(sizeWritten / sizeof(uint32_t));
    std::vector<uint32_t> stagesA(nbStages == 0 ? 1 : nbStages);
    if (nbStages == 0) {
      stagesA[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
      nbStages = 1;
    } else {
      cublasLtMatmulAlgoCapGetAttribute(&algo,
                                        CUBLASLT_ALGO_CAP_STAGES_IDS,
                                        stagesA.data(),
                                        sizeof(uint32_t) * nbStages,
                                        &sizeWritten);
    }

    // Retrieve Other Algo Capabilities attributes
    int32_t splitkSupport, customOptionMax;
    uint32_t redMask, swizzlingMax;
    // cublasLtMatmulInnerShape_t innerShape;
    cublasLtMatmulAlgoCapGetAttribute(&algo,
                                      CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
                                      &splitkSupport,
                                      sizeof(splitkSupport),
                                      &sizeWritten);
    std::clog << "splitkSupport " << splitkSupport;
    cublasLtMatmulAlgoCapGetAttribute(&algo,
                                      CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,
                                      &redMask,
                                      sizeof(redMask),
                                      &sizeWritten);
    cublasLtMatmulAlgoCapGetAttribute(&algo,
                                      CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,
                                      &swizzlingMax,
                                      sizeof(swizzlingMax),
                                      &sizeWritten);
    cublasLtMatmulAlgoCapGetAttribute(&algo,
                                      CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX,
                                      &customOptionMax,
                                      sizeof(customOptionMax),
                                      &sizeWritten);

    /* Loop over the different tiles */
    for (int tileIdx = 0; tileIdx < nbTiles && AlgoCount < AlgoCombinations;
         tileIdx++) {
      /* Loop over different stages count */
      for (int stagesIdx = 0;
           stagesIdx < nbStages && AlgoCount < AlgoCombinations;
           stagesIdx++) {
        /* Loop over the different custom option if any */
        for (int32_t customOption = 0;
             customOption <= customOptionMax && AlgoCount < AlgoCombinations;
             customOption++) {
          /* Loop over the CTAs swizzling support */
          for (uint32_t k = 0;
               k <= swizzlingMax && AlgoCount < AlgoCombinations;
               k++) {
            int splitK_trial = 0;
            if (splitkSupport) {
              splitK_trial +=
                  sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
            }

            for (int l = 0;
                 (l < (1 + splitK_trial)) && (AlgoCount < AlgoCombinations);
                 l++) {
              auto algo_status = cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_TILE_ID,
                  &tileA[tileIdx],
                  sizeof(tileA[tileIdx]));
              if (status != CUBLAS_STATUS_SUCCESS) {
                std::cerr
                    << "GG cublasLtMatmulAlgoConfigSetAttribute with status "
                    << status;
              }
              algo_status = cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_STAGES_ID,
                  &stagesA[stagesIdx],
                  sizeof(stagesA[stagesIdx]));
              if (status != CUBLAS_STATUS_SUCCESS) {
                std::cerr
                    << "GG cublasLtMatmulAlgoConfigSetAttribute with status "
                    << status;
              }
              algo_status = cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                  &customOption,
                  sizeof(customOption));
              if (status != CUBLAS_STATUS_SUCCESS) {
                std::cerr
                    << "GG cublasLtMatmulAlgoConfigSetAttribute with status "
                    << status;
              }
              algo_status = cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k));
              if (status != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cublasLtMatmulAlgoConfigSetAttribute with status "
                          << status;
                continue;
              }
              int splitK_val = 0;
              uint32_t redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
              algo_status = cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                  &splitK_val,
                  sizeof(splitK_val));
              if (status != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cublasLtMatmulAlgoConfigSetAttribute with status "
                          << status;
                continue;
              }
              algo_status = cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                  &redScheme,
                  sizeof(int));
              if (status != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cublasLtMatmulAlgoConfigSetAttribute with status "
                          << status;
                continue;
              }
              if (l > 0) {  // Split-K case
                splitK_val = splitKSequenceA[l - 1];
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo,
                    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                    &splitKSequenceA[l - 1],
                    sizeof(splitKSequenceA[l - 1]));
                for (redScheme = 0;
                     redScheme < (int)CUBLASLT_REDUCTION_SCHEME_MASK &&
                     (AlgoCount < AlgoCombinations);
                     redScheme++) {
                  cublasLtMatmulAlgoConfigSetAttribute(
                      &algo,
                      CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                      &redScheme,
                      sizeof(redScheme));

                  cublasLtMatmulHeuristicResult_t heurResult;
                  cublasStatus_t algoStatus =
                      cublasLtMatmulAlgoCheck(ltHandle,
                                              matmulDesc,
                                              A_desc,
                                              B_desc,
                                              C_desc,
                                              C_desc,
                                              &algo,
                                              &heurResult);
                  if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                    algoSelect_t algoSelect;
                    algoSelect.algo = algo;
                    algoSelect.m = m;
                    algoSelect.n = n;
                    algoSelect.k = k;
                    algoSelect.algoId = algoIdA[idx];
                    algoSelect.tile = tileA[tileIdx];
                    algoSelect.swizzle = k;
                    algoSelect.customOption = customOption;
                    algoSelect.splitK_val = splitK_val;
                    algoSelect.reductionScheme = redScheme;
                    algoSelect.stages = stagesA[stagesIdx];
                    algos.push_back(algoSelect);
                    AlgoCount++;
                  }
                }
              } else {
                // Prepare algos
                cublasLtMatmulHeuristicResult_t heurResult;
                // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoCheck
                cublasStatus_t algoStatus =
                    cublasLtMatmulAlgoCheck(ltHandle,
                                            matmulDesc,
                                            A_desc,
                                            B_desc,
                                            C_desc,
                                            C_desc,
                                            &algo,
                                            &heurResult);
                if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                  algoSelect_t algoSelect;
                  algoSelect.algo = algo;
                  algoSelect.m = m;
                  algoSelect.n = n;
                  algoSelect.k = k;
                  algoSelect.algoId = algoIdA[idx];
                  algoSelect.tile = tileA[tileIdx];
                  algoSelect.swizzle = k;
                  algoSelect.customOption = customOption;
                  algoSelect.splitK_val = splitK_val;
                  algoSelect.reductionScheme = redScheme;
                  algoSelect.stages = stagesA[stagesIdx];
                  algos.push_back(algoSelect);
                  AlgoCount++;
                }
              }
            }
          }
        }
      }
    }
  }
  std::clog << "Got " << AlgoCount << " algos";
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  std::vector<customMatmulPerf_t> perfResults(AlgoCount);
  if (cudaEventCreate(&startEvent) != cudaSuccess) {
    std::cerr << " cudaEventCreate error";
    return perfResults;
  }
  if (cudaEventCreate(&stopEvent) != cudaSuccess) {
    std::cerr << " cudaEventCreate error";
    return perfResults;
  }
  for (int i = 0; i < AlgoCount; i++) {
    status = TestMatmulRun<InT, OutT, ScaleT>(ltHandle,
                                              matmulDesc,
                                              A_desc,
                                              B_desc,
                                              C_desc,
                                              A,
                                              B,
                                              C,
                                              algos[i].algo,
                                              perfResults[i],
                                              startEvent,
                                              stopEvent);
    perfResults[i].status = status;
    algos[i].workspaceSize = perfResults[i].workspaceSize;
    algos[i].time = perfResults[i].time;
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::clog << "algo " << algos[i].algoId << " tile " << algos[i].tile
                << " stages " << algos[i].stages << " splitK_val "
                << algos[i].splitK_val;
      algos[i].time = std::numeric_limits<float>::max();
      std::cerr << "TestMatmulRun with status " << status;
      continue;
    }
  }
  std::sort(algos.begin(), algos.end(), time_compare_algo_para);
  int i = 0;
  while (algos[i].time == 0) i++;
  // return perfResults;
  std::ofstream outfile;
  outfile.open(path, std::ios::app);
  outfile << m << "," << k << "," << n << "," << algos[i].algoId << ","
          << algos[i].swizzle << "," << algos[i].customOption << ","
          << algos[i].tile << "," << algos[i].splitK_val << ","
          << algos[i].reductionScheme << "," << algos[i].stages << ","
          << algos[i].workspaceSize << "," << algos[i].time << "\n";
  outfile.close();
  return perfResults;
}


class DevContext {};

class CPUContext : public DevContext {};

class CUBLASLTContext : public DevContext {
 public:
  CUBLASLTContext() { cublasLtCreate(&handle_); }

  cublasLtHandle_t handle_;

 private:
};

template <typename InT, typename OutT, typename DevContext>
void GEMMInt8(DevContext dev_ctx,
          const std::vector<InT>& A,
          const std::vector<InT>& B,
          std::vector<OutT>& C,
          int m,
          int k,
          int n,
          bool is_test,
          bool is_read_from_csv = false,
          std::string path = "search.csv") {
  std::cerr << "Base Class is not implemented" << std::endl;
}

template <>
void GEMMInt8<int8_t, int32_t, CPUContext>(CPUContext dev_ctx,
                                       const std::vector<int8_t>& A,
                                       const std::vector<int8_t>& B,
                                       std::vector<int32_t>& C,
                                       int m,
                                       int k,
                                       int n,
                                       bool is_test,
                                       bool is_read_from_csv,
                                       std::string path) {
  std::cerr << "CPUContext Class is not implemented" << std::endl;
}

template <>
void GEMMInt8<int8_t, int32_t, CUBLASLTContext>(CUBLASLTContext dev_ctx,
                                            const std::vector<int8_t>& A,
                                            const std::vector<int8_t>& B,
                                            std::vector<int32_t>& C,
                                            int m,
                                            int k,
                                            int n,
                                            bool is_test,
                                            bool is_read_from_csv,
                                            std::string path) {
  int8_t* A_dev;
  int8_t* B_dev;
  int32_t* C_dev;
  char* workspace;

  cudaMalloc((void**)&A_dev, A.size() * sizeof(int8_t));
  cudaMalloc((void**)&B_dev, B.size() * sizeof(int8_t));
  cudaMalloc((void**)&C_dev, m * n * sizeof(int32_t));

  cudaMemcpy(A_dev, A.data(), A.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(B_dev, B.data(), B.size(), cudaMemcpyHostToDevice);

  // init data structure

  cublasLtMatmulDesc_t matmul_desc_;
  cublasLtMatrixLayout_t A_desc_;
  cublasLtMatrixLayout_t B_desc_;
  cublasLtMatrixLayout_t C_desc_;
  int32_t alpha_ = 1;
  int32_t beta_ = 0;

  cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32I;
  cublasLtMatmulDescCreate(&matmul_desc_, cudaComputeType, CUDA_R_32I);
  cublasOperation_t op_transpose = CUBLAS_OP_T;
  cublasLtMatmulDescSetAttribute(matmul_desc_,
                                 CUBLASLT_MATMUL_DESC_TRANSA,
                                 &op_transpose,
                                 sizeof(op_transpose));
  cublasLtMatrixLayoutCreate(&B_desc_, CUDA_R_8I, k, n, k);
  cublasLtMatrixLayoutCreate(&A_desc_, CUDA_R_8I, k, m, k);
  cublasLtMatrixLayoutCreate(&C_desc_, CUDA_R_32I, n, m, n);

  cublasLtMatmulAlgo_t algo;
  int algoId;
  int swizzle;
  int customOption;
  int tile;
  int splitK_val;
  int reductionScheme;
  int stages;
  size_t work_space_size = 0;
  float time_ref;

  auto using_default_config = [&]() {
    algoId = 21;
    swizzle = 0;
    customOption = 0;
    tile = 15;
    splitK_val = 0;
    reductionScheme = 0;
    stages = 23;
    if (m >= 128) {
      tile = 20;
      stages = 17;
    }
  };
  if (is_test) {
    std::vector<algoSelect_t> algos;
    // Select //
    auto results = FindAlgo(dev_ctx.handle_,
                            m,
                            n,
                            k,
                            B_dev,
                            A_dev,
                            C_dev,
                            matmul_desc_,
                            B_desc_,
                            A_desc_,
                            C_desc_,
                            CUBLAS_COMPUTE_32I,
                            CUDA_R_32I,
                            CUDA_R_8I,
                            CUDA_R_8I,
                            CUDA_R_32I,
                            algos,
                            path);
    int i = 0;
    while (algos[i].time == 0) i++;
    algoId = algos[i].algoId;
    swizzle = algos[i].swizzle;
    customOption = algos[i].customOption;
    tile = algos[i].tile;
    splitK_val = algos[i].splitK_val;
    reductionScheme = algos[i].reductionScheme;
    stages = algos[i].stages;
    work_space_size = algos[i].workspaceSize;
  } else if (is_read_from_csv) {
    int m_tmp, k_tmp, n_tmp;
    FILE* fp;
    fp = fopen(path.c_str(), "r");
    if (!fp) {
      using_default_config();
    } else {
      bool match = false;
      int find_cnt = 0;
      while (1) {
        fscanf(fp,
               "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f",
               &m_tmp,
               &k_tmp,
               &n_tmp,
               &algoId,
               &swizzle,
               &customOption,
               &tile,
               &splitK_val,
               &reductionScheme,
               &stages,
               &work_space_size,
               &time_ref);
        if (feof(fp)) break;
        if (k_tmp == k && n_tmp == n && m <= m_tmp) {
          match = true;
          break;
        }
        find_cnt++;
      }
      if (find_cnt == 0) {
        std::cout
            << "Please use test mode to select\n, Now we use default params"
            << std::endl;
        using_default_config();
      }
    }
  } else {
    std::cout << "Please use test mode to select\n, Now we use default params"
              << std::endl;
    using_default_config();
  }

  cudaMalloc((void**)&workspace, work_space_size);
  cublasLtMatmulAlgoInit(dev_ctx.handle_,
                         cudaComputeType,
                         CUDA_R_32I,
                         CUDA_R_8I,
                         CUDA_R_8I,
                         CUDA_R_32I,
                         CUDA_R_32I,
                         algoId,
                         &algo);
  cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                       CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                       &(customOption),
                                       sizeof(customOption));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
  cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                       CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                       &(splitK_val),
                                       sizeof(splitK_val));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
  cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                       CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                       &(reductionScheme),
                                       sizeof(int));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));

  cublasStatus_t status;
  struct timeval start, end;
  gettimeofday(&start, NULL);
  const int repeats = 10;
  for (int loop = 0; loop < repeats; loop++) {
    status = cublasLtMatmul(dev_ctx.handle_,
                            matmul_desc_,
                            &alpha_,
                            B_dev,
                            B_desc_,
                            A_dev,
                            A_desc_,
                            &beta_,
                            C_dev,
                            C_desc_,
                            C_dev,
                            C_desc_,
                            &algo,
                            //  nullptr,
                            (void*)workspace,
                            // 0,
                            work_space_size,
                            0);
    cudaDeviceSynchronize();
  }
  if (status != cudaSuccess) {
    std::cerr << "CUBLASLT runtime error " << status << std::endl;
    exit(status);
  }
  cudaDeviceSynchronize();

  gettimeofday(&end, NULL);
  float time = diffTime(start, end);
  std::cout << "GEMM with cublaslt imma1 int8 spend " << time / repeats
            << " ms in " << m << ", " << k << ", " << n << std::endl;

  cudaMemcpy(C.data(), C_dev, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);
}


void TestBench(const std::vector<int64_t>& M,
                   const std::vector<int64_t>& K,
                   const std::vector<int64_t>& N,
                   const std::string dtype,
                   const std::string path) {

  std::ofstream outfile;
  outfile.open(path, std::ios::out);
  outfile.close();

  for (int j = 0; j < M.size(); j++) {
    int m = M[j];
    for (int i = 0; i < K.size(); ++i) {
      int n = N[i];
      int k = K[i];
      auto A = std::vector<int8_t>(m * k);
      auto B = std::vector<int8_t>(k * n);
      auto C = std::vector<int32_t>(m * n);

      if (dtype == "int8") {
        CUBLASLTContext dev_ctx;
        GEMMInt8(dev_ctx,
                A,
                B,
                C,
                m,
                k,
                n,
                true /*is_test*/,
                false /*is_read_from_csv*/,
                path);
      }else { 
        // other dtype
        std::cout<<"Not currently supported"<<std::endl;
      }
    }
  }
}

PD_BUILD_OP(Tune_gemm)
    .Inputs({})
    .Outputs({})
    .Attrs({"M :std::vector<int64_t>",
            "K :std::vector<int64_t>",
            "N :std::vector<int64_t>",
            "dtype: std::string",
            "path: std::string",})
    .SetKernelFn(PD_KERNEL(TestBench));