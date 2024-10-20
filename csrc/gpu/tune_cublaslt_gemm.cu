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

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <vector>

#include "helper.h"

template <typename T>
void handleError(T status, const char* file, int line) {
  printf("Unknown error type at %s:%d\n", file, line);
  exit(1);
}

// for cudaError_t
template <>
void handleError<cudaError_t>(cudaError_t status, const char* file, int line) {
  if (status != cudaSuccess) {
    printf(
        "CUDA error at %s:%d - %s\n", file, line, cudaGetErrorString(status));
    exit(1);
  }
}

// for cublasStatus_t
template <>
void handleError<cublasStatus_t>(cublasStatus_t status,
                                 const char* file,
                                 int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf(
        "cuBLAS error at %s:%d - %d\n", file, line, static_cast<int>(status));
    exit(1);
  }
}

#define CUDA_CHECK(call)                     \
  do {                                       \
    handleError((call), __FILE__, __LINE__); \
  } while (0)

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

inline double diffTime(
    const std::chrono::high_resolution_clock::time_point& start,
    const std::chrono::high_resolution_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

const int splitKSequenceA[] = {1, 2, 3, 4, 5, 6, 8, 12, 16, 32};

static inline bool time_compare_perf(const customMatmulPerf_t& perf_a,
                                     const customMatmulPerf_t& perf_b) {
  return ((perf_a.status == CUBLAS_STATUS_SUCCESS) &&
          (perf_a.time < perf_b.time));
}

static inline bool time_compare_algo_para(const algoSelect_t& algo_para_a,
                                          const algoSelect_t& algo_para_b) {
  return (algo_para_a.time < algo_para_b.time);
}

// 获取当前 GPU 的剩余显存大小（以字节为单位）
size_t get_remaining_memory() {
  size_t free, total;
  CUDA_CHECK(cudaMemGetInfo(&free, &total));
  return free;
}

template <typename InT, typename OutT, typename ScaleT = OutT>
static void TestMatmulRun(cublasLtHandle_t ltHandle,
                          cublasLtMatmulDesc_t matmulDesc,
                          cublasLtMatrixLayout_t A_desc,
                          cublasLtMatrixLayout_t B_desc,
                          cublasLtMatrixLayout_t C_desc,
                          const InT* A,
                          const InT* B,
                          OutT* C,
                          const cublasLtMatmulAlgo_t& algo,
                          customMatmulPerf_t& perfResults,
                          cudaEvent_t& startEvent,
                          cudaEvent_t& stopEvent) {
  cudaStream_t stream = 0;
  cublasLtMatmulHeuristicResult_t heurResult;
  cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(
      ltHandle, matmulDesc, A_desc, B_desc, C_desc, C_desc, &algo, &heurResult);

  auto remainingMemorySize = 0.95 * get_remaining_memory();
  if (algoStatus == CUBLAS_STATUS_SUCCESS &&
      remainingMemorySize > heurResult.workspaceSize) {
    ScaleT alpha = static_cast<ScaleT>(1), beta = static_cast<ScaleT>(0);
    void* workSpace;
    CUDA_CHECK(cudaMalloc(&workSpace, heurResult.workspaceSize));
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
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
        perfResults.status = currStatus;
        break;
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float time;
    CUDA_CHECK(cudaEventElapsedTime(&time, startEvent, stopEvent));

    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
      perfResults.algo = algo;
      perfResults.time = time / repeats;
      perfResults.workspaceSize = heurResult.workspaceSize;
      perfResults.wavesCount = heurResult.wavesCount;
    }
    CUDA_CHECK(cudaFree(workSpace));
  } else {
    std::cerr << "Not enough workspace! Required "
              << static_cast<double>(heurResult.workspaceSize) / 1024.0 /
                     1024.0 / 1024.0
              << " GiB" << ", But remaining "
              << static_cast<double>(remainingMemorySize) / 1024.0 / 1024.0 /
                     1024.0
              << " GiB" << std::endl;
    perfResults.status = CUBLAS_STATUS_NOT_SUPPORTED;  // Not enough workspace
  }
}

template <typename InT, typename OutT, typename ScaleT = OutT>
void FindAlgo(const cublasLtHandle_t& ltHandle,
              int m,
              int n,
              int k,
              const InT* A,
              const InT* B,
              OutT* C,
              cublasLtMatmulDesc_t matmulDesc,
              cublasLtMatrixLayout_t A_desc,
              cublasLtMatrixLayout_t B_desc,
              cublasLtMatrixLayout_t C_desc,
              cublasComputeType_t computeType,
              cudaDataType_t scaleType,
              cudaDataType_t Atype,
              cudaDataType_t Btype,
              cudaDataType_t Ctype,
              std::vector<algoSelect_t>& algos,
              const std::string& path) {
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
  CUDA_CHECK(cublasLtMatmulAlgoGetIds(ltHandle,
                                      computeType,
                                      scaleType,
                                      Atype,
                                      Btype,
                                      Ctype,
                                      Ctype,
                                      100,
                                      algoIdA,
                                      &nbAlgoIds));

  std::clog << std::endl << "get " << nbAlgoIds << " algoIds" << std::endl;

  for (int idx = 0; idx < nbAlgoIds; idx++) {
    cublasLtMatmulAlgo_t algo;
    std::clog << "Process algo: " << algoIdA[idx] << " ";

    /* Initialize algo structure with given Algp ID */
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoInit
    CUDA_CHECK(cublasLtMatmulAlgoInit(ltHandle,
                                      computeType,
                                      scaleType,
                                      Atype,
                                      Btype,
                                      Ctype,
                                      Ctype,
                                      algoIdA[idx],
                                      &algo));
    // Query the tiles enums supported by that algo which is used to alloc
    // enough space to store it
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoCapGetAttribute
    cublasLtMatmulTile_t tileA[CUBLASLT_MATMUL_TILE_END];
    size_t nbTiles, sizeWritten;
    CUDA_CHECK(cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(tileA), &sizeWritten));
    nbTiles = sizeWritten / sizeof(tileA[0]);

    // Query the stages enums supported by that algo (cuda must >= 11.0)
    CUDA_CHECK(cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL, 0, &sizeWritten));
    int nbStages = int(sizeWritten / sizeof(uint32_t));
    std::vector<uint32_t> stagesA(nbStages == 0 ? 1 : nbStages);
    if (nbStages == 0) {
      stagesA[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
      nbStages = 1;
    } else {
      CUDA_CHECK(cublasLtMatmulAlgoCapGetAttribute(&algo,
                                                   CUBLASLT_ALGO_CAP_STAGES_IDS,
                                                   stagesA.data(),
                                                   sizeof(uint32_t) * nbStages,
                                                   &sizeWritten));
    }

    // Retrieve Other Algo Capabilities attributes
    int32_t splitkSupport, customOptionMax;
    uint32_t redMask, swizzlingMax;
    // cublasLtMatmulInnerShape_t innerShape;
    CUDA_CHECK(
        cublasLtMatmulAlgoCapGetAttribute(&algo,
                                          CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
                                          &splitkSupport,
                                          sizeof(splitkSupport),
                                          &sizeWritten));
    std::clog << "splitkSupport: " << splitkSupport << std::endl;
    CUDA_CHECK(cublasLtMatmulAlgoCapGetAttribute(
        &algo,
        CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,
        &redMask,
        sizeof(redMask),
        &sizeWritten));
    CUDA_CHECK(cublasLtMatmulAlgoCapGetAttribute(
        &algo,
        CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,
        &swizzlingMax,
        sizeof(swizzlingMax),
        &sizeWritten));
    CUDA_CHECK(
        cublasLtMatmulAlgoCapGetAttribute(&algo,
                                          CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX,
                                          &customOptionMax,
                                          sizeof(customOptionMax),
                                          &sizeWritten));

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
              CUDA_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_TILE_ID,
                  &tileA[tileIdx],
                  sizeof(tileA[tileIdx])));
              CUDA_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_STAGES_ID,
                  &stagesA[stagesIdx],
                  sizeof(stagesA[stagesIdx])));
              CUDA_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                  &customOption,
                  sizeof(customOption)));
              CUDA_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k)));
              int splitK_val = 1;
              uint32_t redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
              CUDA_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                  &splitK_val,
                  sizeof(splitK_val)));
              CUDA_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
                  &algo,
                  CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                  &redScheme,
                  sizeof(int)));
              if (l > 0) {  // Split-K case
                splitK_val = splitKSequenceA[l - 1];
                CUDA_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
                    &algo,
                    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                    &splitKSequenceA[l - 1],
                    sizeof(splitKSequenceA[l - 1])));
                for (redScheme = 1;
                     redScheme < (int)CUBLASLT_REDUCTION_SCHEME_MASK &&
                     (AlgoCount < AlgoCombinations);
                     redScheme <<= 1) {
                  CUDA_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
                      &algo,
                      CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                      &redScheme,
                      sizeof(redScheme)));

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
  std::clog << "Got " << AlgoCount << " algos" << std::endl;
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  std::vector<customMatmulPerf_t> perfResults(AlgoCount);
  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&stopEvent));
  for (int i = 0; i < AlgoCount; i++) {
    TestMatmulRun<InT, OutT, ScaleT>(ltHandle,
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
    algos[i].workspaceSize = perfResults[i].workspaceSize;
    algos[i].time = perfResults[i].time;
    if (perfResults[i].status != CUBLAS_STATUS_SUCCESS) {
      std::clog << "algo " << algos[i].algoId << " tile " << algos[i].tile
                << " stages " << algos[i].stages << " splitK_val "
                << algos[i].splitK_val << std::endl;
      algos[i].time = std::numeric_limits<float>::max();
      std::cerr << " TestMatmulRun with status " << perfResults[i].status
                << std::endl;
      continue;
    }
  }
  std::sort(algos.begin(), algos.end(), time_compare_algo_para);
  int i = 0;
  while (algos[i].time == 0) i++;
  std::ofstream outfile;
  outfile.open(path, std::ios::app);
  outfile << m << "," << k << "," << n << "," << algos[i].algoId << ","
          << algos[i].swizzle << "," << algos[i].customOption << ","
          << algos[i].tile << "," << algos[i].splitK_val << ","
          << algos[i].reductionScheme << "," << algos[i].stages << ","
          << algos[i].workspaceSize << "," << algos[i].time << "\n";
  outfile.close();
}

class DevContext {};

class CPUContext : public DevContext {};

class CUBLASLTContext : public DevContext {
public:
  CUBLASLTContext() { CUDA_CHECK(cublasLtCreate(&handle)); }

  cublasLtHandle_t handle;
};

template <typename InT, typename OutT, typename DevContext>
void GEMMInt8(const DevContext& dev_ctx,
              const std::vector<InT>& A,
              const std::vector<InT>& B,
              std::vector<OutT>& C,
              int m,
              int k,
              int n,
              bool is_test,
              bool is_read_from_file = false,
              const std::string& path = "search.csv") {
  std::cerr << "Base Class is not implemented" << std::endl;
}

template <>
void GEMMInt8<int8_t, int32_t, CPUContext>(const CPUContext& dev_ctx,
                                           const std::vector<int8_t>& A,
                                           const std::vector<int8_t>& B,
                                           std::vector<int32_t>& C,
                                           int m,
                                           int k,
                                           int n,
                                           bool is_test,
                                           bool is_read_from_file,
                                           const std::string& path) {
  std::cerr << "CPUContext Class is not implemented" << std::endl;
}

template <>
void GEMMInt8<int8_t, int32_t, CUBLASLTContext>(const CUBLASLTContext& dev_ctx,
                                                const std::vector<int8_t>& AVec,
                                                const std::vector<int8_t>& BVec,
                                                std::vector<int32_t>& CVec,
                                                int m,
                                                int k,
                                                int n,
                                                bool is_test,
                                                bool is_read_from_file,
                                                const std::string& path) {
  int8_t* A_dev;
  int8_t* B_dev;
  int32_t* C_dev;
  char* workSpace;

  CUDA_CHECK(cudaMalloc((void**)&A_dev, AVec.size() * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc((void**)&B_dev, BVec.size() * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc((void**)&C_dev, m * n * sizeof(int32_t)));
  CUDA_CHECK(
      cudaMemcpy(A_dev, AVec.data(), AVec.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(B_dev, BVec.data(), BVec.size(), cudaMemcpyHostToDevice));

  // init data structure
  cublasLtMatmulDesc_t matmul_desc;
  cublasLtMatrixLayout_t A_desc;
  cublasLtMatrixLayout_t B_desc;
  cublasLtMatrixLayout_t C_desc;
  int32_t alpha = 1;
  int32_t beta = 0;

  cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32I;
  CUDA_CHECK(
      cublasLtMatmulDescCreate(&matmul_desc, cudaComputeType, CUDA_R_32I));
  cublasOperation_t op_transpose = CUBLAS_OP_T;
  CUDA_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc,
                                            CUBLASLT_MATMUL_DESC_TRANSA,
                                            &op_transpose,
                                            sizeof(op_transpose)));
  CUDA_CHECK(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_8I, k, n, k));
  CUDA_CHECK(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_8I, k, m, k));
  CUDA_CHECK(cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_32I, n, m, n));

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
    FindAlgo(dev_ctx.handle,
             m,
             n,
             k,
             B_dev,
             A_dev,
             C_dev,
             matmul_desc,
             B_desc,
             A_desc,
             C_desc,
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
  } else if (is_read_from_file) {
    int m_tmp, k_tmp, n_tmp;
    std::ifstream file(path);
    if (!file.is_open()) {
      std::cout << "file not open. Now we use default params" << std::endl;
      using_default_config();
    } else {
      bool match = false;
      int find_cnt = 0;
      std::string line;
      while (std::getline(file, line)) {
        std::istringstream iss(line);
        char comma;
        if (iss >> m_tmp >> comma >> k_tmp >> comma >> n_tmp >> comma >>
            algoId >> comma >> swizzle >> comma >> customOption >> comma >>
            tile >> comma >> splitK_val >> comma >> reductionScheme >> comma >>
            stages >> comma >> work_space_size >> comma >> time_ref) {
          if (k_tmp == k && n_tmp == n && m <= m_tmp) {
            match = true;
            break;
          }
          find_cnt++;
        }
      }
      if (find_cnt == 0) {
        std::cout << "the file is empty. Now we use default params"
                  << std::endl;
        using_default_config();
      }
    }
  } else {
    std::cout << "Please use test mode to select\n, Now we use default params"
              << std::endl;
    using_default_config();
  }

  CUDA_CHECK(cudaMalloc((void**)&workSpace, work_space_size));

  CUDA_CHECK(cublasLtMatmulAlgoInit(dev_ctx.handle,
                                    cudaComputeType,
                                    CUDA_R_32I,
                                    CUDA_R_8I,
                                    CUDA_R_8I,
                                    CUDA_R_32I,
                                    CUDA_R_32I,
                                    algoId,
                                    &algo));
  CUDA_CHECK(
      cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                           CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                           &(customOption),
                                           sizeof(customOption)));
  CUDA_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile)));
  CUDA_CHECK(
      cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                           CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                           &(splitK_val),
                                           sizeof(splitK_val)));
  CUDA_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle)));
  CUDA_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
      &algo,
      CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
      &(reductionScheme),
      sizeof(int)));
  CUDA_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages)));

  auto start = std::chrono::high_resolution_clock::now();
  const int repeats = 10;
  for (int loop = 0; loop < repeats; loop++) {
    CUDA_CHECK(cublasLtMatmul(dev_ctx.handle,
                              matmul_desc,
                              &alpha,
                              B_dev,
                              B_desc,
                              A_dev,
                              A_desc,
                              &beta,
                              C_dev,
                              C_desc,
                              C_dev,
                              C_desc,
                              &algo,
                              //  nullptr,
                              workSpace,
                              // 0,
                              work_space_size,
                              0));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  double time = diffTime(start, end);
  auto now = std::chrono::system_clock::now();
  std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
  std::tm now_tm = *std::localtime(&now_time_t);

  std::cout << "GEMM with cublaslt imma1 int8 spend " << time / repeats
            << " ms in " << m << ", " << k << ", " << n
            << ", current time: " << std::put_time(&now_tm, "%H:%M:%S")
            << std::endl;
  CUDA_CHECK(cudaFree(A_dev));
  CUDA_CHECK(cudaFree(B_dev));
  CUDA_CHECK(cudaFree(C_dev));
  CUDA_CHECK(cudaFree(workSpace));
}

void TuneCublasltGemm(const paddle::Tensor& K,
                      const paddle::Tensor& N,
                      const int M_start,
                      const int M_end,
                      const std::string& dtype,
                      const bool is_test,
                      const bool is_read_from_file,
                      const std::string& path) {
  assert(M_end >= M_start);
  assert(M_start >= 1);
  assert(K.dims().size() == 1 && N.dims().size() == 1);
  assert(is_test != is_read_from_file);

  auto K_cpu = K.copy_to(paddle::CPUPlace(), false);
  auto N_cpu = N.copy_to(paddle::CPUPlace(), false);
  int64_t* K_data = K_cpu.data<int64_t>();
  int64_t* N_data = N_cpu.data<int64_t>();

  int K_size = K.numel();
  int N_size = N.numel();
  assert(K_size == N_size);

  std::vector<int> mm;
  int m = M_start, step = 1;
  while (m <= M_end) {
    // update step
    if (m >= 8192) {
      step = 4096;
    } else if (m >= 1024) {
      step = 1024;
    } else if (m >= 512) {
      step = 128;
    } else if (m >= 256) {
      step = 64;
    } else if (m >= 64) {
      step = 32;
    } else if (m >= 16) {
      step = 16;
    } else if (m >= 4) {
      step = 4;
    } else {
      step = 1;
    }
    mm.push_back(m);
    m += step;
  }

  for (int j = 0; j < mm.size(); j++) {
    int m = mm[j];
    for (int i = 0; i < K_size; ++i) {
      int n = (int)N_data[i];
      int k = (int)K_data[i];
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
                 is_test,           /*is_test*/
                 is_read_from_file, /*is_read_from_file*/
                 path);
      } else {
        // other dtype
        throw std::runtime_error(dtype + "not currently supported");
      }
    }
  }
}

PD_BUILD_OP(tune_cublaslt_gemm)
    .Inputs({"K", "N"})
    .Outputs({})
    .Attrs({"M_start: int",
            "M_end: int",
            "dtype: std::string",
            "is_test: bool",
            "is_read_from_file: bool",
            "path: std::string"})
    .SetKernelFn(PD_KERNEL(TuneCublasltGemm));