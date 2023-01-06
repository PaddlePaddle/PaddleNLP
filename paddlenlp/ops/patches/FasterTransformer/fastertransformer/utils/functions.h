/*
 * Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

namespace fastertransformer {

// for int8 cublasLtMM with algo
// ATransform should be m*k, CUBLASLT_ORDER_COL32
// kernel should be n*k, CUBLASLT_ORDER_COL4_4R2_8C or
// CUBLASLT_ORDER_COL32_2R_4R4
// res is m*n, CUBLASLT_ORDER_COL32
template <typename T>
void cublasLtMM_INT8_withAlgo(
    int32_t *res,
    int batchCount,
    int m,
    int n,
    int k,
    int64_t stridea,
    int64_t strideb,
    int64_t stridec,
    const T *ATransform,
    const T *kernel,
    cublasLtHandle_t cublasLt_handle,
    cudaStream_t stream,
    std::map<std::string, cublasLtMatmulAlgo_info> &cublasLtAlgoMap,
    bool use_ORDER_COL32_2R_4R4) {
  cublasOperation_t opTranspose = CUBLAS_OP_T;
#ifdef CUDA11_MODE
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t computeType = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t AtransformDesc = NULL;
  cublasLtMatrixLayout_t BtransformDesc = NULL;
  cublasLtMatrixLayout_t CtransformDesc = NULL;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;

  cublasLtOrder_t order_matrixB;
#ifdef CUDA11_MODE
  if (use_ORDER_COL32_2R_4R4)
    order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
  else
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#else
  order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#endif

  int ldaTransform = 32 * m;
  int ldbTransform;
  if (use_ORDER_COL32_2R_4R4)
    ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
  else
    ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;
  int ldcTransform = 32 * m;

// create matmulDesc
#ifdef CUDA11_MODE
  cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32I);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc,
                                 CUBLASLT_MATMUL_DESC_TRANSB,
                                 &opTranspose,
                                 sizeof(cublasOperation_t));
  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc,
                                   CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32,
                                   sizeof(order_COL32));
  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
  cublasLtMatrixLayoutSetAttribute(BtransformDesc,
                                   CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_matrixB,
                                   sizeof(order_matrixB));
  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldcTransform);
  cublasLtMatrixLayoutSetAttribute(CtransformDesc,
                                   CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32,
                                   sizeof(order_COL32));
  if (batchCount > 1) {
    cublasLtMatrixLayoutSetAttribute(AtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount,
                                     sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        AtransformDesc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stridea,
        sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount,
                                     sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &strideb,
        sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount,
                                     sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        CtransformDesc,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stridec,
        sizeof(stridec));
  }

  int alphaI = 1;
  int betaI = 0;

  // get algo
  cublasLtMatmulAlgo_t algo;
  char mark[1000];
  sprintf(mark, "%d_%d_%d_%d_%d", batchCount, m, n, k, INT8_DATATYPE);
  std::string markStr(mark);
  int findAlgo = 0;
  if (cublasLtAlgoMap.find(markStr) != cublasLtAlgoMap.end() &&
      cublasLtAlgoMap[markStr].workspaceSize == 0) {
    // printf("find algo %s\n", markStr.c_str());
    findAlgo = 1;

    cublasLtMatmulAlgoInit(cublasLt_handle,
                           computeType,
                           CUDA_R_32I,
                           CUDA_R_8I,
                           CUDA_R_8I,
                           CUDA_R_32I,
                           CUDA_R_32I,
                           cublasLtAlgoMap[markStr].algoId,
                           &algo);
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
        &(cublasLtAlgoMap[markStr].customOption),
        sizeof(cublasLtAlgoMap[markStr].customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                         CUBLASLT_ALGO_CONFIG_TILE_ID,
                                         &(cublasLtAlgoMap[markStr].tile),
                                         sizeof(cublasLtAlgoMap[markStr].tile));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        &(cublasLtAlgoMap[markStr].splitK_val),
        sizeof(cublasLtAlgoMap[markStr].splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        &(cublasLtAlgoMap[markStr].swizzle),
        sizeof(cublasLtAlgoMap[markStr].swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(cublasLtAlgoMap[markStr].reductionScheme),
        sizeof(int));
#ifdef CUDA11_MODE
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_STAGES_ID,
        &(cublasLtAlgoMap[markStr].stages),
        sizeof(cublasLtAlgoMap[markStr].stages));
#endif
  } else {
    findAlgo = 1;
    int algoId;
    if (use_ORDER_COL32_2R_4R4) {
      algoId = 7;
    } else {
      algoId = 6;
    }
    int swizzle = 0;
    int customOption = 0;
    int tile = 20;
    int splitK_val = 0;
    int reductionScheme = 0;
    cublasLtMatmulAlgoInit(cublasLt_handle,
                           computeType,
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
#ifdef CUDA11_MODE
    int stages;
    if (use_ORDER_COL32_2R_4R4)
      stages = 15;
    else
      stages = 13;
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
#endif
  }

  check_cuda_error(cublasLtMatmul(cublasLt_handle,
                                  matmulDesc,
                                  &alphaI,
                                  ATransform,
                                  AtransformDesc,
                                  kernel,
                                  BtransformDesc,
                                  &betaI,
                                  res,
                                  CtransformDesc,
                                  res,
                                  CtransformDesc,
                                  (findAlgo == 1 ? (&algo) : NULL),
                                  NULL,
                                  0,
                                  stream));

  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixLayoutDestroy(AtransformDesc);
  cublasLtMatrixLayoutDestroy(BtransformDesc);
  cublasLtMatrixLayoutDestroy(CtransformDesc);
}

// used in decoder
template <typename T>
void cublasMM_cublasLtMM_wrapper_int8_decoder(cublasLtHandle_t ltHandle,
                                              cublasHandle_t handle,
                                              cublasOperation_t transa,
                                              cublasOperation_t transb,
                                              int m,
                                              int n,
                                              int k,
                                              const void *alpha,
                                              const void *A,
                                              cudaDataType_t Atype,
                                              int lda,
                                              const void *B,
                                              cudaDataType_t Btype,
                                              int ldb,
                                              const void *beta,
                                              T *C,
                                              cudaDataType_t Ctype,
                                              int ldc,
                                              cudaStream_t stream,
                                              void *cublas_workspace) {
  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cudaDataType_t scaleType;

#ifdef CUDA11_MODE
  cublasComputeType_t computeType;
#else
  cudaDataType_t computeType;
#endif

#ifdef CUDA11_MODE
  computeType = CUBLAS_COMPUTE_32I;
#else
  computeType = CUDA_R_32I;
#endif
  scaleType = CUDA_R_32I;

  // --------------------------------------
  // Create descriptors for the original matrices
  cublasLtMatrixLayoutCreate(&Adesc,
                             Atype,
                             transa == CUBLAS_OP_N ? m : k,
                             transa == CUBLAS_OP_N ? k : m,
                             lda);
  cublasLtMatrixLayoutCreate(&Bdesc,
                             Btype,
                             transb == CUBLAS_OP_N ? k : n,
                             transb == CUBLAS_OP_N ? n : k,
                             ldb);
  cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc);

#ifdef CUDA11_MODE
  cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
  cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif

  cublasLtMatmulDescSetAttribute(operationDesc,
                                 CUBLASLT_MATMUL_DESC_TRANSA,
                                 &transa,
                                 sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(operationDesc,
                                 CUBLASLT_MATMUL_DESC_TRANSB,
                                 &transb,
                                 sizeof(cublasOperation_t));

  cublasLtMatmulAlgo_t algo;
  void *workSpace = cublas_workspace;
  int workspaceSize = cublas_workspace == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;

  check_cuda_error(cublasLtMatmul(ltHandle,
                                  operationDesc,
                                  alpha,
                                  A,
                                  Adesc,
                                  B,
                                  Bdesc,
                                  beta,
                                  C,
                                  Cdesc,
                                  C,
                                  Cdesc,
                                  NULL,
                                  workSpace,
                                  workspaceSize,
                                  stream));

  cublasLtMatmulDescDestroy(operationDesc);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
}
}
