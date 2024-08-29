// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <omp.h>

#include <cstdio>
#include <iostream>

#include "paddle/extension.h"

void GreedySearch(const float *probs,
                  int64_t *next_token_ids,
                  int bsz,
                  int vocab_size) {
  int numThreads = 0;
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    if (tid == 0) {
      numThreads = omp_get_num_threads();
    }
  }
  // Max ID and value for each sample
  // std::vector<int> maxIds(batchSize);
  float maxVals[bsz];

  // Small batch size (each sample can have at least 2 threads)
  if (numThreads / bsz >= 2) {
    int thrPerSample = numThreads / bsz;
    int sizePerThr = (vocab_size + thrPerSample - 1) / thrPerSample;
    int maxIndices[bsz * thrPerSample];
    float maxValues[bsz * thrPerSample];

    // TODO: if size is small, possible to cause out of boundary
#pragma omp parallel for collapse(2)
    for (int b = 0; b < bsz; ++b) {
      for (int t = 0; t < thrPerSample;
           ++t) {  // thread index inside the sample
        int start = t * sizePerThr;
        int end = (start + sizePerThr) > vocab_size ? vocab_size
                                                    : (start + sizePerThr);
        const float *p = probs + b * vocab_size;

        int maxIdx = start;
        float maxVal = p[start];
        for (int off = start + 1; off < end; ++off) {
          if (p[off] > maxVal) {
            maxVal = p[off];
            maxIdx = off;
          }
        }

        // False sharing happens, but since only one time, not avoided
        maxIndices[b * thrPerSample + t] = maxIdx;
        maxValues[b * thrPerSample + t] = maxVal;
      }
    }

    // Local reduction
    for (int i = 0; i < bsz; ++i) {
      int *pIndices = maxIndices + i * thrPerSample;
      float *pValues = maxValues + i * thrPerSample;
      int maxIdx = pIndices[0];
      float maxVal = pValues[0];
      for (int j = 1; j < thrPerSample; ++j) {
        if (pValues[j] > maxVal) {
          maxVal = pValues[j];
          maxIdx = pIndices[j];
        }
      }
      next_token_ids[i] = maxIdx;
      maxVals[i] = maxVal;
    }
  }

  // Each thread handle one sample (one row)
  else {
#pragma omp parallel for
    for (int i = 0; i < bsz; ++i) {
      int maxId = 0;
      const float *p = probs + i * vocab_size;
      float maxVal = p[0];
      for (int j = 1; j < vocab_size; ++j) {
        if (p[j] > maxVal) {
          maxVal = p[j];
          maxId = j;
        }
      }
      next_token_ids[i] = maxId;
      maxVals[i] = maxVal;
    }
  }
  return;
}
std::vector<paddle::Tensor> XftGreedySearch(const paddle::Tensor &probs) {
  const int bsz = probs.shape()[0];
  const int vocab_size = probs.shape()[1];
  auto next_tokens =
      paddle::empty({bsz, 1}, paddle::DataType::INT64, probs.place());
  GreedySearch(probs.data<float>(),
               const_cast<int64_t *>(next_tokens.data<int64_t>()),
               bsz,
               vocab_size);
  return {next_tokens};
}
std::vector<std::vector<int64_t>> XftGreedySearchInferShape(
    const std::vector<int64_t> &probs_shape) {
  int64_t bsz = probs_shape[0];
  return {{bsz, 1}};
}
std::vector<paddle::DataType> XftGreedySearchInferDtype(
    const paddle::DataType &probs_dtype) {
  return {paddle::DataType::INT64};
}
PD_BUILD_OP(xft_greedy_search)
    .Inputs({"probs"})
    .Outputs({"next_tokens_ids"})
    .SetInferShapeFn(PD_INFER_SHAPE(XftGreedySearchInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(XftGreedySearchInferDtype))
    .SetKernelFn(PD_KERNEL(XftGreedySearch));
