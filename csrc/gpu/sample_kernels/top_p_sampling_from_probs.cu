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

#include "helper.h"
#include "sample_kernels/sampling.cuh"

std::vector<paddle::Tensor> TopPSamplingReject(const paddle::Tensor& probs,
                                               const paddle::Tensor& top_p) {
  std::vector<int64_t> probs_shape = probs.shape();
  unsigned int batch_size = probs_shape[0];
  unsigned int vocab_size = probs_shape[1];

  // default is 32
  unsigned int max_top_p_rounds = 32;
  std::vector<int64_t> uniform_samples_shape = {batch_size, max_top_p_rounds};
  paddle::Tensor uniform_samples = paddle::experimental::uniform(
      uniform_samples_shape, paddle::DataType::FLOAT32, 0, 1, 0, probs.place());

  // todo: add parameter for deterministic, now default is true
  bool deterministic = true;
  paddle::Tensor probs_input;

  probs_input = paddle::experimental::cast(probs, paddle::DataType::FLOAT32);
  auto cu_stream = probs.stream();

  auto samples =
      paddle::full({batch_size}, 0, paddle::DataType::INT32, probs.place());
  auto success =
      paddle::full({batch_size}, 0, paddle::DataType::BOOL, probs.place());

  cudaError_t status =
      sampling::TopPSamplingFromProb<float, int>(probs_input.data<float>(),
                                                 uniform_samples.data<float>(),
                                                 samples.data<int>(),
                                                 success.data<bool>(),
                                                 nullptr,
                                                 batch_size,
                                                 top_p.data<float>(),
                                                 vocab_size,
                                                 max_top_p_rounds,
                                                 deterministic,
                                                 cu_stream);
  PD_CHECK(status == cudaSuccess,
           "SamplingFromProbs failed with error code " +
               std::string(cudaGetErrorString(status)));

  paddle::Tensor samples_output;
  samples_output = paddle::experimental::cast(samples, paddle::DataType::INT64);
  return {samples_output};
}

std::vector<std::vector<int64_t>> TopPSamplingRejectInferShape(
    const std::vector<int64_t>& probs_shape,
    const std::vector<int64_t>& top_p_shape) {
  int64_t bs = probs_shape[0];
  return {{bs, 1}};
}

std::vector<paddle::DataType> TopPSamplingRejectInferDtype(
    const paddle::DataType& probs_dtype, const paddle::DataType& top_p_shape) {
  return {probs_dtype};
}

PD_BUILD_OP(top_p_sampling_reject)
    .Inputs({"probs", "top_p"})
    .Outputs({"samples"})
    .SetKernelFn(PD_KERNEL(TopPSamplingReject))
    .SetInferShapeFn(PD_INFER_SHAPE(TopPSamplingRejectInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TopPSamplingRejectInferDtype));
