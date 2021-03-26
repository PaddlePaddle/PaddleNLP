#pragma once

#include "fastertransformer/common.h"
#include "fastertransformer/decoding_beamsearch.h"
#include "fastertransformer/decoding_sampling.h"
#include "fastertransformer/open_decoder.h"

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class NotImpleKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW("CPU is not support for this kernel now. Please use GPU. ");
  }
};
}  // namespace operators
}  // namespace paddle
