#pragma once

#include "fastertransformer/common.h"
#include "paddle/fluid/platform/float16.h"

using namespace fastertransformer;
namespace paddle {
template <typename T>
class PDTraits;

template <>
class PDTraits<float> {
public:
  typedef float DataType;
  static const OperationType OpType = OperationType::FP32;
};

template <>
class PDTraits<platform::float16> {
public:
  typedef half DataType;
  static const OperationType OpType = OperationType::FP16;
};

}  // namespace paddle
