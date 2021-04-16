#pragma once

#include "fastertransformer/common.h"

using namespace fastertransformer;

template <paddle::DataType D>
class PDTraits;

template <>
class PDTraits<paddle::DataType::FLOAT32> {
public:
  typedef float DataType;
  typedef float data_t;
  static const OperationType OpType = OperationType::FP32;
};

template <>
class PDTraits<paddle::DataType::FLOAT16> {
public:
  typedef half DataType;
  typedef paddle::float16 data_t;
  static const OperationType OpType = OperationType::FP16;
};
