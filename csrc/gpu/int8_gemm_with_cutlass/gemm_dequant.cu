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

#include "gemm_dequant.h"
#include "cutlass_helper.h"

template <paddle::DataType D, typename T>
void RunGemmDequant(const int8_t* a,
                    const int8_t* b,  // Transposed
                    const float* dequant_scale,
                    T* c,
                    int m,
                    int k,
                    int n,
                    cudaStream_t stream) {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = typename CutlassDtypeTraits<D>::DataType;
  using ElementCompute = int32_t;
  using ElementD = ElementC;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;

  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;

  static int const kStages = 5;

  /// Linear scaling operator
  using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementCompute,
      ElementCompute>;

  using GemmDequantT = cutlass::GemmDequant<ElementA,
                                            LayoutA,
                                            ElementB,
                                            LayoutB,
                                            ElementC,
                                            ElementCompute,
                                            OperatorClass,
                                            ArchTag,
                                            ThreadblockShape,
                                            WarpShape,
                                            InstructionShape,
                                            EpilogueFunctorOp,
                                            kStages>;

  using LayoutC = typename GemmDequantT::LayoutC;

  int64_t lda = LayoutA::packed({m, k}).stride(0);
  int64_t ldb = LayoutB::packed({k, n}).stride(0);
  int64_t ldc = LayoutC::packed({m, n}).stride(0);

  cutlass::gemm::GemmCoord problem_size(m, n, k);

  typename CutlassDtypeTraits<D>::DataType* c_tmp = nullptr;
  typename CutlassDtypeTraits<D>::DataType* d =
      reinterpret_cast<typename CutlassDtypeTraits<D>::DataType*>(c);

  typename GemmDequantT::TensorRefA ref_a(const_cast<int8_t*>(a), lda);
  typename GemmDequantT::TensorRefB ref_b(const_cast<int8_t*>(b), ldb);
  typename GemmDequantT::TensorRefC ref_c(c_tmp, ldc);
  typename GemmDequantT::TensorRefC ref_d(d, ldc);
  typename GemmDequantT::TensorRefScale ref_scale(
      const_cast<float*>(dequant_scale), 0);

  typename GemmDequantT::Arguments args(
      problem_size,
      ref_a,
      ref_b,
      ref_c,
      ref_d,
      ref_scale,
      {ElementCompute(1.0f), ElementCompute(0.0f)});

  GemmDequantT gemm;
  // Initialize
  auto status = gemm.initialize(args);
  PD_CHECK(status == cutlass::Status::kSuccess, "cutlass GemmDequant initialize error");

  // Run
  status = gemm(stream);
  PD_CHECK(status == cutlass::Status::kSuccess, "cutlass GemmDequant runtime error");
}

std::vector<paddle::Tensor> GemmDequant(const paddle::Tensor& x,
                                            const paddle::Tensor& y,
                                            const paddle::Tensor& scale,
                                            const std::string& out_dtype) {
  std::vector<int64_t> x_dims = x.shape(), y_dims = y.shape();
  PD_CHECK(x_dims[x_dims.size() - 1] == y_dims[y_dims.size() - 1], 
        "The last dimension of x and y should be equal. But received x[%d] != y[%d].",
        "Ensure that x is not transposed and y is transposed.",
        x_dims[x_dims.size() - 1],
        y_dims[y_dims.size() - 1]);
  int64_t m = x_dims[x_dims.size() - 2];
  int64_t k = x_dims[x_dims.size() - 1];
  int64_t n = y_dims[y_dims.size() - 2];
  if (out_dtype == "bfloat16") {
    paddle::Tensor out = paddle::empty({m, n}, paddle::DataType::BFLOAT16, x.place());
    RunGemmDequant<paddle::DataType::BFLOAT16, paddle::bfloat16>(x.data<int8_t>(),
                                         y.data<int8_t>(),
                                         scale.data<float>(),
                                         out.data<paddle::bfloat16>(),
                                         m,
                                         k,
                                         n,
                                         x.stream());
    return {out};
  } else if (out_dtype == "float16") {
    paddle::Tensor out = paddle::empty({m, n}, paddle::DataType::FLOAT16, x.place());
    RunGemmDequant<paddle::DataType::FLOAT16, paddle::float16>(x.data<int8_t>(),
                                        y.data<int8_t>(),
                                        scale.data<float>(),
                                        out.data<paddle::float16>(),
                                        m,
                                        k,
                                        n,
                                        x.stream());
    return {out};
  } else {
    PADDLE_THROW(
      phi::errors::InvalidArgument("only support bfloat16 and float16, but got %s", out_dtype));
  }
}

std::vector<std::vector<int64_t>> GemmDequantShape(const std::vector<int64_t>& x,
                                                const std::vector<int64_t>& y,
                                                const std::vector<int64_t>& scale,
                                                const std::string& out_dtype) {
    return {{x[x.size() - 2], y[y.size() - 2]}};
}

std::vector<paddle::DataType> GemmDequantDtype(const paddle::DataType& x,
                                            const paddle::DataType& y,
                                            const paddle::DataType& scale,
                                            const std::string& out_dtype) {
  if (out_dtype == "bfloat16") {
    return {paddle::DataType::BFLOAT16};
  } else if (out_dtype == "float16") {
    return {paddle::DataType::FLOAT16};
  } else {
    PADDLE_THROW(
      phi::errors::InvalidArgument("only support bfloat16 and float16, but got %s", out_dtype));
  }
}

PD_BUILD_OP(gemm_dequant)
    .Inputs({"x" /* transpose_x:false */, "y" /* transpose_y:true */, "scale"})
    .Outputs({"out"})
    .Attrs({"out_dtype: std::string"})
    .SetKernelFn(PD_KERNEL(GemmDequant))
    .SetInferShapeFn(PD_INFER_SHAPE(GemmDequantShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GemmDequantDtype));

