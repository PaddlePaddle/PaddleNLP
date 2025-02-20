
#pragma once

#include <cuda_bf16.h>
#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include <cuda_bf16.h>

#include "utils.h"

using paddle::Tensor;
using namespace tensorrt_llm::kernels::cutlass_kernels;

std::vector<Tensor> batch_symmetric_quantize(
    const paddle::Tensor& weight, const std::string& quant_method="none")
{
    
    const auto _st = weight.dtype();
    QuantType quant_type
            = (quant_method == "weight_only_int4") ? QuantType::W4_A16 : QuantType::W8_A16;

    // std::cout << "1"<< std::endl;

    auto weight_dims = weight.dims();
    const size_t num_experts = weight_dims.size() == 2 ? 1 : weight_dims[0];

    const size_t num_rows = weight_dims[weight_dims.size()-2];
    const size_t num_cols = weight_dims[weight_dims.size()-1];

    const size_t bits_in_type = get_weight_quant_bits(quant_type);
    const size_t bytes_per_out_col = num_cols * bits_in_type / 8;

    const size_t input_mat_size = num_rows * num_cols;
    const size_t quantized_mat_size = num_rows * bytes_per_out_col;

    std::vector<int64_t> quantized_weight_shape;
    std::vector<int64_t> scale_shape;

    if (weight_dims.size() == 2)
    {
        quantized_weight_shape = {int64_t(num_rows), int64_t(bytes_per_out_col)};
        scale_shape = {int64_t(num_cols)};
    }
    else if (weight_dims.size() == 3)
    {
        quantized_weight_shape = {int64_t(num_experts), int64_t(num_rows), int64_t(bytes_per_out_col)};
        scale_shape = {int64_t(num_experts), int64_t(num_cols)};
    }

    // TODO(dastokes) This should be removed if Grouped GEMM is updated to not need interleaved input
    bool force_interleave = weight_dims.size() == 3;

    Tensor unprocessed_quantized_weight
        = paddle::empty(quantized_weight_shape, paddle::DataType::INT8, paddle::CPUPlace());

    Tensor processed_quantized_weight = paddle::empty_like(unprocessed_quantized_weight);

    Tensor scales = paddle::empty(scale_shape, weight.dtype(), paddle::CPUPlace());

    // std::cout << "3"<< std::endl;
    int8_t* unprocessed_quantized_weight_ptr = get_ptr<int8_t>(unprocessed_quantized_weight);
    int8_t* processed_quantized_weight_ptr = get_ptr<int8_t>(processed_quantized_weight);


    if (_st == paddle::DataType::FLOAT32)
    {
        symmetric_quantize<float, float>(processed_quantized_weight_ptr, unprocessed_quantized_weight_ptr,
            reinterpret_cast<float*>(scales.data<float>()), const_cast<float*>(weight.data<float>()), {num_experts, num_rows, num_cols}, quant_type,
            force_interleave);
    }
    else if (_st == paddle::DataType::FLOAT16)
    {
        symmetric_quantize<half, half>(processed_quantized_weight_ptr, unprocessed_quantized_weight_ptr,
            reinterpret_cast<half*>(scales.data<paddle::float16>()), reinterpret_cast<half*>(const_cast<paddle::float16*>(weight.data<paddle::float16>())), {num_experts, num_rows, num_cols}, quant_type,
            force_interleave);
    }
#ifdef ENABLE_BF16
    else if (_st == paddle::DataType::BFLOAT16)
    {
        // std::cout << "4"<< std::endl;
        symmetric_quantize<__nv_bfloat16, __nv_bfloat16>(processed_quantized_weight_ptr,
            unprocessed_quantized_weight_ptr, reinterpret_cast<__nv_bfloat16*>(scales.data<paddle::bfloat16>()), reinterpret_cast<__nv_bfloat16*>(const_cast<paddle::bfloat16*>(weight.data<paddle::bfloat16>())),
            {num_experts, num_rows, num_cols}, quant_type, force_interleave);
    }
#endif
    // std::cout << "5"<< std::endl;
    // unprocessed_quantized_weight_ptr for debug
    return {processed_quantized_weight, scales};
}

PD_BUILD_OP(batch_symmetric_quantize)
    .Inputs({"weight"})
    .Outputs({"processed_quantized_weight", "scales"})
    .Attrs({"quant_method:std::string"})
    .SetKernelFn(PD_KERNEL(batch_symmetric_quantize));