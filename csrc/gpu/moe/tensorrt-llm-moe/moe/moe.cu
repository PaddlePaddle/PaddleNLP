// #include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDACachingAllocator.h>

#pragma once

#include <optional>
#include <algorithm>

#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "utils.h"
#include <cstdio>


template <typename T>
void print_gpu_data(T* gpu_data, size_t num_elements, size_t num) {
    float* host_data = new float[num_elements];
    T* temp_data = new T[num_elements];
    cudaError_t err = cudaMemcpy(temp_data, gpu_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }
    for (size_t i = 0; i < num_elements; i++) {
        host_data[i] = static_cast<float>(temp_data[i]);
    }
    for (size_t i = num; i < num + 1000; i++) {
        printf("gpu_data ？？？ [%zu] = %f\n", i, host_data[i]);
    }

    // 释放内存
    delete[] host_data;
    delete[] temp_data;
}


using paddle::Tensor;

int getSMVersion() {
    int device = -1;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.major * 10 + props.minor;
}


template<typename T, typename WeightType, typename OutputType = T>
std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> getFilteredConfigs(
    tensorrt_llm::kernels::CutlassMoeFCRunner<T, WeightType, OutputType>& moe_runner, int sm) {
    auto tactics = moe_runner.getTactics();
    if (sm == 89) {
        // Filter some unsupported configs for L40S
        auto it = std::remove_if(tactics.begin(), tactics.end(),
            [&](auto conf) {
                using tensorrt_llm::cutlass_extensions::CutlassTileConfig;
                auto checks = std::vector{
                    // Fail for BF16/FP16
                    conf.tile_config == CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64,
                    conf.tile_config == CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64 && conf.stages == 4,
                    // Fail for FP8
                    false && conf.tile_config == CutlassTileConfig::CtaShape16x256x128_WarpShape16x64x128
                        && conf.stages >= 3,
                };

                return std::any_of(checks.begin(), checks.end(), [](auto v) { return v; });
            });
        tactics.erase(it, tactics.end());
    }

    if (tactics.empty()) {
        throw std::runtime_error("No valid GEMM tactics found");
    }

    return tactics;
}


// 第三个模版参数默认是T
template<typename T, typename WeightType, typename OutputType = T>
std::pair<tensorrt_llm::cutlass_extensions::CutlassGemmConfig, tensorrt_llm::cutlass_extensions::CutlassGemmConfig> 
selectTacticsForArch(tensorrt_llm::kernels::CutlassMoeFCRunner<T, WeightType, OutputType>& moe_runner, int sm) {
    bool is_sm90 = sm >= 90;
    auto tactics = getFilteredConfigs(moe_runner, sm);
    auto it = std::find_if(tactics.begin(), tactics.end(), [is_sm90](auto& c) { return c.is_sm90 == is_sm90; });
    if (it == tactics.end()) {
        // Fall back to any tactic
        std::cout << "WARNING: Could not find config for sm version " << sm << std::endl;
        return std::make_pair(tactics[0], tactics[0]);
    }

    return std::make_pair(*it, *it);
}



tensorrt_llm::ActivationType getActivationType(std::string activation_type_str)
{
    if (activation_type_str == "Gelu" || activation_type_str == "gelu") {
        return tensorrt_llm::ActivationType::Gelu;
    }
    else if (activation_type_str == "Relu" || activation_type_str == "relu") {
        return tensorrt_llm::ActivationType::Relu;
    }
    else if (activation_type_str == "Silu" || activation_type_str == "silu") {
        return tensorrt_llm::ActivationType::Silu;
    }
    else if (activation_type_str == "GeGLU" || activation_type_str == "geglu" || activation_type_str == "gated-gelu") {
        return tensorrt_llm::ActivationType::Geglu;
    }
    else if (activation_type_str == "Swiglu") {
        return tensorrt_llm::ActivationType::Swiglu;
    }
    else {
        std::cout << "Activation Type: " <<  activation_type_str << " not supported !";
    }
    return tensorrt_llm::ActivationType::InvalidType;
}

tensorrt_llm::kernels::MOEExpertScaleNormalizationMode getNormalizationMode(int normalization_mode) {
    switch (normalization_mode) {
        case 0:
            return tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;
        case 1:
            return tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE;
        case 2:
            return tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::SPARSE_MIXER;
        case 3:
            return tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::DEVICE_LIMITED;
        case 4:
            return tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::DEVICE_LIMITED_RENORM;
        default:
            std::cerr << "Unknown normalization_mode value: " << normalization_mode << std::endl;
            // Return a default value if an invalid mode is passed
            return tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::NONE;
    }
}

template<typename T, typename WeightType>
Tensor trt_llm_fused_moe_helper(Tensor input_activations, 
                                 Tensor gating_output, 
                                 Tensor fc1_expert_weights, 
                                 tensorrt_llm::ActivationType fc1_activation_type,
                                 Tensor fc2_expert_weights, 
                                 const int active_rows, 
                                 const int k,
                                 paddle::optional<paddle::Tensor> scale1 = nullptr,
                                 paddle::optional<paddle::Tensor> scale2 = nullptr,
                                 paddle::optional<paddle::Tensor> scale3 = nullptr,
                                 int normalization_mode = 0,
                                 const std::string& quant_method = "none")
{
    typedef DataTypeMapper<T> traits_t;
    typedef typename traits_t::DataType DataType_;
    typedef typename traits_t::data_t data_t;

    typedef DataTypeMapper<WeightType> traits_w;
    typedef typename traits_w::DataType DataType_w;
    typedef typename traits_w::data_t data_w;

    const int num_rows = input_activations.shape()[0];//(num_tokens, hidden_size)
    const int hidden_size = input_activations.shape()[1];
    const int inter_size = fc2_expert_weights.shape()[1]; //(num_experts, inter_size, hidden_size)
    // std::cout << "我修改了iner_size "<< std::endl;
    // const int inter_size = fc2_expert_weights.shape()[2];
    const int fc1_inter_size = fc1_expert_weights.shape()[2];

    PD_CHECK(inter_size == fc1_inter_size || inter_size * 2 == fc1_inter_size);
    const int num_experts = gating_output.shape()[1]; //(num_tokens, num_experts)
    
    auto stream = input_activations.stream();
    auto place = input_activations.place();
    phi::Allocator* allocator = paddle::GetAllocator(place);
    T* input_act_ptr = reinterpret_cast<T*>(input_activations.data<data_t>());
    float* gating_output_ptr =  reinterpret_cast<float*>(gating_output.data<float>());

    // bias暂时先不支持
    T* fc1_expert_biases_ptr = nullptr;
    T* fc2_expert_biases_ptr = nullptr;

    bool* finished_ptr = nullptr;

    tensorrt_llm::kernels::MOEParallelismConfig moe_parallel_config = tensorrt_llm::kernels::MOEParallelismConfig(1, 0, 1, 0);

    void* scale1_ptr = nullptr;
    void* scale2_ptr = nullptr;

    const void* fc1_weights_ptr = nullptr;
    const void* fc2_weights_ptr = nullptr;


    bool use_deepseek = false;
    if (quant_method == "fp8_block_wise") {
        use_deepseek = true;
    }
    tensorrt_llm::kernels::QuantParams quant_params;
    if (quant_method == "weight_only_int8" || quant_method == "weight_only_int4") {
        scale1_ptr = get_ptr<data_t>(scale1);
        scale2_ptr = get_ptr<data_t>(scale2);
        quant_params = tensorrt_llm::kernels::QuantParams::Int(scale1_ptr, scale2_ptr);
        fc1_weights_ptr = reinterpret_cast<WeightType*>(fc1_expert_weights.data<data_w>());
        fc2_weights_ptr = reinterpret_cast<WeightType*>(fc2_expert_weights.data<data_w>());
    } else if (quant_method == "fp8_block_wise") {
        // fp8 scale是float
        scale1_ptr = get_ptr<float>(scale1);
        scale2_ptr = get_ptr<float>(scale2);
        std::cout <<"trt nmsl " << std::endl;
        fc1_weights_ptr = reinterpret_cast<const void*>(fc1_expert_weights.data<phi::dtype::float8_e4m3fn>());
        fc2_weights_ptr = reinterpret_cast<const void*>(fc2_expert_weights.data<phi::dtype::float8_e4m3fn>());
    } else {
        fc1_weights_ptr = reinterpret_cast<WeightType*>(fc1_expert_weights.data<data_w>());
        fc2_weights_ptr = reinterpret_cast<WeightType*>(fc2_expert_weights.data<data_w>());
    }

    tensorrt_llm::kernels::BlockScaleParams deepseek_params;
    if (use_deepseek)
    {    
        using BlockScaleGemmImplPtr = std::shared_ptr<tensorrt_llm::kernels::small_m_gemm::CutlassFp8BlockScaleGemmRunnerInterface>;
        BlockScaleGemmImplPtr mBlockScaleGemmImplPtr;

        if (std::is_same_v<WeightType, __nv_bfloat16>) {
            mBlockScaleGemmImplPtr
                    = std::make_shared<tensorrt_llm::kernels::small_m_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16,
                        __nv_bfloat16, __nv_bfloat16>>();
        } else {
            mBlockScaleGemmImplPtr = std::make_shared<tensorrt_llm::kernels::small_m_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16,
                        __nv_fp8_e4m3, __nv_bfloat16>>();;
        }
        
        size_t deepseek_workspace_size = 0;
        cudaEvent_t mMemcpyEvent;
        bool is_gated_activation = isGatedActivation(fc1_activation_type);
        int factor = is_gated_activation ? 2 : 1;
        size_t deepseek_fc1_size = mBlockScaleGemmImplPtr->getWorkspaceSize(
            num_rows * k, factor * inter_size, hidden_size, num_experts);
        size_t deepseek_fc2_size = mBlockScaleGemmImplPtr->getWorkspaceSize(
            num_rows * k, hidden_size, inter_size, num_experts);
        deepseek_workspace_size = std::max(deepseek_fc1_size, deepseek_fc2_size);

        auto deepseek_ws = allocator->Allocate(deepseek_workspace_size)->ptr();
        std::cout <<"in 3" << std::endl;
        // print_gpu_data<float>(scale1_ptr, 256 * 2 * 56, 15);
        auto fc1_scales_ptr = static_cast<float const*>(scale1_ptr);
        auto fc2_scales_ptr = static_cast<float const*>(scale2_ptr);

        deepseek_params = tensorrt_llm::kernels::BlockScaleParams(
            fc1_scales_ptr, fc2_scales_ptr, mBlockScaleGemmImplPtr, reinterpret_cast<char*>(deepseek_ws), &mMemcpyEvent);
    }

    // deepseek相关参数
    int sm = getSMVersion();
    tensorrt_llm::kernels::CutlassMoeFCRunner<T, WeightType> moe_runner;
    // std::cout <<"in 5" << std::endl;

    auto [tactic1, tactic2] = selectTacticsForArch(moe_runner, sm);
    moe_runner.setTactic(std::make_optional(tactic1), std::make_optional(tactic2));

    tensorrt_llm::kernels::MOEExpertScaleNormalizationMode normalization_mode_enum = getNormalizationMode(normalization_mode);

    auto bytes = moe_runner.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k, fc1_activation_type, 
                                             normalization_mode_enum, moe_parallel_config, use_deepseek);

    auto workspace_ptr = allocator->Allocate(bytes)->ptr();
    auto expert_scales = paddle::empty({num_rows, k}, input_activations.dtype(), place);
    T* expert_scales_ptr = reinterpret_cast<T*>(expert_scales.data<data_t>());

    auto expanded_source_row_to_expanded_dest_row = paddle::empty({num_rows, k}, paddle::DataType::INT32, place);
    int* expanded_source_row_to_expanded_dest_row_ptr = reinterpret_cast<int*>(expanded_source_row_to_expanded_dest_row.data<int>());

    // topk的结果
    auto expert_for_source_row = paddle::empty({num_rows, k}, paddle::DataType::INT32, place);

    int* expert_for_source_row_ptr = reinterpret_cast<int*>(expert_for_source_row.data<int>());

    auto output_tensor = paddle::empty({num_rows, hidden_size}, input_activations.dtype(), place);
    T* output_tensor_ptr = reinterpret_cast<T*>(output_tensor.data<data_t>());

    moe_runner.runMoe(input_act_ptr,
                    gating_output_ptr,
                    fc1_weights_ptr,
                    fc1_expert_biases_ptr, // nullptr
                    fc1_activation_type,
                    fc2_weights_ptr,
                    fc2_expert_biases_ptr, // nullptr
                    quant_params,
                    num_rows,
                    hidden_size,
                    inter_size,
                    num_experts,
                    k,
                    reinterpret_cast<char*>(workspace_ptr),
                    output_tensor_ptr,
                    finished_ptr,
                    active_rows,
                    expert_scales_ptr,
                    expanded_source_row_to_expanded_dest_row_ptr,
                    expert_for_source_row_ptr,
                    0.2f,  // sparse_mixer_epsilon
                    moe_parallel_config,
                    normalization_mode_enum,
                    use_deepseek,
                    deepseek_params,
                    stream);
    return output_tensor;
}

template<typename T, typename WeightType>
Tensor trt_llm_fused_moe_helper_fp8_per_tensor(Tensor input_activations, 
                                 Tensor gating_output, 
                                 Tensor fc1_expert_weights, 
                                 tensorrt_llm::ActivationType fc1_activation_type,
                                 Tensor fc2_expert_weights, 
                                 const int active_rows, 
                                 const int k,
                                 paddle::optional<paddle::Tensor> scale1 = nullptr,
                                 paddle::optional<paddle::Tensor> scale2 = nullptr,
                                 paddle::optional<paddle::Tensor> scale3 = nullptr,
                                 const std::string& quant_method = "none")
{
    typedef DataTypeMapper<T> traits_t;
    typedef typename traits_t::DataType DataType_;
    typedef typename traits_t::data_t data_t;

    typedef DataTypeMapper<WeightType> traits_w;
    typedef typename traits_w::DataType DataType_w;
    typedef typename traits_w::data_t data_w;

    const int num_rows = input_activations.shape()[0];
    const int hidden_size = input_activations.shape()[1];
    const int inter_size = fc2_expert_weights.shape()[1];
    const int num_experts = gating_output.shape()[1];
    auto stream = input_activations.stream();
    auto place = input_activations.place();

    data_t* input_act_ptr = get_ptr<data_t>(input_activations);
    float* gating_output_ptr = get_ptr<float>(gating_output);

    float* scale1_ptr = scale1 ? get_ptr<float>(scale1) : nullptr;
    float* scale2_ptr = scale2 ? get_ptr<float>(scale2) : nullptr;
    float* scale3_ptr = scale3 ? get_ptr<float>(scale3) : nullptr;

    data_w* fc1_expert_weights_ptr = get_ptr<data_w>(fc1_expert_weights);
    data_t* fc1_expert_biases_ptr = nullptr;

    data_w* fc2_expert_weights_ptr = get_ptr<data_w>(fc2_expert_weights);
    data_t* fc2_expert_biases_ptr = nullptr;

    bool* finished_ptr = nullptr;

    tensorrt_llm::kernels::MOEParallelismConfig moe_parallel_config = tensorrt_llm::kernels::MOEParallelismConfig(1, 0, 1, 0);

    // 根据启用的量化方法设置量化参数
    tensorrt_llm::kernels::QuantParams quant_params;
    quant_params = tensorrt_llm::kernels::QuantParams::FP8(scale1_ptr, scale2_ptr, scale3_ptr);

    int sm = getSMVersion();
    tensorrt_llm::kernels::CutlassMoeFCRunner<T, WeightType, __nv_bfloat16> moe_runner;
    tensorrt_llm::kernels::BlockScaleParams deepseek_params;
    bool use_deepseek = false;

    auto [tactic1, tactic2] = selectTacticsForArch(moe_runner, sm);
    moe_runner.setTactic(std::make_optional(tactic1), std::make_optional(tactic2));

    auto bytes = moe_runner.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k, fc1_activation_type, 
                                             tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE, moe_parallel_config, false);

    auto workspace_tensor = paddle::empty({static_cast<int>(bytes)}, paddle::DataType::UINT8, place);
    uint8_t* uint8_ptr = get_ptr<uint8_t>(workspace_tensor);
    char* workspace_ptr = reinterpret_cast<char*>(uint8_ptr);

    auto fc2_output = paddle::empty({k * num_rows, hidden_size}, input_activations.dtype(), place);
    auto expert_scales = paddle::empty({num_rows, k}, input_activations.dtype(), place);
    data_t* expert_scales_ptr = get_ptr<data_t>(expert_scales);

    auto expanded_source_row_to_expanded_dest_row = paddle::empty({num_rows, k}, paddle::DataType::INT32, place);
    int* expanded_source_row_to_expanded_dest_row_ptr = get_ptr<int>(expanded_source_row_to_expanded_dest_row);

    auto expert_for_source_row = paddle::empty({num_rows, k}, paddle::DataType::INT32, place);
    int* expert_for_source_row_ptr = get_ptr<int>(expert_for_source_row);

    auto output_tensor = paddle::empty({num_rows, hidden_size}, input_activations.dtype(), place);
    data_t* output_tensor_ptr = get_ptr<data_t>(output_tensor);

    moe_runner.runMoe(input_act_ptr,
                      gating_output_ptr,
                      fc1_expert_weights_ptr,
                      fc1_expert_biases_ptr,
                      fc1_activation_type,
                      fc2_expert_weights_ptr,
                      fc2_expert_biases_ptr,
                      quant_params,
                      num_rows,
                      hidden_size,
                      inter_size,
                      num_experts,
                      k,
                      workspace_ptr,
                      output_tensor_ptr,
                      finished_ptr,
                      active_rows,
                      expert_scales_ptr,
                      expanded_source_row_to_expanded_dest_row_ptr,
                      expert_for_source_row_ptr,
                      0.2f,  // sparse_mixer_epsilon
                      moe_parallel_config,
                      tensorrt_llm::kernels::MOEExpertScaleNormalizationMode::RENORMALIZE,
                      use_deepseek,
                      deepseek_params,
                      stream);

    return output_tensor;
}



std::vector<paddle::Tensor> TrtLLMFusedMoe(const paddle::Tensor&     input_activations, //(num_tokens, hidden_size)
                const paddle::Tensor&      gating_output, //(num_tokens, num_experts)
                const paddle::Tensor&      fc1_expert_weights, //(num_experts, hidden_size, inter_size * 2)
                const paddle::Tensor&      fc2_expert_weights, //(num_experts, inter_size, hidden_size)
                const paddle::optional<paddle::Tensor>& scale1,
                const paddle::optional<paddle::Tensor>& scale2,
                const paddle::optional<paddle::Tensor>& scale3,
                int     k,
                int normalization_mode = 1,
                const std::string& quant_method="none",
                const std::string& fc1_activation_type_str="Swiglu")
{

    const auto _st = input_activations.dtype();
    const auto weight_type = fc1_expert_weights.dtype();
    int     active_rows;
    const int num_rows    = input_activations.shape()[0];
    // const int hidden_size = input_activations.shape()[1];
    // const int num_experts = gating_output.shape()[1];

    const auto quant_type = fc2_expert_weights.dtype();
    // const int inter_size = fc2_expert_weights.shape()[1];

    Tensor output_tensor;

    tensorrt_llm::ActivationType fc1_activation_type = tensorrt_llm::ActivationType::Swiglu;;
    active_rows = num_rows;
    if (fc1_activation_type_str == "identity") {
        fc1_activation_type = tensorrt_llm::ActivationType::Identity;
    }
    else {
        fc1_activation_type = getActivationType(fc1_activation_type_str);
    }

    std::cout << "start ! "<< std::endl;
    std::cout<< quant_method  << std::endl;
    switch (_st) {
         case paddle::DataType::FLOAT32: {
            if (quant_type == _st) {
                output_tensor = trt_llm_fused_moe_helper<float, float>(input_activations,
                                                                gating_output,
                                                                fc1_expert_weights,
                                                                fc1_activation_type,
                                                                fc2_expert_weights,
                                                                active_rows,
                                                                k,
                                                                nullptr,
                                                                nullptr,
                                                                nullptr,
                                                                normalization_mode,
                                                                quant_method);
            }
            else {
                std::string err_msg = "Unsupported weight type ";
                throw std::runtime_error(err_msg);
            }
            break;
        }
        case paddle::DataType::FLOAT16: {
            if (quant_type == _st) {
                output_tensor = trt_llm_fused_moe_helper<half, half>(input_activations,
                                                                    gating_output,
                                                                    fc1_expert_weights,
                                                                    fc1_activation_type,
                                                                    fc2_expert_weights,
                                                                    active_rows,
                                                                    k,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    normalization_mode,
                                                                    quant_method);
            }
            else {
                std::string err_msg = "Unsupported weight type ";
                throw std::runtime_error(err_msg);
            }
            break;
        }
        case paddle::DataType::BFLOAT16: {
            if (quant_type == _st) {
                output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16, __nv_bfloat16>(input_activations,
                                                                                gating_output,
                                                                                fc1_expert_weights,
                                                                                fc1_activation_type,
                                                                                fc2_expert_weights,
                                                                                active_rows,
                                                                                k,
                                                                                nullptr,
                                                                                nullptr,
                                                                                nullptr,
                                                                                normalization_mode,
                                                                                quant_method);
            } else {
                if (quant_method == "weight_only_int8") {
                    output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16, uint8_t>(input_activations,
                                                                                gating_output,
                                                                                fc1_expert_weights,
                                                                                fc1_activation_type,
                                                                                fc2_expert_weights,
                                                                                active_rows,
                                                                                k,
                                                                                scale1,
                                                                                scale2,
                                                                                nullptr, //scale3不需要
                                                                                normalization_mode,
                                                                                quant_method);
            } else if (quant_method == "weight_only_int4") {
                output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16, cutlass::uint4b_t>(input_activations,
                                                                            gating_output,
                                                                            fc1_expert_weights,
                                                                            fc1_activation_type,
                                                                            fc2_expert_weights,
                                                                            active_rows,
                                                                            k,
                                                                            scale1,
                                                                            scale2,
                                                                            nullptr, //scale3不需要
                                                                            normalization_mode,
                                                                            quant_method);
                } else if (quant_method == "fp8_block_wise") {
                    if (quant_type == _st)  {
                        output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16,__nv_bfloat16>(input_activations,
                                                                                gating_output,
                                                                                fc1_expert_weights,
                                                                                fc1_activation_type,
                                                                                fc2_expert_weights,
                                                                                active_rows,
                                                                                k,
                                                                                scale1,
                                                                                scale2,
                                                                                nullptr, //scale3不需要
                                                                                normalization_mode,
                                                                                quant_method);
                    } else {
                        std::cout << "i want" << std::endl;
                        output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16, __nv_fp8_e4m3>(input_activations,
                                                                                gating_output,
                                                                                fc1_expert_weights,
                                                                                fc1_activation_type,
                                                                                fc2_expert_weights,
                                                                                active_rows,
                                                                                k,
                                                                                scale1,
                                                                                scale2,
                                                                                nullptr, //scale3不需要
                                                                                normalization_mode,
                                                                                quant_method);
                    }
                    

                } else {
                    std::string err_msg = "Unsupported weight type ";
                    throw std::runtime_error(err_msg);
                }
            }
            break;
        }
        case paddle::DataType::FLOAT8_E4M3FN: {
            std::cout << "fp8_per_tensor"<< std::endl;
            // if (quant_type == _st) {
            //     output_tensor = trt_llm_fused_moe_helper_fp8_per_tensor<__nv_fp8_e4m3, __nv_fp8_e4m3>(input_activations,
            //                                                                     gating_output,
            //                                                                     fc1_expert_weights,
            //                                                                     fc1_activation_type,
            //                                                                     fc2_expert_weights,
            //                                                                     active_rows,
            //                                                                     k,
            //                                                                     scale1,
            //                                                                     scale2,
            //                                                                     scale3,
            //                                                                     quant_method);
            // }
            // else {
            //     std::string err_msg = "Unsupported weight type ";
            //     throw std::runtime_error(err_msg);
            // }
            break;
        }
        
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
    return {output_tensor};
}


PD_BUILD_OP(trt_llm_fused_moe)
    .Inputs({"input_activations", "gating_output", "fc1_expert_weights", "fc2_expert_weights", paddle::Optional("scale1"), paddle::Optional("scale2"), paddle::Optional("scale3"),})
    .Outputs({"output_tensor"})
    .Attrs({"k: int", "normalization_mode: int", "quant_method:std::string", "fc1_activation_type_str:std::string"})
    .SetKernelFn(PD_KERNEL(TrtLLMFusedMoe));