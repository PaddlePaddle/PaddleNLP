// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <optional>
#include <algorithm>
#include "cutlass_helper.h"
#include "tensorrt_llm/kernels/mixtureOfExperts/utils.h"
#include "tensorrt_llm/kernels/mixtureOfExperts/profile.h"
#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"


kernels::MOEExpertScaleNormalizationMode getNormalizationMode(int normalization_mode) {
    switch (normalization_mode) {
        case 0:
            return kernels::MOEExpertScaleNormalizationMode::NONE;
        case 1:
            return kernels::MOEExpertScaleNormalizationMode::RENORMALIZE;
        case 2:
            return kernels::MOEExpertScaleNormalizationMode::SPARSE_MIXER;
        case 3:
            return kernels::MOEExpertScaleNormalizationMode::DEVICE_LIMITED;
        case 4:
            return kernels::MOEExpertScaleNormalizationMode::DEVICE_LIMITED_RENORM;
        default:
            std::cerr << "Unknown normalization_mode value: " << normalization_mode << std::endl;
            // Return a default value if an invalid mode is passed
            return kernels::MOEExpertScaleNormalizationMode::NONE;
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
                                 const std::string& quant_method = "none",
                                 int    tune_max_num_tokens=40960)
{
    typedef DataTypeMapper<T> traits_t;
    typedef typename traits_t::DataType DataType_;
    typedef typename traits_t::data_t data_t;

    typedef DataTypeMapper<WeightType> traits_w;
    typedef typename traits_w::DataType DataType_w;
    typedef typename traits_w::data_t data_w;

    // 初始化一些参数，获得data_ptr
    const int num_rows = input_activations.shape()[0];//(num_tokens, hidden_size)
    const int hidden_size = input_activations.shape()[1];
    int inter_size = fc2_expert_weights.shape()[1]; //(num_experts, inter_size, hidden_size)
    if (quant_method == "fp8_block_wise") {
       inter_size = fc2_expert_weights.shape()[2];
    }
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

    // 暂不支持并行策略
    kernels::MOEParallelismConfig moe_parallel_config = kernels::MOEParallelismConfig(1, 0, 1, 0);

    void* scale1_ptr = nullptr;
    void* scale2_ptr = nullptr;
    bool use_deepseek = false;
    kernels::BlockScaleParams deepseek_params;
    void* fc1_weights_ptr = reinterpret_cast<WeightType*>(fc1_expert_weights.data<data_w>());
    void* fc2_weights_ptr = reinterpret_cast<WeightType*>(fc2_expert_weights.data<data_w>());

    kernels::QuantParams quant_params;
    if (quant_method == "weight_only_int8" || quant_method == "weight_only_int4") {
        scale1_ptr = get_ptr<data_t>(scale1);
        scale2_ptr = get_ptr<data_t>(scale2);
        quant_params = kernels::QuantParams::Int(scale1_ptr, scale2_ptr);
    } else if (quant_method == "fp8_block_wise") {
        // fp8 blockwisescale是float
        scale1_ptr = get_ptr<float>(scale1);
        scale2_ptr = get_ptr<float>(scale2);
        use_deepseek = true;
    } 

    // 初始化moe_runner
    std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> moe_runner_ptr = std::make_shared<kernels::CutlassMoeFCRunner<T, WeightType>>();
      
    if (use_deepseek)
    {    
        using BlockScaleGemmImplPtr = std::shared_ptr<kernels::small_m_gemm::CutlassFp8BlockScaleGemmRunnerInterface>;
        BlockScaleGemmImplPtr mBlockScaleGemmImplPtr= std::make_shared<kernels::small_m_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16,
                    __nv_fp8_e4m3, __nv_bfloat16>>();
        
        size_t deepseek_workspace_size = 0;
        cudaEvent_t mMemcpyEvent;
        size_t deepseek_fc1_size = mBlockScaleGemmImplPtr->getWorkspaceSize(
            num_rows * k, 2 * inter_size, hidden_size, num_experts);
        size_t deepseek_fc2_size = mBlockScaleGemmImplPtr->getWorkspaceSize(
            num_rows * k, hidden_size, inter_size, num_experts);
        deepseek_workspace_size = std::max(deepseek_fc1_size, deepseek_fc2_size);
        auto deepseek_ws = allocator->Allocate(deepseek_workspace_size)->ptr();

        deepseek_params = kernels::BlockScaleParams(
            static_cast<float *>(scale1_ptr), static_cast<float *>(scale2_ptr), mBlockScaleGemmImplPtr, reinterpret_cast<char*>(deepseek_ws), &mMemcpyEvent);
    }

    
    // profile相关
    std::string profile_file = "./trt_moe_profile_results.json";
    CutlassGemmConfigMannager& best_config_mannager = CutlassGemmConfigMannager::getInstance();
    std::string efficientllm_op_configs = getenv("FLAGS_efficientllm_op_configs");
    if (efficientllm_op_configs == "tune" && !std::filesystem::exists(profile_file))  {
        FusedMoeRunnerPofiler profiler(/* activation_dtype= */ input_activations.dtype(), 
                                /* weight_dtype= */ fc1_expert_weights.dtype(), 
                                /* output_dtype= */ input_activations.dtype(), 
                                /* moe_runner= */ moe_runner_ptr, 
                                /* quant_method= */ quant_method);

        std::vector<int64_t> num_token_buckets = get_power_of_2_num_tokens_buckets(tune_max_num_tokens);
        profiler.runProfile(fc2_expert_weights, k, 1, 0, 1, 0, num_token_buckets);
        profiler.saveProfileResultsToFile(profile_file);
        std::vector<int64_t> profile_ids = profiler.getProfileIds(next_positive_power_of_2(num_rows), fc2_expert_weights, k, num_experts);
        for (int i = 0; i < profile_ids.size(); ++i){
            std::cout <<  "profile_ids : "<< profile_ids[i] << std::endl;
        }
        setRunnerProfiles(moe_runner_ptr, profile_ids, quant_method);
    } else {
        if (!std::filesystem::exists(profile_file)) {
            printf("[Warning]Moe not tune cutlass kernel, using defualt config slowly !\n");
            auto [tactic1, tactic2] = selectTacticsForArch(moe_runner_ptr);
            moe_runner_ptr->setTactic(std::make_optional(tactic1), std::make_optional(tactic2));
        } else {
            auto* configs_json = best_config_mannager.get_gemm_best_configs(profile_file);
            std::vector<int64_t> profile_ids = loadProfileResultsFromFile(*configs_json, next_positive_power_of_2(num_rows));
            setRunnerProfiles(moe_runner_ptr, profile_ids, quant_method);
        }
    }


    // 分配中间tensor
    kernels::MOEExpertScaleNormalizationMode normalization_mode_enum = getNormalizationMode(normalization_mode);

    auto bytes = moe_runner_ptr->getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k, fc1_activation_type, 
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
    
    // run
    moe_runner_ptr->runMoe(input_act_ptr,
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
                                 const std::string& quant_method = "none",
                                 int     tune_max_num_tokens=40960)
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

    kernels::MOEParallelismConfig moe_parallel_config = kernels::MOEParallelismConfig(1, 0, 1, 0);

    // 根据启用的量化方法设置量化参数
    kernels::QuantParams quant_params;
    quant_params = kernels::QuantParams::FP8(scale1_ptr, scale2_ptr, scale3_ptr);

    int sm = getSMVersion();
    kernels::CutlassMoeFCRunner<T, WeightType, __nv_bfloat16> moe_runner;
    kernels::BlockScaleParams deepseek_params;
    bool use_deepseek = false;

    auto [tactic1, tactic2] = selectTacticsForArch(moe_runner, sm);
    moe_runner.setTactic(std::make_optional(tactic1), std::make_optional(tactic2));

    auto bytes = moe_runner.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k, fc1_activation_type, 
                                             kernels::MOEExpertScaleNormalizationMode::RENORMALIZE, moe_parallel_config, false);

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
                      kernels::MOEExpertScaleNormalizationMode::RENORMALIZE,
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
                const paddle::optional<paddle::Tensor>& scale3, // fp8-per-tensor需要
                int     k,
                int normalization_mode = 1, // 决定是否做norm / softmax_topk / only topk
                const std::string& quant_method="none",
                int     tune_max_num_tokens=40960)
{

    const auto _st = input_activations.dtype();
    const auto weight_type = fc1_expert_weights.dtype();
    const int num_rows    = input_activations.shape()[0];
    const auto quant_type = fc2_expert_weights.dtype();

    Tensor output_tensor;
    tensorrt_llm::ActivationType fc1_activation_type = tensorrt_llm::ActivationType::Swiglu;;
    std::cout << "start ! "<< std::endl;
    std::cout<< quant_method  << std::endl;
    switch (_st) {
        case paddle::DataType::FLOAT16: {
            if (quant_type == _st) {
                output_tensor = trt_llm_fused_moe_helper<half, half>(input_activations,
                                                                    gating_output,
                                                                    fc1_expert_weights,
                                                                    fc1_activation_type,
                                                                    fc2_expert_weights,
                                                                    num_rows,
                                                                    k,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    normalization_mode,
                                                                    quant_method);
            }
            else {
                std::string err_msg = "Unsupported weight type, wint8/wint4 only support bfloat16 noew ";
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
                                                                                num_rows,
                                                                                k,
                                                                                nullptr,
                                                                                nullptr,
                                                                                nullptr,
                                                                                normalization_mode,
                                                                                quant_method,
                                                                                tune_max_num_tokens);
            } else {
                if (quant_method == "weight_only_int8") {
                    output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16, uint8_t>(input_activations,
                                                                                gating_output,
                                                                                fc1_expert_weights,
                                                                                fc1_activation_type,
                                                                                fc2_expert_weights,
                                                                                num_rows,
                                                                                k,
                                                                                scale1,
                                                                                scale2,
                                                                                nullptr, //scale3不需要
                                                                                normalization_mode,
                                                                                quant_method,
                                                                                tune_max_num_tokens);
                } else if (quant_method == "weight_only_int4") {
                output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16, cutlass::uint4b_t>(input_activations,
                                                                            gating_output,
                                                                            fc1_expert_weights,
                                                                            fc1_activation_type,
                                                                            fc2_expert_weights,
                                                                            num_rows,
                                                                            k,
                                                                            scale1,
                                                                            scale2,
                                                                            nullptr, //scale3不需要
                                                                            normalization_mode,
                                                                            quant_method,
                                                                            tune_max_num_tokens);
                } else if (quant_method == "fp8_block_wise") {
                    output_tensor = trt_llm_fused_moe_helper<__nv_bfloat16, __nv_fp8_e4m3>(input_activations,
                                                                            gating_output,
                                                                            fc1_expert_weights,
                                                                            fc1_activation_type,
                                                                            fc2_expert_weights,
                                                                            num_rows,
                                                                            k,
                                                                            scale1,
                                                                            scale2,
                                                                            nullptr, //scale3不需要
                                                                            normalization_mode,
                                                                            quant_method,
                                                                            tune_max_num_tokens);
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
            //                                                                     num_rows,
            //                                                                     k,
            //                                                                     scale1,
            //                                                                     scale2,
            //                                                                     scale3,
            //                                                                     quant_method,
                                                                                    // tune_max_num_tokens);
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

// std::vector<paddle::DataType> TrtLLMFusedMoeInferDtype(
//         const paddle::DataType&     input_activations_dtype, //(num_tokens, hidden_size)
//         const paddle::DataType&      gating_output_dtype, //(num_tokens, num_experts)
//         const paddle::DataType&      fc1_expert_weights_dtype, //(num_experts, hidden_size, inter_size * 2)
//         const paddle::DataType&     fc2_expert_weight_dtypes, //(num_experts, inter_size, hidden_size)
//         const paddle::optional<paddle::DataType>& scale1_dtype,
//         const paddle::optional<paddle::DataType>& scale2_dtype,
//         const paddle::optional<paddle::DataType>& scale3_dtype, 
//         int     k,
//         int normalization_mode,
//         const std::string& quant_method,
//         int     tune_max_num_tokens){
        
//         return {input_activations_dtype};
// }

// std::vector<std::vector<int64_t>> TrtLLMFusedMoeInferShape(
//         const std::vector<int64_t>&     input_activations_shape, //(num_tokens, hidden_size)
//         const std::vector<int64_t>&      gating_output_shape, //(num_tokens, num_experts)
//         const std::vector<int64_t>&      fc1_expert_weights_shape, //(num_experts, hidden_size, inter_size * 2)
//         const std::vector<int64_t>&      fc2_expert_weights_shape, //(num_experts, inter_size, hidden_size)
//         const paddle::optional<std::vector<int64_t>&>& scale1_shape,
//         const paddle::optional<std::vector<int64_t>&>& scale2_shape,
//         const paddle::optional<std::vector<int64_t>&>& scale3_shape){
    
//     return {input_activations_shape};
// }

PD_BUILD_OP(trt_llm_fused_moe)
    .Inputs({"input_activations", "gating_output", "fc1_expert_weights", "fc2_expert_weights", paddle::Optional("scale1"), paddle::Optional("scale2"), paddle::Optional("scale3"),})
    .Outputs({"output_tensor"})
    .Attrs({"k: int", "normalization_mode: int", "quant_method:std::string", "tune_max_num_tokens: int"})
    .SetKernelFn(PD_KERNEL(TrtLLMFusedMoe));
    // .SetInferShapeFn(PD_INFER_SHAPE(TrtLLMFusedMoeInferShape))
    // .SetInferDtypeFn(PD_INFER_DTYPE(TrtLLMFusedMoeInferDtype));