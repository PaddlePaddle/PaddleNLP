// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>

#include "fusion_gpt_op.h"
#include "pd_traits.h"

#ifdef WITH_FT5
#include "src/fastertransformer5/utils/nvtx_utils.h"

#include "src/fastertransformer5/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer5/utils/cuda_bf16_wrapper.h"
#endif

#include "utils.h"
#include "cublas_handle.h"

// TODO(guosheng): `HOST` conflict exists in float.h of paddle and mpi.h of mpi
#ifdef HOST
#undef HOST
#endif

#ifndef CUB_NS_QUALIFIER
#define CUB_NS_QUALIFIER ::cub
#endif

#include "fastertransformer/cuda/cub/cub.cuh"
#include "fastertransformer/gpt.h"
#include "fastertransformer/utils/common.h"

#ifdef BUILD_GPT  // consistent with FasterTransformer
#include "parallel_utils.h"
#endif

#include <sys/time.h>

template <paddle::DataType D>
std::vector<paddle::Tensor> gpt_kernel(
        const paddle::Tensor& input,
        const paddle::Tensor& attn_mask,
        const paddle::Tensor& start_length,
        const paddle::Tensor& word_emb,
        const std::vector<paddle::Tensor>& self_ln_weight,
        const std::vector<paddle::Tensor>& self_ln_bias,
        const std::vector<paddle::Tensor>& self_q_weight,
        const std::vector<paddle::Tensor>& self_q_bias,
        const std::vector<paddle::Tensor>& self_k_weight,
        const std::vector<paddle::Tensor>& self_k_bias,
        const std::vector<paddle::Tensor>& self_v_weight,
        const std::vector<paddle::Tensor>& self_v_bias,
        const std::vector<paddle::Tensor>& self_out_weight,
        const std::vector<paddle::Tensor>& self_out_bias,
        const std::vector<paddle::Tensor>& ffn_ln_weight,
        const std::vector<paddle::Tensor>& ffn_ln_bias,
        const std::vector<paddle::Tensor>& ffn_inter_weight,
        const std::vector<paddle::Tensor>& ffn_inter_bias,
        const std::vector<paddle::Tensor>& ffn_out_weight,
        const std::vector<paddle::Tensor>& ffn_out_bias,
        const paddle::Tensor& decoder_ln_weight,
        const paddle::Tensor& decoder_ln_bias,
        const paddle::Tensor& positional_embedding_weight,
        const paddle::Tensor& emb_weight,
        paddle::Tensor& output_ids,
        const int& topk,
        const float& topp,
        const int& max_len,
        const int& n_head,
        const int& size_per_head,
        const int& num_layer,
        const int& bos_id,
        const int& eos_id,
        const float& temperature,
        cudaStream_t stream,
        const int& tensor_para_size = 1,
        const int& layer_para_size = 1,
        const int& layer_para_batch_size = 1) {
#ifdef WITH_FT5
    static char* enable_ft5_env_char = std::getenv("ENABLE_FT5");
    bool is_enable_ft5 = (enable_ft5_env_char != nullptr && (std::string(enable_ft5_env_char) == "ON" || std::string(enable_ft5_env_char) == "1")) ? true : false;

    if (is_enable_ft5) {
        namespace ft = fastertransformer5;

        typedef ft::PDTraits<D> traits_;
        typedef typename traits_::DataType DataType_;
        typedef typename traits_::data_t data_t_;

        std::mutex* cublas_wrapper_mutex = CublasWrapperMutex::GetInstance()->mutex_;
        ft::cublasAlgoMap* cublas_algo_map = new ft::cublasAlgoMap(GEMM_CONFIG);

        ft::ParallelGptWeight<DataType_> gpt_weights;

        ft::gptVariantParams gpt_variant_params{
            (float)1e-6,
            ft::getLayerNormType("pre_layernorm"),
            ft::getActivationType("gelu"),
            true,  // has_positional_encoding
            false,  // has_pre_decoder_layernorm
            true,  // has_post_decoder_layernorm
            false,  // has_adapters
            (size_t) 0,  // adapter_inter_size
            false  // use_attention_linear_bias
        };

        gpt_weights.resizeLayer(self_ln_weight.size());

        for (int i = 0; i < self_ln_weight.size(); ++i) {
            gpt_weights.decoder_layer_weights[i]->pre_layernorm_weights.gamma =
                get_ptr<DataType_, data_t_>(self_ln_weight[i]);
            gpt_weights.decoder_layer_weights[i]->pre_layernorm_weights.beta =
                get_ptr<DataType_, data_t_>(self_ln_bias[i]);

            gpt_weights.decoder_layer_weights[i]->self_attention_weights.query_weight.kernel =
                get_ptr<DataType_, data_t_>(self_q_weight[i]);
            gpt_weights.decoder_layer_weights[i]->self_attention_weights.query_weight.bias =
                get_ptr<DataType_, data_t_>(self_q_bias[i]);

            gpt_weights.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel =
                get_ptr<DataType_, data_t_>(self_out_weight[i]);
            gpt_weights.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias =
                get_ptr<DataType_, data_t_>(self_out_bias[i]);

            gpt_weights.decoder_layer_weights[i]->self_attn_layernorm_weights.gamma =
                get_ptr<DataType_, data_t_>(ffn_ln_weight[i]);
            gpt_weights.decoder_layer_weights[i]->self_attn_layernorm_weights.beta =
                get_ptr<DataType_, data_t_>(ffn_ln_bias[i]);

            gpt_weights.decoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel =
                get_ptr<DataType_, data_t_>(ffn_inter_weight[i]);
            gpt_weights.decoder_layer_weights[i]->ffn_weights.intermediate_weight.bias =
                get_ptr<DataType_, data_t_>(ffn_inter_bias[i]);

            gpt_weights.decoder_layer_weights[i]->ffn_weights.output_weight.kernel =
                get_ptr<DataType_, data_t_>(ffn_out_weight[i]);
            gpt_weights.decoder_layer_weights[i]->ffn_weights.output_weight.bias =
                get_ptr<DataType_, data_t_>(ffn_out_bias[i]);
        }

        if (gpt_variant_params.has_post_decoder_layernorm) {
            gpt_weights.post_decoder_layernorm.gamma = get_ptr<DataType_, data_t_>(decoder_ln_weight);
            gpt_weights.post_decoder_layernorm.beta = get_ptr<DataType_, data_t_>(decoder_ln_bias);
        }

        if (gpt_variant_params.has_positional_encoding) {
            gpt_weights.position_encoding_table = get_ptr<DataType_, data_t_>(positional_embedding_weight);
            gpt_weights.setMaxSeqLen(positional_embedding_weight.shape()[0]);
        }

        gpt_weights.pre_decoder_embedding_table = get_ptr<DataType_, data_t_>(word_emb);
        gpt_weights.post_decoder_embedding.kernel = get_ptr<DataType_, data_t_>(emb_weight);

        struct cudaDeviceProp props = CudaDeviceProp::GetInstance()->prop_;

        ft::Allocator<ft::AllocatorType::PD> allocator = ft::Allocator<ft::AllocatorType::PD>();
        ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(CublasHandle::GetInstance()->cublas_handle_,
                                                                 CublasHandle::GetInstance()->cublaslt_handle_,
                                                                 stream,
                                                                 cublas_algo_map,
                                                                 cublas_wrapper_mutex,
                                                                 &allocator);

        if (std::is_same<DataType_, half>::value) {
            cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F);
        }
#ifdef ENABLE_BF16
        else if (std::is_same<DataType_, __nv_bfloat16>::value) {
            cublas_wrapper.setBF16GemmConfig();
        }
#endif
        else if (std::is_same<DataType_, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        const size_t request_batch_size = (size_t)input.shape()[0];
        const size_t max_input_length = (size_t)input.shape()[1];
        const size_t total_output_len = (int)max_len;

        ft::NcclParam tensor_para;
        ft::NcclParam pipeline_para;

        ft::AttentionType attention_type =
            ft::getAttentionType<DataType_>(size_per_head,
                                    props.major * 10 + props.minor,
                                    true,
                                    max_input_length,  // gpt supports any-seq-length fmha
                                    true,              // is_fuse
                                    false,             // with_relative_position_bias
                                    true);             // causal_mask

        ft::ParallelGpt<DataType_> gpt = ft::ParallelGpt<DataType_>(
            request_batch_size,
            max_len,
            max_input_length,
            1,  // beam_width
            n_head,
            size_per_head,
            ffn_inter_weight[0].shape()[1],  // inter_size
            num_layer,
            0,  // expert num
            0,  // moe_k
            {},  // moe_layer_index
            word_emb.shape()[0],  // vocab_size
            bos_id,
            eos_id,
            eos_id + 1,  // p/prompt tuning virtual token start id
            ft::PromptLearningType::no_prompt,
            gpt_variant_params,   // gpt variant params --> meta opt
            0.0f,                 // beam_search_diversity_rate,
            topk,                 // top_k,
            topp,                 // top_p,
            0,                    // random_seed,
            temperature,          // temperature,
            0.0f,                 // len_penalty,
            1.0f,                 // repetition_penalty,
            tensor_para,
            pipeline_para,
            stream,
            &cublas_wrapper,
            &allocator,
            false,
            &props,
            attention_type,
            false,  // sparse
            0);
        std::vector<uint32_t> output_seq_len(request_batch_size, total_output_len);

        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids",
                ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, max_input_length},
                        get_ptr<int32_t, int32_t>(input)}},
            {"input_lengths",
                ft::Tensor{
                    ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, get_ptr<int32_t, int32_t>(start_length)}},
            {"output_seq_len",
                ft::Tensor{
                    ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{request_batch_size}, output_seq_len.data()}}};

        if (topp > 0.0f) {
            input_tensors.insert(
                {"runtime_top_p", ft::Tensor(ft::MemoryType::MEMORY_CPU, ft::TYPE_FP32, {1}, &topp)});
        }
        if (topk > 0) {
            input_tensors.insert(
                {"runtime_top_k", ft::Tensor(ft::MemoryType::MEMORY_CPU, ft::TYPE_INT32, {1}, &topk)});
        }
        if (temperature != 1.0) {
            input_tensors.insert(
                {"temperature", ft::Tensor(ft::MemoryType::MEMORY_CPU, ft::TYPE_FP32, {1}, &temperature)});
        }

        std::vector<int64_t> sequence_lengths_shape({static_cast<int64_t>(request_batch_size), 1});
        paddle::Tensor sequence_lengths = paddle::empty(sequence_lengths_shape, paddle::DataType::INT32, paddle::GPUPlace());

        std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"output_ids",
                ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, 1, (size_t)total_output_len},
                        get_ptr<int32_t, int32_t>(output_ids)}},
            {"sequence_length",
                ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, 1},
                        get_ptr<int32_t, int32_t>(sequence_lengths)}},
            {"output_log_probs",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_FP32,
                        {(size_t)request_batch_size, (size_t)1, (size_t)total_output_len},
                        nullptr}}};

        gpt.forward(&output_tensors, &input_tensors, &gpt_weights);

        delete cublas_algo_map;
    } else {
#endif

        namespace ft = fastertransformer;

        auto input_dims = input.shape();
        int batch_size_ = input_dims[0];
        int start_len = input_dims[1];
        const int vocab_size = word_emb.shape()[0];

        typedef ft::PDTraits<D> traits_;
        typedef typename traits_::DataType DataType_;
        typedef typename traits_::data_t data_t_;

        ft::DecodingInitParam<DataType_> decoding_params;
        decoding_params.cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
        decoding_params.cublaslt_handle = CublasHandle::GetInstance()->cublaslt_handle_;

        decoding_params.output_ids = output_ids.data<int>();

        typedef ft::DecoderTransformerTraits<traits_::OpType> DecodingTraits_;
        decoding_params.stream = stream;
        ft::Allocator<ft::AllocatorType::PD> allocator_(stream);

        const int hidden_unit = size_per_head * n_head;

#ifdef BUILD_GPT
        auto* model_para_desc = ModelParaDescFactory::CreateModelParaDesc(
            n_head,
            size_per_head,
            num_layer,
            tensor_para_size,
            layer_para_size,
            layer_para_batch_size,
            const_cast<data_t_*>(word_emb.data<data_t_>()));
        auto& tensor_parallel_param = model_para_desc->tensor_parallel_param;
        auto& layer_parallel_param = model_para_desc->layer_parallel_param;
        auto seed = model_para_desc->dist(model_para_desc->gen);
#else
        TensorParallelParam tensor_parallel_param;
        LayerParallelParam layer_parallel_param;
        tensor_parallel_param.rank = 0;
        tensor_parallel_param.world_size = 1;
        tensor_parallel_param.local_head_num_ = n_head;
        tensor_parallel_param.local_hidden_units_ = hidden_unit;

        layer_parallel_param.rank = 0;
        layer_parallel_param.world_size = 1;
        layer_parallel_param.layers_per_group = num_layer;
        layer_parallel_param.local_batch_size = batch_size_;
        int seed = -1;
#endif

        ft::DecodingGpt<DecodingTraits_::OpType>* gpt_decoding;

        decoding_params.request_batch_size = batch_size_;
        decoding_params.max_input_len = start_len;
        decoding_params.request_input_len = start_len;
        decoding_params.request_output_len = max_len - start_len;

        decoding_params.d_start_ids = const_cast<int *>(input.data<int>());
        decoding_params.d_attn_mask =
            reinterpret_cast<DataType_*>(const_cast<data_t_ *>(attn_mask.data<data_t_>()));
        decoding_params.d_start_lengths = start_length.data<int>();

        gpt_decoding =
            new ft::DecodingGpt<DecodingTraits_::OpType>(allocator_,
                                                    batch_size_,
                                                    max_len,
                                                    n_head,
                                                    size_per_head,
                                                    vocab_size,
                                                    num_layer,
                                                    bos_id,
                                                    eos_id,
                                                    topk,
                                                    topp,
                                                    temperature,
                                                    tensor_para_size,
                                                    layer_para_size,
                                                    true, /*is_fuse_QKV*/
                                                    1.0,  /*repetition_penalty*/
                                                    seed);

        gpt_decoding->set_tensor_parallel_param(tensor_parallel_param);
        gpt_decoding->set_layer_parallel_param(layer_parallel_param);

        ft::DecoderInitParam<DataType_>* params =
            new ft::DecoderInitParam<DataType_>[num_layer];

        for (int i = 0; i < self_ln_weight.size(); ++i) {
            // Allow python passing weights of all layers or only passing the
            // corresponding layers to save memory.
            int layer_idx = self_ln_weight.size() != num_layer
                                ? layer_parallel_param.rank *
                                        layer_parallel_param.layers_per_group +
                                    i
                                : i;

            params[layer_idx].stream = stream;
            params[layer_idx].cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
            params[layer_idx].cublaslt_handle = CublasHandle::GetInstance()->cublaslt_handle_;

            params[layer_idx].request_batch_size = batch_size_;
            params[layer_idx].request_max_mem_seq_len = start_len;

            params[layer_idx].self_layernorm.gamma =
                reinterpret_cast<const DataType_*>(self_ln_weight[i].data<data_t_>());
            params[layer_idx].self_layernorm.beta =
                reinterpret_cast<const DataType_*>(self_ln_bias[i].data<data_t_>());

            params[layer_idx].self_attention.query_weight.kernel =
                reinterpret_cast<const DataType_*>(self_q_weight[i].data<data_t_>());
            params[layer_idx].self_attention.query_weight.bias =
                reinterpret_cast<const DataType_*>(self_q_bias[i].data<data_t_>());
            // For `is_fuse_QKV == true`, ignore weight and bias of key and value to
            // remove requirements on python passing weights to save memory.
            // params[layer_idx].self_attention.key_weight.kernel =
            //     reinterpret_cast<const DataType_*>(self_k_weight[i].data<data_t_>());
            // params[layer_idx].self_attention.key_weight.bias =
            //     reinterpret_cast<const DataType_*>(self_k_bias[i].data<data_t_>());
            // params[layer_idx].self_attention.value_weight.kernel =
            //     reinterpret_cast<const DataType_*>(self_v_weight[i].data<data_t_>());
            // params[layer_idx].self_attention.value_weight.bias =
            //     reinterpret_cast<const DataType_*>(self_v_bias[i].data<data_t_>());

            params[layer_idx].self_attention.attention_output_weight.kernel =
                reinterpret_cast<const DataType_*>(self_out_weight[i].data<data_t_>());
            params[layer_idx].self_attention.attention_output_weight.bias =
                reinterpret_cast<const DataType_*>(self_out_bias[i].data<data_t_>());

            params[layer_idx].ffn_layernorm.gamma =
                reinterpret_cast<const DataType_*>(ffn_ln_weight[i].data<data_t_>());
            params[layer_idx].ffn_layernorm.beta =
                reinterpret_cast<const DataType_*>(ffn_ln_bias[i].data<data_t_>());

            params[layer_idx].ffn.intermediate_weight.kernel =
                reinterpret_cast<const DataType_*>(ffn_inter_weight[i].data<data_t_>());
            params[layer_idx].ffn.intermediate_weight.bias =
                reinterpret_cast<const DataType_*>(ffn_inter_bias[i].data<data_t_>());
            params[layer_idx].ffn.output_weight.kernel =
                reinterpret_cast<const DataType_*>(ffn_out_weight[i].data<data_t_>());
            params[layer_idx].ffn.output_weight.bias =
                reinterpret_cast<const DataType_*>(ffn_out_bias[i].data<data_t_>());
        }

        decoding_params.layernorm.gamma =
            reinterpret_cast<const DataType_*>(decoder_ln_weight.data<data_t_>());
        decoding_params.layernorm.beta =
            reinterpret_cast<const DataType_*>(decoder_ln_bias.data<data_t_>());
        decoding_params.embedding_table =
            reinterpret_cast<const DataType_*>(word_emb.data<data_t_>());
        decoding_params.embedding_kernel =
            reinterpret_cast<const DataType_*>(emb_weight.data<data_t_>());
        decoding_params.position_encoding_table = reinterpret_cast<const DataType_*>(
            positional_embedding_weight.data<data_t_>());

        gpt_decoding->forward_context(params, decoding_params);
        gpt_decoding->forward(params, decoding_params);

        delete gpt_decoding;
        delete[] params;

#ifdef WITH_FT5
    }
#endif

    return {output_ids};
}

std::vector<paddle::Tensor> GPT2CUDAForward(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& start_length,
    const paddle::Tensor& word_embedding,
    const std::vector<paddle::Tensor>& self_ln_weight,
    const std::vector<paddle::Tensor>& self_ln_bias,
    const std::vector<paddle::Tensor>& self_q_weight,
    const std::vector<paddle::Tensor>& self_q_bias,
    const std::vector<paddle::Tensor>& self_k_weight,
    const std::vector<paddle::Tensor>& self_k_bias,
    const std::vector<paddle::Tensor>& self_v_weight,
    const std::vector<paddle::Tensor>& self_v_bias,
    const std::vector<paddle::Tensor>& self_out_weight,
    const std::vector<paddle::Tensor>& self_out_bias,
    const std::vector<paddle::Tensor>& ffn_ln_weight,
    const std::vector<paddle::Tensor>& ffn_ln_bias,
    const std::vector<paddle::Tensor>& ffn_inter_weight,
    const std::vector<paddle::Tensor>& ffn_inter_bias,
    const std::vector<paddle::Tensor>& ffn_out_weight,
    const std::vector<paddle::Tensor>& ffn_out_bias,
    const paddle::Tensor& decoder_ln_weight,
    const paddle::Tensor& decoder_ln_bias,
    const paddle::Tensor& positional_embedding_weight,
    const paddle::Tensor& emb_weight,
    paddle::Tensor& output_ids,
    const int& topk,
    const float& topp,
    const int& max_len,
    const int& n_head,
    const int& size_per_head,
    const int& num_layer,
    const int& bos_id,
    const int& eos_id,
    const float& temperature,
    const bool& use_fp16 = false,
    const int& tensor_para_size = 1,
    const int& layer_para_size = 1,
    const int& layer_para_batch_size = 1) {
  auto stream = word_embedding.stream();
  cublasSetStream(CublasHandle::GetInstance()->cublas_handle_, stream);
  std::vector<paddle::Tensor> ret;

  if (use_fp16) {
    ret = gpt_kernel<paddle::DataType::FLOAT16>(input,
                                                 attn_mask,
                                                 start_length,
                                                 word_embedding,
                                                 self_ln_weight,
                                                 self_ln_bias,
                                                 self_q_weight,
                                                 self_q_bias,
                                                 self_k_weight,
                                                 self_k_bias,
                                                 self_v_weight,
                                                 self_v_bias,
                                                 self_out_weight,
                                                 self_out_bias,
                                                 ffn_ln_weight,
                                                 ffn_ln_bias,
                                                 ffn_inter_weight,
                                                 ffn_inter_bias,
                                                 ffn_out_weight,
                                                 ffn_out_bias,
                                                 decoder_ln_weight,
                                                 decoder_ln_bias,
                                                 positional_embedding_weight,
                                                 emb_weight,
                                                 output_ids,
                                                 topk,
                                                 topp,
                                                 max_len,
                                                 n_head,
                                                 size_per_head,
                                                 num_layer,
                                                 bos_id,
                                                 eos_id,
                                                 temperature,
                                                 stream,
                                                 tensor_para_size,
                                                 layer_para_size,
                                                 layer_para_batch_size);
  } else {
    ret = gpt_kernel<paddle::DataType::FLOAT32>(input,
                                                 attn_mask,
                                                 start_length,
                                                 word_embedding,
                                                 self_ln_weight,
                                                 self_ln_bias,
                                                 self_q_weight,
                                                 self_q_bias,
                                                 self_k_weight,
                                                 self_k_bias,
                                                 self_v_weight,
                                                 self_v_bias,
                                                 self_out_weight,
                                                 self_out_bias,
                                                 ffn_ln_weight,
                                                 ffn_ln_bias,
                                                 ffn_inter_weight,
                                                 ffn_inter_bias,
                                                 ffn_out_weight,
                                                 ffn_out_bias,
                                                 decoder_ln_weight,
                                                 decoder_ln_bias,
                                                 positional_embedding_weight,
                                                 emb_weight,
                                                 output_ids,
                                                 topk,
                                                 topp,
                                                 max_len,
                                                 n_head,
                                                 size_per_head,
                                                 num_layer,
                                                 bos_id,
                                                 eos_id,
                                                 temperature,
                                                 stream,
                                                 tensor_para_size,
                                                 layer_para_size,
                                                 layer_para_batch_size);
  }

  return ret;
}
