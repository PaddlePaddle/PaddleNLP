#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>

#include "cublas_handle.h"
#include "fastertransformer/cuda/cub/cub.cuh"
#include "fusion_ernie3_op.h"
#include "pd_traits.h"

template <paddle::DataType D>
std::vector<paddle::Tensor> ernie3_kernel(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& start_length,
    const paddle::Tensor& word_embedding,
    const std::vector<paddle::Tensor>& sharing_self_ln_weight,
    const std::vector<paddle::Tensor>& sharing_self_ln_bias,
    const std::vector<paddle::Tensor>& sharing_self_q_weight,
    const std::vector<paddle::Tensor>& sharing_self_q_bias,
    const std::vector<paddle::Tensor>& sharing_self_k_weight,
    const std::vector<paddle::Tensor>& sharing_self_k_bias,
    const std::vector<paddle::Tensor>& sharing_self_v_weight,
    const std::vector<paddle::Tensor>& sharing_self_v_bias,
    const std::vector<paddle::Tensor>& sharing_self_out_weight,
    const std::vector<paddle::Tensor>& sharing_self_out_bias,
    const std::vector<paddle::Tensor>& sharing_ffn_ln_weight,
    const std::vector<paddle::Tensor>& sharing_ffn_ln_bias,
    const std::vector<paddle::Tensor>& sharing_ffn_inter_weight,
    const std::vector<paddle::Tensor>& sharing_ffn_inter_bias,
    const std::vector<paddle::Tensor>& sharing_ffn_out_weight,
    const std::vector<paddle::Tensor>& sharing_ffn_out_bias,
    const paddle::Tensor& sharing_decoder_ln_weight,
    const paddle::Tensor& sharing_decoder_ln_bias,
    const paddle::Tensor&  sharing_to_nlg_weight,
    const std::vector<paddle::Tensor>& nlg_self_ln_weight,
    const std::vector<paddle::Tensor>& nlg_self_ln_bias,
    const std::vector<paddle::Tensor>& nlg_self_q_weight,
    const std::vector<paddle::Tensor>& nlg_self_q_bias,
    const std::vector<paddle::Tensor>& nlg_self_k_weight,
    const std::vector<paddle::Tensor>& nlg_self_k_bias,
    const std::vector<paddle::Tensor>& nlg_self_v_weight,
    const std::vector<paddle::Tensor>& nlg_self_v_bias,
    const std::vector<paddle::Tensor>& nlg_self_out_weight,
    const std::vector<paddle::Tensor>& nlg_self_out_bias,
    const std::vector<paddle::Tensor>& nlg_ffn_ln_weight,
    const std::vector<paddle::Tensor>& nlg_ffn_ln_bias,
    const std::vector<paddle::Tensor>& nlg_ffn_inter_weight,
    const std::vector<paddle::Tensor>& nlg_ffn_inter_bias,
    const std::vector<paddle::Tensor>& nlg_ffn_out_weight,
    const std::vector<paddle::Tensor>& nlg_ffn_out_bias,
    const paddle::Tensor& nlg_decoder_ln_weight,
    const paddle::Tensor& nlg_decoder_ln_bias,
    const paddle::Tensor& trans_weight,
    const paddle::Tensor& trans_bias,
    const paddle::Tensor& lm_ln_weight,
    const paddle::Tensor& lm_ln_bias,
    const paddle::Tensor& lm_out_weight, // embedding_weight
    const paddle::Tensor& lm_out_bias, // embedding_bias, gpt have no
    const paddle::Tensor& positional_embedding_weight,
    paddle::Tensor& output_ids,
    const int topk,
    const float topp,
    const int max_len,
    const int sharing_n_head,
    const int sharing_size_per_head,
    const int sharing_num_layer,
    const int nlg_n_head,
    const int nlg_size_per_head,
    const int nlg_num_layer,
    const int bos_id,
    const int eos_id,
    const float temperature,
    const float repetition_penalty,
    const int min_length,
    cudaStream_t stream) {
  auto input_dims = input.shape();
  int batch_size_ = input_dims[0];
  int start_len = input_dims[1];
  const int vocab_size = word_embedding.shape()[0];

  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t_;

  DecodingInitParam<DataType_> decoding_params;
  decoding_params.cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
  decoding_params.cublaslt_handle = CublasHandle::GetInstance()->cublaslt_handle_;

  decoding_params.output_ids = output_ids.mutable_data<int>(word_embedding.place());

  typedef DecoderTransformerTraits<traits_::OpType> DecodingTraits_;
  decoding_params.stream = stream;
  fastertransformer::Allocator<AllocatorType::PD> allocator_(stream);

  // sharing ParallelParam
  const int sharing_hidden_unit = sharing_size_per_head * sharing_n_head;
  TensorParallelParam sharing_tensor_parallel_param;
  LayerParallelParam sharing_layer_parallel_param;
  // TODO: multi-cards supports.
  // ncclComm_t tensor_para_nccl_comm, layer_para_nccl_comm;

  sharing_tensor_parallel_param.rank = 0;
  sharing_tensor_parallel_param.world_size = 1;
  // TODO: multi-cards supports.
  // tensor_parallel_param.nccl_comm = tensor_para_nccl_comm;
  sharing_tensor_parallel_param.local_head_num_ = sharing_n_head;
  sharing_tensor_parallel_param.local_hidden_units_ = sharing_hidden_unit;

  sharing_layer_parallel_param.rank = 0;
  sharing_layer_parallel_param.world_size = 1;
  // TODO: multi-cards supports.
  // layer_parallel_param.nccl_comm = layer_para_nccl_comm;
  sharing_layer_parallel_param.layers_per_group = sharing_num_layer;
  sharing_layer_parallel_param.local_batch_size = batch_size_;

  // nlg ParallelParam
  const int nlg_hidden_unit = nlg_size_per_head * nlg_n_head;
  TensorParallelParam nlg_tensor_parallel_param;
  LayerParallelParam nlg_layer_parallel_param;
  // TODO: multi-cards supports.
  // ncclComm_t tensor_para_nccl_comm, layer_para_nccl_comm;

  nlg_tensor_parallel_param.rank = 0;
  nlg_tensor_parallel_param.world_size = 1;
  // TODO: multi-cards supports.
  // tensor_parallel_param.nccl_comm = tensor_para_nccl_comm;
  nlg_tensor_parallel_param.local_head_num_ = nlg_n_head;
  nlg_tensor_parallel_param.local_hidden_units_ = nlg_hidden_unit;

  nlg_layer_parallel_param.rank = 0;
  nlg_layer_parallel_param.world_size = 1;
  // TODO: multi-cards supports.
  // layer_parallel_param.nccl_comm = layer_para_nccl_comm;
  nlg_layer_parallel_param.layers_per_group = nlg_num_layer;
  nlg_layer_parallel_param.local_batch_size = batch_size_;

  DecodingErnie3<DecodingTraits_::OpType>* ernie3_decoding;

  decoding_params.request_batch_size = batch_size_;
  decoding_params.max_input_len = start_len;
  decoding_params.request_input_len = start_len;
  decoding_params.request_output_len = max_len - start_len;

  decoding_params.d_start_ids = const_cast<int *>(input.data<int>());
  decoding_params.d_attn_mask =
      reinterpret_cast<DataType_*>(const_cast<data_t_ *>(attn_mask.data<data_t_>()));
  decoding_params.d_start_lengths = start_length.data<int>();

  ernie3_decoding =
      new DecodingErnie3<DecodingTraits_::OpType>(allocator_,
                                               batch_size_,
                                               max_len,
                                               sharing_n_head,
                                               nlg_n_head,
                                               sharing_size_per_head,
                                               nlg_size_per_head,
                                               vocab_size,
                                               sharing_num_layer,
                                               nlg_num_layer,
                                               bos_id,
                                               eos_id,
                                               topk,
                                               topp,
                                               temperature,
                                               1, /*tensor_para_size*/
                                               1, /*layer_para_size*/
                                               true /*is_fuse_QKV*/,
                                               repetition_penalty,
                                               min_length);

  ernie3_decoding->set_tensor_parallel_param(sharing_tensor_parallel_param, nlg_tensor_parallel_param);
  ernie3_decoding->set_layer_parallel_param(sharing_layer_parallel_param, nlg_layer_parallel_param);

  // sharing decoder params
  DecoderInitParam<DataType_>* sharing_params =
      new DecoderInitParam<DataType_>[sharing_num_layer];

  for (int i = 0; i < sharing_num_layer; ++i) {
    if (sharing_layer_parallel_param.is_valid(i) == false) {
      continue;
    }

    sharing_params[i].stream = stream;
    sharing_params[i].cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
    sharing_params[i].cublaslt_handle = CublasHandle::GetInstance()->cublaslt_handle_;

    sharing_params[i].request_batch_size = batch_size_;
    sharing_params[i].request_max_mem_seq_len = start_len;

    sharing_params[i].self_layernorm.gamma =
        reinterpret_cast<const DataType_*>(sharing_self_ln_weight[i].data<data_t_>());
    sharing_params[i].self_layernorm.beta =
        reinterpret_cast<const DataType_*>(sharing_self_ln_bias[i].data<data_t_>());

    sharing_params[i].self_attention.query_weight.kernel =
        reinterpret_cast<const DataType_*>(sharing_self_q_weight[i].data<data_t_>());
    sharing_params[i].self_attention.query_weight.bias =
        reinterpret_cast<const DataType_*>(sharing_self_q_bias[i].data<data_t_>());
    sharing_params[i].self_attention.key_weight.kernel =
        reinterpret_cast<const DataType_*>(sharing_self_k_weight[i].data<data_t_>());
    sharing_params[i].self_attention.key_weight.bias =
        reinterpret_cast<const DataType_*>(sharing_self_k_bias[i].data<data_t_>());
    sharing_params[i].self_attention.value_weight.kernel =
        reinterpret_cast<const DataType_*>(sharing_self_v_weight[i].data<data_t_>());
    sharing_params[i].self_attention.value_weight.bias =
        reinterpret_cast<const DataType_*>(sharing_self_v_bias[i].data<data_t_>());

    sharing_params[i].self_attention.attention_output_weight.kernel =
        reinterpret_cast<const DataType_*>(sharing_self_out_weight[i].data<data_t_>());
    sharing_params[i].self_attention.attention_output_weight.bias =
        reinterpret_cast<const DataType_*>(sharing_self_out_bias[i].data<data_t_>());

    sharing_params[i].ffn_layernorm.gamma =
        reinterpret_cast<const DataType_*>(sharing_ffn_ln_weight[i].data<data_t_>());
    sharing_params[i].ffn_layernorm.beta =
        reinterpret_cast<const DataType_*>(sharing_ffn_ln_bias[i].data<data_t_>());

    sharing_params[i].ffn.intermediate_weight.kernel =
        reinterpret_cast<const DataType_*>(sharing_ffn_inter_weight[i].data<data_t_>());
    sharing_params[i].ffn.intermediate_weight.bias =
        reinterpret_cast<const DataType_*>(sharing_ffn_inter_bias[i].data<data_t_>());
    sharing_params[i].ffn.output_weight.kernel =
        reinterpret_cast<const DataType_*>(sharing_ffn_out_weight[i].data<data_t_>());
    sharing_params[i].ffn.output_weight.bias =
        reinterpret_cast<const DataType_*>(sharing_ffn_out_bias[i].data<data_t_>());
  }
  // nlg decoder params
  DecoderInitParam<DataType_>* nlg_params =
      new DecoderInitParam<DataType_>[nlg_num_layer];

  for (int i = 0; i < nlg_num_layer; ++i) {
    if (nlg_layer_parallel_param.is_valid(i) == false) {
      continue;
    }

    nlg_params[i].stream = stream;
    nlg_params[i].cublas_handle = CublasHandle::GetInstance()->cublas_handle_;
    nlg_params[i].cublaslt_handle = CublasHandle::GetInstance()->cublaslt_handle_;

    nlg_params[i].request_batch_size = batch_size_;
    nlg_params[i].request_max_mem_seq_len = start_len;

    nlg_params[i].self_layernorm.gamma =
        reinterpret_cast<const DataType_*>(nlg_self_ln_weight[i].data<data_t_>());
    nlg_params[i].self_layernorm.beta =
        reinterpret_cast<const DataType_*>(nlg_self_ln_bias[i].data<data_t_>());

    nlg_params[i].self_attention.query_weight.kernel =
        reinterpret_cast<const DataType_*>(nlg_self_q_weight[i].data<data_t_>());
    nlg_params[i].self_attention.query_weight.bias =
        reinterpret_cast<const DataType_*>(nlg_self_q_bias[i].data<data_t_>());
    nlg_params[i].self_attention.key_weight.kernel =
        reinterpret_cast<const DataType_*>(nlg_self_k_weight[i].data<data_t_>());
    nlg_params[i].self_attention.key_weight.bias =
        reinterpret_cast<const DataType_*>(nlg_self_k_bias[i].data<data_t_>());
    nlg_params[i].self_attention.value_weight.kernel =
        reinterpret_cast<const DataType_*>(nlg_self_v_weight[i].data<data_t_>());
    nlg_params[i].self_attention.value_weight.bias =
        reinterpret_cast<const DataType_*>(nlg_self_v_bias[i].data<data_t_>());

    nlg_params[i].self_attention.attention_output_weight.kernel =
        reinterpret_cast<const DataType_*>(nlg_self_out_weight[i].data<data_t_>());
    nlg_params[i].self_attention.attention_output_weight.bias =
        reinterpret_cast<const DataType_*>(nlg_self_out_bias[i].data<data_t_>());

    nlg_params[i].ffn_layernorm.gamma =
        reinterpret_cast<const DataType_*>(nlg_ffn_ln_weight[i].data<data_t_>());
    nlg_params[i].ffn_layernorm.beta =
        reinterpret_cast<const DataType_*>(nlg_ffn_ln_bias[i].data<data_t_>());

    nlg_params[i].ffn.intermediate_weight.kernel =
        reinterpret_cast<const DataType_*>(nlg_ffn_inter_weight[i].data<data_t_>());
    nlg_params[i].ffn.intermediate_weight.bias =
        reinterpret_cast<const DataType_*>(nlg_ffn_inter_bias[i].data<data_t_>());
    nlg_params[i].ffn.output_weight.kernel =
        reinterpret_cast<const DataType_*>(nlg_ffn_out_weight[i].data<data_t_>());
    nlg_params[i].ffn.output_weight.bias =
        reinterpret_cast<const DataType_*>(nlg_ffn_out_bias[i].data<data_t_>());
  }

  decoding_params.layernorm.gamma =
      reinterpret_cast<const DataType_*>(sharing_decoder_ln_weight.data<data_t_>());
  decoding_params.layernorm.beta =
      reinterpret_cast<const DataType_*>(sharing_decoder_ln_bias.data<data_t_>());
  decoding_params.embedding_table =
      reinterpret_cast<const DataType_*>(word_embedding.data<data_t_>());
  decoding_params.embedding_kernel =
      reinterpret_cast<const DataType_*>(lm_out_weight.data<data_t_>());
  decoding_params.embedding_bias =
      reinterpret_cast<const DataType_*>(lm_out_bias.data<data_t_>());
  decoding_params.position_encoding_table = reinterpret_cast<const DataType_*>(
      positional_embedding_weight.data<data_t_>());

 decoding_params.sharing_to_nlg_kernel = reinterpret_cast<const DataType_*>(
      sharing_to_nlg_weight.data<data_t_>());
 decoding_params.trans_kernel = reinterpret_cast<const DataType_*>(
      trans_weight.data<data_t_>());
 decoding_params.trans_bias = reinterpret_cast<const DataType_*>(
      trans_bias.data<data_t_>());
  decoding_params.nlg_decoder_layernorm.gamma =
      reinterpret_cast<const DataType_*>(nlg_decoder_ln_weight.data<data_t_>());
  decoding_params.nlg_decoder_layernorm.beta =
      reinterpret_cast<const DataType_*>(nlg_decoder_ln_bias.data<data_t_>());
  decoding_params.lm_out_layernorm.gamma =
      reinterpret_cast<const DataType_*>(lm_ln_weight.data<data_t_>());
  decoding_params.lm_out_layernorm.beta =
      reinterpret_cast<const DataType_*>(lm_ln_bias.data<data_t_>());

  ernie3_decoding->forward_context(sharing_params, nlg_params, decoding_params);
  ernie3_decoding->forward(sharing_params, nlg_params, decoding_params);

  delete ernie3_decoding;
  delete[] sharing_params;
  delete[] nlg_params;

  return {output_ids};
}

std::vector<paddle::Tensor> Ernie3CUDAForward(
    const paddle::Tensor& input,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& start_length,
    const paddle::Tensor& word_embedding,
    const std::vector<paddle::Tensor>& sharing_self_ln_weight,
    const std::vector<paddle::Tensor>& sharing_self_ln_bias,
    const std::vector<paddle::Tensor>& sharing_self_q_weight,
    const std::vector<paddle::Tensor>& sharing_self_q_bias,
    const std::vector<paddle::Tensor>& sharing_self_k_weight,
    const std::vector<paddle::Tensor>& sharing_self_k_bias,
    const std::vector<paddle::Tensor>& sharing_self_v_weight,
    const std::vector<paddle::Tensor>& sharing_self_v_bias,
    const std::vector<paddle::Tensor>& sharing_self_out_weight,
    const std::vector<paddle::Tensor>& sharing_self_out_bias,
    const std::vector<paddle::Tensor>& sharing_ffn_ln_weight,
    const std::vector<paddle::Tensor>& sharing_ffn_ln_bias,
    const std::vector<paddle::Tensor>& sharing_ffn_inter_weight,
    const std::vector<paddle::Tensor>& sharing_ffn_inter_bias,
    const std::vector<paddle::Tensor>& sharing_ffn_out_weight,
    const std::vector<paddle::Tensor>& sharing_ffn_out_bias,
    const paddle::Tensor& sharing_decoder_ln_weight,
    const paddle::Tensor& sharing_decoder_ln_bias,
    const paddle::Tensor&  sharing_to_nlg_weight,
    const std::vector<paddle::Tensor>& nlg_self_ln_weight,
    const std::vector<paddle::Tensor>& nlg_self_ln_bias,
    const std::vector<paddle::Tensor>& nlg_self_q_weight,
    const std::vector<paddle::Tensor>& nlg_self_q_bias,
    const std::vector<paddle::Tensor>& nlg_self_k_weight,
    const std::vector<paddle::Tensor>& nlg_self_k_bias,
    const std::vector<paddle::Tensor>& nlg_self_v_weight,
    const std::vector<paddle::Tensor>& nlg_self_v_bias,
    const std::vector<paddle::Tensor>& nlg_self_out_weight,
    const std::vector<paddle::Tensor>& nlg_self_out_bias,
    const std::vector<paddle::Tensor>& nlg_ffn_ln_weight,
    const std::vector<paddle::Tensor>& nlg_ffn_ln_bias,
    const std::vector<paddle::Tensor>& nlg_ffn_inter_weight,
    const std::vector<paddle::Tensor>& nlg_ffn_inter_bias,
    const std::vector<paddle::Tensor>& nlg_ffn_out_weight,
    const std::vector<paddle::Tensor>& nlg_ffn_out_bias,
    const paddle::Tensor& nlg_decoder_ln_weight,
    const paddle::Tensor& nlg_decoder_ln_bias,
    const paddle::Tensor& trans_weight,
    const paddle::Tensor& trans_bias,
    const paddle::Tensor& lm_ln_weight,
    const paddle::Tensor& lm_ln_bias,
    const paddle::Tensor& lm_out_weight, // embedding_weight
    const paddle::Tensor& lm_out_bias, // embedding_bias, gpt have no
    const paddle::Tensor& positional_embedding_weight,
    paddle::Tensor& output_ids,
    const int topk,
    const float topp,
    const int max_len,
    const int sharing_n_head,
    const int sharing_size_per_head,
    const int sharing_num_layer,
    const int nlg_n_head,
    const int nlg_size_per_head,
    const int nlg_num_layer,
    const int bos_id,
    const int eos_id,
    const float temperature,
    const float repetition_penalty,
    const int min_length,
    const bool use_fp16 = false) {
  auto stream = word_embedding.stream();
  cublasSetStream(CublasHandle::GetInstance()->cublas_handle_, stream);

  std::vector<paddle::Tensor> ret;

  if (use_fp16) {
    ret = ernie3_kernel<paddle::DataType::FLOAT16>(input,
                                                attn_mask,
                                                start_length,
                                                word_embedding,
                                                sharing_self_ln_weight,
                                                sharing_self_ln_bias,
                                                sharing_self_q_weight,
                                                sharing_self_q_bias,
                                                sharing_self_k_weight,
                                                sharing_self_k_bias,
                                                sharing_self_v_weight,
                                                sharing_self_v_bias,
                                                sharing_self_out_weight,
                                                sharing_self_out_bias,
                                                sharing_ffn_ln_weight,
                                                sharing_ffn_ln_bias,
                                                sharing_ffn_inter_weight,
                                                sharing_ffn_inter_bias,
                                                sharing_ffn_out_weight,
                                                sharing_ffn_out_bias,
                                                sharing_decoder_ln_weight,
                                                sharing_decoder_ln_bias,
                                                sharing_to_nlg_weight,
                                                nlg_self_ln_weight,
                                                nlg_self_ln_bias,
                                                nlg_self_q_weight,
                                                nlg_self_q_bias,
                                                nlg_self_k_weight,
                                                nlg_self_k_bias,
                                                nlg_self_v_weight,
                                                nlg_self_v_bias,
                                                nlg_self_out_weight,
                                                nlg_self_out_bias,
                                                nlg_ffn_ln_weight,
                                                nlg_ffn_ln_bias,
                                                nlg_ffn_inter_weight,
                                                nlg_ffn_inter_bias,
                                                nlg_ffn_out_weight,
                                                nlg_ffn_out_bias,
                                                nlg_decoder_ln_weight,
                                                nlg_decoder_ln_bias,
                                                trans_weight,
                                                trans_bias,
                                                lm_ln_weight,
                                                lm_ln_bias,
                                                lm_out_weight,
                                                lm_out_bias,
                                                positional_embedding_weight,
                                                output_ids,
                                                topk,
                                                topp,
                                                max_len,
                                                sharing_n_head,
                                                sharing_size_per_head,
                                                sharing_num_layer,
                                                nlg_n_head,
                                                nlg_size_per_head,
                                                nlg_num_layer,
                                                bos_id,
                                                eos_id,
                                                temperature,
                                                repetition_penalty,
                                                min_length,
                                                stream);
  } else {
    ret = ernie3_kernel<paddle::DataType::FLOAT32>(input,
                                                attn_mask,
                                                start_length,
                                                word_embedding,
                                                sharing_self_ln_weight,
                                                sharing_self_ln_bias,
                                                sharing_self_q_weight,
                                                sharing_self_q_bias,
                                                sharing_self_k_weight,
                                                sharing_self_k_bias,
                                                sharing_self_v_weight,
                                                sharing_self_v_bias,
                                                sharing_self_out_weight,
                                                sharing_self_out_bias,
                                                sharing_ffn_ln_weight,
                                                sharing_ffn_ln_bias,
                                                sharing_ffn_inter_weight,
                                                sharing_ffn_inter_bias,
                                                sharing_ffn_out_weight,
                                                sharing_ffn_out_bias,
                                                sharing_decoder_ln_weight,
                                                sharing_decoder_ln_bias,
                                                sharing_to_nlg_weight,
                                                nlg_self_ln_weight,
                                                nlg_self_ln_bias,
                                                nlg_self_q_weight,
                                                nlg_self_q_bias,
                                                nlg_self_k_weight,
                                                nlg_self_k_bias,
                                                nlg_self_v_weight,
                                                nlg_self_v_bias,
                                                nlg_self_out_weight,
                                                nlg_self_out_bias,
                                                nlg_ffn_ln_weight,
                                                nlg_ffn_ln_bias,
                                                nlg_ffn_inter_weight,
                                                nlg_ffn_inter_bias,
                                                nlg_ffn_out_weight,
                                                nlg_ffn_out_bias,
                                                nlg_decoder_ln_weight,
                                                nlg_decoder_ln_bias,
                                                trans_weight,
                                                trans_bias,
                                                lm_ln_weight,
                                                lm_ln_bias,
                                                lm_out_weight,
                                                lm_out_bias,
                                                positional_embedding_weight,
                                                output_ids,
                                                topk,
                                                topp,
                                                max_len,
                                                sharing_n_head,
                                                sharing_size_per_head,
                                                sharing_num_layer,
                                                nlg_n_head,
                                                nlg_size_per_head,
                                                nlg_num_layer,
                                                bos_id,
                                                eos_id,
                                                temperature,
                                                repetition_penalty,
                                                min_length,
                                                stream);
  }
  return ret;
}
