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

#include "append_attn/encoder_write_cache_with_rope_kernel.h"

template <paddle::DataType D>
void encoder_write_cache_with_rope(const paddle::Tensor& qkv, // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 * gqa_group_size, head_dim] if GQA)
														const paddle::Tensor& rotary_emb,
														const paddle::Tensor& seq_lens_this_time,
														const paddle::Tensor& seq_lens_encoder,
														const paddle::Tensor& seq_lens_decoder,
														const paddle::Tensor& padding_offsets,
														const paddle::Tensor& cum_offsets,
														const paddle::Tensor& block_table,
														const paddle::Tensor& batch_ids,
														const paddle::Tensor& tile_ids_per_batch,
														const paddle::Tensor& num_blocks_x_cpu,
														const paddle::Tensor& max_enc_len,
														const paddle::optional<paddle::Tensor>& qkv_out_scales,
														const paddle::optional<paddle::Tensor>& qkv_biases,
														const paddle::optional<paddle::Tensor>& cache_k_scale,
														const paddle::optional<paddle::Tensor>& cache_v_scale,
														const paddle::optional<paddle::Tensor>& cache_k_zp,
														const paddle::optional<paddle::Tensor>& cache_v_zp,
														paddle::Tensor& qkv_out_tmp,
														paddle::Tensor& cache_k,
														paddle::Tensor& cache_v,
														const std::string& cache_quant_type_str,
														const int max_input_len,
														const int num_heads,
														const int kv_num_heads,
														const int head_dim) {
	int num_blocks_data = num_blocks_x_cpu.data<int>()[0];
	int max_enc_len_data = max_enc_len.data<int>()[0];

	if (max_enc_len_data <= 0) return;
	typedef PDTraits<D> traits_;
	typedef typename traits_::DataType DataType_;
	typedef typename traits_::data_t data_t;
	auto stream = qkv.stream();

	
  	// auto qkv_out = paddle::empty_like(qkv, D, paddle::GPUPlace());

	if (qkv_out_scales) {
		EncoderWriteCacheWithRopeKernel<data_t, int>(qkv, // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 * gqa_group_size, head_dim] if GQA)
														rotary_emb,
														seq_lens_this_time,
														seq_lens_encoder,
														seq_lens_decoder,
														padding_offsets,
														cum_offsets,
														block_table,
														batch_ids,
														tile_ids_per_batch,
														qkv_out_scales,
														qkv_biases,
														cache_k_scale,
														cache_v_scale,
														cache_k_zp,
														cache_v_zp,
														cache_quant_type_str,
														num_blocks_data,
														max_input_len,
														num_heads,
														kv_num_heads,
														head_dim,
														stream,
														&qkv_out_tmp,
														&cache_k, 
														&cache_v);
	} else {
		EncoderWriteCacheWithRopeKernel<data_t, data_t>(qkv, // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 * gqa_group_size, head_dim] if GQA)
															rotary_emb,
															seq_lens_this_time,
															seq_lens_encoder,
															seq_lens_decoder,
															padding_offsets,
															cum_offsets,
															block_table,
															batch_ids,
															tile_ids_per_batch,
															qkv_out_scales,
															qkv_biases,
															cache_k_scale,
															cache_v_scale,
															cache_k_zp,
															cache_v_zp,
															cache_quant_type_str,
															num_blocks_data,
															max_input_len,
															num_heads,
															kv_num_heads,
															head_dim,
															stream,
															&qkv_out_tmp,
															&cache_k, 
															&cache_v);
	}
}

void EncoderWriteCacheWithRope(const paddle::Tensor& qkv, // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 * gqa_group_size, head_dim] if GQA)
													const paddle::Tensor& rotary_emb,
													const paddle::Tensor& seq_lens_this_time,
													const paddle::Tensor& seq_lens_encoder,
													const paddle::Tensor& seq_lens_decoder,
													const paddle::Tensor& padding_offsets,
													const paddle::Tensor& cum_offsets,
													const paddle::Tensor& block_table,
													const paddle::Tensor& batch_ids,
													const paddle::Tensor& tile_ids_per_batch,
													const paddle::Tensor& num_blocks_x_cpu,
													const paddle::Tensor& max_enc_len,
													paddle::Tensor& qkv_out_tmp,
													paddle::Tensor& cache_k, 
													paddle::Tensor& cache_v,
													const paddle::optional<paddle::Tensor>& qkv_out_scales,
													const paddle::optional<paddle::Tensor>& qkv_biases,
													const paddle::optional<paddle::Tensor>& cache_k_scale,
													const paddle::optional<paddle::Tensor>& cache_v_scale,
													const paddle::optional<paddle::Tensor>& cache_k_zp,
													const paddle::optional<paddle::Tensor>& cache_v_zp,
													const std::string& cache_quant_type_str,
													const int max_input_len,
													const int num_heads,
													const int kv_num_heads,
													const int head_dim) {
	
	switch (qkv_out_tmp.type()) {
		case paddle::DataType::FLOAT16: {
			return encoder_write_cache_with_rope<paddle::DataType::FLOAT16>(
				qkv, // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 * gqa_group_size, head_dim] if GQA)
				rotary_emb,
				seq_lens_this_time,
				seq_lens_encoder,
				seq_lens_decoder,
				padding_offsets,
				cum_offsets,
				block_table,
				batch_ids,
				tile_ids_per_batch,
				num_blocks_x_cpu,
				max_enc_len,
				qkv_out_scales,
				qkv_biases,
				cache_k_scale,
				cache_v_scale,
				cache_k_zp,
				cache_v_zp,
				qkv_out_tmp,
				cache_k,
				cache_v,
				cache_quant_type_str,
				max_input_len,
				num_heads,
				kv_num_heads,
				head_dim);
		}
		case paddle::DataType::BFLOAT16: {
			return encoder_write_cache_with_rope<paddle::DataType::BFLOAT16>(
				qkv, // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 * gqa_group_size, head_dim] if GQA)
				rotary_emb,
				seq_lens_this_time,
				seq_lens_encoder,
				seq_lens_decoder,
				padding_offsets,
				cum_offsets,
				block_table,
				batch_ids,
				tile_ids_per_batch,
				num_blocks_x_cpu,
				max_enc_len,
				qkv_out_scales,
				qkv_biases,
				cache_k_scale,
				cache_v_scale,
				cache_k_zp,
				cache_v_zp,
				qkv_out_tmp,
				cache_k,
				cache_v,
				cache_quant_type_str,
				max_input_len,
				num_heads,
				kv_num_heads,
				head_dim);
		}
		default: {
			PD_THROW(
					"NOT supported data type. "
					"Only float16 and bfloat16 are supported. ");
			break;
		}
	}
}

// std::vector<paddle::DataType> EncoderWriteCacheWithRopeInferDtype(const paddle::DataType& qkv_dtype, // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 * gqa_group_size, head_dim] if GQA)
// 																const paddle::DataType& rotary_emb_dtype,
// 																const paddle::DataType& seq_lens_this_time_dtype,
// 																const paddle::DataType& seq_lens_encoder_dtype,
// 																const paddle::DataType& seq_lens_decoder_dtype,
// 																const paddle::DataType& padding_offsets_dtype,
// 																const paddle::DataType& cum_offsets_dtype,
// 																const paddle::DataType& block_table_dtype,
// 																const paddle::DataType& batch_ids_dtype,
// 																const paddle::DataType& tile_ids_per_batch_dtype,
// 																const paddle::DataType& num_blocks_x_cpu_dtype,
// 																const paddle::DataType& max_enc_len_dtype,
// 																const paddle::DataType& cache_k_dtype, 
// 																const paddle::DataType& cache_v_dtype,
// 																const paddle::optional<paddle::DataType>& qkv_out_scales_dtype,
// 																const paddle::optional<paddle::DataType>& qkv_biases_dtype,
// 																const paddle::optional<paddle::DataType>& cache_k_scale_dtype,
// 																const paddle::optional<paddle::DataType>& cache_v_scale_dtype,
// 																const paddle::optional<paddle::DataType>& cache_k_zp_dtype,
// 																const paddle::optional<paddle::DataType>& cache_v_zp_dtype
// 																) {
//     if (qkv_out_scales_dtype) {
// 		return  {*qkv_out_scales_dtype};
// 	} else {
// 		return {qkv_dtype};
// 	}
// }

// std::vector<std::vector<int64_t>> EncoderWriteCacheWithRopeInferShape(const std::vector<int64_t>& qkv_shape, // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 * gqa_group_size, head_dim] if GQA)
// 																	const std::vector<int64_t>& rotary_emb_shape,
// 																	const std::vector<int64_t>& seq_lens_this_time_shape,
// 																	const std::vector<int64_t>& seq_lens_encoder_shape,
// 																	const std::vector<int64_t>& seq_lens_decoder_shape,
// 																	const std::vector<int64_t>& padding_offsets_shape,
// 																	const std::vector<int64_t>& cum_offsets_shape,
// 																	const std::vector<int64_t>& block_table_shape,
// 																	const std::vector<int64_t>& batch_ids_shape,
// 																	const std::vector<int64_t>& tile_ids_per_batch_shape,
// 																	const std::vector<int64_t>& num_blocks_x_cpu_shape,
// 																	const std::vector<int64_t>& max_enc_len_shape,
// 																	const std::vector<int64_t>& cache_k_shape, 
// 																	const std::vector<int64_t>& cache_v_shape,
// 																	const paddle::optional<std::vector<int64_t>>& qkv_out_scales_shape,
// 																	const paddle::optional<std::vector<int64_t>>& qkv_biases_shape,
// 																	const paddle::optional<std::vector<int64_t>>& cache_k_scale_shape,
// 																	const paddle::optional<std::vector<int64_t>>& cache_v_scale_shape,
// 																	const paddle::optional<std::vector<int64_t>>& cache_k_zp_shape,
// 																	const paddle::optional<std::vector<int64_t>>& cache_v_zp_shape) {
// 	return {qkv_shape};
// }

PD_BUILD_OP(encoder_write_cache_with_rope)
    .Inputs({"qkv", "rotary_emb", "seq_lens_this_time", "seq_lens_encoder", "seq_lens_decoder", "padding_offsets", "cum_offsets", "block_table", "batch_ids", "tile_ids_per_batch", "num_blocks_x_cpu", "max_enc_len", "qkv_out_tmp", "cache_k", "cache_v", 
			paddle::Optional("qkv_out_scales"), paddle::Optional("qkv_biases"), paddle::Optional("cache_k_scale"), paddle::Optional("cache_v_scale"), paddle::Optional("cache_k_zp"), paddle::Optional("cache_v_zp")})
    .Outputs({"qkv_out", "key_cache_out", "value_cache_out"})
	.SetInplaceMap({{"qkv_out_tmp", "qkv_out"},
					{"cache_k", "key_cache_out"},
					{"cache_v", "value_cache_out"}})
    .Attrs({"cache_quant_type_str: std::string",
			"max_input_len: int",
			"num_heads: int",
			"kv_num_heads: int",
			"head_dim: int"})
    .SetKernelFn(PD_KERNEL(EncoderWriteCacheWithRope));
	// .SetInferShapeFn(PD_INFER_SHAPE(EncoderWriteCacheWithRopeInferShape))
	// .SetInferDtypeFn(PD_INFER_DTYPE(EncoderWriteCacheWithRopeInferDtype));
