#include "paddle/extension.h"


//
//  ============== fp16 kernels registry, for sm80 arch ==============
//
// impl: sageattn_qk_int_sv_f16_kernel.cu
// attn buffer kernel
// std::vector<paddle::Tensor>  qk_int8_sv_f16_accum_f16_attn_buf_fwd(
//                     paddle::Tensor& query,
//                     paddle::Tensor& key,
//                     paddle::Tensor& value,
//                     paddle::Tensor& output,
//                     paddle::Tensor& query_scale,
//                     paddle::Tensor& key_scale,
//                     int tensor_layout,
//                     int is_causal,
//                     int qk_quant_gran,
//                     float sm_scale,
//                     int return_lse);

// std::vector<std::vector<int64_t>> qk_int8_sv_f16_accum_f16_attn_buf_InferShape(
//   std::vector<int64_t> query_shape, 
//   std::vector<int64_t> key_shape, 
//   std::vector<int64_t> value_shape, 
//   std::vector<int64_t> output_shape, 
//   std::vector<int64_t> query_scale_shape, 
//   std::vector<int64_t> key_scale_shape) {

//     // force layout: NHD: [bsz, seq_len, num_heads, head_dim]
//     int64_t bsz = query_shape[0];
//     int64_t seq_len = query_shape[1];
//     int64_t h_qo = query_shape[2];

//     std::vector<int64_t> return_shape = {bsz, h_qo, seq_len};
//     return {return_shape};
// }

// std::vector<paddle::DataType> qk_int8_sv_f16_accum_f16_attn_buf_InferDtype(
//   paddle::DataType A_dtype,
//   paddle::DataType B_dtype,
//   paddle::DataType C_dtype,
//   paddle::DataType D_dtype,
//   paddle::DataType E_dtype,
//   paddle::DataType F_dtype) {
//   return {paddle::DataType::FLOAT32};
// }

// PD_BUILD_OP(qk_int8_sv_f16_accum_f16_attn_buf)
//     .Inputs({"query", "key", "value", "output", "query_scale", "key_scale"})
//     .Outputs({"out1", "out2", "out3", "out4", "out5", "out6", "lse"})
//     .SetInplaceMap({{"query", "out1"}, {"key", "out2"}, {"value", "out3"}, {"output", "out4"}, {"query_scale", "out5"}, {"key_scale", "out6"}}) // Inplace
//     .Attrs({"tensor_layout: int",
//             "is_causal: int",
//             "qk_quant_gran: int",
//             "sm_scale: float",
//             "return_lse: int"})
//     .SetKernelFn(PD_KERNEL(qk_int8_sv_f16_accum_f16_attn_buf_fwd))
//     .SetInferShapeFn(PD_INFER_SHAPE(qk_int8_sv_f16_accum_f16_attn_buf_InferShape))
//     .SetInferDtypeFn(PD_INFER_DTYPE(qk_int8_sv_f16_accum_f16_attn_buf_InferDtype));


// attn forward kernel: sv f16 accumulator f32
std::vector<paddle::Tensor> qk_int8_sv_f16_accum_f32_attn_fwd(
                    paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);


std::vector<std::vector<int64_t>> qk_int8_sv_f16_accum_f32_attn_InferShape(
  std::vector<int64_t> query_shape, 
  std::vector<int64_t> key_shape, 
  std::vector<int64_t> value_shape, 
  std::vector<int64_t> output_shape, 
  std::vector<int64_t> query_scale_shape, 
  std::vector<int64_t> key_scale_shape) {

    // force layout: NHD: [bsz, seq_len, num_heads, head_dim]
    int64_t bsz = query_shape[0];
    int64_t seq_len = query_shape[1];
    int64_t h_qo = query_shape[2];

    std::vector<int64_t> return_shape = {bsz, h_qo, seq_len};
    return {return_shape};
}

std::vector<paddle::DataType> qk_int8_sv_f16_accum_f32_attn_InferDtype(
  paddle::DataType A_dtype,
  paddle::DataType B_dtype,
  paddle::DataType C_dtype,
  paddle::DataType D_dtype,
  paddle::DataType E_dtype,
  paddle::DataType F_dtype) {
  return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(qk_int8_sv_f16_accum_f32_attn)
    .Inputs({"query", "key", "value", "output", "query_scale", "key_scale"})
    .Outputs({"out1", "out2", "out3", "out4", "out5", "out6", "lse"})
    .SetInplaceMap({{"query", "out1"}, {"key", "out2"}, {"value", "out3"}, {"output", "out4"}, {"query_scale", "out5"}, {"key_scale", "out6"}}) // Inplace
    .Attrs({"tensor_layout: int",
            "is_causal: int",
            "qk_quant_gran: int",
            "sm_scale: float",
            "return_lse: int"})
    .SetKernelFn(PD_KERNEL(qk_int8_sv_f16_accum_f32_attn_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(qk_int8_sv_f16_accum_f32_attn_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(qk_int8_sv_f16_accum_f32_attn_InferDtype));

//
//  ============== fp8 kernels registry, for sm89 arch ==============
//

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_fwd(
                    paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    paddle::Tensor& value_scale,
                    paddle::Tensor& value_mean,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<std::vector<int64_t>> qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_InferShape(
  std::vector<int64_t> query_shape, 
  std::vector<int64_t> key_shape, 
  std::vector<int64_t> value_shape, 
  std::vector<int64_t> output_shape, 
  std::vector<int64_t> query_scale_shape, 
  std::vector<int64_t> key_scale_shape,
  std::vector<int64_t> value_scale_shape,
  std::vector<int64_t> value_mean_shape) {

    // force layout: NHD: [bsz, seq_len, num_heads, head_dim]
    int64_t bsz = query_shape[0];
    int64_t seq_len = query_shape[1];
    int64_t h_qo = query_shape[2];

    std::vector<int64_t> return_shape = {bsz, h_qo, seq_len};
    return {return_shape};
}

std::vector<paddle::DataType> qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_InferDtype(
  paddle::DataType A_dtype,
  paddle::DataType B_dtype,
  paddle::DataType C_dtype,
  paddle::DataType D_dtype,
  paddle::DataType E_dtype,
  paddle::DataType F_dtype,
  paddle::DataType G_dtype,
  paddle::DataType H_dtype) {
  return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn)
    .Inputs({"query", "key", "value", "output", "query_scale", "key_scale", "value_scale", "value_mean"})
    .Outputs({"out1", "out2", "out3", "out4", "out5", "out6", "out7", "out8", "lse"})
    .SetInplaceMap({{"query", "out1"}, {"key", "out2"}, {"value", "out3"}, {"output", "out4"}, {"query_scale", "out5"}, {"key_scale", "out6"}, {"value_scale", "out7"}, {"value_mean", "out8"}}) // Inplace
    .Attrs({"tensor_layout: int",
            "is_causal: int",
            "qk_quant_gran: int",
            "sm_scale: float",
            "return_lse: int"})
    .SetKernelFn(PD_KERNEL(qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_InferDtype));

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_fwd(
                    paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    paddle::Tensor& value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<std::vector<int64_t>> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_InferShape(
  std::vector<int64_t> query_shape, 
  std::vector<int64_t> key_shape, 
  std::vector<int64_t> value_shape, 
  std::vector<int64_t> output_shape, 
  std::vector<int64_t> query_scale_shape, 
  std::vector<int64_t> key_scale_shape,
  std::vector<int64_t> value_scale_shape) {

    // force layout: NHD: [bsz, seq_len, num_heads, head_dim]
    int64_t bsz = query_shape[0];
    int64_t seq_len = query_shape[1];
    int64_t h_qo = query_shape[2];

    std::vector<int64_t> return_shape = {bsz, h_qo, seq_len};
    return {return_shape};
}

std::vector<paddle::DataType> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_InferDtype(
  paddle::DataType A_dtype,
  paddle::DataType B_dtype,
  paddle::DataType C_dtype,
  paddle::DataType D_dtype,
  paddle::DataType E_dtype,
  paddle::DataType F_dtype,
  paddle::DataType G_dtype) {
  return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn)
    .Inputs({"query", "key", "value", "output", "query_scale", "key_scale", "value_scale"})
    .Outputs({"out1", "out2", "out3", "out4", "out5", "out6", "out7", "lse"})
    .SetInplaceMap({{"query", "out1"}, {"key", "out2"}, {"value", "out3"}, {"output", "out4"}, {"query_scale", "out5"}, {"key_scale", "out6"}, {"value_scale", "out7"}}) // Inplace
    .Attrs({"tensor_layout: int",
            "is_causal: int",
            "qk_quant_gran: int",
            "sm_scale: float",
            "return_lse: int"})
    .SetKernelFn(PD_KERNEL(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_InferDtype));


std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm89_fwd(
                    paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    paddle::Tensor& value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<std::vector<int64_t>> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm89_InferShape(
  std::vector<int64_t> query_shape, 
  std::vector<int64_t> key_shape, 
  std::vector<int64_t> value_shape, 
  std::vector<int64_t> output_shape, 
  std::vector<int64_t> query_scale_shape, 
  std::vector<int64_t> key_scale_shape,
  std::vector<int64_t> value_scale_shape) {

    // force layout: NHD: [bsz, seq_len, num_heads, head_dim]
    int64_t bsz = query_shape[0];
    int64_t seq_len = query_shape[1];
    int64_t h_qo = query_shape[2];

    std::vector<int64_t> return_shape = {bsz, h_qo, seq_len};
    return {return_shape};
}

std::vector<paddle::DataType> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm89_InferDtype(
  paddle::DataType A_dtype,
  paddle::DataType B_dtype,
  paddle::DataType C_dtype,
  paddle::DataType D_dtype,
  paddle::DataType E_dtype,
  paddle::DataType F_dtype,
  paddle::DataType G_dtype) {
  return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm89)
    .Inputs({"query", "key", "value", "output", "query_scale", "key_scale", "value_scale"})
    .Outputs({"out1", "out2", "out3", "out4", "out5", "out6", "out7", "lse"})
    .SetInplaceMap({{"query", "out1"}, {"key", "out2"}, {"value", "out3"}, {"output", "out4"}, {"query_scale", "out5"}, {"key_scale", "out6"}, {"value_scale", "out7"}}) // Inplace
    .Attrs({"tensor_layout: int",
            "is_causal: int",
            "qk_quant_gran: int",
            "sm_scale: float",
            "return_lse: int"})
    .SetKernelFn(PD_KERNEL(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm89_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm89_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm89_InferDtype));

//
//  ============== fp8 kernels registry, for sm90 arch ==============
//

std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90_fwd(
                  paddle::Tensor& query,
                  paddle::Tensor& key,
                  paddle::Tensor& value,
                  paddle::Tensor& output,
                  paddle::Tensor& query_scale,
                  paddle::Tensor& key_scale,
                  int tensor_layout,
                  int is_causal,
                  int qk_quant_gran,
                  float sm_scale,
                  int return_lse);

std::vector<std::vector<int64_t>> qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90_InferShape(
  std::vector<int64_t> query_shape, 
  std::vector<int64_t> key_shape, 
  std::vector<int64_t> value_shape, 
  std::vector<int64_t> output_shape, 
  std::vector<int64_t> query_scale_shape, 
  std::vector<int64_t> key_scale_shape) {

    // force layout: NHD: [bsz, seq_len, num_heads, head_dim]
    int64_t bsz = query_shape[0];
    int64_t seq_len = query_shape[1];
    int64_t h_qo = query_shape[2];

    std::vector<int64_t> return_shape = {bsz, h_qo, seq_len};
    return {return_shape};
}

std::vector<paddle::DataType> qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90_InferDtype(
  paddle::DataType A_dtype,
  paddle::DataType B_dtype,
  paddle::DataType C_dtype,
  paddle::DataType D_dtype,
  paddle::DataType E_dtype,
  paddle::DataType F_dtype) {
  return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90)
    .Inputs({"query", "key", "value", "output", "query_scale", "key_scale"})
    .Outputs({"out1", "out2", "out3", "out4", "out5", "out6", "lse"})
    .SetInplaceMap({{"query", "out1"}, {"key", "out2"}, {"value", "out3"}, {"output", "out4"}, {"query_scale", "out5"}, {"key_scale", "out6"}}) // Inplace
    .Attrs({"tensor_layout: int",
            "is_causal: int",
            "qk_quant_gran: int",
            "sm_scale: float",
            "return_lse: int"})
    .SetKernelFn(PD_KERNEL(qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(qk_int8_sv_f8_accum_f32_attn_inst_buf_sm90_InferDtype));


std::vector<paddle::Tensor> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90_fwd(
                    paddle::Tensor& query,
                    paddle::Tensor& key,
                    paddle::Tensor& value,
                    paddle::Tensor& output,
                    paddle::Tensor& query_scale,
                    paddle::Tensor& key_scale,
                    paddle::Tensor& value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

std::vector<std::vector<int64_t>> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90_InferShape(
  std::vector<int64_t> query_shape, 
  std::vector<int64_t> key_shape, 
  std::vector<int64_t> value_shape, 
  std::vector<int64_t> output_shape, 
  std::vector<int64_t> query_scale_shape, 
  std::vector<int64_t> key_scale_shape,
  std::vector<int64_t> value_scale_shape) {

    // force layout: NHD: [bsz, seq_len, num_heads, head_dim]
    int64_t bsz = query_shape[0];
    int64_t seq_len = query_shape[1];
    int64_t h_qo = query_shape[2];

    std::vector<int64_t> return_shape = {bsz, h_qo, seq_len};
    return {return_shape};
}

std::vector<paddle::DataType> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90_InferDtype(
  paddle::DataType A_dtype,
  paddle::DataType B_dtype,
  paddle::DataType C_dtype,
  paddle::DataType D_dtype,
  paddle::DataType E_dtype,
  paddle::DataType F_dtype,
  paddle::DataType G_dtype) {
  return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90)
    .Inputs({"query", "key", "value", "output", "query_scale", "key_scale", "value_scale"})
    .Outputs({"out1", "out2", "out3", "out4", "out5", "out6", "out7", "lse"})
    .SetInplaceMap({{"query", "out1"}, {"key", "out2"}, {"value", "out3"}, {"output", "out4"}, {"query_scale", "out5"}, {"key_scale", "out6"}, {"value_scale", "out7"}}) // Inplace
    .Attrs({"tensor_layout: int",
            "is_causal: int",
            "qk_quant_gran: int",
            "sm_scale: float",
            "return_lse: int"})
    .SetKernelFn(PD_KERNEL(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90_fwd))
    .SetInferShapeFn(PD_INFER_SHAPE(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90_InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_sm90_InferDtype));


//
//  ============== fused kernels registry ==============
//

void quant_per_block_int8_fuse_sub_mean_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& mean,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int block_size,
                int tensor_layout);

// quant_per_block_int8_fuse_sub_mean_cuda_fwd does not have any return
// so we don't implement infer type & shape function here.

PD_BUILD_OP(quant_per_block_int8_fuse_sub_mean_cuda)
    .Inputs({"input", "mean", "output", "scale"})
    .Outputs({"out1", "out2", "out3", "out4"})
    .SetInplaceMap({{"input", "out1"}, {"mean", "out2"}, {"output", "out3"}, {"scale", "out4"}}) // Inplace
    .Attrs({"block_size: int", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(quant_per_block_int8_fuse_sub_mean_cuda_fwd));


void quant_per_warp_int8_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int block_size,
                int warp_block_size,
                int tensor_layout);

// quant_per_warp_int8_cuda_fwd does not have any return
// so we don't implement infer type & shape function here.

PD_BUILD_OP(quant_per_warp_int8_cuda)
    .Inputs({"input", "output", "scale"})
    .Outputs({"out1", "out2", "out3"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}, {"scale", "out3"}}) // Inplace
    .Attrs({"block_size: int", "warp_block_size: int", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(quant_per_warp_int8_cuda_fwd));


void quant_per_block_int8_cuda_scale_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                float sm_scale,
                int block_size,
                int tensor_layout);

// quant_per_block_int8_cuda_scale does not have any return
// so we don't implement infer type & shape function here.

PD_BUILD_OP(quant_per_block_int8_cuda_scale)
    .Inputs({"input", "output", "scale"})
    .Outputs({"out1", "out2", "out3"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}, {"scale", "out3"}}) // Inplace
    .Attrs({"sm_scale: float", "block_size: int", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(quant_per_block_int8_cuda_scale_fwd));


void quant_per_block_int8_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int block_size,
                int tensor_layout);

// quant_per_block_int8_cuda does not have any return
// so we don't implement infer type & shape function here.

PD_BUILD_OP(quant_per_block_int8_cuda)
    .Inputs({"input", "output", "scale"})
    .Outputs({"out1", "out2", "out3"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}, {"scale", "out3"}}) // Inplace
    .Attrs({"sm_scale: float", "block_size: int", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(quant_per_block_int8_cuda_fwd));


void transpose_pad_permute_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                int tensor_layout);

// transpose_pad_permute_cuda_fwd does not have any return
// so we don't implement infer type & shape function here.

PD_BUILD_OP(transpose_pad_permute_cuda)
    .Inputs({"input", "output"})
    .Outputs({"out1", "out2"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}}) // Inplace
    .Attrs({"tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(transpose_pad_permute_cuda_fwd));


void scale_fuse_quant_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);

// scale_fuse_quant_cuda_fwd does not have any return
// so we don't implement infer type & shape function here.

PD_BUILD_OP(scale_fuse_quant_cuda)
    .Inputs({"input", "output", "scale"})
    .Outputs({"out1", "out2", "out3"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}, {"scale", "out3"}}) // Inplace
    .Attrs({"num_tokens: int", "scale_max: float", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(scale_fuse_quant_cuda_fwd));


void mean_scale_fuse_quant_cuda_fwd(
                paddle::Tensor& input,
                paddle::Tensor& output,
                paddle::Tensor& mean,
                paddle::Tensor& scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);

// mean_scale_fuse_quant_cuda_fwd does not have any return
// so we don't implement infer type & shape function here.

PD_BUILD_OP(mean_scale_fuse_quant_cuda)
    .Inputs({"input", "output", "mean", "scale"})
    .Outputs({"out1", "out2", "out3", "out4"})
    .SetInplaceMap({{"input", "out1"}, {"output", "out2"}, {"mean", "out3"}, {"scale", "out4"}}) // Inplace
    .Attrs({"num_tokens: int", "scale_max: float", "tensor_layout: int"})
    .SetKernelFn(PD_KERNEL(mean_scale_fuse_quant_cuda_fwd));