


import paddle

from paddle.incubate.nn.functional import (
    fused_bias_act,
    fused_layer_norm,
    fused_moe)

from paddle.nn.quant import weight_quantize

def contiguous(tensor):
    """ return contiguous tensor """
    if hasattr(tensor, "contiguous"):
        print("haha")
    return tensor.contiguous()


def GetQuantizedWeights(quant_method, w1, w2, arch=80):
    """
    Quantizes the weights for the experts' layers and returns the quantized weights and scales.
    :param quant_method: The quantization method to use.
    :return: Quantized bmm_w0, bmm_w1, scale0, scale1
    """
    num_expert = 64
    bmm_w0 = w1
    bmm_w1 = w2
    d_model = 2048
    d_feedforward = 1408
    
    if quant_method != "None":
        fc0_expert_weights_for_ref_list = []
        scale0 = []
        for i in range(num_expert):
            fc0_expert_weights_for_ref_i, fc0_expert_weights_scale_for_ref_i = weight_quantize(bmm_w0[i], algo=quant_method)
            # print(fc0_expert_weights_for_ref_i.shape) [2816, 2048]
            fc0_expert_weights_for_ref_list.append(
                fc0_expert_weights_for_ref_i.reshape(
                    [d_model, d_feedforward * 2]
                    if quant_method == "weight_only_int8"
                    else [d_model, d_feedforward]
                )
            )
            scale0.append(fc0_expert_weights_scale_for_ref_i)

        fc1_expert_weights_for_ref_list = []
        scale1 = []
        for i in range(num_expert):
            fc1_expert_weights_for_ref_i, fc1_expert_weights_scale_for_ref_i = weight_quantize(bmm_w1[i], algo=quant_method)
            fc1_expert_weights_for_ref_list.append(
                fc1_expert_weights_for_ref_i.reshape(
                    [d_feedforward, d_model]
                    if quant_method == "weight_only_int8"
                    else [d_feedforward, d_model // 2]
                )
            )
            scale1.append(fc1_expert_weights_scale_for_ref_i)
        
        bmm_w0_quantized = paddle.to_tensor(fc0_expert_weights_for_ref_list)
        bmm_w1_quantized = paddle.to_tensor(fc1_expert_weights_for_ref_list)
        scale0 = paddle.to_tensor(scale0)
        scale1 = paddle.to_tensor(scale1)
        
        return bmm_w0_quantized, bmm_w1_quantized, scale0, scale1
    else:
        return bmm_w0, bmm_w1, None, None


def block_quant(input_tensor, weight_block_size=[128, 128]):
    n, k = input_tensor.shape
    block_size_n, block_size_k = weight_block_size
    num_blocks_rows = (n + block_size_n - 1) // block_size_n
    num_blocks_cols = (k + block_size_k - 1) // block_size_k
    quant_data = paddle.zeros([n, k], dtype=paddle.float32)
    quant_scale = paddle.ones([num_blocks_rows, num_blocks_cols], dtype=paddle.float32)
    tensor_fp32 = paddle.cast(input_tensor, paddle.float32)
    for row in range(quant_scale.shape[0]):
        start_row = row * block_size_n
        end_row = min((row + 1) * block_size_n, n)
        for col in range(quant_scale.shape[1]):
            start_col = col * block_size_k
            end_col = min((col + 1) * block_size_k, k)
            data = tensor_fp32[start_row:end_row, start_col:end_col]
            max_val = data.abs().max()
            scale = max_val.clip(0.000001) / 448.0
            quant_scale[row, col] = scale
            quant_val = paddle.clip(data / scale, min=-448.0, max=448.0)
            quant_data[start_row:end_row, start_col:end_col] = quant_val
    return quant_data, quant_scale


path = "/root/paddlejob/workspace/env_run/output/gaoziyuan/PaddleNLP/csrc/gpu/moe/tensorrt-llm-moe/moe/moe_input"
input_down = paddle.load(path)

# 64, 1408, 1024]
from paddlenlp_ops import trt_llm_fused_moe

gate_weight = input_down["gate_weights[i]"]
tmp_out = input_down["tmp_out"]
ffn1_weights = input_down["ffn1_weights[i]"]
ffn2_weights = input_down["ffn2_weights[i]"]

quant_method = "none"


def permute(x):
    shape_x = x.shape
    if len(shape_x) == 3:
        print("yahei")
        return paddle.transpose(x, perm=[0, 2, 1]).reshape(shape_x).contiguous()
        # a = paddle.reshape(x, [-1, shape_x[1], shape_x[2]])
        # return paddle.transpose(a, perm=[0, 2, 1]).contiguous()
    else:
        return paddle.transpose(x, perm=[1, 0]).reshape(shape_x)

import numpy as np
np.set_printoptions(threshold=np.inf)

def quant(ffn_weight):
    a_s = []
    b_s = []
    for i in range(64):
        tmp = ffn_weight[i]
        a , b = block_quant(tmp)
        a_s.append(a)
        b_s.append(b)

    quant_ffn = paddle.to_tensor(a_s).cast("float8_e4m3fn")
    scale = paddle.to_tensor(b_s)
    return  quant_ffn, scale


ffn1_fp8, scale_1 = quant(ffn1_weights)

ffn2_fp8, scale_2 = quant(ffn2_weights)



gate_out = paddle.matmul(tmp_out.cast("float32"), gate_weight)
# fused_moe_out_1 = trt_llm_fused_moe(
#             tmp_out,
#             gate_out,
#             ffn1_weights,
#             ffn2_weights,
#             None,
#             None,
#             # scale0,
#             # scale1,
#             None,
#             6,
#             0,
#             # quant_method,
#             "none",
#             "Swiglu"
#         )
# print(fused_moe_out_1)

print(ffn1_fp8.shape)
print(ffn2_fp8.shape)
print(scale_1.shape)
print(scale_2.shape)

# [64, 2048, 2816]

# [64, 1408, 2048]
# [64, 16, 22]
# [64, 11, 16]

# exit(0)
quant_method = "fp8_block_wise"
tmp_out = tmp_out[:64]
fused_moe_out_1 = trt_llm_fused_moe(
            tmp_out,
            gate_out,
            ffn1_fp8.reshape([64, 2048,-1]),
            ffn2_fp8.reshape([64, -1,2048]),
            scale_1,
            scale_2,
            # scale0,
            # scale1,
            None,
            6,
            0,
            quant_method,
            # "none",
            "Swiglu"
        )
print(fused_moe_out_1)


# print(fused_moe_out)

# diff = fused_moe_out - fused_moe_out_1

# # # diff = diff
# # print(diff)
# print(diff[:10,:10])

# print(paddle.max(paddle.abs(diff)))






# fused_moe_out = fused_moe(
#             tmp_out,
#             gate_weight,
#             # bmm_w0_quantized,
#             # bmm_w1_quantized,
#             ffn1_weights, 
#             ffn2_weights,
#             None,
#             # original_scale0,
#             # scale0,
#             None,
#             None,
#             None,
#             # scale1,
#             quant_method,
#             # "none",
#             6,
#             False,
#         )