


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

path = "/root/paddlejob/workspace/env_run/gaoziyuan/PaddleNLP/moe_input"
input_down = paddle.load(path)

# 64, 1408, 1024]
from paddlenlp_ops import trt_llm_fused_moe, batch_symmetric_quantize

gate_weight = input_down["gate_weights[i]"]
# print(gate_weight.shape) [2048, 64]
tmp_out = input_down["tmp_out"]
ffn1_weights = input_down["ffn1_weights[i]"]
ffn2_weights = input_down["ffn2_weights[i]"]

# tmp_out = paddle.ones([110, 2048]).cast("bfloat16")

quant_method = "weight_only_int4"

def permute(x):
    shape_x = x.shape
    if len(shape_x) == 3:
        print("yahei")
        return paddle.transpose(x, perm=[0, 2, 1]).reshape(shape_x).contiguous()
        # a = paddle.reshape(x, [-1, shape_x[1], shape_x[2]])
        # return paddle.transpose(a, perm=[0, 2, 1]).contiguous()
    else:
        return paddle.transpose(x, perm=[1, 0]).reshape(shape_x)


# ffn1_weights = paddle.ones([64, 2048, 2816]).cast("bfloat16")*0.01
# ffn1_weights = paddle.arange(2048).cast("bfloat16").reshape([1,2048,1]).tile([64,1,2816])

# ffn2_weights = paddle.ones([64, 1408, 2048]).cast("bfloat16")
# ffn1_weights = permute(ffn1_weights)
import numpy as np
np.set_printoptions(threshold=np.inf)


# a = ffn1_weights[:32]
# b = ffn1_weights[32:]

# # c = paddle.concat([a,a],axis=0)
# ww = []
# for i in range(64):
#     ww.append(ffn1_weights[0].reshape([1,2048,2816]))

# c = paddle.concat(ww, axis=0)

# ffn1_weights = c



bmm_w0_quantized, bmm_w1_quantized, scale0, scale1 = GetQuantizedWeights(quant_method, contiguous(ffn1_weights), contiguous(ffn2_weights))




a, b = paddle.chunk(ffn1_weights, 2, axis=-1)
trt_weight_1 = paddle.concat([b,a], axis=-1)
print("拼接后的shape", trt_weight_1.shape)


bmm_w0_quantized_trt, bmm_w1_quantized_trt,scale0_trt, scale1_trt = GetQuantizedWeights(quant_method, trt_weight_1, contiguous(ffn2_weights))
# bmm_w1
# [64, 2048, 1408]#
# print(bmm_w0_quantized_trt.shape) # [64, 1408, 1024]
# exit(0)
# _, bmm_w0_quantized_trt, scale0_trt = symmetric_quantize(trt_weight_1.cpu(), quant_method)

# _, bmm_w1_quantized_trt, scale1_trt = symmetric_quantize(ffn2_weights.cpu(), quant_method)


gate_out = paddle.matmul(tmp_out.cast("float32"), gate_weight)
fused_moe_out_1 = trt_llm_fused_moe(
            tmp_out,
            gate_out,
            # permute(bmm_w0_quantized),
            # contiguous(bmm_w0_quantized),
            # contiguous(bmm_w1_quantized),
            bmm_w0_quantized_trt.cuda().contiguous(),
            bmm_w1_quantized_trt.cuda().contiguous(),
            # permute(bmm_w1_quantized),
            # ffn1_weights_trt_transpose,
            # ffn1_weights.reshape([64,2048, 2816]),
            # ffn1_weights,
            # permute(ffn1_weights),
            # ffn1_weights.transpose([0,2,1]),
            # permute(ffn1_weights),
            # permute(ffn2_weights),
            # w1,
            # w2,
            # bmm_w0_quantized_trt.reshape([64, 2816, 2048]),
            # bmm_w0_quantized_trt,
            # permute(bmm_w0_quantized_trt),
            # bmm_w1_quantized_trt,
            # permute(ffn1_weights),
            # permute(ffn2_weights),
            # scale0_trt,
            # scale1_trt,
            contiguous(scale0_trt).cuda().contiguous(),
            contiguous(scale1_trt).cuda().contiguous(),
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

fused_moe_out = fused_moe(
            tmp_out,
            gate_weight,
            bmm_w0_quantized,
            bmm_w1_quantized,
            # ffn1_weights, 
            # ffn2_weights,
            None,
            # original_scale0,
            scale0,
            None,
            scale1,
            quant_method,
            # "none",
            6,
            False,
        )
print(fused_moe_out)
# bmm_w0_quantized =bmm_w0_quantized.reshape([-1])
# print(bmm_w0_quantized[10000:11000])


# print(fused_moe_out[:10,:10])
# print(fused_moe_out_1[:10,:10])
diff = fused_moe_out - fused_moe_out_1

# # diff = diff
# print(diff)
print(diff[:10,:10])

print(paddle.max(paddle.abs(diff)))


# out1 = paddle.count_nonzero(diff)
# print(out1)


