
import paddle
import numpy as np
import paddle.nn.functional as F
from paddle import nn
from paddle.incubate.nn.functional import moe_ffn
from paddle.nn.quant import weight_quantize
from paddle.incubate.nn.functional import fused_moe, swiglu
import paddle.nn.functional as F
from paddle.incubate.nn.functional import (
            moe_dispatch,
            moe_ffn,
            moe_reduce,
        )

from paddlenlp_ops import trt_llm_fused_moe
from enum import IntEnum


class Expert(nn.Layer):
    def __init__(self, d_model, d_feedforward):
        super().__init__()
        self.fc1 = nn.Linear(
            d_model, d_feedforward * 2
        )  # Swiglu expects twice the hidden_dim
        self.swiglu = swiglu
        self.fc2 = nn.Linear(d_feedforward, d_model)

    def forward(self, x, idx):
        x = self.fc1(x)
        x = swiglu(x)
        x = self.fc2(x)
        return x



# Configuration
x_type = paddle.bfloat16
batch_size = 1
seq_len = 16

M = seq_len * batch_size
num_expert = 32 # Reduced for simplicity

E = num_expert
d_model = 64
d_feedforward = 2048
top_k = 4
quant_method = "weight_only_int8"  # Quantization method for testing

# Set default dtype
paddle.set_default_dtype(x_type)
paddle.disable_static(place=paddle.CUDAPlace(0))

# Initialize experts and layers
# experts = nn.LayerList([nn.Linear(d_model, d_feedforward) for _ in range(num_expert)])

experts = nn.LayerList(
            [
                Expert(d_model, d_feedforward)
                for _ in range(num_expert)
            ]
        )

# Initialize tensors for weights and biases
bmm_w0 = paddle.to_tensor(
    np.array([expert.fc1.weight.numpy() for expert in experts]), dtype=x_type
)
bmm_b0 = paddle.to_tensor(
    np.array([expert.fc1.bias.numpy() for expert in experts]).reshape(num_expert, 1, -1),
    dtype=x_type
)

bmm_w1 = paddle.to_tensor(
    np.array([expert.fc2.weight.numpy() for expert in experts]), dtype=x_type
)
bmm_b1 = paddle.to_tensor(
    np.array([expert.fc2.bias.numpy() for expert in experts]).reshape(num_expert, 1, -1),
    dtype=x_type
)



tensor_x = paddle.to_tensor(np.random.randn(batch_size, seq_len, d_model) * 0.1, dtype=x_type)

# Define the gate layer
gate = nn.Linear(d_model, num_expert)



def GetQuantizedWeights(quant_method):
    """
    Quantizes the weights for the experts' layers and returns the quantized weights and scales.
    :param quant_method: The quantization method to use.
    :return: Quantized bmm_w0, bmm_w1, scale0, scale1
    """
    if quant_method != "None":
        fc0_expert_weights_for_ref_list = []
        scale0 = []
        for i in range(num_expert):
            fc0_expert_weights_for_ref_i, fc0_expert_weights_scale_for_ref_i = weight_quantize(bmm_w0[i], algo=quant_method)
            # print(fc0_expert_weights_for_ref_i.shape)
            # # print(quant_method) # [3072, 768]
            # exit(0)

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


# Quantize weights

quant_method = "weight_only_int8"

bmm_w0_quantized, bmm_w1_quantized, scale0, scale1 = GetQuantizedWeights(quant_method)

print(bmm_w0.shape) # [2, 768, 6144]
print(bmm_w1.shape)

print(bmm_w0_quantized.shape) 
print(bmm_w1_quantized.shape) 
# print(bmm_w0_quantized)
# print(bmm_w0_quantized.cast("uint8"))
# exit(0)
# [16, 768, 4096]
# [16, 2048, 768]

# [16, 768, 2048]
# [16, 2048, 384]
# print(bmm_w0_quantized)
# exit(0)

# [256, 4096]
# [256, 1024]

# print(scale0.shape)
# print(scale1.shape)

# # [16, 4096]
# # [16, 768]

# Run the moe_ffn function with the quantized weights
tensor_x = tensor_x.reshape([-1, d_model])  # Shape: [1280, 768]


# Gate output
gate_out = paddle.matmul(tensor_x.cast("float32"), gate.weight.cast("float32"))
gate_out = gate_out.reshape([-1, num_expert])

activation_str = "Swiglu"
factor_for_scale = 1e-2
active_rows = M



class ExpertScaleNormalizationMode(IntEnum):
    NONE = 0
    RENORMALIZE = 1
    SPARSE_MIXER = 2
    DEVICE_LIMITED = 3
    DEVICE_LIMITED_RENORM = 4

# normalization_mode: ExpertScaleNormalizationMode = ExpertScaleNormalizationMode.RENORMALIZE

normalization_mode = ExpertScaleNormalizationMode.RENORMALIZE
print(normalization_mode)
print(ExpertScaleNormalizationMode.RENORMALIZE)

gate_out = F.softmax(gate_out, axis=-1).cast(paddle.float32)
# out = trt_llm_fused_moe(
#             tensor_x,
#             gate_out,
#             bmm_w0,
#             bmm_w1,
#             None,
#             None,
#             None,
#             top_k,
#             "none",
#             3,
#         )
# print(out)


# scale0 = paddle.ones([256, 4096])
# scale1 = paddle.ones([256, 1024])
# s1 = bmm_w0_quantized.shape
# s2 = bmm_w1_quantized.shape
# bmm_w0_quantized=paddle.transpose(bmm_w0_quantized, perm=[0, 2, 1]).reshape(s1)
# bmm_w1_quantized=paddle.transpose(bmm_w1_quantized, perm=[0, 2, 1]).reshape(s2)

print("使用weight only int 8*************************")
out1 = trt_llm_fused_moe(
            tensor_x,
            gate_out,
            bmm_w0_quantized,
            bmm_w1_quantized,
            scale0,
            scale1,
            None,
            top_k,
            3,
            quant_method,
            "Swiglu"
        )
print(out1)
if paddle.isnan(out1).sum().item() > 0:
    print("fuck !")

print("使用weight only int4*************************")
quant_method = "weight_only_int4"
bmm_w0_quantized, bmm_w1_quantized, scale0, scale1 = GetQuantizedWeights(quant_method)

out = trt_llm_fused_moe(
            tensor_x,
            gate_out,
            bmm_w0_quantized,
            bmm_w1_quantized,
            scale0,
            scale1,
            None,
            top_k,
            3,
            quant_method,
            "Swiglu"
        )
print(out)


if paddle.isnan(out).sum().item() > 0:
    print("fuck !")

print(out1 - out)