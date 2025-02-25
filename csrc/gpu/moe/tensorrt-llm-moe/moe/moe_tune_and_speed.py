

import paddle
import time
import argparse
from paddle.incubate.nn.functional import fused_moe
from paddle.nn.quant import weight_quantize
from paddlenlp_ops import trt_llm_fused_moe

paddle.seed(2)

parser = argparse.ArgumentParser(description='Run MoE model with specified M')
parser.add_argument('--M', type=int, default=256, help='Batch size, token_num (M)')
args = parser.parse_args()

# Constants
DTYPES = paddle.bfloat16
# deepseek v3 parameters
M = args.M # (Batch size * token_num)
TP = 16
N = 2048 // TP  # Intermediate size
K = 7168  # Hidden size
E = 256  # Number of experts
topk = 8
block_size = [128, 128]
SEEDS = 0

def create_random_cuda_tensor(shape, dtype, mean: float = 0, std: float = 1):    
    return paddle.empty(shape, dtype=dtype).normal_(mean, std)

# Gate logits (score after gate)
score = create_random_cuda_tensor([M, E], paddle.float32).cast("float32")

# Bfloat16 input
a = paddle.randn((M, K), dtype=paddle.bfloat16) / 100
w1 = paddle.rand((E, K, 2 * N), dtype=paddle.bfloat16) / 100
w2 = paddle.rand((E , N, K), dtype=paddle.bfloat16)/ 100
gate_weight = paddle.rand((K, E), dtype=paddle.float32) / 100

def GetQuantizedWeights(quant_method, w1, w2, arch=80):
    """
    Quantizes the weights for the experts' layers and returns the quantized weights and scales.
    :param quant_method: The quantization method to use.
    :return: Quantized bmm_w0, bmm_w1, scale0, scale1
    """
    num_expert = E
    bmm_w0 = w1
    bmm_w1 = w2
    d_model = K
    d_feedforward = N
    
    if quant_method != "None":
        fc0_expert_weights_for_ref_list = []
        scale0 = []
        for i in range(num_expert):
            fc0_expert_weights_for_ref_i, fc0_expert_weights_scale_for_ref_i = weight_quantize(bmm_w0[i], algo=quant_method)
            # print(fc0_expert_weights_for_ref_i.shape)
            # exit(0) [256, 7168]
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
    

def trt_bf16():
    paddle.device.synchronize()
    start = time.time()
    out = trt_llm_fused_moe(
            a,
            score,
            w1,
            w2,
            None,
            None,
            None,
            topk,
            0,
            "none",
            "Swiglu",
            8192,
        )
    paddle.device.synchronize()
    end = time.time()
    print(f"trt bf16 : {((end - start) * 1000)} ms")


quant_method = "weight_only_int8"
bmm_w0_quantized, bmm_w1_quantized, scale0, scale1 = GetQuantizedWeights(quant_method, w1, w2)


def trt_win8(quant_method):    
    paddle.device.synchronize()
    start = time.time()
    gate_out = paddle.matmul(a.cast("float32"), gate_weight)
    out = trt_llm_fused_moe(
            a,
            score,
            bmm_w0_quantized,
            bmm_w1_quantized,
            scale0,
            scale1,
            None,
            topk,
            0,
            quant_method,
            8192 * 64,
        )
    paddle.device.synchronize()
    end = time.time()
    print(f"trt wint8 : {((end - start) * 1000 * 1000)} us")

def paddle_bf16():
    paddle.device.synchronize()
    start = time.time()
    fused_moe_out = fused_moe(
                a,
                score,
                w1,
                w2,
                None,
                None,
                None,
                None,
                # quant_method,
                quant_method,
                topk,
                False,
            )
    paddle.device.synchronize()
    end = time.time()
    print(f"paddle bf16 : {((end - start) * 1000)} ms")



def paddle_win8(quant_method):
    paddle.device.synchronize()
    start = time.time()
    fused_moe_out = fused_moe(
                a,
                score,
                bmm_w0_quantized,
                bmm_w1_quantized,
                None,
                scale0,
                None,
                scale1,
                quant_method,
                topk,
                False,
            )
    paddle.device.synchronize()
    end = time.time()
    print(f"paddle win8 : {((end - start) * 1000 * 1000)} us")


for i in range(1):
    trt_win8(quant_method)

# for i in range(20):
#     paddle_win8(quant_method)



