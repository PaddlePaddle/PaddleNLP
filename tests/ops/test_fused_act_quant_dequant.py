import numpy as np
import paddle
import paddle.incubate.nn.functional as F
import FusedQuantOps as FQO 
from enum import Enum

def dynamic_range(x: paddle.Tensor) -> float:
    # 一次性转换为float32并取绝对值
    x_abs = paddle.abs(x.astype('float32'))
    
    # 找到非零元素
    non_zero = x_abs != 0.0
    if not paddle.any(non_zero).item():
        return 0.0
    
    # 一次性计算最大值和最小值
    amax = paddle.max(x_abs)
    amin = paddle.min(x_abs.masked_select(non_zero))
    
    # 计算对数范围
    d_range = (paddle.log2(amax) - paddle.log2(amin)).item()
    return d_range

class QuantGranularity(Enum):
    PER_1x128 = 3

def show_dynamic_range_stats(x: paddle.Tensor, granularity: QuantGranularity):
    drs = []
    if granularity == QuantGranularity.PER_1x128:
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[1], 128):
                drs.append(dynamic_range(x[i, j : j + 128]))
    else:
        raise ValueError("Unsupported Granularity")
    n_quantize_groups = len(drs)
    drs = paddle.to_tensor(drs)
    percentile_50 = paddle.quantile(drs, 0.50)
    percentile_90 = paddle.quantile(drs, 0.90)
    percentile_95 = paddle.quantile(drs, 0.95)
    percentile_100 = paddle.quantile(drs, 1.0)
    print(
        f"n_quantize_groups: {n_quantize_groups}, dynamic_range_percentile_50: {percentile_50:.2f}, "
        f"dynamic_range_percentile_90: {percentile_90:.2f}, dynamic_range_percentile_95: {percentile_95:.2f}, "
        f"dynamic_range_percentile_100: {percentile_100:.2f}"
    )
    return (
        n_quantize_groups,
        percentile_50,
        percentile_90,
        percentile_95,
        percentile_100,
    )

def compare(x, x_q, x_qdq):
    original_n_zeros = x.numel() - paddle.count_nonzero(x)
    print(f"Orignal zeros: {original_n_zeros}; zero rate {original_n_zeros / x.numel() * 100.0} %")
    original_rms = paddle.sqrt(paddle.mean(x**2)).item()
    print(f"Orignal rms: {original_rms}")
    n_nonzeros = paddle.count_nonzero(x_q.to(paddle.float32))
    n_zeros = x_q.astype("float32").numel() - n_nonzeros
    ftz_rate = (n_zeros - original_n_zeros) * 100.0 / x.numel()
    print(f"after_quant_n_zeros {n_zeros}; ftz_rate: {ftz_rate:.5f} %")

    diff_squared = (x_qdq - x.to(paddle.float32)) ** 2
    recovered_rms= paddle.sqrt(paddle.mean(x_qdq**2)).item()
    print(f"recovered rms: {recovered_rms}")
    rmse = paddle.sqrt(paddle.sum(diff_squared) / x.numel())
    print(f"quantize_rmse: {rmse}")
    show_dynamic_range_stats(x_qdq, QuantGranularity.PER_1x128)
    return ftz_rate, rmse

""" Eval of various quantization schemes """


def eval_quant(x: paddle.Tensor):
    x_q, descale = FQO.fused_act_quant(x, transpose_output=False, padding_last_dim_to_8x=False, using_pow2_scaling=False) 
    x_qdq = FQO.fused_act_dequant(x_q, descale)
    return compare(x, x_q, x_qdq)

def verify_act_dequant():
    for width in [7168]:
        for height in [4096, 16384, 32768]:
            print("#"*60 + f" Testing width:{width}, height:{height} " + "#"*60)
            x= paddle.randn([height, width]).astype("bfloat16")
            eval_quant(x)
            
def run():
    verify_act_dequant()

if __name__ == "__main__":
    run()