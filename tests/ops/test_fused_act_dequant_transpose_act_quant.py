import numpy as np
import paddle
import paddle.incubate.nn.functional as F
#import test_quant
import FusedQuantOps as FQO 
from paddle.base import core
REP=1

'''
Swiglu Function:
out = silu(x) * y when y is not None
out = silu(xs[0]) * xs[1] when y is None, where xs = paddle.chunk(x, 2, axis=-1)
'''

def dequantize_fp8_to_bf16(fp8_tensor: paddle.Tensor, 
                           scale: paddle.Tensor) -> paddle.Tensor:
    expanded_scale = paddle.repeat_interleave(
        scale, 
        repeats=128, 
        axis=-1
    )
    # 非规整情况，需要截断
    expanded_scale = expanded_scale[:, :fp8_tensor.shape[-1]]
    return (fp8_tensor.astype('float32') * expanded_scale)

def compare_tensors(a, b):
    # 形状一致性检查
    assert a.shape == b.shape, "输入张量形状不一致"
    
    # 计算绝对差距
    abs_diff = np.abs(a - b)
    max_abs_val = np.max(abs_diff)
    max_abs_flat_idx = np.argmax(abs_diff)
    max_abs_idx = np.unravel_index(max_abs_flat_idx, a.shape)
    
    # 计算相对差距（防止除以零）
    denominator = np.maximum(np.abs(a), np.abs(b))
    rel_diff = np.divide(
        abs_diff, 
        denominator, 
        out=np.zeros_like(abs_diff),
        where=(denominator != 0)
    )
    max_rel_val = np.max(rel_diff)
    max_rel_flat_idx = np.argmax(rel_diff)
    max_rel_idx = np.unravel_index(max_rel_flat_idx, a.shape)
    

    # 打印结果
    print("\n[最大绝对差距]" f"位置: {max_abs_idx}")
    print(f"a[{max_abs_idx}] = {a[max_abs_idx]:.6g}" + f"\t b[{max_abs_idx}] = {b[max_abs_idx]:.6g}" + f"\t 绝对差值: {max_abs_val:.6g}\n")
    
    print("[最大相对差距]" f"位置: {max_rel_idx}")
    print(f"a[{max_rel_idx}] = {a[max_rel_idx]:.6g}" + f"\t b[{max_rel_idx}] = {b[max_rel_idx]:.6g}" + f"\t 相对差值: {max_rel_val:.6g}\n")
    print("周围元素比较-a:")
    print("周围元素比较-b:")
    print(f"{a[max_rel_idx[0], (max_rel_idx[1] - 10):(max_rel_idx[1] + 10)]} ")
    print(f"{b[max_rel_idx[0], (max_rel_idx[1] - 10):(max_rel_idx[1] + 10)]} ")
    
    # 返回结构化结果
    return {
        'max_absolute': {
            'index': max_abs_idx,
            'a_value': a[max_abs_idx],
            'b_value': b[max_abs_idx],
            'difference': max_abs_val
        },
        'max_relative': {
            'index': max_rel_idx,
            'a_value': a[max_rel_idx],
            'b_value': b[max_rel_idx],
            'difference': max_rel_val
        }
    }

def printany(te):
    for i in range(te.shape[0]):
        for j in range(te.shape[1]):
            print(te[i][j], end=", ")
        print()
    print("-"*20)

def verify_swiglu_quant_result():
    for width in [7168]:
        for height in [32768]:
            #print("#"*60 + f" Testing width:{width}, height:{height} " + "#"*60)
            x= paddle.clip(paddle.randn([height, width]).astype("bfloat16"), min=-50, max=50)
            for padding in [False]:
                pad_tag = "Padded: True" if padding is not None else "Padded: False"
                print("-" * 20 + f"Testing with {pad_tag}" + "-" * 20)
                x_fp8, scale = FQO.fused_act_quant(x, transpose_output=False, padding_last_dim_to_8x=padding, using_pow2_scaling=False)
                scale_t = scale.T.contiguous()
                core.nvprof_start()
                for i in range(REP):
                    if i > 1: core.nvprof_nvtx_push("fused_dtaq")
                    fused_res, fused_scales = FQO.fused_act_dequant_transpose_act_quant(x_fp8,scale_t,padding_last_dim_to_8x=padding,using_pow2_scaling=False)
                    if i > 1: core.nvprof_nvtx_pop()
                np_results=[]
                for i in range(REP):
                    if i > 1: core.nvprof_nvtx_push("original")
                    golden_res, golden_scales = FQO.fused_act_quant(FQO.fused_act_dequant(x_fp8,scale).T.contiguous(),transpose_output=False, padding_last_dim_to_8x=padding, using_pow2_scaling=False)
                    if i > 1: core.nvprof_nvtx_pop()

                golden_res = dequantize_fp8_to_bf16(golden_res, golden_scales)
                #golden_res = x.T.contiguous()
                np_results.append(golden_res.astype("float32").numpy())
                if padding:
                    dequanted_sliced_result = dequantize_fp8_to_bf16(fused_res, fused_scales)
                    np_results.append(dequanted_sliced_result[:, :height].astype("float32").numpy())
                else:
                    np_results.append(dequantize_fp8_to_bf16(fused_res, fused_scales.T.contiguous().astype("float32")).numpy())
                nan_cnt_golden, nan_cnt_fused= np.sum(np.isnan(np_results[0])), np.sum(np.isnan(np_results[1]))
                print(f"Nan count of Golden result: {nan_cnt_golden}; Nan count of Fused result: {nan_cnt_fused}")
                try:
                    np.testing.assert_allclose(np_results[0], np_results[1], rtol=0.01, atol=1e-2) #存在截断误差，atol=1，通常在1e-6
                    print("+++++++ Passed ++++++++")
                    print(np_results[0])
                    print("-----------------------")
                    print(np_results[1])
                except AssertionError as err:
                    print(err)
                    print(np_results[0])
                    print("-----------------------")
                    print(np_results[1])
                    compare_tensors(np_results[0], np_results[1])
            
def run():
    verify_swiglu_quant_result()

if __name__ == "__main__":
    run()