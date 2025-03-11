import numpy as np
import paddle
import paddle.incubate.nn.functional as F
#import test_quant
import FusedQuantOps as FQO 

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
    print(f"{a[max_rel_idx[0], (max_rel_idx[1] - 10):(max_rel_idx[1] + 10)]} ")
    print("周围元素比较-b:")
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
    for width in [130, 4098, 7168]:
        for height in [256, 1026, 4098]:
            print("#"*60 + f" Testing width:{width}, height:{height} " + "#"*60)
            x= paddle.clip(paddle.randn([height, width]).astype("bfloat16"), min=-50, max=50)
            y= paddle.clip(paddle.randn([height, width]).astype("bfloat16"), min=-50, max=50)
            for transposing in [True,False]:
                for padding in [True]:
                    for optional_y in [y,None]:
                        y_tag = "is_combined: False" if optional_y is not None else "is_combined: True"
                        pad_tag = "Padded: True" if padding is not None else "Padded: False"
                        print("-" * 20 + f"Testing swiglu with {y_tag} , {pad_tag} and transposing: {transposing}" + "-" * 20)
                        fused_res, fused_scales = FQO.fused_swiglu_act_quant(x,optional_y,transpose_output=transposing, to_e4m3=True, using_pow2_scaling=False, padding_last_dim_to_8x=padding)
                        np_results=[]
                        if optional_y is None:
                            golden_res = F.swiglu(x) if not transposing else F.swiglu(x).T
                            np_results.append(golden_res.astype("float").numpy())
                        else:
                            golden_res = F.swiglu(x,y) if not transposing else F.swiglu(x,y).T
                            np_results.append(golden_res.astype("float").numpy())
                        if padding:
                            if transposing:
                                dequanted_sliced_result = dequantize_fp8_to_bf16(fused_res, fused_scales)
                                np_results.append(dequanted_sliced_result[:, :height].numpy())
                            else:
                                rank = width//2 if optional_y is None else width
                                dequanted_sliced_result = dequantize_fp8_to_bf16(fused_res, fused_scales)
                                np_results.append(dequanted_sliced_result[:, :rank].numpy())
                        else:
                            np_results.append(dequantize_fp8_to_bf16(fused_res, fused_scales).numpy())
                        nan_cnt_golden, nan_cnt_fused= np.sum(np.isnan(np_results[0])), np.sum(np.isnan(np_results[1]))
                        print(f"Nan count of Golden result: {nan_cnt_golden}; Nan count of Fused result: {nan_cnt_fused}")
                        try:
                            np.testing.assert_allclose(np_results[0], np_results[1], rtol=0.01, atol=1) #存在截断误差，atol=1，通常在1e-6
                            print("+++++++ Passed ++++++++")
                        except AssertionError as err:
                            print(err)
                            compare_tensors(np_results[0], np_results[1])
            
def run():
    verify_swiglu_quant_result()

if __name__ == "__main__":
    run()