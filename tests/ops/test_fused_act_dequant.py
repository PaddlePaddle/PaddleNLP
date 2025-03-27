import numpy as np
import paddle
import paddle.incubate.nn.functional as F
import FusedQuantOps as FQO 

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

def verify_act_dequant():
    for width in [7168]:
        for height in [4096, 16384, 32768]:
            print("#"*60 + f" Testing width:{width}, height:{height} " + "#"*60)
            x= paddle.clip(paddle.randn([height, width]).astype("bfloat16"), min=-50, max=50)
            print("-" * 20 + f"Testing with {width} * {height}" + "-" * 20)
            x_fp8, scale = FQO.fused_act_quant(x, transpose_output=False, padding_last_dim_to_8x=False, using_pow2_scaling=False)
            dequant_result= FQO.fused_act_dequant(x_fp8,scale)
            np_results=[]
            golden_res = x
            np_results.append(golden_res.astype("float").numpy())
            np_results.append(dequant_result.astype("float").numpy())
            nan_cnt_golden, nan_cnt_fused= np.sum(np.isnan(np_results[0])), np.sum(np.isnan(np_results[1]))
            print(f"Nan count of Golden result: {nan_cnt_golden}; Nan count of Fused result: {nan_cnt_fused}")
            try:
                np.testing.assert_allclose(np_results[0], np_results[1], rtol=0.01, atol=1) #存在截断误差，atol=1，通常在1e-6
                print("+++++++ Passed ++++++++")
            except AssertionError as err:
                print(err)
                compare_tensors(np_results[0], np_results[1])
            print(np_results[0])
            print("------------")
            print( np_results[1])
            
def run():
    verify_act_dequant()

if __name__ == "__main__":
    run()