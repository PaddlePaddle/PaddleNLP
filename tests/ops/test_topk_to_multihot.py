import numpy as np
import paddle
import paddle.incubate.nn.functional as F
import TokenDispatherUtils as TDU

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

def verify_topk_to_multihot():
    expert_num = 4
    topk = 8
    routemap_topk = [ 
        [-1,-1,0,1,-1,-1,-1,-1],
        [1,-1,-1,-1,-1,-1,-1,-1],
        [-1,0,-1,-1,-1,-1,-1,-1],
    ]
    probs_topk = [ 
        [0,0,0.3,0.5,0,0,0,0],
        [0.5,0,0,0,0,0,0,0],
        [0,0.7,0,0,0,0,0,0],
    ]
    expected_routemap_topk = [
        [1,1,0,0],
        [0,1,0,0],
        [1,0,0,0]
    ]
    expected_probs_topk = [
        [0.3, 0.5, 0,0],
        [0,0.5,0,0],
        [0.7,0,0,0]
    ]
    routemap_topk = paddle.to_tensor(routemap_topk, dtype="int32")
    probs_topk = paddle.to_tensor(probs_topk,dtype="float32")
    print(routemap_topk)
    print(probs_topk)
    route_onehot, probs_onehot = TDU.fused_topk_to_multihot(routemap_topk, probs_topk,seqlen=3, topk=topk, num_experts=expert_num)
    route_onehot_only, _= TDU.fused_topk_to_multihot(routemap_topk,None, seqlen=3, topk=topk, num_experts=expert_num)
    print(route_onehot)
    print(route_onehot_only)
    print(probs_onehot)
    
def run():
    verify_topk_to_multihot()

if __name__ == "__main__":
    run()