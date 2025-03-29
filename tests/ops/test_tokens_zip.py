import numpy as np
import paddle
import paddle.incubate.nn.functional as F
import TokenDispatcherUtils as TDU

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

def verify_tokens_unzip():
    expert_num = 4
    topk = 8
    seqlen = 3
    token_len = 8
    tokens_zipped = [
        [1,1,1,1,1,1,1,1],
        [2,2,2,2,2,2,2,2],
        [3,3,3,3,3,3,3,3]
    ]
    routemap_topk = [ 
        [-1,-1,0,1,-1,-1,-1,-1],
        [1,-1,-1,-1,-1,-1,-1,-1],
        [-1,0,-1,-1,-1,-1,-1,-1],
    ]
    probs_topk = [ 
        [0,0,0.5,0.5,0,0,0,0],
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
    ]
    total_unzipped_tokens_num = 4
    expected_unzipped_tokens= [
        [1,1,1,1,1,1,1,1],
        [2,2,2,2,2,2,2,2],
        [3,3,3,3,3,3,3,3],
        [1,1,1,1,1,1,1,1],
    ]
    expected_unzipped_probs= [
        0.5,
        1,
        1,
        0.5
    ]
    expected_zipped_expertwise_rowmap= [
        [0,3,-1,-1],
        [-1,1,-1,-1],
        [2,-1,-1,-1],
    ]
    expected_unzipped_expert_idx = [
        0,
        1,
        0,
        1 
    ]
    tokens_simple_zipped = [
        [2,2,2,2,2,2,2,2],
        [2,2,2,2,2,2,2,2],
        [3,3,3,3,3,3,3,3]
    ]
    tokens_zipped = paddle.to_tensor(tokens_zipped, dtype='bfloat16')
    routemap_topk = paddle.to_tensor(routemap_topk, dtype='int32')
    probs_topk = paddle.to_tensor(probs_topk, dtype='float32')
    expected_unzipped_probs = paddle.to_tensor(expected_unzipped_probs, dtype='float32')
    expected_zipped_expertwise_rowmap = paddle.to_tensor(expected_zipped_expertwise_rowmap, dtype='int32')
    expected_unzipped_tokens = paddle.to_tensor(expected_unzipped_tokens, dtype='bfloat16')
    expected_unzipped_expert_idx = paddle.to_tensor(expected_unzipped_expert_idx, dtype='int32')

    unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs, unzipped_expert_idx = TDU.tokens_unzip(tokens_zipped,routemap_topk, probs_topk,total_unzipped_tokens_num=total_unzipped_tokens_num, topk=topk, num_experts=expert_num)

    # 本算子
    zipped_tokens, zipped_probs_topk = TDU.tokens_zip(unzipped_tokens, zipped_expertwise_rowmap, routemap_topk, unzipped_probs, total_zipped_tokens=seqlen, num_experts=expert_num)
    # ------------------------- 前向验证 ------------------------
    print("-------- Tokens unzipped by customed op: ------------")
    print(unzipped_tokens)
    print("-------- Tokens expected : ------------")
    print(expected_unzipped_tokens)
    print("-------- Probs unzipped by customed op: ------------")
    print(unzipped_probs)
    print("-------- Probs expected: ------------")
    print(expected_unzipped_probs)
    print("-------- zipped expertwize rowmap by customed op: ------------")
    print(zipped_expertwise_rowmap)
    print("-------- rowmap expected: ------------")
    print(expected_zipped_expertwise_rowmap)
    print("-------- expert_idx unzipped by customed op: ------------")
    print(unzipped_expert_idx)
    print("-------- expert_idx expected: ------------")
    print(expected_unzipped_expert_idx)
    print("-------- zipped by customed op: ------------")
    print(zipped_tokens)
    print("-------- zipped expected: ------------")
    print(tokens_simple_zipped)

    print("-------- zipped probs_topk by customed op: ------------")
    print(zipped_probs_topk)
    print("-------- probs_topk expected: ------------")
    print(probs_topk)

    
def run():
    verify_tokens_unzip()

if __name__ == "__main__":
    run()