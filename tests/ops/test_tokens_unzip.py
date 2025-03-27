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

def check_expert_idx(unzipped_expert_idx, zipped_expertwise_rowmap, expert_num=4, zipped_tokens_num=3):
    unzipped_expert_idx = unzipped_expert_idx.reshape((1, -1))
    print(unzipped_expert_idx)
    assert zipped_tokens_num == zipped_expertwise_rowmap.shape[0]
    for token_idx in range(zipped_tokens_num):
        for expert_idx in range(expert_num):
            this_expert_row = zipped_expertwise_rowmap[token_idx][expert_idx]
            if this_expert_row != -1:
                expected_expert_idx = expert_idx
                this_unzipped_expert_idx = unzipped_expert_idx[0][this_expert_row]
                if this_unzipped_expert_idx != expected_expert_idx:
                    print(f"第{token_idx}行第{expert_idx}号专家不匹配， rowmap里是{expected_expert_idx}, unzipped里是{this_unzipped_expert_idx}")
                    return  
                else:
                    print(f"rowmap [{token_idx}][{expert_idx}] 指向{this_expert_row}, unzipped[{this_expert_row}] = {this_unzipped_expert_idx}")
    print("检查结束，通过")

def check_indices(dispatched_indices, zipped_expertwise_rowmap, unzipped_expert_idx):
    topk = dispatched_indices.shape[1]
    zipped_token_num = zipped_expertwise_rowmap.shape[0]
    for token_idx in range(zipped_token_num):
        for k in range(topk):
            this_expert = dispatched_indices[token_idx][k]
            if this_expert != -1: #有效专家
                this_expert_row = zipped_expertwise_rowmap[token_idx][this_expert]
                unzipped_expert = unzipped_expert_idx[this_expert_row]
                if unzipped_expert != this_expert:
                    print(f"unzipped[{this_expert_row}] 为{unzipped_expert}, 而原始数据[{token_idx}][{k}]为{this_expert}")
                    return 
                else:
                    print(f"dispatched_indices[{token_idx}][{k}] 派发到专家{this_expert}, unzipped[{this_expert_row}]结果为{unzipped_expert}")
    print("检查结束，通过")

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
        [0,0,0.3,0.5,0,0,0,0],
        [0.5,0,0,0,0,0,0,0],
        [0,0.7,0,0,0,0,0,0],
    ]
    total_unzipped_tokens_num = 4
    expected_unzipped_tokens= [
        [1,1,1,1,1,1,1,1],
        [2,2,2,2,2,2,2,2],
        [3,3,3,3,3,3,3,3],
        [1,1,1,1,1,1,1,1],
    ]
    expected_unzipped_probs= [
        0.3,
        0.5,
        0.7,
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
    # TODO: 加入fp8单测
    for prec in ['bfloat16']:
        tokens_zipped = paddle.to_tensor(tokens_zipped, dtype=prec)
        routemap_topk = paddle.to_tensor(routemap_topk, dtype='int32')
        probs_topk = paddle.to_tensor(probs_topk, dtype='bfloat16')
        expected_unzipped_probs = paddle.to_tensor(expected_unzipped_probs, dtype='bfloat16')
        expected_zipped_expertwise_rowmap = paddle.to_tensor(expected_zipped_expertwise_rowmap, dtype='int32')
        expected_unzipped_tokens = paddle.to_tensor(expected_unzipped_tokens, dtype=prec)
        expected_unzipped_expert_idx = paddle.to_tensor(expected_unzipped_expert_idx, dtype='int32')

        unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs, unzipped_expert_idx = TDU.tokens_unzip(tokens_zipped,routemap_topk, probs_topk,total_unzipped_tokens_num=total_unzipped_tokens_num, topk=topk, num_experts=expert_num)

        check_expert_idx(unzipped_expert_idx,zipped_expertwise_rowmap)
        check_indices(routemap_topk, zipped_expertwise_rowmap, unzipped_expert_idx)
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

    

    
def run():
    verify_tokens_unzip()

if __name__ == "__main__":
    run()