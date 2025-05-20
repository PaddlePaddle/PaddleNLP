import numpy as np
import paddle
import TokenDispatcherUtils as TDU


def fabricate_dispatch_result(
    seqlen, token_length, topk, num_experts, data_type="bfloat32", broadcast_ratio=0.5
):
    tokens = paddle.randn([seqlen, token_length], dtype=data_type)

    tokens_scale = paddle.empty([0])
    if data_type == "float8_e4m3fn":
        scale_cols = (token_length + 127) // 128
        tokens_scale = paddle.randn([seqlen, scale_cols], dtype="float32")

    # 计算每个token选择的专家数量，更集中在期望值附近
    expected_experts = max(1, min(broadcast_ratio * num_experts, topk))

    # 使用正态分布生成专家数量，确保集中在期望值附近
    std_dev = max(1, expected_experts / 6)  # 标准差设为期望值的1/4，确保集中分布
    experts_count = paddle.normal(expected_experts, std_dev, [seqlen])
    # 四舍五入并裁剪到合理范围
    # experts_count = paddle.clip(paddle.round(experts_count), 1, min(topk, num_experts))
    experts_count = paddle.clip(paddle.round(experts_count), 1, min(topk, num_experts))
    experts_count = paddle.cast(experts_count, "int32")

    # 预分配结果数组
    dispatched_indices = paddle.full([seqlen, topk], -1, dtype="int32")
    dispatched_probs = paddle.zeros([seqlen, topk], dtype="float32")

    # 批量生成专家索引和概率
    for i in range(seqlen):
        count = experts_count[i].item()

        # 高效生成随机不重复专家索引
        indices = paddle.randperm(num_experts)[:count]
        dispatched_indices[i, :count] = indices

        # 高效设置概率值
        prob_value = 1.0 / count
        dispatched_probs[i, :count] = paddle.full([count], prob_value, dtype=data_type)

    # 高效计算每个专家的最大token数
    # 创建one-hot编码
    valid_indices = dispatched_indices.reshape([-1])
    valid_mask = valid_indices >= 0
    valid_experts = valid_indices[valid_mask]

    # 使用histogram统计每个专家的token数
    expert_counts = paddle.histogram(
        valid_experts, bins=num_experts, min=0, max=num_experts - 1
    )
    expert_counts = paddle.cast(expert_counts, "int32")
    expert_counts = list(expert_counts)
    print("expert counts: ", expert_counts)

    return (
        tokens,
        tokens_scale,
        dispatched_indices,
        dispatched_probs,
        expert_counts,
    )


def tensor_max_abs_rel_err(a, b, eps=1e-8):
    max_abs_err = paddle.max(paddle.abs(a - b))
    denom = paddle.maximum(paddle.abs(a), paddle.abs(b))
    denom = paddle.maximum(denom, paddle.to_tensor(eps, dtype=denom.dtype))
    max_rel_err = paddle.max(paddle.abs(a - b) / denom)
    return max_abs_err, max_rel_err


def test_unzip_zip():
    SEQLEN = 16384
    TOKEN_LEN = 7168
    for dt in ["bfloat16"]:
        for expert_num in [4, 8, 16, 32]:
            for topk in [4, 8, 12]:
                print("###################################")
                print(
                    "testing with {} experts and topk {}, datatype is {}".format(
                        expert_num, topk, dt
                    )
                )
                (
                    tokens,
                    tokens_scale,
                    dispatched_indices,
                    dispatched_probs,
                    expert_tokens_count,
                ) = fabricate_dispatch_result(
                    SEQLEN,
                    TOKEN_LEN,
                    topk,
                    expert_num,
                    data_type=dt,
                    broadcast_ratio=0.5,
                )
                if dt == "bfloat16":
                    tokens_scale = None
                (
                    unzipped_tokens,
                    zipped_expertwise_rowmap,
                    unzipped_probs,
                    unzipped_scales,
                ) = TDU.tokens_unzip_stable(
                    tokens,
                    tokens_scale,
                    dispatched_indices,
                    dispatched_probs,
                    topk=topk,
                    num_experts=expert_num,
                    tokens_per_expert=expert_tokens_count,
                    padding_multiplex=128
                )
                tokens_recovered, probs_recovered = TDU.tokens_zip(
                    (unzipped_tokens * unzipped_probs.unsqueeze(-1)).astype("bfloat16"),
                    zipped_expertwise_rowmap,
                    dispatched_indices,
                    unzipped_probs,
                    total_zipped_tokens=SEQLEN,
                    num_experts=expert_num,
                )
                print(
                    "unzip-zip tokens 最大绝对误差：{}, 相对误差：{}".format(
                        *tensor_max_abs_rel_err(tokens, tokens_recovered)
                    )
                )
                print(
                    "unzip-zip probs 最大绝对误差：{}, 相对误差：{}".format(
                        *tensor_max_abs_rel_err(dispatched_probs, probs_recovered)
                    )
                )


# core.nvprof_enable_record_event()

if __name__ == "__main__":
    test_unzip_zip()
