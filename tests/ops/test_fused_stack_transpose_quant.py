import FusedQuantOps as FQO
import numpy as np

import paddle


def restore_stack_quant(out, scale):
    scale = paddle.repeat_interleave(scale, repeats=128, axis=0)
    scale = paddle.repeat_interleave(scale, repeats=128, axis=1)
    x = out.astype('float32') * scale
    return x


def test_fused_stack_transpose_quant(
    num_experts, seq_len, hidden_size, transpose
):
    print(num_experts, seq_len, hidden_size, transpose)

    x_vec = []
    for _ in range(num_experts):
        x = paddle.randn([seq_len, hidden_size], dtype='bfloat16')
        x = paddle.clip(x, min=-50, max=50)
        x_vec.append(x)

    if transpose:
        out, scale = FQO.fused_stack_transpose_quant(x_vec)
    else:
        out, scale = FQO.fused_stack_quant(x_vec)

    x_fp32 = paddle.stack(x_vec).reshape([-1, hidden_size]).astype('float32')
    x_restored = restore_stack_quant(out, scale)

    if transpose:
        x_restored = (
            x_restored.reshape([num_experts, hidden_size, seq_len])
            .transpose([0, 2, 1])
            .reshape([-1, hidden_size])
        )

    np.testing.assert_allclose(
        x_fp32, x_restored, rtol=0.01, atol=0.2
    )  # 存在截断误差，atol=0.2，通常在1e-6


def run():
    for batch_size in [1, 4]:
        for seq_len in [2048, 7168]:
            for hidden_size in [128, 4096]:
                for transpose in [False, True]:
                    test_fused_stack_transpose_quant(
                        batch_size, seq_len, hidden_size, transpose
                    )


if __name__ == "__main__":
    run()
