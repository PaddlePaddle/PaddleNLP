import FusedQuantOps as FQO
import numpy as np

import paddle


def restore_transpose_quant(out, scale):
    out = out.transpose([0, 2, 1]).astype('float32')
    scale = paddle.repeat_interleave(scale, repeats=128, axis=1)
    x = out * scale
    return x


def test_fused_transpose_quant(batch_size, seq_len, hidden_size):
    print(batch_size, seq_len, hidden_size)
    x = paddle.randn([batch_size, seq_len, hidden_size], dtype='bfloat16')
    x = paddle.clip(x, min=-50, max=50)

    out, scale = FQO.fused_transpose_quant(x)

    x_fp32 = x.astype('float32')
    x_restored = restore_transpose_quant(out, scale)

    np.testing.assert_allclose(
        x_fp32, x_restored, rtol=0.01, atol=0.2
    )  # 存在截断误差，atol=0.2，通常在1e-6


def run():
    for batch_size in [1, 4]:
        for seq_len in [2048, 7168]:
            for hidden_size in [1, 257, 2114, 4096]:
                test_fused_transpose_quant(batch_size, seq_len, hidden_size)


if __name__ == "__main__":
    run()
