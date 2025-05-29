import FusedQuantOps as FQO
import numpy as np

import paddle


def restore_transpose_split_quant(out, scale):
    out = [t.astype('float32') for t in out]
    out = paddle.concat(out, axis=1).transpose([1, 0])
    scale = paddle.concat(scale, axis=0)
    scale = paddle.repeat_interleave(scale, repeats=128, axis=0)
    return out * scale


def test_fused_transpose_split_quant(tokens_per_expert, seq_len, pow_2_scales):
    print(tokens_per_expert, seq_len, pow_2_scales)

    x = paddle.randn([sum(tokens_per_expert), seq_len], dtype='bfloat16')
    x = paddle.clip(x, min=-50, max=50)

    out, scale = [], []
    for tokens in tokens_per_expert:
        out.append(paddle.empty([seq_len, tokens], dtype='float8_e4m3fn'))
        scale.append(paddle.empty([tokens//128, seq_len], dtype='float32'))

    FQO.fused_transpose_split_quant(x, out, scale, pow_2_scales)

    x_restore = restore_transpose_split_quant(out, scale)
    x_cast = x.astype('float32')

    np.testing.assert_allclose(x_cast, x_restore, rtol=0.01, atol=0.3)


def run():
    test_fused_transpose_split_quant([0, 0], 1024, False)
    test_fused_transpose_split_quant([128, 2*128], 0, True)
    test_fused_transpose_split_quant([128], 1, False)
    test_fused_transpose_split_quant([0, 128, 0, 2*128], 127, True)
    test_fused_transpose_split_quant([3*128, 4*128, 5*128], 233, False)
    test_fused_transpose_split_quant(
        [24*128, 128, 50*128, 16*128], 2162, True
    )
    test_fused_transpose_split_quant(
        [7*128, 29*128, 3*128, 128*128, 13*128], 4000, False
    )
    test_fused_transpose_split_quant(
        [18*128, 5*128, 24*128, 128, 6*128, 0, 27*128, 7*128], 7168, True
    )


if __name__ == '__main__':
    run()
