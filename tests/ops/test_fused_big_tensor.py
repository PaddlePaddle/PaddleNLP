import paddle
import paddle.incubate.nn.functional as F
import FusedQuantOps as FQO
import numpy as np
paddle.seed(0)


def test_fused_spaq(height, width):
    print(f'test_fused_spaq', (height, width))

    x = paddle.randn([height, width], dtype='bfloat16').clip_(min=-50, max=50)
    prob = paddle.randn([height, 1]).astype("float32")

    out, scale = FQO.fused_spaq(x, prob, using_pow2_scaling=False)
    paddle.base.core.eager._for_test_check_cuda_error()

    out_golden = F.swiglu(x) * prob
    out_dequant = (
        out.astype('float32') *
        scale.repeat_interleave(128, axis=1)[:, :out.shape[-1]]
    )
    paddle.base.core.eager._for_test_check_cuda_error()

    np.testing.assert_allclose(out_dequant, out_golden, atol=1, rtol=1e-2)


def test_fused_act_dequant(height, width):
    print(f"test_fused_act_dequant height:{height}, width:{width}")

    x_fp8 = paddle.ones([height, width], dtype="float8_e4m3fn")
    scale = paddle.randn([height, width // 128], dtype="float32")

    x_dequant = FQO.fused_act_dequant(x_fp8, scale)
    paddle.base.core.eager._for_test_check_cuda_error()

    x_gold = scale.repeat_interleave(128, axis=1)
    x_dequant = x_dequant.astype('float32')
    np.testing.assert_allclose(x_gold, x_dequant, atol=1e-2, rtol=1e-2)


def test_fused_swiglu_probs_bwd(topk, seq_len, moe_intermediate_size):
    print(f'test_fused_swiglu_probs_bwd topk:{topk} seq_len:{seq_len} moe_intermediate_size:{moe_intermediate_size}')

    o1 = paddle.rand([topk, seq_len, moe_intermediate_size * 2], dtype="bfloat16")
    unzipped_probs = paddle.rand([topk, seq_len, 1], dtype="float32")
    do2_s = paddle.rand([topk, seq_len , moe_intermediate_size], dtype="bfloat16")

    do1, pg, o2_s = FQO.fused_swiglu_probs_bwd(o1, do2_s, unzipped_probs)
    paddle.base.core.eager._for_test_check_cuda_error()

    def fn_gold():
        o2 = F.swiglu(o1)
        o2_s = (o2 * unzipped_probs)
        do2 = (do2_s.cast(paddle.float32) * unzipped_probs)
        do2 = do2.cast(paddle.bfloat16)
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        probs_grad = (do2_s.cast(paddle.float32) * (o2.cast(paddle.float32))).sum(axis=-1)
        return do1, probs_grad, o2_s

    do1_gold, pg_gold, o2_s_gold = fn_gold()
    paddle.base.core.eager._for_test_check_cuda_error()

    np.testing.assert_allclose(do1.astype('float32'), do1_gold.astype('float32'), atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(pg, pg_gold.flatten(), atol=1e-2, rtol=1e-3)
    np.testing.assert_allclose(o2_s.astype('float32'), o2_s_gold, atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
    for height in [8192, 16384, 32768, 128000, 510336]:
        for width in [4096, 7168]:
            test_fused_spaq(width, height)

    for height in [4096, 16384, 32768, 128000, 510336]:
        for width in [4096, 7168]:
            test_fused_act_dequant(height, width)

    for topk in [8]:
        for seq_len in [4096, 7168]:
            for moe_intermediate_size in [2048, 20480, 40960]:
                test_fused_swiglu_probs_bwd(topk, seq_len, moe_intermediate_size)
