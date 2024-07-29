import paddle
from paddlenlp_ops import cutlass_fp8_fp8_half_gemm_fused

A = paddle.ones([2,32, 64], dtype="float8_e4m3fn")
B = paddle.ones([2,32, 64], dtype="float8_e4m3fn")

res0 = cutlass_fp8_fp8_half_gemm_fused(
                    A, 
                    B, 
                    bias=None,
                    transpose_x=False,
                    transpose_y=True,
                    output_dtype="float16",
                    scale=0.5,
                    activation_type = "identity"
                )
print("res0: ",res0)

A = paddle.ones([2,32, 64], dtype="float8_e5m2")
B = paddle.ones([2,128, 64], dtype="float8_e5m2")

res1 = cutlass_fp8_fp8_half_gemm_fused(
                    A, 
                    B, 
                    bias=None,
                    transpose_x=False,
                    transpose_y=True,
                    output_dtype="bfloat16",
                    scale=0.5,
                    activation_type = "identity"
                )

A = paddle.ones([2,32, 64], dtype="float32")
B = paddle.ones([2,128, 64], dtype="float32")
expect_result = 0.5*paddle.matmul(A, B.transpose([0, 2, 1]))

result0 = paddle.equal_all(
                    paddle.cast(res0, "float32"),
                    paddle.to_tensor(expect_result),
                )

result1 = paddle.equal_all(
                    paddle.cast(res1, "float32"),
                    paddle.to_tensor(expect_result),
                )

print("result0: ",result0)
print("result1: ",result1)