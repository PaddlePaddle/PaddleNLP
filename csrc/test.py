import paddle
from paddlenlp_ops import cutlass_fp8_fp8_fp8_gemm_fused
from paddlenlp_ops import cutlass_fp8_fp8_half_gemm_fused


def fp32_gemm(A, B):
    fp32_res = paddle.matmul(A.cast("float32"), B.cast("float32"), transpose_x=False, transpose_y=True)
    fp32_res = fp32_res.cast("float8_e4m3fn").cast("float32")
    return fp32_res

def fp16_gemm(A, B, output_scale):
    fp8_fp8_half_res = cutlass_fp8_fp8_half_gemm_fused(
        A, B, bias=None, transpose_x=False, transpose_y=True, scale=1.0,output_dtype="float16", act="identity"
    )
    fp8_fp8_half_res = fp8_fp8_half_res.cast("float32").cast("float8_e4m3fn").cast("float32")
    return fp8_fp8_half_res

def fp8_gemm(A, B, output_scale=1.0):
    output_scale_tensor = paddle.to_tensor([output_scale], dtype="float32")
    fp8_fp8_fp8_res = cutlass_fp8_fp8_fp8_gemm_fused(
        A, B, bias=None, scale_out=output_scale_tensor, transpose_x=False, transpose_y=True, scale=1,output_dtype="float8_e4m3fn", act="identity"
    ).cast("float32")
    return fp8_fp8_fp8_res
def gemm(m, n, k):
    paddle.seed(1)
    A = paddle.rand(shape=[m, k]).cast("float8_e4m3fn")
    B = paddle.rand(shape=[n, k]).cast("float8_e4m3fn")

    a = 34.03216553
    print(paddle.to_tensor(a).cast("float8_e4m3fn").cast("float32"))
    

    output_scale = 1.0
    
    fp32_res = fp32_gemm(A, B)
    fp32_res = fp32_res.numpy().reshape([-1])

    fp16_res = fp16_gemm(A, B, output_scale)
    fp16_res = fp16_res.numpy().reshape([-1])
    
    fp8_res = fp8_gemm(A, B, output_scale)
    fp8_res = fp8_res.numpy().reshape([-1])

    print(fp32_res[130881])
    print(fp16_res[130881])
    print(fp8_res[130881])
    
    count = 0
    for i in range(fp32_res.shape[0]):
        if fp8_res[i] != fp32_res[i] and fp8_res[i] != fp16_res[i]:
            count += 1
            print(f'{i}: fp32_res: {fp32_res[i]}, fp16_res: {fp16_res[i]}, fp8_res: {fp8_res[i]}')
    print(f'total different elements is : {count}')
    
    return None
if __name__ == "__main__":
    m = 512
    n = 256
    k = 128
    for i in range(32, m + 32, 32):
        gemm(m, n, k)
        paddle.device.cuda.empty_cache()
        exit()
    