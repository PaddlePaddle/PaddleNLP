# PaddleNLP 自定义 OP

此文档介绍如何编译安装 PaddleNLP 自定义 OP。

## 安装 C++ 依赖

```shell
pip install -r requirements.txt
```

## 编译 Cuda 算子

```shell
python setup_cuda.py install
```

## 生成 FP8的 cutlass 算子
```shell
python utils/auto_gen_fp8_fp8_gemm_fused_kernels.py

python utils/auto_gen_fp8_fp8_dual_gemm_fused_kernels.py
```

### FP8 GEMM 自动调优
```shell
sh tune_fp8_gemm.sh
```
