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

### 手动安装 Cutlass 库
1. 访问 Cutlass 仓库: [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)

2. 拉取代码:
    git clone -b v3.5.0 --single-branch https://github.com/NVIDIA/cutlass.git

3. 将下载的 `cutlass` 目录放在 `csrc/gpu/cutlass_kernels/cutlass`下

4. 重新编译 Cuda 算子
```shell
python setup_cuda.py install
```
