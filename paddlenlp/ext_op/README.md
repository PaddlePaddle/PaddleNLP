# 自定义OP编译使用

## 子目录结构

```text
.
├── sample/                 # 基于 Transformer 机器翻译使用样例（beam search）
├── src/                    # 自定义 OP C++ CUDA 代码
└── transformer/            # Python API 封装脚本
```

## 使用环境说明

* 本项目依赖于 PaddlePaddle 2.0.1 及以上版本或适当的 develop 版本
* CMake >= 3.10
* CUDA 10.1 或是更新的版本（需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* [Faster Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer/v3.1#setup) 使用必要的环境

## 快速开始

### 编译自定义OP

自定义 OP 需要将实现的 C++、CUDA 代码编译成动态库，我们提供对应的 CMakeLists.txt ，可以参考使用如下的方式完成编译。

#### 克隆 PaddleNLP

首先，因为需要基于当前环境重新编译，当前的 paddlenlp 的 python 包里面并不包含 Faster Transformer 相关 lib，需要克隆一个 PaddleNLP，并重新编译:
``` sh
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

其次，配置环境变量，让我们可以使用当前 clone 的 paddlenlp，并进入到自定义 OP 的路径，准备后续的编译操作：

``` sh
export PYTHONPATH=$PWD/PaddleNLP/:$PYTHONPATH
cd PaddleNLP/paddlenlp/ext_op/
```

#### 编译

编译之前，请确保安装的 PaddlePaddle 的版本需要大于 2.0.1，并且正常可用。

编译自定义 OP 可以参照一下步骤：

``` sh
mkdir build
cd build/
cmake .. -DSM=xx -DCMAKE_BUILD_TYPE=Release
make -j
cd ../
```

注意：`xx` 是指的所用 GPU 的 compute capability。举例来说，可以将之指定为 70(V100) 或是 75(T4)。

最终，编译会在 `./build/lib/` 路径下，产出 `libdecoding_op.so`，即需要的 Faster Transformer decoding 执行的库。

#### 使用

编写 python 脚本的时候，调用 `FasterTransformer` API 并传入 `libdecoding_op.so` 的位置即可实现将 Faster Transformer 用于当前的预测。

举例如下：

``` python
from paddlenlp.ext_op import FasterTransformer

transformer = FasterTransformer(
    src_vocab_size=args.src_vocab_size,
    trg_vocab_size=args.trg_vocab_size,
    max_length=args.max_length + 1,
    n_layer=args.n_layer,
    n_head=args.n_head,
    d_model=args.d_model,
    d_inner_hid=args.d_inner_hid,
    dropout=args.dropout,
    weight_sharing=args.weight_sharing,
    bos_id=args.bos_idx,
    eos_id=args.eos_idx,
    beam_size=args.beam_size,
    max_out_len=args.max_out_len,
    decoding_lib=args.decoding_lib,
    use_fp16_decoding=args.use_fp16_decoding)
```

更详细的例子可以参考 `./sample/decoding_sample.py` 以及 `./sample/encoder_decoding_sample.py`，我们提供了更详细用例。

#### 执行 decoding on PaddlePaddle

使用 PaddlePaddle 仅执行 decoding 测试（float32）：

``` sh
export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.1
./build/third-party/build/bin/decoding_gemm 32 4 8 64 30000 32 512 0
python sample/decoding_sample.py --config ./sample/config/decoding.sample.yaml --decoding-lib ./build/lib/libdecoding_op.so
```

使用 PaddlePaddle 仅执行 decoding 测试（float16）：
执行 float16 的 decoding，需要在执行的时候，加上 `--use-fp16-decoding` 选项。

``` sh
export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.1
./build/third-party/build/bin/decoding_gemm 32 4 8 64 30000 32 512 1
python sample/decoding_sample.py --config ./sample/config/decoding.sample.yaml --decoding-lib ./build/lib/libdecoding_op.so --use-fp16-decoding
```

其中，`decoding_gemm` 不同参数的意义可以参考 [FasterTransformer 文档](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer/v3.1#execute-the-decoderdecoding-demos)。
