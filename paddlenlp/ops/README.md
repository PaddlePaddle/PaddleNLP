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
* CUDA 10.1（需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* [Faster Transformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#setup) 使用必要的环境

## 快速开始

我们实现了基于 GPU 的 Faster Transformer 的自定义 op 的接入。接下来，我们将分别介绍基于 Python 动态图和预测库使用 Faster Transformer 自定义 op 的方式，包括 op 的编译与使用。

## Python 动态图使用自定义 op

### 编译自定义OP

在 Python 动态图下使用自定义 OP 需要将实现的 C++、CUDA 代码编译成动态库，我们已经提供对应的 CMakeLists.txt ，可以参考使用如下的方式完成编译。

#### 克隆 PaddleNLP

首先，因为需要基于当前环境重新编译，当前的 paddlenlp 的 python 包里面并不包含 Faster Transformer 相关 lib，需要克隆一个 PaddleNLP，并重新编译:

``` sh
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

其次，配置环境变量，让我们可以使用当前 clone 的 paddlenlp，并进入到自定义 OP 的路径，准备后续的编译操作：

``` sh
export PYTHONPATH=$PWD/PaddleNLP/:$PYTHONPATH
cd PaddleNLP/paddlenlp/ops/
```

#### 编译

编译之前，请确保安装的 PaddlePaddle 的版本是基于最新的 develop 分支的代码编译，并且正常可用。

编译自定义 OP 可以参照一下步骤：

``` sh
mkdir build
cd build/
cmake .. -DSM=xx -DCMAKE_BUILD_TYPE=Release -DPY_CMD=python3.x
make -j
cd ../
```

注意：`xx` 是指的所用 GPU 的 compute capability。举例来说，可以将之指定为 70(V100) 或是 75(T4)。若未指定 `-DPY_CMD` 将会默认使用系统命令 `python` 对应的 Python。

最终，编译会在 `./build/lib/` 路径下，产出 `libdecoding_op.so`，即需要的 Faster Transformer decoding 执行的库。

#### 使用

编写 python 脚本的时候，调用 `FasterTransformer` API 并传入 `libdecoding_op.so` 的位置即可实现将 Faster Transformer 用于当前的预测。

举例如下：

``` python
from paddlenlp.ops import FasterTransformer

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
    decoding_strategy=args.decoding_strategy,
    beam_size=args.beam_size,
    topk=args.topk,
    topp=args.topp,
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

其中，`decoding_gemm` 不同参数的意义可以参考 [FasterTransformer 文档](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#execute-the-decoderdecoding-demos)。


## C++ 预测库使用自定义 op

### 编译自定义OP

在 C++ 预测库使用自定义 OP 需要将实现的 C++、CUDA 代码**以及 C++ 预测的 demo**编译成一个可执行文件。因预测库支持方式与 Python 不同，这个过程将不会产生自定义 op 的动态库，将直接得到可执行文件。我们已经提供对应的 CMakeLists.txt ，可以参考使用如下的方式完成编译。并获取执行 demo。

#### 克隆 PaddleNLP

首先，仍然是需要克隆一个 PaddleNLP:

``` sh
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

其次，让我们可以使用当前 clone 的 paddlenlp，并进入到自定义 OP 的路径，准备后续的编译操作：

``` sh
cd PaddleNLP/paddlenlp/ops/
```

#### 编译

编译之前，请确保安装的 PaddlePaddle 预测库的版本是基于最新的 develop 分支的代码编译，并且正常可用。

编译自定义 OP 可以参照一下步骤：

``` sh
mkdir build
cd build/
cmake .. -DSM=xx -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB=/path/to/paddle_inference_lib/ -DDEMO=./demo/transformer_e2e.cc -DWITH_STATIC_LIB=OFF -DON_INFER=ON -DWITH_MKL=ON
make -j
cd ../
```

注意：
* `xx` 是指的所用 GPU 的 compute capability。举例来说，可以将之指定为 70(V100) 或是 75(T4)。
* `-DPADDLE_LIB` 需要指明使用的 PaddlePaddle 预测库的路径 `/path/to/paddle_inference_install_dir/`，并且在该路径下，预测库的组织结构满足：
  ```text
  .
  ├── CMakeCache.txt
  ├── paddle/
    ├── include/
    └── lib/
  ├── third_party/
    ├── cudaerror/
    ├── install/
    └── threadpool/
  └── version.txt
  ```
* `-DDEMO` 说明预测库使用 demo 的位置。
* **当使用预测库的自定义 op 的时候，请务必开启 `-DON_INFER=ON` 选项，否则，不会得到预测库的可执行文件。**

#### 执行 decoding on PaddlePaddle

编译完成后，在 `build/bin/` 路径下将会看到 `transformer_e2e` 的一个可执行文件。通过设置对应的设置参数完成执行的过程。

``` sh
cd bin/
./transformer_e2e <batch_size> <gpu_id> <model_directory> <dict_directory> <input_data>
```

举例说明：

``` sh
cd bin/
../third-party/build/bin/decoding_gemm 8 5 8 64 38512 256 512 0
./transformer_e2e 8 0 ./infer_model/ DATA_HOME/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708 DATA_HOME/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en
```

其中：
* `decoding_gemm` 不同参数的意义可以参考 [FasterTransformer 文档](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#execute-the-decoderdecoding-demos)。
* `DATA_HOME` 则是 `paddlenlp.utils.env.DATA_HOME` 返回的路径。

预测所需要的模型文件，可以通过 `PaddleNLP/examples/machine_translation/transformer/faster_transformer/README.md` 文档中所记述的方式导出。
