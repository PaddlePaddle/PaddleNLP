# FasterGenerartion 使用

## 使用环境说明

* 本项目依赖于 PaddlePaddle 2.2.2 及以上版本或适当的 develop 版本
* CMake >= 3.10
* CUDA 10.1 以及以上 (需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/v4.0#setup) 使用必要的环境
* 环境依赖
  - jieba
  - h5py
  - colorlog
  - colorama
  - seqeval
  - multiprocess
  - attrdict
  - pyyaml
  - tqdm
  - datasets
  ```shell
  pip install jieba h5py colorlog colorama seqeval multiprocess attrdict pyyaml datasets
  pip install -U tqdm
  ```

## 目录结构

```text
ernie3-prompt/
├── infer.py                          # 动态图前向Demo代码
└── README.md                         # 当前README.md文件
```

## 快速开始

### 环境准备

首先，如果需要从源码自行编译，可以直接使用 Python 的 package 下的 paddlenlp，或是可从 github 克隆一个 PaddleNLP:

``` sh
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

其次，配置环境变量，让我们可以使用当前 clone 的 paddlenlp：

``` sh
export PYTHONPATH=$PWD/PaddleNLP/:$PYTHONPATH
```

### ERNIE3 decoding 示例代码(动态图)

使用 PaddlePaddle 仅执行 decoding 测试（float16）：

``` sh
export CUDA_VISIBLE_DEVICES=0
python infer.py
```

若当前环境下没有需要的自定义 op 的动态库，将会使用 JIT 自动编译需要的动态库。如果需要自行编译自定义 op 所需的动态库，可以参考 [文本生成高性能加速](../../../paddlenlp/ops/README.md)。编译好后，可以在执行 `ernie3_export_model_sample.py` 时使用 `--decoding_lib ../../../paddlenlp/ops/build/lib/libdecoding_op.so` 可以完成导入。
注意：如果是自行编译的话，这里的 `libdecoding_op.so` 的动态库是参照文档 [文本生成高性能加速](../../../paddlenlp/ops/README.md) 中 **`Python 动态图使用自定义 op`** 编译出来的 lib
