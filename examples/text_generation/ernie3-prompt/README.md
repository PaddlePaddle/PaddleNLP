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
├── faster_ernie3_prompt
│   ├── ernie3_prompt_export_model_sample.py # 动转静模型导出代码
│   ├── ernie3_prompt_inference.py           # 静态图推理代码
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

### ERNIE3-Prompt decoding 示例代码(动态图)

使用 PaddlePaddle 仅执行 decoding 测试（float16）：

``` sh
export CUDA_VISIBLE_DEVICES=0
python infer.py
```

若当前环境下没有需要的自定义 op 的动态库，将会使用 JIT 自动编译需要的动态库。如果需要自行编译自定义 op 所需的动态库，可以参考 [文本生成高性能加速](../../../paddlenlp/ops/README.md)。编译好后，可以在执行 `ernie3_export_model_sample.py` 时使用 `--decoding_lib ../../../paddlenlp/ops/build/lib/libdecoding_op.so` 可以完成导入。
注意：如果是自行编译的话，这里的 `libdecoding_op.so` 的动态库是参照文档 [文本生成高性能加速](../../../paddlenlp/ops/README.md) 中 **`Python 动态图使用自定义 op`** 编译出来的 lib

### 导出基于 FasterErnie3Prompt 的预测库使用模型文件

Ernie3Prompt的FasterGenerartion高性能预测功能底层依托于`FasterErnie3Prompt()`。
编写 python 脚本的时候，调用 `FasterErnie3Prompt()` API 即可创建可用于导出的高性能预测模型。

通过 `faster_ernie3_prompt/ernie3_prompt_export_model_sample.py` 脚本获取预测库用模型，执行方式如下所示：

``` sh
python faster_ernie3_prompt/ernie3_prompt_export_model_sample.py
```

各个选项的意义与上文的 `infer.py` 的选项相同。额外新增一个 `--inference_model_dir` 选项用于指定保存的模型文件、词表等文件。
若是使用的模型是 ernie3-prompt，保存之后，`./infer_model/` 目录下组织的结构如下：

``` text
.
├── ernie3_prompt.pdiparams       # 保存的参数文件
├── ernie3_prompt.pdiparams.info  # 保存的一些变量描述信息，预测不会用到
└── ernie3_prompt.pdmodel         # 保存的模型文件
```

### Python 预测库使用高性能加速

使用上一步导出的模型，来进行推理

``` sh
python faster_ernie3_prompt/ernie3_prompt_inference.py \
--inference_model_dir ./infer_model \
```

### C++ 预测库使用高性能加速

C++ 预测库使用 FasterErnie3Prompt 的高性能加速需要自行编译，可以参考 [文本生成高性能加速](../../../paddlenlp/ops/README.md) 文档完成基于 C++ 预测库的编译，同时也可以参考相同文档执行对应的 C++ 预测库的 demo 完成预测。
