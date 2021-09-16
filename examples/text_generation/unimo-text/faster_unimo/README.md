# FasterUNIMOText 预测

在这里我们集成并升级了 NVIDIA [Faster Transformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1) 用于预测加速。以下是使用 FasterUNIMOText 的说明。

**需要注意的是：**这里的说明仅涉及使用 FasterUNIMOText 进行预测，而预测所需的模型，需要经过上一级目录 `../run_gen.py` finetuning 获取。

## 使用环境说明

* 本项目依赖于 PaddlePaddle 2.1.2 或是最新的 develop 版本，可能需要自行编译 PaddlePaddle
* CMake >= 3.10
* CUDA 10.1（需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* [Faster Transformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#setup) 使用必要的环境

## 快速开始

我们实现了基于 GPU 的 FasterUNIMOText 的自定义 op 的接入。接下来，我们将介绍基于 Python 动态图使用 FasterUNIMOText 自定义 op 的方式，包括 op 的编译与使用。

## Python 动态图使用自定义 op

### 编译自定义OP

在 Python 动态图下使用自定义 OP 需要将实现的 C++、CUDA 代码编译成动态库，我们已经提供对应的 CMakeLists.txt ，可以参考使用如下的方式完成编译。同样的自定义 op 编译的说明也可以在自定义 op 对应的路径 `PaddleNLP/paddlenlp/ops/` 下面找到。

#### 克隆 PaddleNLP

首先，因为需要基于当前环境重新编译，当前的 paddlenlp 的 python 包里面并不包含 FasterUNIMOText 相关 lib，需要从源码自行编译，可以直接使用 Python 的 package 下的 paddlenlp，或是可从 github 克隆一个 PaddleNLP，并重新编译。

以下以从 github 上 clone 一个新版 PaddleNLP 为例:

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
cmake .. -DSM=xx -DCMAKE_BUILD_TYPE=Release -DPY_CMD=python3.x -DWITH_UNIFIED=ON
make -j
cd ../
```

其中，
* `-DSM`: 是指的所用 GPU 的 compute capability。举例来说，可以将之指定为 70(V100) 或是 75(T4)。
* `-DPY_CMD`: 是指编译所使用的 python，若未指定 `-DPY_CMD` 将会默认使用系统命令 `python` 对应的 Python 版本。
* `-DWITH_UNIFIED`: 是指是否编译带有 FasterUNIMOText 自定义 op 的动态库。

最终，编译会在 `./build/lib/` 路径下，产出 `libdecoding_op.so`，即需要的 FasterUNIMOText decoding 执行的库。

## 使用 FasterUNIMOText 完成预测

编写 python 脚本的时候，调用 `FasterUNIMOText` API 并传入 `libdecoding_op.so` 的位置即可实现将 FasterUNIMOText 用于当前的预测。

举例如下：

``` python
from paddlenlp.ops import FasterUNIMOText

model = UNIMOLMHeadModel.from_pretrained(args.model_name_or_path)
tokenizer = UNIMOTokenizer.from_pretrained(args.model_name_or_path)

model = FasterUNIMOText(
    model,
    decoding_strategy=args.decode_strategy,
    decoding_lib=args.decoding_lib,
    use_fp16_decoding=args.use_fp16_decoding)
```

更详细的例子可以参考 `infer.py`，我们提供了更详细用例。需要注意的是，当前 FasterUNIMOText 只支持基础的策略，后续我们会进一步丰富像是 length penalty 等策略以提升生成式 API 推理性能。


### 数据准备

比赛使用三个任务数据集测试参赛系统的生成能力，包括文案生成(AdvertiseGen)、摘要生成(LCSTS_new)和问题生成(DuReaderQG)：

- 文案生成根据结构化的商品信息生成合适的广告文案；
- 摘要生成是为输入文档生成简洁且包含关键信息的简洁文本；
- 问题生成则是根据给定段落以及答案生成适合的问题。

为了方便用户快速使用基线，PaddleNLP Dataset API内置了数据集，一键即可完成数据集加载，示例代码如下：

```python
from paddlenlp.datasets import load_dataset
train_ds, dev_ds = load_dataset('dureader_qg', splits=('train', 'dev'))
```


### 模型预测

运行下方脚本可以使用训练好的模型进行预测。

```shell
export CUDA_VISIBLE_DEVICES=0
python infer.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=your_model_path \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --decode_strategy=sampling \
    --device=gpu
```

程序运行结束后会将预测结果保存在`output_path`中。将预测结果准备成比赛官网要求的格式，提交评估即可得评估结果。
