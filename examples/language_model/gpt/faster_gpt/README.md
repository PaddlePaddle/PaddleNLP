# Faster GPT 使用

在这里我们集成了 NVIDIA [Faster Transformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1) 用于预测加速。同时集成了 FasterGPT float32 以及 float16 预测。以下是使用 FasterGPT 的使用说明。

## 使用环境说明

* 本项目依赖于 PaddlePaddle 2.1.0 及以上版本或适当的 develop 版本
* CMake >= 3.10
* CUDA 10.1（需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* [Faster Transformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#setup) 使用必要的环境

## 快速开始

我们实现了基于 GPU 的 FasterGPT 的自定义 op 的接入。接下来，我们将分别介绍基于 Python 动态图和预测库使用 FasterGPT 自定义 op 的方式，包括 op 的编译与使用。

## Python 动态图使用自定义 op

### 编译自定义OP

在 Python 动态图下使用自定义 OP 需要将实现的 C++、CUDA 代码编译成动态库，我们已经提供对应的 CMakeLists.txt ，可以参考使用如下的方式完成编译。同样的自定义 op 编译的说明也可以在自定义 op 对应的路径 `PaddleNLP/paddlenlp/ops/` 下面找到。

#### 克隆 PaddleNLP

首先，因为需要基于当前环境重新编译，当前的 paddlenlp 的 python 包里面并不包含 FasterGPT 相关 lib，需要从源码自行编译，可以直接使用 Python 的 package 下的 paddlenlp，或是可从 github 克隆一个 PaddleNLP，并重新编译。

``` sh
git clone https://github.com/PaddlePaddle/PaddleNLP.git
```

其次，配置环境变量，让我们可以使用当前 clone 的 paddlenlp，并进入到自定义 OP 的路径，准备后续的编译操作：

``` sh
export PYTHONPATH=$PWD/PaddleNLP/:$PYTHONPATH
cd PaddleNLP/paddlenlp/ops/
```

#### 编译

编译之前，请确保安装的 PaddlePaddle 的版本是大于 2.1.0 或是最新的 develop 分支的代码编译，并且正常可用。

编译自定义 OP 可以参照一下步骤：

``` sh
mkdir build
cd build/
cmake .. -DSM=xx -DCMAKE_BUILD_TYPE=Release -DPY_CMD=python3.x -DWITH_GPT=ON
make -j
cd ../
```

其中，
* `-DSM`: 是指的所用 GPU 的 compute capability。举例来说，可以将之指定为 70(V100) 或是 75(T4)。
* `-DPY_CMD`: 是指编译所使用的 python，若未指定 `-DPY_CMD` 将会默认使用系统命令 `python` 对应的 Python 版本。
* `-DWITH_GPT`: 是指是否编译带有 FasterGPT 自定义 op 的动态库。


最终，编译会在 `./build/lib/` 路径下，产出 `libdecoding_op.so`，即需要的 FasterGPT decoding 执行的库。

### 使用 GPT-2 decoding 高性能推理

编写 python 脚本的时候，调用 `FasterGPT` API 并传入 `libdecoding_op.so` 的位置即可实现将 FasterGPT 用于当前的预测。

``` python
from paddlenlp.ops import FasterGPT
from paddlenlp.transformers import GPTModel, GPTForPretraining

MODEL_CLASSES = {
    "gpt2-medium-en": (GPTLMHeadModel, GPTTokenizer),
}

model_class, tokenizer_class = MODEL_CLASSES[args.model_name]
tokenizer = tokenizer_class.from_pretrained(args.model_name)
model = model_class.from_pretrained(args.model_name)

# Define model
gpt = FasterGPT(
    model=model,
    candidate_num=args.candidate_num,
    probability_threshold=args.probability_threshold,
    max_seq_len=args.max_seq_len,
    start_id=start_id,
    end_id=end_id,
    temperature=args.temperature,
    decoding_lib=args.decoding_lib,
    use_fp16_decoding=args.use_fp16_decoding)
```

目前，GPT-2 的例子仅支持 `batch size` 为 `1` 或是 batch 内输入的样本的长度都是相同的情况。并且，仅支持 topk-sampling 和 topp-sampling，不支持 beam-search。

更详细的例子可以参考 `./infer.py`，我们提供了更详细用例。

#### 执行 GPT-2 decoding on PaddlePaddle

使用 PaddlePaddle 仅执行 decoding 测试（float32）：

``` sh
export CUDA_VISIBLE_DEVICES=0
python infer.py --model_name_or_path gpt2-medium-en --decoding_lib ./build/lib/libdecoding_op.so --batch_size 1 --topk 4 --topp 0.0 --max_out_len 32 --start_token "<|endoftext|>" --end_token "<|endoftext|>" --temperature 1.0
```

其中，各个选项的意义如下：
* `--model_name_or_path`: 预训练模型的名称或是路径。
* `--decoding_lib`: 指向 `libdecoding_op.so` 的路径。需要包含 `libdecoding_op.so`。若不存在则将自动进行 jit 编译产出该 lib。
* `--batch_size`: 一个 batch 内，样本数目的大小。
* `--candidate_num`: 执行 topk-sampling 的时候的 `k` 的大小，默认是 4。
* `--probability_threshold`: 执行 topp-sampling 的时候的阈值的大小，默认是 0.0 表示不执行 topp-sampling。
* `--max_seq_len`: 最长的生成长度。
* `--start_token`: 字符串，表示任意生成的时候的开始 token。
* `--end_token`: 字符串，生成的结束 token。
* `--temperature`: temperature 的设定。
* `--use_fp16_decoding`: 是否使用 fp16 进行推理。


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
cmake .. -DSM=xx -DWITH_GPT=ON -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB=/path/to/paddle_inference_lib/ -DDEMO=./demo/gpt.cc -DON_INFER=ON -DWITH_MKL=ON
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
* `-DDEMO` 说明预测库使用 demo 的位置。比如指定 -DDEMO=./demo/gpt.cc。最好使用绝对路径，若使用相对路径，需要是相对于 `PaddleNLP/paddlenlp/ops/faster_transformer/src/` 的相对路径。
* `-DWITH_GPT`，如果是编译 GPT 的预测库可执行文件，需要加上 `-DWITH_GPT=ON`。
* **当使用预测库的自定义 op 的时候，请务必开启 `-DON_INFER=ON` 选项，否则，不会得到预测库的可执行文件。**

#### 执行 GPT decoding on PaddlePaddle

如果需要使用 Paddle Inference 预测库针对 GPT 进行预测，首先，需要导出预测模型，可以通过 `./export_model.py` 脚本获取预测库用模型，执行方式如下所示：

``` sh
python ./export_model.py --model_name_or_path gpt2-medium-en --decoding_lib ./build/lib/libdecoding_op.so --topk 4 --topp 0.0 --max_out_len 32 --start_token "<|endoftext|>" --end_token "<|endoftext|>" --temperature 1.0 --inference_model_dir ./infer_model/
```

各个选项的意义与上文的 `infer.py` 的选项相同。额外新增一个 `--inference_model_dir` 选项用于指定保存的模型文件、词表等文件。若是使用的模型是 gpt2-medium-en，保存之后，`./infer_model/` 目录下组织的结构如下：

``` text
.
├── gpt.pdiparams       # 保存的参数文件
├── gpt.pdiparams.info  # 保存的一些变量描述信息，预测不会用到
├── gpt.pdmodel         # 保存的模型文件
├── merges.txt          # bpe
└── vocab.txt           # 词表
```

同理，完成编译后，可以在 `PaddleNLP/paddlenlp/ops/build/bin/` 路径下将会看到 `gpt` 的一个可执行文件。通过设置对应的设置参数完成执行的过程。

``` sh
cd bin/
./gpt -batch_size 1 -gpu_id 0 -model_dir path/to/model -vocab_dir path/to/vocab -start_token "<|endoftext|>" -end_token "<|endoftext|>"
```
