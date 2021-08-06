# 自定义OP编译使用

## 子目录结构

```text
.
├── faster_transformer/       # 基于自定义 op Faster Transformer 子路径
  ├── sample/                 # 基于 Faster Transformer 使用样例
  ├── src/                    # 自定义 OP C++ CUDA 代码
  └── transformer/            # Python API 封装脚本
└── patches                   # 自定义 op 第三方库自定义补丁代码
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

注意：
* `xx` 是指的所用 GPU 的 compute capability。举例来说，可以将之指定为 70(V100) 或是 75(T4)
* 若未指定 `-DPY_CMD` 将会默认使用系统命令 `python` 对应的 Python。
* 若使用 GPT-2 高性能推理，需要加上 -DWITH_GPT=ON。


最终，编译会在 `./build/lib/` 路径下，产出 `libdecoding_op.so`，即需要的 Faster Transformer decoding 执行的库。

### 使用 Transformer decoding 高性能推理

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

更详细的例子可以参考 `./faster_transformer/sample/decoding_sample.py` 以及 `./sample/encoder_decoding_sample.py`，我们提供了更详细用例。

#### 执行 Transformer decoding on PaddlePaddle

使用 PaddlePaddle 仅执行 decoding 测试（float32）：

``` sh
export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.1
./build/third-party/build/bin/decoding_gemm 32 4 8 64 30000 32 512 0
python ./faster_transformer/sample/decoding_sample.py --config ./faster_transformer/sample/config/decoding.sample.yaml --decoding_lib ./build/lib/libdecoding_op.so
```

使用 PaddlePaddle 仅执行 decoding 测试（float16）：
执行 float16 的 decoding，需要在执行的时候，加上 `--use_fp16_decoding` 选项。

``` sh
export CUDA_VISIBLE_DEVICES=0
export FLAGS_fraction_of_gpu_memory_to_use=0.1
./build/third-party/build/bin/decoding_gemm 32 4 8 64 30000 32 512 1
python ./faster_transformer/sample/decoding_sample.py --config ./faster_transformer/sample/config/decoding.sample.yaml --decoding_lib ./build/lib/libdecoding_op.so --use_fp16_decoding
```

其中，`decoding_gemm` 不同参数的意义可以参考 [FasterTransformer 文档](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#execute-the-decoderdecoding-demos)。

### 使用 GPT-2 decoding 高性能推理

与 `FasterTransformer` 类似，可以通过一下方式调用 GPT-2 相关优化：

``` python
from paddlenlp.ops import FasterGPT
from paddlenlp.transformers import GPTModel, GPTForPretraining

MODEL_CLASSES = {
    "gpt2-medium-en": (GPTForPretraining, GPTTokenizer),
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

目前，GPT-2 的例子仅支持 `batch size` 为 `1` 或是 batch 内输入的序列长度相等的情况。并且，仅支持 topk-sampling 和 topp-sampling，不支持 beam-search。

更详细的例子可以参考 `./faster_transformer/sample/gpt_sample.py`，我们提供了更详细用例。

#### 执行 GPT-2 decoding on PaddlePaddle

使用 PaddlePaddle 仅执行 decoding 测试（float32）：

``` sh
export CUDA_VISIBLE_DEVICES=0
python ./faster_transformer/sample/gpt_sample.py --model_name_or_path gpt2-medium-en --decoding_lib ./build/lib/libdecoding_op.so --batch_size 1 --topk 4 --topp 0.0 --max_out_len 32 --start_token "<|endoftext|>" --end_token "<|endoftext|>" --temperature 1.0
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
* `-DDEMO` 说明预测库使用 demo 的位置。比如指定 -DDEMO=./demo/transformer_e2e.cc 或是 -DDEMO=./demo/gpt.cc。最好使用绝对路径，若使用相对路径，需要是相对于 `PaddleNLP/paddlenlp/ops/faster_transformer/src/` 的相对路径。
* `-DWITH_GPT`，如果是编译 GPT 的预测库可执行文件，需要加上 `-DWITH_GPT=ON`。
* **当使用预测库的自定义 op 的时候，请务必开启 `-DON_INFER=ON` 选项，否则，不会得到预测库的可执行文件。**

#### 执行 Transformer decoding on PaddlePaddle

编译完成后，在 `build/bin/` 路径下将会看到 `transformer_e2e` 的一个可执行文件。通过设置对应的设置参数完成执行的过程。

``` sh
cd bin/
./transformer_e2e -batch_size <batch_size> -beam_size <beam_size> -gpu_id <gpu_id> -model_dir <model_directory> -vocab_dir <dict_directory> -data_dir <input_data>
```

举例说明：

``` sh
cd bin/
../third-party/build/bin/decoding_gemm 8 5 8 64 38512 256 512 0
./transformer_e2e -batch_size 8 -beam_size 5 -gpu_id 0 -model_dir ./infer_model/ -vocab_dir DATA_HOME/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708 -data_dir DATA_HOME/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en
```

其中：
* `decoding_gemm` 不同参数的意义可以参考 [FasterTransformer 文档](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#execute-the-decoderdecoding-demos)。
* `DATA_HOME` 则是 `paddlenlp.utils.env.DATA_HOME` 返回的路径。

预测所需要的模型文件，可以通过 `PaddleNLP/examples/machine_translation/transformer/faster_transformer/README.md` 文档中所记述的方式导出。

#### 执行 GPT decoding on PaddlePaddle

如果需要使用 Paddle Inference 预测库针对 GPT 进行预测，首先，需要导出预测模型，可以通过 `./faster_transformer/sample/gpt_export_model_sample.py` 脚本获取预测库用模型，执行方式如下所示：

``` sh
python ./faster_transformer/sample/gpt_export_model_sample.py --model_name_or_path gpt2-medium-en --decoding_lib ./build/lib/libdecoding_op.so --topk 4 --topp 0.0 --max_out_len 32 --start_token "<|endoftext|>" --end_token "<|endoftext|>" --temperature 1.0 --inference_model_dir ./infer_model/
```

各个选项的意义与上文的 `gpt_sample.py` 的选项相同。额外新增一个 `--inference_model_dir` 选项用于指定保存的模型文件、词表等文件。若是使用的模型是 gpt2-medium-en，保存之后，`./infer_model/` 目录下组织的结构如下：

``` text
.
├── gpt.pdiparams   # 保存的参数文件
├── gpt.pdmodel     # 保存的模型文件
├── merges.txt      # bpe
└── vocab.txt       # 词表
```

同理，完成编译后，可以在 `build/bin/` 路径下将会看到 `gpt` 的一个可执行文件。通过设置对应的设置参数完成执行的过程。

``` sh
cd bin/
./gpt -batch_size 1 -gpu_id 0 -model_dir path/to/model -vocab_dir path/to/vocab -start_token "<|endoftext|>" -end_token "<|endoftext|>"
```
