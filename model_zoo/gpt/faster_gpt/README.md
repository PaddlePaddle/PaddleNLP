# FasterGenerartion 使用

在这里我们集成了 NVIDIA [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1) 用于预测加速。以下是FasterGenerartion相关的使用说明。

## 使用环境说明

* 本项目依赖于 PaddlePaddle 2.1.0 及以上版本或适当的 develop 版本
* CMake >= 3.10
* CUDA 10.1 或 10.2（需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#setup) 使用必要的环境

## 快速开始

### GPT-2 decoding 示例代码

使用 PaddlePaddle 仅执行 decoding 测试（float32）：

``` sh
export CUDA_VISIBLE_DEVICES=0
python infer.py --model_name_or_path gpt2-medium-en --batch_size 1 --topk 4 --topp 0.0 --max_length 32 --start_token "<|endoftext|>" --end_token "<|endoftext|>" --temperature 1.0
```

其中，各个选项的意义如下：
* `--model_name_or_path`: 预训练模型的名称或是路径。
* `--batch_size`: 一个 batch 内，样本数目的大小。
* `--topk`: 执行 topk-sampling 的时候的 `k` 的大小，默认是 4。
* `--topp`: 执行 topp-sampling 的时候的阈值的大小，默认是 0.0 表示不执行 topp-sampling。
* `--max_length`: 最长的生成长度。
* `--start_token`: 字符串，表示任意生成的时候的开始 token。
* `--end_token`: 字符串，生成的结束 token。
* `--temperature`: temperature 的设定。
* `--use_fp16_decoding`: 是否使用 fp16 进行推理。

### 导出基于 FasterGPT 的预测库使用模型文件

GPT的FasterGenerartion高性能预测功能底层依托于`FasterGPT()`。
编写 python 脚本的时候，调用 `FasterGPT()` API 即可创建可用于导出的高性能预测模型。

示例如下：

``` python
from paddlenlp.ops import FasterGPT
from paddlenlp.transformers import GPTLMHeadModel
from paddlenlp.transformers import GPTTokenizer

MODEL_CLASSES = {
    "gpt2-medium-en": (GPTLMHeadModel, GPTTokenizer),
}

model_class, tokenizer_class = MODEL_CLASSES[args.model_name]
tokenizer = tokenizer_class.from_pretrained(args.model_name)
model = model_class.from_pretrained(args.model_name)

# Define model
gpt = FasterGPT(
    model=model,
    temperature=args.temperature,
    use_fp16_decoding=args.use_fp16_decoding)
```

通过 `export_model.py` 脚本获取预测库用模型，执行方式如下所示：

``` sh
python export_model.py --model_name_or_path gpt2-medium-en --topk 4 --topp 0.0 --max_out_len 32 --temperature 1.0 --inference_model_dir ./infer_model/
```

各个选项的意义与上文的 `infer.py` 的选项相同。额外新增一个 `--inference_model_dir` 选项用于指定保存的模型文件、词表等文件。

若当前环境下没有需要的自定义 op 的动态库，将会使用 JIT 自动编译需要的动态库。如果需要自行编译自定义 op 所需的动态库，可以参考 [文本生成高性能加速](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/ops/README.md)。编译好后，可以在执行 `export_model.py` 时使用 `--decoding_lib ../../../../paddlenlp/ops/build/lib/libdecoding_op.so` 可以完成导入。

注意：如果是自行编译的话，这里的 `libdecoding_op.so` 的动态库是参照文档 [文本生成高性能加速](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/ops/README.md) 中 **`Python 动态图使用自定义 op`** 编译出来的 lib，与相同文档中 **`C++ 预测库使用自定义 op`** 编译产出不同。因此，在使用预测库前，还需要额外导出模型：
  * 一次用于获取 Python 动态图下的 lib，用到 Python 端进行模型导出。
  * 一次获取编译的基于预测库的可执行文件

若是使用的模型是 gpt2-medium-en，保存之后，`./infer_model/` 目录下组织的结构如下：

``` text
.
├── gpt.pdiparams       # 保存的参数文件
├── gpt.pdiparams.info  # 保存的一些变量描述信息，预测不会用到
├── gpt.pdmodel         # 保存的模型文件
├── merges.txt          # bpe
└── vocab.txt           # 词表
```

### C++ 预测库使用高性能加速

C++ 预测库使用 FasterGPT 的高性能加速需要自行编译，可以参考 [文本生成高性能加速](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/ops/README.md) 文档完成基于 C++ 预测库的编译，同时也可以参考相同文档执行对应的 C++ 预测库的 demo 完成预测。

具体的使用 demo 可以参考 [GPT-2 预测库 C++ demo](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/ops/faster_transformer/src/demo/gpt.cc)。
