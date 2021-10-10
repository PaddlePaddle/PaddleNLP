# Faster GPT 使用

在这里我们集成了 NVIDIA [Faster Transformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1) 用于预测加速。同时集成了 FasterGPT float32 以及 float16 预测。以下是使用 FasterGPT 的使用说明。

## 使用环境说明

* 本项目依赖于 PaddlePaddle 2.1.0 及以上版本或适当的 develop 版本
* CMake >= 3.10
* CUDA 10.1 或 10.2（需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* [Faster Transformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#setup) 使用必要的环境

## 快速开始

我们实现了基于 Faster Transformer 的 FasterGPT 的自定义 op 的接入，用于 GPT-2 模型在 GPU 上的预测性能。接下来，我们将介绍基于 Python 动态图使用 FasterGPT 自定义 op 的方式。

## Python 动态图使用自定义 op

### 使用 GPT-2 decoding 高性能推理

编写 python 脚本的时候，调用 `FasterGPT()` API 即可实现 GPT-2 模型的高性能预测。

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
    topk=args.topk,
    topp=args.topp,
    max_out_len=args.max_out_len,
    bos_id=bos_id,
    eos_id=eos_id,
    temperature=args.temperature,
    use_fp16_decoding=args.use_fp16_decoding)
```

目前，GPT-2 的高性能预测接口 `FasterGPT()` 要求 batch 内输入的样本的长度都是相同的。并且，仅支持 topk-sampling 和 topp-sampling，不支持 beam-search。

若当前环境下没有需要的自定义 op 的动态库，将会使用 JIT 自动编译需要的动态库。如果需要自行编译自定义 op 所需的动态库，可以参考 [文本生成高性能加速](../../../paddlenlp/ops/README.md)。编译好后，使用 `FasterGPT(decoding_lib="/path/to/lib", ...)` 可以完成导入。

更详细的例子可以参考 `./infer.py`，我们提供了更详细用例。

#### GPT-2 decoding 示例代码

使用 PaddlePaddle 仅执行 decoding 测试（float32）：

``` sh
export CUDA_VISIBLE_DEVICES=0
python infer.py --model_name_or_path gpt2-medium-en --decoding_lib ./build/lib/libdecoding_op.so --batch_size 1 --topk 4 --topp 0.0 --max_out_len 32 --start_token "<|endoftext|>" --end_token "<|endoftext|>" --temperature 1.0
```

其中，各个选项的意义如下：
* `--model_name_or_path`: 预训练模型的名称或是路径。
* `--decoding_lib`: 指向 `libdecoding_op.so` 的路径。需要包含 `libdecoding_op.so`。若不存在则将自动进行 jit 编译产出该 lib。
* `--batch_size`: 一个 batch 内，样本数目的大小。
* `--topk`: 执行 topk-sampling 的时候的 `k` 的大小，默认是 4。
* `--topp`: 执行 topp-sampling 的时候的阈值的大小，默认是 0.0 表示不执行 topp-sampling。
* `--max_out_len`: 最长的生成长度。
* `--start_token`: 字符串，表示任意生成的时候的开始 token。
* `--end_token`: 字符串，生成的结束 token。
* `--temperature`: temperature 的设定。
* `--use_fp16_decoding`: 是否使用 fp16 进行推理。

#### C++ 预测库使用高性能加速

C++ 预测库使用 FasterGPT 的高性能加速需要自行编译，可以参考 [文本生成高性能加速](../../../paddlenlp/ops/README.md) 文档完成基于 C++ 预测库的编译。

具体的使用 demo 可以参考 [GPT-2 C++ 预测库使用 demo](../../../paddlenlp/ops/faster_transformer/src/demo/gpt.cc)。
