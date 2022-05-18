# FasterGeneration

FasterGeneration是PaddleNLP v2.2版本加入的文本生成高性能加速功能，其支持GPT、BART、UnifiedTransformer等多种NLP生成类预训练模型，并且支持多种解码策略，可以用于机器翻译、文本续写、文本摘要、对话生成等多种NLG任务的GPU场景预测加速。

功能底层依托于[NV FasterTransformer](https://github.com/NVIDIA/FasterTransformer)，该库针对标准的Transformer和GPT模型、beam search和sampling解码策略进行了性能优化。PaddleNLP FasterGeneration在其之上进行了扩展，实现了更多模型和生成策略的优化支持，并将功能入口封装于`model.generate`函数。功能的开启和关闭通过传入`use_faster`参数进行控制（默认为关闭状态）。通过调用generate函数，用户可以简单的使用模型高性能推理功能。下图展示了FasterGeneration的启动流程：


<p align="center">
  <img src="../docs/imgs/faster_generation.png" width="400" height ="600" />
</p>

## Featrues

- 全面支持生成式预训练模型。包括GPT、BART、mBART、UnifiedTransformer和UNIMO-text。
- 支持大多数主流解码策略。包括Beam Search、Sampling、Greedy Search。以及Diverse Sibling Search、Length Penalty等子策略。
- 解码速度快。最高可达非加速版generate函数的**18倍**。**并支持FP16混合精度计算**。
- 易用性强。功能的入口为`model.generate`，与非加速版生成api的使用方法相同，当满足加速条件时使用jit即时编译高性能算子并用于生成，不满足则自动切换回非加速版生成api。
- GPT、UnifiedTransformer和UNIMO-text模型支持高性能并行推理，在具备MPI和NCCL的环境中一行代码即可开启使用，允许通过多张小显存容量的 GPU 使用百亿大模型，预测速度较单卡也进一步提升。百亿模型四卡并行高性能推理速度达单卡高性能推理速度2+倍。

### Inference Model Support
下表为PaddleNLP FasterGeneration对预训练模型和解码策略的支持情况（GPU）。

| Model Name | GPT2 |BART |mBART | UnifiedTransformer |
|:---:| :----:| :-----:|:-----:| :------------------: |
| Model Structure| Decoder |Encoder-Decoder |  Encoder-Decoder | Prefix-LM  |
|Beam Search           | ❌  | ✅  | ✅  | ✅  |
|Top-K Sampling        | ✅  | ✅  | ✅  | ✅  |
|Top-P Sampling        | ✅  | ✅  | ✅  | ✅  |
|Diverse Sibling Search| ❌  | ✅  | ✅  | ✅  |
|Forced Decoding       | ❌  | ❌  | ✅  | ❌  |
|Length Penalty        | ❌  | ✅  | ✅  | ✅  |
|Temperature           | ✅  | ✅  | ✅  | ✅  |
|Repetition Penalty    | ✅  | ❌  | ❌  | ❌  |


## Performence

FasterGeneration的高性能解码相比原版generate方法加速明显，并且与竞品相比有也有极大的速度优势。以下为性能对比图：

- **batch_size = 4, out_seq_len = 32**
- Device: Tesla V100-SXM2-16GB
- CUDA version 11.2
- cudnn version 8
- torch version 1.10.0+cu113
- transformers version 4.12.5

**BART** (bart-base, batch_size=4, max_length=32)

<p align="left">
  <img src="../docs/imgs/bart_perf.png" width="800" height ="400" />
</p>

**GPT** (gpt2, batch_size=4, max_length=32)

<p align="left">
  <img src="../docs/imgs/gpt_perf.png" width="800" height ="400" />
</p>

更详细的性能数据请参见[这里](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/faster_generation/perf)

## Quick Start

### 高性能推理

为体现FasterGeneration的易用性，我们在`samples`文件夹中内置了几个典型任务示例，下面以基于GPT模型的中文文本续写任务为例：

```sh
python samples/gpt_sample.py
```

如果是第一次执行，PaddleNLP会启动即时编译（[JIT Compile](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/new_op/new_custom_op_cn.html#jit-compile)）自动编译高性能解码算子。

```sh
...
2021-11-17 13:42:56,771 - INFO - execute command: cd /10.2/hub/PaddleNLP/paddlenlp/ops/extenstions && /usr/local/bin/python FasterTransformer_setup.py build
INFO:utils.cpp_extension:execute command: cd /10.2/hub/PaddleNLP/paddlenlp/ops/extenstions && /usr/local/bin/python FasterTransformer_setup.py build
grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
running build
running build_ext
-- The C compiler identification is GNU 8.2.0
-- The CXX compiler identification is GNU 8.2.0
-- The CUDA compiler identification is NVIDIA 10.2.89
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr
...
```

编译过程通常会花费几分钟的时间编译只会进行一次，之后再次使用高性能解码就不需要重新编译了，编译完成后会继续运行，可以看到生成的结果如下：

```
Model input: 花间一壶酒，独酌无相亲。举杯邀明月，
Result: 对影成三人。
```

打开示例代码 `samples/gpt_sample.py` ，我们可以看到如下代码：

```
...
model = GPTLMHeadModel.from_pretrained(model_name)
...
outputs, _ = model.generate(
    input_ids=inputs_ids, max_length=10, decode_strategy='greedy_search',
    use_faster=True)
...
```

可以看到，FasterGeneration的使用方法与 `model.generate()` 相同，只需传入输入tensor和解码相关参数即可，使用非常简便。如果要使用非加速版的 `model.generate()` 方法，只需传入 `use_faster=False` 即可，示例如下：

```
...
outputs, _ = model.generate(
    input_ids=inputs_ids, max_length=10, decode_strategy='greedy_search', use_faster=False)
...
```

**NOTE:** 需要注意的是，如果传入 `model.generate()` 的参数不满足高性能版本的要求。程序会做出提示并自动切换为非加速版本，例如我们在上面的例子中传入 `min_length=1` ，会得到如下提示：

```
...
[2021-11-17 14:21:06,132] [ WARNING] - 'min_length != 0' is not supported yet in the faster version
[2021-11-17 14:21:06,132] [ WARNING] - FasterGeneration is not available, and the original version would be used instead.
...
```

关于该函数的详细介绍可以参考API文档[generate](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.generation_utils.html)和**Aistudio教程[文本生成任务实战：如何使用PaddleNLP实现各种解码策略](https://aistudio.baidu.com/aistudio/projectdetail/3243711?contributionType=1)。**`samples`文件夹中的其他示例的使用方法相同。

### 并行推理

FasterGeneration对GPT、UnifiedTransformer和UNIMO-text模型在高性能推理的基础上还实现了模型并行功能，其中GPT支持Tensor Parallel和Layer Parallel（Pipeline Parallel）两种并行策略的组合，UnifiedTransformer和UNIMO-text支持Tensor Parallel。关于这两种并行策略的详细介绍请参考[Megatron论文](https://arxiv.org/pdf/2104.04473.pdf)。

并行推理当前依赖MPI（[MPICH](https://www.mpich.org)、[OpenMPI](https://www.open-mpi.org)均可）和[NCCL](https://developer.nvidia.com/nccl)，如需使用还请先安装依赖。在使用时，相比上面的单卡高性能加速代码中也只增加了`from_pretrained`创建加载模型之前加上`enable_ft_para()`一行。
#### GPT 并行推理

GPT高性能并行推理的完整使用示例已在`gpt_mp_sample.py`中提供，按照如下方式启动即可：

```sh
mpirun -n 4 python gpt_mp_sample.py --tensor_para_size 4 --layer_para_size 1
```

其中`-n 4`指明使用的进程和GPU数，`tensor_para_size`和`tensor_para_size`分别指明Tensor Parallel和Layer Parallel各自使用的GPU数，均设置为1则进行单卡预测。另外加上`--use_fp16`以使用FP16，加上`--profile`可以进行相应设置的性能测试。其他生成相关的参数设置释义如下：
- `model_name` 指定使用的GPT模型，默认为[`gpt-cpm-larg-cn`](https://github.com/TsinghuaAI/CPM-1-Generate)。
- `max_length` 指定生成的最大长度，默认为50。
- `topk` 用于Top-K采样策略，采样时将只从概率最高K个token中采样，默认为1，即greedy search。
- `topp` 用于Top-P采样策略，采样时将只从概率最高且累加概率不超过该值的token中采样，默认为1.0。
- `temperature` 用于调整预测概率分布，默认为1.0，即保持模型原有的预测概率。

使用`gpt-cpm-larg-cn`(2.6B)和默认设置，在V100上4卡Tensor Parallel较单卡高性能预测速度提升约40%。

#### PLATO-XL 并行推理

PLATO-XL百亿对话预训练模型(11B UnifiedTransformer模型)高性能并行推理的完整使用示例已在`plato_xl_sample.py`中提供(当前只支持Tensor Parallel)，按照如下方式启动即可：

```shell
mpirun -n 4 python plato_xl_sample.py
```

参数释义基本同上。在V100上4卡Tensor Parallel高性能预测为单卡高性能预测速度的2倍。

## Generate Examples

除了以上示例之外，PaddleNLP的examples中大多使用了`model.generate`的示例都可以通过调整到合适的参数使用高性能推理。具体如下：

- [examples/dialogue/unified_transformer](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dialogue/unified_transformer)
- [model_zoo/gpt/faster_gpt](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/gpt/faster_gpt)
- [examples/text_generation/unimo-text](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_generation/unimo-text)
- [examples/text_summarization/bart](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_summarization/bart)

下面我们以基于 `Unified Transformer` 的任务型对话为例展示一下FasterGeneration的加速效果：

打开以上链接中Unified Transformer对应的example，找到README中对应预测的脚本。稍作修改如下：

```sh
export CUDA_VISIBLE_DEVICES=0
    python infer.py \
    --model_name_or_path=unified_transformer-12L-cn-luge \
    --output_path=./predict.txt \
    --logging_steps=10 \
    --seed=2021 \
    --max_seq_len=512 \
    --max_knowledge_len=256 \
    --batch_size=4 \
    --min_dec_len=1 \
    --max_dec_len=64 \
    --num_return_sequences=1 \
    --decode_strategy=sampling \
    --top_k=5 \
    --faster
    --device=gpu
```

由于这里只是展示性能，我们直接在 `model_name_or_path` 填入PaddleNLP预训练模型名称 `unified_transformer-12L-cn-luge` 。

可以看到，由于该任务为对话任务，我们为了防止模型生成过多安全回复（如：哈哈哈、不错等），保证生成结果具有更多的随机性，我们选择TopK-sampling作为解码策略，并让k=5。

打开 `infer.py` ，可以看到我们传入的脚本参数大多都提供给了 `model.generate()` 方法：

```
output = model.generate(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        seq_len=seq_len,
        max_length=args.max_dec_len,
        min_length=args.min_dec_len,
        decode_strategy=args.decode_strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        early_stopping=args.early_stopping,
        num_return_sequences=args.num_return_sequences,
        use_fp16_decoding=args.use_fp16_decoding,
        use_faster=args.faster)
```

运行脚本，输出结果如下：

```sh
step 10 - 1.695s/step
step 20 - 1.432s/step
step 30 - 1.435s/step
```

可以看到，非加速版 `generate()` 方法的预测速度为每个step耗时1.5秒左右。

下面我们在启动脚本中传入 `--faster` 参数，该参数会向 `generate()` 方法传入 `use_faster=True` ，启动加速模式。同时我们需要设置 `--min_dec_len=0` ，因为FasterGeneration当前还不支持该参数。新的脚本启动参数如下：

```sh
export CUDA_VISIBLE_DEVICES=0
    python infer.py \
    --model_name_or_path=unified_transformer-12L-cn-luge \
    --output_path=./predict.txt \
    --logging_steps=10 \
    --seed=2021 \
    --max_seq_len=512 \
    --max_knowledge_len=256 \
    --batch_size=4 \
    --min_dec_len=0 \
    --max_dec_len=64 \
    --num_return_sequences=1 \
    --decode_strategy=sampling \
    --top_k=5 \
    --device=gpu \
    --faster
```

再次运行脚本，输出结果如下（由于我们已经编译过高性能算子，所以这里不会重新编译）：

```sh
[2021-11-23 13:38:09,200] [   DEBUG] - skipping 'FasterTransformer' extension (up-to-date) build
step 10 - 0.250s/step
step 20 - 0.156s/step
step 30 - 0.141s/step
```

可以看到，FasterGeneration的预测速度为每个step耗时0.15秒左右，相比非加速版提速超过9倍。
