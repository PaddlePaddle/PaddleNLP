# FasterUNIMOText 预测

在这里我们集成并升级了 NVIDIA [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1) 用于预测加速。以下是使用 FasterUNIMOText 的说明。

**需要注意的是：**这里的说明仅涉及使用 FasterUNIMOText 进行预测，而预测所需的模型，需要经过上一级目录 `../run_gen.py` finetuning 获取。

## 使用环境说明

* 本项目依赖于 PaddlePaddle 2.1.2 或是最新的 develop 版本，可能需要自行编译 PaddlePaddle
* CMake >= 3.10
* CUDA 10.1 或 10.2（需要 PaddlePaddle 框架一致）
* gcc 版本需要与编译 PaddlePaddle 版本一致，比如使用 gcc8.2
* 推荐使用 Python3
* [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/tree/v3.1#setup) 使用必要的环境

## 快速开始

我们实现了基于 FasterTransformer 的 FasterUNIMOText 的自定义 op 的接入，用于加速 UNIMOText 模型在 GPU 上的预测性能。接下来，我们将介绍基于 Python 动态图使用 FasterUNIMOText 自定义 op 的方式。

## 使用 FasterUNIMOText 完成预测

编写 python 脚本的时候，调用 `FasterUNIMOText` API 即可实现 UNIMOText 的高性能预测。

举例如下：

``` python
from paddlenlp.ops import FasterUNIMOText

model = UNIMOLMHeadModel.from_pretrained(args.model_name_or_path)
tokenizer = UNIMOTokenizer.from_pretrained(args.model_name_or_path)

model = FasterUNIMOText(
    model,
    decoding_strategy=args.decode_strategy,
    use_fp16_decoding=args.use_fp16_decoding)
```

若当前环境下没有需要的自定义 op 的动态库，将会使用 JIT 自动编译需要的动态库。如果需要自行编译自定义 op 所需的动态库，可以参考 [文本生成高性能加速](../../../../paddlenlp/ops/README.md)。编译好后，使用 `FasterUNIMOText(decoding_lib="/path/to/lib", ...)` 可以完成导入。

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
