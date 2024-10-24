# PaddleNLP 模型压缩 API

 **目录**
   * [模型压缩 API 功能简介](#模型压缩 API 功简介)
   * [三大场景快速启动模型压缩示例](#三大场景快速启动模型压缩示例)
   * [四步启动模型压缩](#四步启动模型压缩)
       * [Step1：获取模型压缩参数 compression_args](#获取模型压缩参数 compression_args)
       * [Step2：实例化 Trainer 并调用 compress()](#实例化 Trainer 并调用 compress())
           * [Trainer 实例化参数介绍](#Trainer 实例化参数介绍)
       * [Step3：实现自定义评估函数（按需可选）](#实现自定义评估函数（按需可选）)
       * [Step4：传参并运行压缩脚本](#传参并运行压缩脚本)
           * [CompressionArguments 参数介绍](#CompressionArguments 参数介绍)
   * [模型评估与部署](#模型评估与部署)
   * [FAQ](#FAQ)
   * [参考文献](#References)


<a name="模型压缩 API 功能简介"></a>

## 模型压缩 API 功能简介

PaddleNLP 模型压缩 API 功能支持对 ERNIE 类下游任务上微调后的模型进行裁剪、量化，以缩小模型体积、减少内存占用、减少计算、提升推理速度从而减少部署难度。模型压缩 API 效果好，且简洁易用。目前裁剪功能现在支持 DynaBERT 中的宽度自适应裁剪策略；量化现在支持静态离线量化方法（PTQ）、量化训练（QAT）和 Embedding 量化。PTQ 无需训练，只需少量校准数据，即可导出量化模型，QAT 类似 FP32 模型的训练过程，也基本能够做到精度无损，Embedding 量化过程较为简单，不需要训练也不需要校准数据即可完成。

- **效果好**：目前已经在分类（包含文本分类、文本匹配、自然语言推理、代词消歧、阅读理解等任务）、序列标注、抽取式阅读理解任务上进行过验证，基本达到精度无损。例如，对于 12L768H 和 6L768H 结构的模型，进行宽度保留比例为 2/3 的裁剪基本可以达到精度无损，模型裁剪后推理速度能够达到原先的 1-2 倍；6L768H 结构的模型量化后推理速度能够达到量化前的 2-3 倍。

- **简洁易用**：只需要简单几步即可开展模型压缩任务

##### ERNIE 3.0 压缩效果
如下表所示，ERNIE 3.0-Medium (6-layer, 384-hidden, 12-heads) 模型在三类任务（文本分类、序列标注、抽取式阅读理解）经过裁剪 + 量化后加速比均达到 3 倍左右，所有任务上平均精度损失可控制在 0.5 以内（0.46）。

|                            | TNEWS 性能    | TNEWS 精度   | MSRA_NER 性能 | MSRA_NER 精度 | CMRC2018 性能 | CMRC2018 精度 |
|----------------------------|---------------|--------------|---------------|---------------|---------------|---------------|
| ERNIE 3.0-Medium+FP32      | 1123.85(1.0x) | 57.45        | 366.75(1.0x)  | 93.04         | 146.84(1.0x)  | 66.95         |
| ERNIE 3.0-Medium+INT8      | 3226.26(2.9x) | 56.99(-0.46) | 889.33(2.4x)  | 92.70(-0.34)  | 348.84(2.4x)  | 66.32(-0.63)  |
| ERNIE 3.0-Medium+裁剪+FP32 | 1424.01(1.3x) | 57.31(-0.14) | 454.27(1.2x)  | 93.27(+0.23)  | 183.77(1.3x)  | 65.92(-1.03)  |
| ERNIE 3.0-Medium+裁剪+INT8 | 3635.48(3.2x) | 57.26(-0.19) | 1105.26(3.0x) | 93.20(+0.16)  | 444.27(3.0x)  | 66.17(-0.78)  |

(以上数据来自 [ERNIE 3.0 性能测试文档](../slm/model_zoo/ernie-3.0/README.md#性能测试)，文档包含测试环境介绍)

##### UIE 压缩效果

以报销工单信息抽取任务为例，使用 `uie-base` 进行微调，先得到原始 FP32 模型，然后使用 QAT 策略进一步量化。量化后的模型比原始 FP32 模型的 F1 值高 2.19。

| Models                  |  F1   |
|-------------------------|:-----:|
| uie-base+微调+FP32      | 91.93 |
| uie-base+微调+量化+INT8 | 94.12 |


<a name="三大场景快速启动模型压缩示例"></a>

### 三大场景快速启动模型压缩示例

本项目提供了压缩 API 在分类（包含文本分类、文本匹配、自然语言推理、代词消歧等任务）、序列标注、抽取式阅读理解三大场景下的使用样例，可以分别参考 [ERNIE 3.0](../slm/model_zoo/ernie-3.0) 目录下的 [compress_seq_cls.py](../slm/model_zoo/ernie-3.0/compress_seq_cls.py) 、[compress_token_cls.py](../slm/model_zoo/ernie-3.0/compress_token_cls.py)、[compress_qa.py](../slm/model_zoo/ernie-3.0/compress_qa.py) 脚本，启动方式如下：

```shell
# 分类任务
# 该脚本共支持 CLUE 中 7 个分类任务，超参不全相同，因此分类任务中的超参配置利用 config.yml 配置
python compress_seq_cls.py \
    --dataset "clue tnews"  \
    --model_name_or_path best_models/TNEWS  \
    --output_dir ./

# 序列标注任务
python compress_token_cls.py \
    --dataset "msra_ner"  \
    --model_name_or_path best_models/MSRA_NER \
    --output_dir ./ \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 0.00005 \
    --remove_unused_columns False \
    --num_train_epochs 3

# 阅读理解任务
python compress_qa.py \
    --dataset "clue cmrc2018" \
    --model_name_or_path best_models/CMRC2018  \
    --output_dir ./ \
    --max_seq_length 512 \
    --learning_rate 0.00003 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --max_answer_length 50 \

```

示例代码中压缩使用的是 datasets 内置的数据集，若想要使用自定义数据集压缩，可参考 [datasets 加载自定义数据集文档](https://huggingface.co/docs/datasets/loading)。


<a name="四步启动模型压缩"></a>

## 四步启动模型压缩

### 环境依赖

- paddlepaddle-gpu >=2.4.1
- paddlenlp >= 2.5
- paddleslim >= 2.4.0

模型压缩 API 中的压缩功能依赖最新的 `paddleslim` 包。可运行以下命令安装：

```shell
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

模型压缩 API 的使用大致分为四步：

- Step 1: 使用 `PdArgumentParser` 解析从命令行传入的超参数，以获取压缩参数 `compression_args`；
- Step 2: 实例化 Trainer 并调用 `compress()` 压缩 API
- Step 3: 实现自定义评估函数和 loss 计算函数（按需可选），以适配自定义压缩任务
- Step 4：传参并运行压缩脚本

**示例代码**

```python
from paddlenlp.trainer import PdArgumentParser, CompressionArguments

# Step1: 使用 `PdArgumentParser` 解析从命令行传入的超参数，以获取压缩参数 `compression_args`；
parser = PdArgumentParser(CompressionArguments)
compression_args = parser.parse_args_into_dataclasses()

# Step2: 实例化 Trainer 并调用 compress()
trainer = Trainer(
    model=model,
    args=compression_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    criterion=criterion)

# Step 3: 使用内置模型和评估方法，则不需要实现自定义评估函数和 loss 计算函数
trainer.compress()
```

```shell
# Step4: 传参并运行压缩脚本
python compress.py \
    --output_dir ./compress_models  \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 4 \
    --width_mult_list 0.75 \
    --batch_size_list 4 8 16 \
    --batch_num_list 1 \

```


<a name="获取模型压缩参数 compression_args"></a>

### Step 1：获取模型压缩参数 compression_args

使用 `PdArgumentParser` 对象解析从命令行得到的超参数，从而得到 `compression_args`，并将 `compression_args` 传给 `Trainer` 对象。获取 `compression_args` 的方法通常如下：

```python
from paddlenlp.trainer import PdArgumentParser, CompressionArguments

# Step1: 使用 `PdArgumentParser` 解析从命令行传入的超参数，以获取压缩参数 `compression_args`；
parser = PdArgumentParser(CompressionArguments)
compression_args = parser.parse_args_into_dataclasses()
```

<a name="实例化 Trainer 并调用 compress()"></a>

### Step 2：实例化 Trainer 并调用 compress

<a name="Trainer 实例化参数介绍"></a>

#### Trainer 实例化参数介绍

- **--model** 待压缩的模型，目前支持 ERNIE、BERT、RoBERTa、ERNIE-M、ELECTRA、ERNIE-Gram、PP-MiniLM、TinyBERT 等结构相似的模型，是在下游任务中微调后的模型，当预训练模型选择 ERNIE 时，需要继承 `ErniePretrainedModel`。以分类任务为例，可通过`AutoModelForSequenceClassification.from_pretrained(model_name_or_path)` 等方式来获取，这种情况下，`model_name_or_path`目录下需要有 model_config.json, model_state.pdparams 文件；
- **--data_collator** 三类任务均可使用 PaddleNLP 预定义好的 [DataCollator 类](../paddlenlp/data/data_collator.py)，`data_collator` 可对数据进行 `Pad` 等操作。使用方法参考 [示例代码](../slm/model_zoo/ernie-3.0/compress_seq_cls.py) 即可；
- **--train_dataset** 裁剪训练需要使用的训练集，是任务相关的数据。自定义数据集的加载可参考 [文档](https://huggingface.co/docs/datasets/loading)。不启动裁剪时，可以为 None；
- **--eval_dataset** 裁剪训练使用的评估集，也是量化使用的校准数据，是任务相关的数据。自定义数据集的加载可参考 [文档](https://huggingface.co/docs/datasets/loading)。是 Trainer 的必选参数；
- **--tokenizer** 模型 `model` 对应的 `tokenizer`，可使用 `AutoTokenizer.from_pretrained(model_name_or_path)` 来获取。
- **--criterion** 模型的 loss 计算方法，可以是一个 nn.Layer 对象，也可以是一个函数，用于在 ofa_utils.py 计算模型的 loss 用于计算梯度从而确定神经元重要程度。

其中，`criterion` 函数定义示例：

```python
# 支持的形式一：
def criterion(logits, labels):
    loss_fct = paddle.nn.BCELoss()
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = paddle.cast(start_ids, 'float32')
    end_ids = paddle.cast(end_ids, 'float32')
    loss_start = loss_fct(start_prob, start_ids)
    loss_end = loss_fct(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    return loss

# 支持的形式二：
class CrossEntropyLossForSQuAD(paddle.nn.Layer):

    def __init__(self):
        super(CrossEntropyLossForSQuAD, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits = y
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.cross_entropy(input=start_logits,
                                                        label=start_position)
        end_loss = paddle.nn.functional.cross_entropy(input=end_logits,
                                                      label=end_position)
        loss = (start_loss + end_loss) / 2
        return loss
```

用以上参数实例化 Trainer 对象，之后直接调用 `compress()` 。`compress()` 会根据选择的策略进入不同的分支，以进行裁剪或者量化的过程。

**示例代码**

```python
from paddlenlp.trainer import PdArgumentParser, CompressionArguments

# Step1: 使用 `PdArgumentParser` 解析从命令行传入的超参数，以获取压缩参数 `compression_args`；
parser = PdArgumentParser(CompressionArguments)
compression_args = parser.parse_args_into_dataclasses()

# Step2: 实例化 Trainer 并调用 compress()
trainer = Trainer(
    model=model,
    args=compression_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    criterion=criterion)

trainer.compress()
```

<a name="实现自定义评估函数(按需可选）"></a>

### Step3：实现自定义评估函数，以适配自定义压缩任务

当使用 DynaBERT 裁剪功能时，如果模型、Metrics 不符合下表的情况，那么模型压缩 API 中评估函数需要自定义。

目前 DynaBERT 裁剪功能只支持 SequenceClassification 等三类 PaddleNLP 内置 class，并且内置评估器对应为 Accuracy、F1、Squad。

| Model class name | SequenceClassification | TokenClassification | QuestionAnswering |
|------------------|------------------------|---------------------|-------------------|
| Metrics          | Accuracy               | F1                  | Squad             |

需要注意以下三个条件：

- 如果模型是自定义模型，需要继承 `XXXPretrainedModel`，例如当预训练模型选择 ERNIE 时，继承 `ErniePretrainedModel`，模型需要支持调用 `from_pretrained()` 导入模型，且只含 `pretrained_model_name_or_path` 一个必选参数，`forward` 函数返回 `logits` 或者 `tuple of logits`；

- 如果模型是自定义模型，或者数据集比较特殊，压缩 API 中 loss 的计算不符合使用要求，需要自定义 `custom_evaluate` 评估函数，需要同时支持 `paddleslim.nas.ofa.OFA` 模型和 `paddle.nn.layer` 模型。可参考下方示例代码。
    - 输入`model` 和 `dataloader`，返回模型的评价指标（单个 float 值）。
    - 将该函数传入 `compress()` 中的 `custom_evaluate` 参数；

`custom_evaluate()` 函数定义示例：

```python
    import paddle
    from paddle.metric import Accuracy

    @paddle.no_grad()
    def evaluate_seq_cls(self, model, data_loader):
        metric = Accuracy()
        model.eval()
        metric.reset()
        for batch in data_loader:
            logits = model(input_ids=batch['input_ids'],
                           token_type_ids=batch['token_type_ids'])
            # Supports paddleslim.nas.ofa.OFA model and nn.layer model.
            if isinstance(model, paddleslim.nas.ofa.OFA):
                logits = logits[0]
            correct = metric.compute(logits, batch['labels'])
            metric.update(correct)
        res = metric.accumulate()
        logger.info("acc: %s, " % res)
        model.train()
        return res
```


在调用 `compress()` 时传入这个自定义函数：

```python
trainer.compress(custom_evaluate=evaluate_seq_cls)
```


<a name="传参并运行压缩脚本"></a>

### Step 4：传参并运行压缩脚本

这一步主要是将压缩需要用到的参数通过命令行传入，并启动压缩脚本。

压缩启动命令：

**示例代码**

```shell
# Step4: 运行压缩脚本
python compress.py \
    --output_dir ./compress_models  \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 4 \
    --width_mult_list 0.75 \
    --batch_size_list 4 8 16 \
    --batch_num_list 1 \

```

下面会介绍模型压缩启动命令可以传递的超参数。

<a name="CompressionArguments 参数介绍"></a>

#### CompressionArguments 参数介绍

`CompressionArguments` 中的参数一部分是模型压缩功能特定参数，另一部分继承自 `TrainingArguments`，是压缩训练时需要设置的超参数。下面会进行具体介绍，

**公共参数**

公共参数中的参数和具体的压缩策略无关。

- **--strategy** 模型压缩策略，目前支持 `'dynabert+qat+embeddings'`、`'dynabert+qat'`、`'dynabert+embeddings'`、`'dynabert+ptq'`、 `'dynabert'` 、 `'ptq'` 和 `'qat'`。
其中 `'dynabert'` 代表基于 DynaBERT 的宽度裁剪策略，`'qat'` 表示量化训练，`'ptq'` 表示静态离线量化，`'embeddings'` 表示词表量化，并且 `--strategy` 支持选择它们之间所有合理的策略组合。默认是 `'dynabert+ptq'`；

- **--output_dir** 模型压缩后模型保存目录；

- **--input_infer_model_path** 待压缩的静态图模型，该参数是为了支持对静态图模型的压缩。不需使用时可忽略。默认为 `None`；

- **--input_dtype** 导出模型的输入类型，一般是 `int64` 或者是 `int32`。默认为 `int64`；

**DynaBERT 裁剪参数**

当用户使用了 DynaBERT 裁剪、PTQ 量化策略（即策略中包含 'dynabert'、'qat' 时需要传入以下可选参数：

- **--width_mult_list** 裁剪宽度保留的搜索列表，对 6 层模型推荐 `3/4` ，对 12 层模型推荐 `2/3`，表示对 `q`、`k`、`v` 以及 `ffn` 权重宽度的保留比例，假设 12 层模型原先有 12 个 attention heads，裁剪后只剩 9 个 attention heads。默认是 `[3/4]`；

- **--per_device_train_batch_size**  用于裁剪训练的每个 GPU/CPU 核心 的 batch 大小。默认是 8；

- **--per_device_eval_batch_size** 用于裁剪评估的每个 GPU/CPU 核心 的 batch 大小。默认是 8；

- **--num_train_epochs** 裁剪训练所需要的 epochs 数。默认是 3.0；

- **--max_steps** 如果设置为正数，则表示要执行的训练步骤总数。覆盖 `num_train_epochs`。默认为 -1；

- **--logging_steps** 两个日志之间的更新步骤数。默认为 500；

- **--save_steps** 评估模型的步数。默认为 100；

- **--optim** 裁剪训练使用的优化器名称，默认为 adamw，默认为 'adamw'；

- **--learning_rate** 裁剪训练使用优化器的初始学习率，默认为 5e-05；

- **--weight_decay** 除了所有 bias 和 LayerNorm 权重之外，应用于所有层裁剪训练时的权重衰减数值。 默认为 0.0；

- **--adam_beta1** 裁剪训练使用 AdamW 的优化器时的 beta1 超参数。默认为 0.9；

- **--adam_beta2** 裁剪训练使用 AdamW 优化器时的 beta2 超参数。默认为 0.999；

- **--adam_epsilon** 裁剪训练使用 AdamW 优化器时的 epsilon 超参数。默认为 1e-8；

- **--max_grad_norm** 最大梯度范数（用于梯度裁剪）。默认为 1.0；

- **--lr_scheduler_type** 要使用的学习率调度策略。默认为 'linear'；

- **--warmup_ratio** 用于从 0 到 `learning_rate` 的线性 warmup 的总训练步骤的比例。 默认为 0.0；

- **--warmup_steps** 用于从 0 到 `learning_rate` 的线性 warmup 的步数。覆盖 warmup_ratio 参数。默认是 0；

- **--seed** 设置的随机种子。为确保多次运行的可复现性。默认为 42；

- **--device** 运行的设备名称。支持 cpu/gpu。默认为 'gpu'；

- **--remove_unused_columns** 是否去除 Dataset 中不用的字段数据。默认是 True；

**量化公共参数**


**PTQ 量化参数**

当用户使用了 PTQ 量化策略时需要传入以下可选参数：

- **--algo_list** 量化策略搜索列表，目前支持 `'KL'`、`'abs_max'`、`'min_max'`、`'avg'`、`'hist'`、`'mse'` 和 `'emd'`，不同的策略计算量化比例因子的方法不同。建议传入多种策略，可批量得到由多种策略产出的多个量化模型，可从中选择效果最优模型。ERNIE 类模型较推荐 `'hist'`, `'mse'`, `'KL'`，`'emd'` 等策略。默认是 ['mse', 'KL']；

- **--batch_num_list** batch_nums 的超参搜索列表，batch_nums 表示采样需要的 batch 数。校准数据的总量是 batch_size * batch_nums。如 batch_num 为 None，则 data loader 提供的所有数据均会被作为校准数据。默认是 [1]；

- **--batch_size_list** 校准样本的 batch_size 搜索列表。并非越大越好，也是一个超参数，建议传入多种校准样本数，最后可从多个量化模型中选择最优模型。默认是 `[4]`；

- **--weight_quantize_type** 权重的量化类型，支持 `'abs_max'` 和 `'channel_wise_abs_max'` 两种方式。通常使用 'channel_wise_abs_max'， 这种方法得到的模型通常精度更高；

- **activation_quantize_type** 激活 tensor 的量化类型。支持 'abs_max', 'range_abs_max' 和 'moving_average_abs_max'。在 'ptq' 策略中，默认是 'range_abs_max'；

- **--round_type** 权重值从 FP32 到 INT8 的转化方法，目前支持 `'round'` 和 '[adaround](https://arxiv.org/abs/2004.10568.)'，默认是 `'round'`；

- **--bias_correction** 如果是 True，表示使用 [bias correction](https://arxiv.org/abs/1810.05723) 功能，默认为 False。

**QAT 量化参数**

当用户使用了 QAT 量化策略时，除了可以设置上面训练相关的参数，还可以传入以下可选参数：

- **--weight_quantize_type** 权重的量化类型，支持 `'abs_max'` 和 `'channel_wise_abs_max'` 两种方式。通常使用 'channel_wise_abs_max'， 这种方法得到的模型通常精度更高；

- **activation_quantize_type** 激活 tensor 的量化类型。支持 'abs_max', 'range_abs_max' 和 'moving_average_abs_max'。在'qat'策略中，它默认是 'moving_average_abs_max'；

- **use_pact** 是否使用 PACT 量化策略，是对普通方法的改进，参考论文[PACT: Parameterized Clipping Activation for Quantized Neural Networks](https://arxiv.org/abs/1805.06085)，打开后精度更高，默认是 True。

- **moving_rate** 'moving_average_abs_max' 量化方法中的衰减系数，默认为 0.9；

<a name="模型评估与部署"></a>

## 模型评估与部署

裁剪、量化后的模型不能再通过 `from_pretrained` 导入进行预测，而是需要使用 Paddle 部署工具才能完成预测。

压缩后的模型部署可以参考 [部署文档](../slm/model_zoo/ernie-3.0/deploy) 完成。

### Python 部署

服务端部署可以从这里开始。可以参考 [seq_cls_infer.py](../slm/model_zoo/ernie-3.0/deploy/python/seq_cls_infer.py) 或者 [token_cls_infer.py](../slm/model_zoo/ernie-3.0/deploy/python/token_cls_infer.py) 来编写自己的预测脚本。并根据 [Python 部署指南](../slm/model_zoo/ernie-3.0/deploy/python/README.md) 的介绍安装预测环境，对压缩后的模型进行精度评估、性能测试以及部署。


<a name="服务化部署"></a>

### 服务化部署

- [FastDeploy ERNIE 3.0 模型 Serving 部署示例](../slm/model_zoo/ernie-3.0/deploy/serving/README.md)
- [基于 PaddleNLP SimpleServing 的服务化部署](../slm/model_zoo/ernie-3.0/deploy/simple_serving/README.md)

### 移动端部署


<a name="FAQ"></a>

## FAQ

**Q：模型压缩需要数据吗？**

A：DynaBERT 裁剪和量化训练 QAT 需要使用训练集进行训练，验证集进行评估，其过程类似微调；静态离线量化 PTQ 只需要验证集（对样本量要求较低，一般 4-16 个样本就可能可以满足要求）；

**Q：示例代码里是内置的数据集，如何使用我自己的数据呢**

A：可以参考 UIE 的例子，也可以参考 [datasets 加载自定义数据集文档](https://huggingface.co/docs/datasets/loading)；

**Q：模型压缩后的模型还能继续训练吗？**

A：模型压缩主要用于推理加速，因此压缩后的模型都是静态图（预测）模型，不能再通过 `from_pretrained()` API 导入继续训练；

**Q：裁剪和量化怎么选？**

A：可以设置参数 `--strategy` 来选择压缩的策略，默认是裁剪和量化同时选择，先裁剪后量化。目前裁剪策略有训练过程，需要下游任务的训练数据，其训练时间视下游任务数据量而定，且和微调的训练时间是一个量级。静态离线量化则不需要额外的训练，更快，通常来说量化的加速比比裁剪更明显。建议裁剪和量化同时选择，有些情况下可能比单独量化效果更好；

**Q：裁剪中也有训练过程吗？**

A：DynaBERT 裁剪类似蒸馏过程，也会有模型训练时用到的超参，方便起见，可以直接使用微调时所用的最佳的超参。如果想进一步提升精度，可以对 `batch_size`、`learning_rate`、`epoch` 等超参数进行 Grid Search；

**Q：使用 `TensorDataset` 对象做量化报错了，为什么？**

A：使用量化时，`eval_dataset` 不可以是 `TensorDataset` 对象，因为量化功能内部在静态图模式下执行，而 `TensorDataset` 只能在动态图下使用，两者同时使用会导致错误；

<a name="References"></a>

## 参考文献
- Hou L, Huang Z, Shang L, Jiang X, Chen X and Liu Q. DynaBERT: Dynamic BERT with Adaptive Width and Depth[J]. arXiv preprint arXiv:2004.04037, 2020.

- Cai H, Gan C, Wang T, Zhang Z, and Han S. Once for all: Train one network and specialize it for efficient deployment[J]. arXiv preprint arXiv:1908.09791, 2020.

- Wu H, Judd P, Zhang X, Isaev M and Micikevicius P. Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation[J]. arXiv preprint arXiv:2004.09602v1, 2020.
