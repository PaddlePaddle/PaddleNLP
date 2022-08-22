# PaddleNLP Compression API

## 一、压缩 API 使用介绍

### 1. 获取压缩功能所需参数

压缩功能需要参数 `CompressionArguments`，主要通过 python 脚本启动命令传入，下方是 `CompressionArguments`中的参数介绍。

#### 1.1 CompressionArguments 参数介绍

CompressionArguments 中的参数一部分是压缩所使用的参数，另一部分继承自 TrainingArguments，用于设置压缩训练时的超参数。

**公共参数**

公共参数中的参数和具体的压缩策略无关。
```python
--strategy
                        压缩策略，目前支持 'dynabert+ptq'、 'dynabert' 和 'ptq'。其中 'dynabert' 代表基于 DynaBERT 的宽度裁剪策略，'ptq' 表示静态离线量化。'dynabert+ptq'，代表先裁剪后量化。三种策略可以都选择，然后选最优结果。默认是 'dynabert+ptq'；

--output_dir
                        压缩后模型保存目录；


--input_infer_model_path
                        待压缩的静态图模型，该参数是为了支持对静态图模型的压缩。不需使用时可忽略。默认为 None；
```

**DynaBERT 裁剪参数**

当用户使用了 DynaBERT 裁剪策略时需要传入以下可选参数：

```python
--width_mult_list
                        裁剪宽度保留的搜索列表，对 6 层模型推荐 `3/4` ，对 12 层模型推荐 `2/3`，表示对 `q`、`k`、`v` 以及 `ffn` 权重宽度的保留比例，假设 12 层模型原先有 12 个 attention heads，裁剪后只剩 9 个 attention heads。默认是 `[3/4]`；

--per_device_train_batch_size
                        用于裁剪训练的每个 GPU/CPU 核心 的 batch 大小。默认是 8；

--per_device_eval_batch_size
                        用于裁剪评估的每个 GPU/CPU 核心 的 batch 大小。默认是 8；

--num_train_epochs
                        裁剪训练所需要的 epochs 数。默认是 3.0；

--max_steps
                        如果设置为正数，则表示要执行的训练步骤总数。
                        覆盖 `num_train_epochs`。默认为 -1；

--logging_steps
                        两个日志之间的更新步骤数。默认为 500；

--save_steps
                        评估模型的步数。默认为 500；

--optim
                        裁剪训练使用的优化器名称，默认为adamw，默认为 'adamw'；

--learning_rate
                        裁剪训练使用优化器的初始学习率，默认为 5e-05；

--weight_decay
                        除了所有 bias 和 LayerNorm 权重之外，应用于所有层裁剪训练时的权重衰减数值。默认为 0.0；

--adam_beta1
                        裁剪训练使用 AdamW 的优化器时的 beta1 超参数。默认为 0.9；

--adam_beta2
                        裁剪训练使用 AdamW 优化器时的 beta2 超参数。默认为 0.999；

--adam_epsilon
                        裁剪训练使用 AdamW 优化器时的 epsilon 超参数。默认为 1e-8；

--max_grad_norm
                        最大梯度范数（用于梯度裁剪）。默认为 1.0；

--lr_scheduler_type
                        要使用的学习率调度策略。默认为 'linear'；

--warmup_ratio
                        用于从 0 到 `learning_rate` 的线性 warmup 的总训练步骤的比例。默认为 0.0；

--warmup_steps
                        用于从 0 到 `learning_rate` 的线性 warmup 的步数。覆盖warmup_ratio 参数。默认是 0；

--seed
                        设置的随机种子。为确保多次运行的可复现性。默认为 42；
--device
                        运行的设备名称。支持 cpu/gpu，默认为 'gpu'；

--remove_unused_columns
                        是否去除 Dataset 中不用的字段数据。默认是 True；

```



**PTQ 量化参数**

当用户使用了 PTQ 量化策略时需要传入以下可选参数：

```python

--algo_list
                        量化策略搜索列表，目前支持 'KL'、'abs_max'、'min_max'、'avg'、'hist'、'mse' 和 'emd'，不同的策略计算量化比例因子的方法不同。建议传入多种策略，可批量得到由多种策略产出的多个量化模型，可从中选择效果最优模型。ERNIE 类模型较推荐 'hist', 'mse', 'KL'，'emd' 等策略。默认是 ['mse', 'KL']；

--batch_num_list
                        batch_nums 的超参搜索列表，batch_nums 表示采样需要的 batch 数。校准数据的总量是 batch_size * batch_nums。如 batch_num 为 None，则 data loader 提供的所有数据均会被作为校准数据。默认是 [1]；

--batch_size_list
                        校准样本的 batch_size 搜索列表。并非越大越好，也是一个超参数，建议传入多种校准样本数，最后可从多个量化模型中选择最优模型。默认是 `[4]`；

--weight_quantize_type
                        权重的量化类型，支持 'abs_max' 和 'channel_wise_abs_max' 两种方式。通常使用 'channel_wise_abs_max'， 这种方法得到的模型通常精度更高；

--round_type
                        权重值从 FP32 到 INT8 的转化方法，目前支持 'round' 和 '[adaround](https://arxiv.org/abs/2004.10568.)'，默认是 'round'；

--bias_correction
                        如果是 True，表示使用[bias correction](https://arxiv.org/abs/1810.05723)功能，默认为 False。
```
#### 1.2 获取 CompressionArguments 对象

```python
from paddlenlp.trainer import PdArgumentParser, CompressionArguments
parser = PdArgumentParser(CompressionArguments)
compression_args = parser.parse_args_into_dataclasses()
```

### 2. 实例化 Trainer

#### 2.1 Trainer 实例化参数介绍

```python
--model
                        待压缩的模型，目前支持 ERNIE 等模型，是在下游任务中微调后的模型。以 seq_cls 任务为例，可通过`AutoModelForSequenceClassification.from_pretrained(model_name_or_path)` 等方式来获取，这种情况下，`model_name_or_path`目录下需要有 model_config.json, model_state.pdparams 文件；
--data_collator
                        三类任务均可使用 PaddleNLP 预定义好的[DataCollator 类](../../paddlenlp/data/data_collator.py)，`data_collator` 可对数据进行 `Pad` 等操作。使用方法参考[示例代码](../model_zoo/ernie-3.0/compress_seq_cls.py)即可；
--train_dataset
                        裁剪训练需要使用的训练集；
--eval_dataset
                        裁剪训练使用的评估集，也是量化使用的校准数据；
--tokenizer
                        模型 `model`对应的 `tokenizer`，可使用 `AutoTokenizer.from_pretrained(model_name_or_path)` 来获取。
```

**示例代码**

```python
trainer = Trainer(
    model=model,
    args=compression_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    criterion=criterion)
```


### 3. 调用 compress()

Trainer 只需直接调用 `compress()` 即可，可以通过传入命令行参数来控制模型压缩的一些超参数：

```python
trainer.compress()
```

### 4. 运行压缩脚本

这一步主要是将压缩需要用到的参数通过命令行传入，并启动压缩脚本。

```shell
python compress.py \
    --dataset   "clue cluewsc2020"   \
    --model_name_or_path best_models/CLUEWSC2020 \
    --output_dir ./compress_models  \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 4
    --width_mult_list 0.75 \
    --batch_size_list 4 8 16 \
    --batch_num_list 1 \

```

### 5. 压缩自定义 ERNIE 类模型（按需可选）

如果使用 DynaBERT 裁剪功能，并且待压缩的模型是自定义模型，即非 PaddleNLP 定义的模型，还需要满足以下三个条件：

- 能够通过 `from_pretrained()` 导入模型，且只含 `pretrained_model_name_or_path` 一个必选参数；

- 实现自定义 `custom_dynabert_evaluate` 评估函数，需要同时支持 `paddleslim.nas.ofa.OFA` 模型和 `paddle.nn.layer` 模型。可参考下方示例代码；
    - 输入`model` 和 `dataloader`，返回模型的评价指标（单个 float 值）。
    - 将该函数传入 `compress()` 中的 `custom_dynabert_evaluate` 参数；
- 实现自定义 `custom_dynabert_calc_loss` 函数。便于反向传播计算梯度，从而计算神经元的重要性以便后续裁剪使用。可参考下方示例代码；
    - 输入每个batch的数据，返回模型的loss。
    - 将该函数传入 `compress()` 中的 `custom_dynabert_calc_loss` 参数；


`custom_dynabert_evaluate()` 函数定义示例：

```python
    import paddle
    from paddle.metric import Accuracy
    from paddleslim.nas.ofa import OFA

    @paddle.no_grad()
    def evaluate_seq_cls(model, data_loader):
        metric = Accuracy()
        model.eval()
        metric.reset()
        for batch in data_loader:
            logits = model(batch['input_ids'],
                           batch['token_type_ids'],
                           attention_mask=[None, None])
            # Supports paddleslim.nas.ofa.OFA model and nn.layer model.
            if isinstance(model, OFA):
                logits = logits[0]
            correct = metric.compute(logits, batch['labels'])
            metric.update(correct)
        res = metric.accumulate()
        logger.info("acc: %s, " % res)
        model.train()
        return res
```

`custom_dynabert_calc_loss` 函数定义示例：

```python
def calc_loss(loss_fct, model, batch, head_mask):
    logits = model(batch["input_ids"],
                batch["token_type_ids"],
                attention_mask=[None, head_mask])
    loss = loss_fct(logits, batch["labels"])
    return loss
```
在调用 `compress()` 时传入这 2 个自定义函数：

```python
trainer.compress(custom_dynabert_evaluate=evaluate_seq_cls,
                 custom_dynabert_calc_loss=calc_loss
                 )
```

## 二、压缩 API 使用 TIPS

1. 模型压缩主要用于推理加速，因此压缩后的模型都是静态图模型，不能再通过 `from_pretrained()` API 导入继续训练；

2. 压缩 API `compress()` 默认会启动裁剪和量化，用户可以设置参数 `--strategy` 来选择压缩的策略。目前裁剪策略有训练过程，需要下游任务的训练数据，其训练时间视下游任务数据量而定，且和微调的训练时间是一个量级。量化则不需要额外的训练，更快，通常来说量化的加速比比裁剪更明显。建议裁剪和量化同时选择，有些情况下可能比单独量化效果更好；

3. DynaBERT 裁剪类似蒸馏过程，方便起见，可以直接使用微调时的超参。如果想要进一步提升精度，可以对 `batch_size`、`learning_rate`、`epoch` 等超参进行 Grid Search；

4. 使用量化时，`eval_dataset` 不可以是 `TensorDataset` 对象，因为量化功能内部在静态图模式下执行，而 `TensorDataset` 只能在动态图下使用，两者同时使用会导致错误；

## 三、压缩 API 使用案例

本项目提供了压缩 API 在分类（包含文本分类、文本匹配、自然语言推理、代词消歧等任务）、序列标注、阅读理解三大场景下的使用样例，可以分别参考 [ernie-3.0](../model_zoo/ernie-3.0/) 目录下的 `compress_seq_cls.py` 、`compress_token_cls.py`、`compress_qa.py` 脚本，启动方式如下：

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
