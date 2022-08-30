# PaddleNLP Trainer API

PaddleNLP提供了Trainer训练API，针对训练过程的通用训练配置做了封装，比如：

- 优化器、学习率调度等训练配置
- 多卡，混合精度，梯度累积等功能
- checkpoint断点，断点重启（数据集，随机数恢复）
- 日志显示，loss可视化展示等

用户输入模型，数据集，就可以使用Trainer API高效快速的实现预训练、微调等任务。


## Trainer基本使用方法介绍

下面是用户使用 Trainer API进行finetune任务的简单示例，这里以中文情感分类数据集`chnsenticorp`为例。
更详细的使用可以参考[CLUE Trainer](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/benchmark/clue/classification/run_clue_classifier_trainer.py)版本。

1. 导入需要用到的头文件。
    - 主要是模型、Tokenizer
    - 还有Trainer组件
        - 其中`Trainer`是训练主要入口，用户传入模型，数据集，即可进行训练
        - `TrainingArguments` 包含了用户需要的大部分训练参数。
        - `PdArgumentParser` 是用户输出参数的工具
```python
from functools import partial
import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.trainer import Trainer, TrainingArguments, PdArgumentParser
```
2. 设置好用户参数
    - PdArgumentParser 可以接受多个类似`TrainingArguments`的参数。用户可以自定义所需要的`ModelArguments`, `DataArguments`为为 tuple 传入 PdArgumentParser即可。
    - 这些参数都是通过`python xxx.py --dataset xx --max_seq_length xx`的方式传入。`TrainingArguments`的所有可配置参数见后文。
```python
from dataclasses import dataclass
@dataclass
class DataArguments:
    dataset: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use."})

    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization."})

parser = PdArgumentParser(TrainingArguments, DataArguments)
(training_args, data_args) = parser.parse_args_into_dataclasses()
```

3. 加载模型，tokenizer, 数据集
    - 注意，这里的数据集，需要输出的是一个dict。dict中的key，需要和模型的输入名称对应。
    - 这里的，`labels`如果模型没有使用到，我们还需要额外定义`criterion`，计算最后的loss损失。
```python
train_dataset = load_dataset("chnsenticorp", splits=["train"])
model = AutoModelForSequenceClassification.from_pretrained("ernie-3.0-medium-zh", num_classes=len(train_dataset.label_list))
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")

def convert_example(example, tokenizer):
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=128, pad_to_max_seq_len=True)
    encoded_inputs["labels"] = int(example["label"])
    return encoded_inputs

train_dataset = train_dataset.map(partial(convert_example, tokenizer=tokenizer))
```

4. 构造Trainer实例，进行模型训练。
    - 这里传入`model,criterion,args,train_dataset,tokenizer`这些训练需要的组件，构建了实例化的trainer
    - 使用trainer.train()接口开始训练过程。训练完成后，可以保存模型，保存一些日志。
```python
trainer = Trainer(
    model=model,
    criterion=paddle.nn.loss.CrossEntropyLoss(),
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    tokenizer=tokenizer)

if training_args.do_train:
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_state()
```
预训练的使用方式可以参考[ERNIE-1.0 Trainer](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/ernie-1.0/run_pretrain_trainer.py)版本。


## Trainer 实例化参数介绍
Trainer 是一个简单，但功能完整的 Paddle训练和评估模块，并针对 PaddleNLP 模型进行了优化。

```python
参数：
    model（[`PretrainedModel`] 或 `paddle.nn.Layer`，可选）：
        用于训练、评估或预测的模型。
        [`Trainer`] 对PaddleNLP的 [`PretrainedModel`] 一起使用进行了优化。你仍然可以使用
        您自己的模型定义为`paddle.nn.Layer`，只要它们的工作方式与 PaddleNLP 模型相同。

        ([`PretrainedModel`] or `paddle.nn.Layer`, *optional*):
        The model to train, evaluate or use for predictions.
    criterion (`paddle.nn.Layer`，*可选*）：
        model可能只输出中间结果loggit，如果想对模型的输出做更多的计算，可以添加criterion层。

        The model may only output the loggit, if you want do more computation for the output of model,
        you can add the criterion Layer.

    args（[`TrainingArguments`]，可选）：
        训练时需要用到的参数。将默认使用 [`TrainingArguments`] 初始化。
        `output_dir` 设置为当前目录中名为 *tmp_trainer* 的目录（如果未提供）。

        ([`TrainingArguments`], *optional*):
        The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
        `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.

    data_collator（`DataCollator`，可选）：
        用于将 `train_dataset` 或 `eval_dataset` 的数据，组合为batch的函数。
        如果没有提供 `tokenizer`，则默认为 [`default_data_collator`], 否则为
        [`DataCollatorWithPadding`]。

         (`DataCollator`, *optional*):
        The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
        default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
        [`DataCollatorWithPadding`] otherwise.


    train_dataset（`paddle.io.Dataset` 或 `paddle.io.IterableDataset`，可选）：
        用于训练的数据集。如果是 `datasets.Dataset`，那么
        `model.forward()` 不需要的输入字段会被自动删除。

        (`paddle.io.Dataset` or `paddle.io.IterableDataset`, *optional*):
        The dataset to use for training. If it is an `datasets.Dataset`, columns not accepted by the
        `model.forward()` method are automatically removed.

    eval_dataset（`paddle.io.Dataset`，可选）：
        用于评估的数据集。如果是 `datasets.Dataset`，那么
        `model.forward()` 不需要的输入字段会被自动删除。

        The dataset to use for evaluation.

    tokenizer（[`PretrainedTokenizer`]，可选）：
        用于数据预处理的tokenizer。如果传入，将用于自动Pad输入
        batch输入的最大长度，它随模型保存，可以重新运行中断的训练过程。

         ([`PretrainedTokenizer`], *optional*):
        The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
        maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
        interrupted training or reuse the fine-tuned model.

    compute_metrics (`Callable[[EvalPrediction], Dict]`, 可选):
        用于评估的计算指标的函数。必须采用 [`EvalPrediction`] 并返回
        dict形式的metrics结果。

        (`Callable[[EvalPrediction], Dict]`, *optional*):
        The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
        a dictionary string to metric values.

    callbacks (List of [`TrainerCallback`]，*可选*）：
        用于自定义训练call列表函数。将这些函数会被添加到默认回调函数列表。
        如果要删除使用的回调函数，请使用 [`Trainer.remove_callback`] 方法。

        A list of callbacks to customize the training loop. Will add those to the list of default callbacks.
        If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.

    optimizers (`Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler]`, 可选）：
        一个tuple, 包含要使用Optimizer和LRScheduler。将默认为模型上的 [`AdamW`] 实例
        和LinearDecayWithWarmup。

        (`Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler]`, *optional*)
        A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your model
        and a scheduler  [`LinearDecayWithWarmup`].
```


## TrainingArguments 参数介绍
```python
  --output_dir
                        保存模型输出和和中间checkpoints的输出目录。(`str`, 必须, 默认为 `None`)

                        The output directory where the model predictions and
                        checkpoints will be written. (default: None)

  --overwrite_output_dir
                        如果 `True`，覆盖输出目录的内容。如果 `output_dir` 指向检查点
                        目录，则使用它继续训练。(`bool`, 可选, 默认为 `False`)

                        Overwrite the content of the output directory. Use
                        this to continue training if output_dir points to a
                        checkpoint directory. (default: False)

  --do_train
                        是否进行训练任务。 注：`Trainer`不直接使用此参数，而是提供给用户
                        的训练/评估脚本使用。(`bool`, 可选, 默认为 `False`)

                        Whether to run training. (default: False)

  --do_eval
                        是否进行评估任务。同上。(`bool`, 可选, 默认为 `False`)

                        Whether to run eval on the dev set. (default: False)

  --do_predict
                        是否进行预测任务。同上。(`bool`, 可选, 默认为 `False`)

                        Whether to run predictions on the test set. (default:False)

  --do_export
                        是否进行模型导出任务。同上。(`bool`, 可选, 默认为 `False`)

                        Whether to export infernece model. (default: False)

  --evaluation_strategy {no,steps,epoch}
                        评估策略，（`str`, 可选，默认为 `"no"`）：
                        训练期间采用的评估策略。可能的值为：
                            - `"no"`：训练期间不进行评估。
                            - `"steps"`：评估在每个`eval_steps`完成（并记录）。
                            - `"epoch"`：在每个 epoch 结束时进行评估。

                        The evaluation strategy to use. (default: no)

  --prediction_loss_only
                        在执行评估和预测任务时，只返回loss的值。(`bool`, 可选, 默认为 `False`)

                        When performing evaluation and predictions, only
                        returns the loss. (default: False)

  --per_device_train_batch_size
                        用于训练的每个 GPU 核心/CPU 的batch大小.（`int`，可选，默认为 8）

                        Batch size per GPU core/CPU for training. (default: 8)

  --per_device_eval_batch_size
                        用于评估的每个 GPU 核心/CPU 的batch大小.（`int`，可选，默认为 8）

                        Batch size per GPU core/CPU for evaluation. (default:8)

  --gradient_accumulation_steps
                        在执行反向，更新回传梯度之前，累积梯度的更新步骤数（`int`，可选，默认为 1）

                        Number of updates steps to accumulate before
                        performing a backward/update pass. (default: 1)

  --learning_rate
                        优化器的初始学习率, （`float`，可选，默认为 5e-05）

                        The initial learning rate for optimizer. (default: 5e-05)

  --weight_decay
                        除了所有bias和 LayerNorm 权重之外，应用于所有层的权重衰减数值。（`float`，可选，默认为 0.0）

                        Weight decay for AdamW if we apply some. (default:
                        0.0)

  --adam_beta1
                        AdamW的优化器的 beta1 超参数。（`float`，可选，默认为 0.9）

                        Beta1 for AdamW optimizer (default: 0.9)

  --adam_beta2
                        AdamW的优化器的 beta2 超参数。（`float`，可选，默认为 0.999）

                        Beta2 for AdamW optimizer (default: 0.999)

  --adam_epsilon
                        AdamW的优化器的 epsilon 超参数。（`float`，可选，默认为 1e-8)

                        Epsilon for AdamW optimizer. (default: 1e-08)

  --max_grad_norm
                        最大梯度范数（用于梯度裁剪）。（`float`，可选，默认为 1.0）

                        Max gradient norm. (default: 1.0)

  --num_train_epochs
                        要执行的训练 epoch 总数（如果不是整数，将在停止训练
                        之前执行最后一个 epoch 的小数部分百分比）。
                        (`float`, 可选, 默认为 3.0):

                        Total number of training epochs to perform. (default:3.0)

  --max_steps
                        如果设置为正数，则表示要执行的训练步骤总数。
                        覆盖`num_train_epochs`。（`int`，可选，默认为 -1）

                        If > 0: set total number of training steps to
                        perform.Override num_train_epochs. (default: -1

  --lr_scheduler_type
                        要使用的学习率调度策略。 (`str`, 可选, 默认为 `"linear"`)

                        The scheduler type to use. (default: linear) 支持，linear, cosine, constant, constant_with_warmup.

  --warmup_ratio
                        用于从 0 到 `learning_rate` 的线性warmup的总训练步骤的比例。（`float`，可选，默认为 0.0）

                        Linear warmup over warmup_ratio fraction of total
                        steps. (default: 0.0)

  --warmup_steps
                        用于从 0 到 `learning_rate` 的线性warmup的步数。覆盖warmup_ratio参数。
                        （`int`，可选，默认为 0）

                        Linear warmup over warmup_steps. (default: 0)

  --log_on_each_node
                        在多节点分布式训练中，是在每个节点上记录一次，还是仅在主节点上记录节点。（`bool`，可选，默认为`True`）

                        When doing a multinode distributed training, whether
                        to log once per node or just once on the main node.
                        (default: True)

  --logging_dir
                        VisualDL日志目录。（`str`，可选，默认为None）
                        None情况下会修改为 *output_dir/runs/**CURRENT_DATETIME_HOSTNAME**

                        VisualDL log dir. (default: None)

  --logging_strategy {no,steps,epoch}
                        (`str`, 可选，默认为 `"steps"`)
                        训练期间采用的日志记录策略。可能的值为：
                            - `"no"`：训练期间不进行记录。
                            - `"epoch"`：记录在每个 epoch 结束时完成。
                            - `"steps"`：记录是每 `logging_steps` 完成的。

                        The logging strategy to use. (default: steps)

  --logging_first_step
                        是否记录和评估第一个 `global_step`。（`bool`，可选，默认为`False`）

                        Log the first global_step (default: False)

  --logging_steps
                        如果 `logging_strategy="steps"`，则两个日志之间的更新步骤数。
                        （`int`，可选，默认为 500）

                        Log every X updates steps. (default: 500)

  --save_strategy {no,steps,epoch}
                        (`str`, 可选，默认为 `"steps"`)
                        训练期间采用的checkpoint保存策略。可能的值为：
                            - `"no"`：训练期间不保存。
                            - `"epoch"`：保存在每个 epoch 结束时完成。
                            - `"steps"`：保存是每`save_steps`完成。
                        The checkpoint save strategy to use. (default: steps)

  --save_steps
                        如果 `save_strategy="steps"`，则在两个checkpoint保存之间的更新步骤数。
                        （`int`，可选，默认为 500）

                        Save checkpoint every X updates steps. (default: 500)

  --save_total_limit
                        如果设置次参数，将限制checkpoint的总数。删除旧的checkpoints
                        `输出目录`。(`int`，可选）

                        Limit the total amount of checkpoints. Deletes the
                        older checkpoints in the output_dir. Default is
                        unlimited checkpoints (default: None)

  --save_on_each_node
                        在做多节点分布式训练时，是在每个节点上保存模型和checkpoints，
                        还是只在主节点上。当不同的节点使用相同的存储时，不应激活此功能，
                        因为每个节点的文件将以相同的名称保存。(`bool`, 可选, 默认为 `False`)

                        When doing multi-node distributed training, whether to
                        save models and checkpoints on each node, or only on
                        the main one (default: False)

  --no_cuda
                        是否不使用 CUDA，即使CUDA环境可用。(`bool`, 可选, 默认为 `False`)
                        Do not use CUDA even when it is available (default:
                        False)
  --seed
                        设置的随机种子。为确保多次运行的可复现性。（`int`，可选，默认为 42）

                        Random seed that will be set at the beginning of
                        training. (default: 42)

  --fp16
                        是否使用 fp16 混合精度训练而不是 fp32 训练。
                        (`bool`, 可选, 默认为 `False`)

                        Whether to use fp16 (mixed) precision instead of
                        32-bit (default: False)

  --fp16_opt_level
                        混合精度训练模式，可为``O1``或``O2``模式，默认``O1``模式，默认O1. 只在fp16选项开启时候生效
                        (`str`, 可选, 默认为 `O1`)

                        For fp16: AMP optimization level selected in
                        ['O0', 'O1', and 'O2']. See details at https://www.pad
                        dlepaddle.org.cn/documentation/docs/zh/develop/api/pad
                        dle/amp/auto_cast_cn.html (default: O1)

  --scale_loss
                        FP16训练时，scale_loss的初始值。
                        （`float`，可选，默认为 32768）

                        The value of initial scale_loss for fp16. (default: 32768)

  --minimum_eval_times
                        最少评估次数，如果当前设置的eval_steps，评估次数少于minimum_eval_times，
                        此选项会覆盖eval_steps参数。
                        （`int`，可选，默认为 None）

                        If under eval_steps, the valid time is less then
                        minimum_eval_times, the config of override eval_steps.
                        (default: None)

  --local_rank
                        分布式训练时，设备的本地rank值。
                        For distributed training: local_rank (default: -1)

  --dataloader_drop_last
                        是否丢弃最后一个不完整的批次（如果数据集的长度不能被批次大小整除）
                        （`bool`，可选，默认为 False）

                        Drop the last incomplete batch if it is not divisible
                        by the batch size. (default: False)

  --eval_steps
                        如果 `evaluation_strategy="steps"`，则两次评估之间的更新步骤数。将默认为相同如果未设置，则值为 `logging_steps`。
                        （`int`，可选，默认为 None）

                        Run an evaluation every X steps. (default: None)

  --dataloader_num_workers
                        用于数据加载的子进程数。 0 表示数据将在主进程制造。
                        （`int`，可选，默认为 0）

                        Number of subprocesses to use for data loading. 0 means
                        that the data will be loaded in the main process. (default: 0)

  --past_index
                        If >=0, uses the corresponding part of the output as
                        the past state for next step. (default: -1)

  --run_name
                        An optional descriptor for the run. (default: None)
  --device
                        运行的设备名称。支持cpu/gpu, 默认gpu
                        （`str`，可选，默认为 'gpu'）

                        select cpu, gpu, xpu devices. (default: gpu)

  --disable_tqdm
                        是否使用tqdm进度条
                        Whether or not to disable the tqdm progress bars.
                        (default: None)

  --remove_unused_columns
                        去除Dataset中不用的字段数据
                        Remove columns not required by the model when using an
                        nlp.Dataset. (default: True)

  --label_names
                        训练数据标签label的名称
                        The list of keys in your dictionary of inputs that
                        correspond to the labels. (default: None)

  --load_best_model_at_end
                        训练结束后是否加载最优模型，通常与`metric_for_best_model`配合使用
                        Whether or not to load the best model found during
                        training at the end of training. (default: False)

  --metric_for_best_model
                        最优模型指标，如`eval_accuarcy`等，用于比较模型好坏。
                        The metric to use to compare two different models.
                        (default: None)

  --greater_is_better
                        与`metric_for_best_model`配合使用。
                        Whether the `metric_for_best_model` should be
                        maximized or not. (default: None)

  --ignore_data_skip
                        重启训练时候，不略过已经训练的数据。
                        When resuming training, whether or not to skip the
                        first epochs and batches to get to the same training
                        data. (default: False)

  --optim
                        优化器名称，默认为adamw，，(`str`, 可选，默认为 `adamw`)
                        The optimizer to use. (default: adamw)

  --report_to
                        日志可视化显示，默认使用visualdl可视化展示。(可选，默认为 None，展示所有)
                        The list of integrations to report the results and
                        logs to. (default: None)

  --resume_from_checkpoint
                        是否从断点重启恢复训练，(可选，默认为 None)
                        The path to a folder with a valid checkpoint for your
                        model. (default: None)

```
