trainer.md
# PaddleNLP Trainer API

PaddleNLP 提供了 Trainer 训练 API，针对训练过程的通用训练配置做了封装，比如：

- 优化器、学习率调度等训练配置
- 多卡，混合精度，梯度累积等功能
- checkpoint 断点，断点重启（数据集，随机数恢复）
- 日志显示，loss 可视化展示等

用户输入模型，数据集，就可以使用 Trainer API 高效快速的实现预训练、微调等任务。


## Trainer 基本使用方法介绍

下面是用户使用 Trainer API 进行 finetune 任务的简单示例，这里以中文情感分类数据集`chnsenticorp`为例。
更详细的使用可以参考[CLUE Trainer](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/slm/examples/benchmark/clue/classification/run_clue_classifier_trainer.py)版本。

1. 导入需要用到的头文件。
    - 主要是模型、Tokenizer
    - 还有 Trainer 组件
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
    - PdArgumentParser 可以接受多个类似`TrainingArguments`的参数。用户可以自定义所需要的`ModelArguments`, `DataArguments`为 tuple 传入 PdArgumentParser 即可。
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
    - 注意，这里的数据集，需要输出的是一个 dict。dict 中的 key，需要和模型的输入名称对应。
    - 这里的，`labels`如果模型没有使用到，我们还需要额外定义`criterion`，计算最后的 loss 损失。
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

4. 构造 Trainer 实例，进行模型训练。
    - 这里传入`model,criterion,args,train_dataset,tokenizer`这些训练需要的组件，构建了实例化的 trainer
    - 使用 trainer.train()接口开始训练过程。训练完成后，可以保存模型，保存一些日志。
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
预训练的使用方式可以参考[ERNIE-1.0 Trainer](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/slm/model_zoo/ernie-1.0/run_pretrain_trainer.py)版本。


## Trainer 进阶分布式能力使用介绍

**通用分布式能力**
对于通用的分布式能力, PaddleNLP 主要做了数据并行 data_parallel, 分布式参数 sharding 功能的支持.
这类功能无需用户修改组网, 直接多卡即可运行.

用户使用 `paddle.distruted.launch --devices "0,1,2,3" train.py`即可将运行的程序切换为多卡数据并行.
如果想要使用 sharding 功能, 减少模型显存占用, 指定参数`--sharding "stage2"`即可. 更多 sharding 功能配置见参数介绍部分.


**混合并行分布式能力**

飞桨4D 并行, 即: data parallel +  sharding parallel + tensor parallel + pipeline parallel.

混合并行这里, 主要添加了 tensor parallel (TP) 和 pipeline parallel(PP)支持.
目前, PaddleNLP 主要对一些大模型, 如 GPT, Llama 等做了 TP PP 支持, 用户可以使用这些策略.

相关代码实现可以参考 llama 训练的[例子](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm)

流水线并行的组网改造可以参见[modeling_pp.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/llama/modeling_pp.py)


当组网适配好 张量并行(TP), 流水线并行(PP)之后, 用户使用 `--tensor_parallel_degree` `--pipeline_parallel_degree` 即可启用混合并行训练.




## Trainer 实例化参数介绍
Trainer 是一个简单，但功能完整的 Paddle 训练和评估模块，并针对 PaddleNLP 模型进行了优化。

```text
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

    eval_dataset（`paddle.io.Dataset` 或 `Dict[str, paddle.io.Dataset]`，可选）：
        用于评估的数据集。如果是 `datasets.Dataset`，那么
        `model.forward()` 不需要的输入字段会被自动删除。
        如果它是一个字典，则将对字典中每个数据集进行评估，
        并将字典中的键添加到评估指标名称前。

        The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
        `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
        dataset prepending the dictionary key to the metric name.

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

    preprocess_logits_for_metrics (`Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]`, 可选）)：
        一个函数, 在每次评估之前对logits进行预处理。

        (`Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]`, *optional*)
        A function that preprocess the logits right before caching them at each evaluation step. Must take two
        tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
        by this function will be reflected in the predictions received by `compute_metrics`.
```


## TrainingArguments 参数介绍
```text
  --output_dir
                        保存模型输出和中间checkpoints的输出目录。(`str`, 必须, 默认为 `None`)

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

  --eval_accumulation_steps
                        在将结果移动到CPU之前，累积输出张量的预测步骤数。如果如果未设置，
                        则在移动到CPU之前，整个预测都会在GPU上累积（速度更快需要更多的显存）。
                        （`int`，可选，默认为 None 不设置）

                        Number of predictions steps to accumulate the output tensors for,
                        before moving the results to the CPU. If left unset, the whole predictions are
                        accumulated on GPU before being moved to the CPU (faster butrequires more memory)
                        (default: None)

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
                        (`float`, 可选, 默认为 1.0):

                        Total number of training epochs to perform. (default:1.0)

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

  --bf16
                        是否使用 bf16 混合精度训练而不是 fp32 训练。需要 Ampere 或更高的 NVIDIA
                        显卡架构支持。这是实验性质的API，以后可能会修改。
                        (`bool`, 可选, 默认为 `False`)

                        Whether to use bf16 (mixed) precision instead of
                        32-bit. Requires Ampere or higher NVIDIA architecture.
                        This is an experimental API and it may change.
                        (default: False)

  --fp16
                        是否使用 fp16 混合精度训练而不是 fp32 训练。
                        (`bool`, 可选, 默认为 `False`)

                        Whether to use fp16 (mixed) precision instead of
                        32-bit (default: False)

  --fp16_opt_level
                        混合精度训练模式，可为``O1``或``O2``模式，默认``O1``模式，默认O1.
                        O1表示混合精度训练，O2表示纯fp16/bf16训练。
                        只在fp16或bf16选项开启时候生效.
                        (`str`, 可选, 默认为 `O1`)

                        For fp16: AMP optimization level selected in
                        ['O0', 'O1', and 'O2']. See details at https://www.pad
                        dlepaddle.org.cn/documentation/docs/zh/develop/api/pad
                        dle/amp/auto_cast_cn.html (default: O1)
  --amp_custom_black_list
                       飞桨有默认的黑名单，可以根据模型特点设置自定义黑名单。自定义黑名单中的算子在计算时会被认为是数值危险的，它们的影响也可能会在下游算子中观察到。该名单中的算子不会转为 float16/bfloat16 计算。(可选，默认为None)

                       The custom black_list. The set of ops that support fp16/bf16 calculation and are considered numerically-dangerous and whose effects may also be observed in downstream ops. These ops will not be converted to fp16/bf16. (default:None)

  --amp_custom_white_list
                       飞桨有默认的白名单，通常不需要设置自定义白名单。自定义白名单中的算子在计算时会被认为是数值安全的，并且对性能至关重要。如果设置了该名单，其中的算子会使用 float16/bfloat16 计算。(可选，默认为None)

                       The custom white_list. It’s the set of ops that support fp16/bf16 calculation and are considered numerically-safe and performance-critical. These ops will be converted to fp16/bf16. (default:None)

  --amp_master_grad
                        当使用pure fp16/bf16的时候, 可能对梯度的数值精度有更高要求,
                        例如梯度裁剪, weight decay, 权重更新的时候.
                        打开此选项, 梯度的数值精度会变成float32类型.
                        只在 --fp16_opt_level O2 生效, 默认为 False

                        For amp opt level=’O2’, whether to use float32 weight gradients
                        for calculations such as gradient clipping, weight decay, and weight updates.
                        If master_grad is enabled, the weight gradients will be float32 dtype after the backpropagation.
                        Note: only support model parallel and pipeline parallel for now !!! (default: False)

  --scale_loss
                        fp16/bf16训练时，scale_loss的初始值。
                        （`float`，可选，默认为 32768）

                        The value of initial scale_loss for fp16. (default: 32768)

  --sharding
                        是否使用Paddle的Sharding数据并行功能，用户的参数。支持sharding `stage1`, `stage2` or `stage3`。
                        其中`stage2``stage3`可以和`offload`组合使用。
                        每个种策略分别为：
                            stage1 : optimizer 中的参数切分到不同卡
                            stage2 : optimizer  + gradient 中的参数切分到不同卡
                            stage3 : parameter + gradient + optimizer  中的参数都切分到不同卡
                            offload ： offload parameters to cpu 部分参数存放到cpu中
                         (`str`,  可选, 默认为 `` 不使用sharding)

                        Whether or not to use Paddle Sharding Data Parallel training (in distributed training
                        only). The base option should be `stage1`, `stage2` or `stage3` and you can add
                        CPU-offload to `stage2` or `stage3` like this: `stage2 offload` or `stage3 offload`.
                        Each stage means:
                            stage1 : optimizer state segmentation
                            stage2 : optimizer state + gradient segmentation
                            stage3 : parameter + gradient + optimizer state segmentation
                            offload ： offload parameters to cpu

  --sharding_parallel_degree
                        设置sharding的通信组参数，表示通信组的大小。同一个sharding通信组内的参数，进行sharding，分布到不同卡上。
                        不同sharding通信组之间，相当于单纯的数据并行。此选项只在sharding选项开启时候生效。
                        默认值为-1，表示所有训练的卡在同一个通信组内。
                        (`int`, 可选, 默认为 `-1`)

                        Sharding parameter in certain cards group. For example, aussume we use 2 machines each
                        with 8 cards, then set sharding_degree=8, sharding will only communication inside machine.
                        default -1 means sharding parameters between all workers. (`int`, *optional*, defaults to `-1`)

  --sharding_comm_buffer_size_MB
                        设置sharding的通信中fuse梯度的大小。此选项只在sharding选项开启时候生效。
                        默认值为-1，表示所有通信fuse的梯度大小按照默认配置，默认配置是256MB。
                        (`int`, 可选, 默认为 `-1`)

                        Set the size of the fuse gradient in sharding communication. This option only takes effect when the sharding option is turned on.The default value is -1, which means that the gradient size of all communication fuses follows the default configuration, which is 256MB.
                        (`int`, optional, default `-1`)

  --tensor_parallel_degree
                        张量并行是Megatron论文针对Transformer结构的张量切分方法.
                        此方法将一层transformer的计算划分到了不同卡上.
                        此参数tensor_parallel_degree表示将一层transformer结构的份数.
                        默认值-1, 表示不启用张量并行,
                        (`int`, 可选, 默认为 `-1`)
                        (注: 该方法需要修改模型结构, 目前支持GPT/BLOOM/LLAMA/BLOOM/CLM/CHATGLM)
                        (注: 该方法对通信开销较大, 建议 tensor_parallel_degree<=8, 尽量使用机器内部通信)

                        Tensor parallelism is a parallel technique which proposed in (https://arxiv.org/pdf/2104.04473.pdf see 2.3 Tensor Model Parallelism).
                        This techique splits one transformer layer into multi-cards (For examples, tensor_parallel_degree=4, will split a layer to 4-parts)
                        tensor_parallel_degree means split the transformer layer to how many parts.
                        default -1 for not use tensor parallel,  Suggest tensor_parallel_degree<=8 for better proformance.
                        Note, this need model support in source code, currently GPT/BLOOM/LLAMA/BLOOM/CLM/CHATGLM is supported.

  --tensor_parallel_config
                        对于张量并行,一些选项会影响训练性能,这里将一些选项配置集中管理,以str形式传入配置.
                        支持如下选项:
                            enable_delay_scale_loss : 在优化器阶段做梯度累加，将所有梯度除以累加次数，而不是直接对loss除以累加次数。
                            sync_param : 在优化器阶段使用broadcast同步所有is_distributed=False的参数
                            sync_grad : 在优化器阶段使用broadcast同步所有is_distributed=False的梯度
                            sync_moment : 在优化器阶段使用broadcast同步所有is_distributed=False的momentum

                        Some additional config it highly affect the usage of tensor parallel, we provide some option to config it.
                        following config is support:
                            enable_delay_scale_loss, accumulate gradients until optimizer step, all gradients div by accumute step. instead of div accumute step on loss directly.
                            sync_param, in optimizer step, use broadcast to sync parameters those attr 'is_distributed' is False.
                            sync_grad, in optimizer step, use broadcast to sync gradients those attr 'is_distributed' is False.
                            sync_moment, in optimizer step, use broadcast to sync momentums those attr 'is_distributed' is False.

  --pipeline_parallel_degree
                        流水线并行是Megatron论文针对多层Transformer结构提出的按层划分方法.
                        该方法将多层的transformer结构,按照不同层,均匀划分到不同的卡上.
                        然后数据流先后在不同的卡上传递, 形成流水线.
                        参数pipeline_parallel_degree表示划分流水线的大小.(假设该参数为4, 模型12层, 则每一个pp stage 包含3层模型)
                        默认值-1, 表示不启用流水线并行,
                        (`int`, 可选, 默认为 `-1`)
                        (注, 使用此功能需要修改源码,请参见language_model/llama/modeling_pp.py文件)

                        Pipeline parallelism is parallel technique proposed in (https://arxiv.org/pdf/2104.04473.pdf see 2.2 Pipeline Model Parallelism).
                        Pipeline parallelism assigns multi-transformer layers to different cards, the micro batch data stream passed between cards like pipelines.
                        pipeline_parallel_degree means split all transformer layers to how many stages.
                        default -1 for not use pipeline parallel.
                        Note. this need model support in source code, see llama modeling_pp.py file

  --pipeline_parallel_config
                        对于流水线并行,一些选项会影响训练性能,这里将一些选项配置集中管理,以str形式传入配置.
                        支持如下选项:
                            disable_p2p_cache_shape : 关闭通信时候的tensor shape cache, 如果你的模型输入的tensor, shape 是不断变化的(如sequence length) 必须配置此选项
                            disable_partial_send_recv : 关闭与张量并行合用时候的通信优化.
                            enable_dp_comm_overlap : 开启PP+DP使用时候的通信优化.
                            enable_delay_scale_loss : 开启, 使得梯度累积, 先累积最后除以累积次数. 而不是每次除以累积次数.

                        Some additional config it highly affect the useage of pipeline parallel, we provide some option to config it.
                        following config is support:
                          disable_p2p_cache_shape, if you max sequence length is varying, please set disable_p2p_cache_shape.
                          disable_partial_send_recv, optmize send speed for tensor parallel.
                          enable_delay_scale_loss, accumulate gradients until optimizer step, all gradients div by inner pipeline accumute step. instead of div accumute step on loss directly.
                          enable_dp_comm_overlap, fuse data parallel gradient communication.

  --data_parallel_config
                        对于数据并行,一些选项会影响训练性能,这里将一些选项配置集中管理,以str形式传入配置.
                        支持如下选项:
                            enable_allreduce_avg_in_gradinent_scale : 在数据并行中, 替换`allreduce_sum + scale`模式为`allreduce_avg`, 以提高性能. 仅支持auto模式.
                            gradient_sync_after_accumulate : 当梯度累积开启时, 将梯度同步操作从backward阶段移动到optimizer阶段, 以减少同步次数, 提高性能, 但会增加显存占用. 仅支持auto模式.

                        Some additional configs which affect data parallel performance, we provide some option to config it.
                        following config is support:
                            enable_allreduce_avg_in_gradinent_scale, it replace `allreduce_sum + scale` pattern with `allreduce_avg` when scale gradient in data_parallel, which improve the performance. ONLY supported for auto mode now.
                            gradient_sync_after_accumulate, move gradient sync operations from backward into optimizer step when gradient accumulate enabling, which reduce the sync times to improve performance, but will increase the memory usage. ONLY supported for auto mode now.
  --context_parallel_degree
                        上下文并行是将训练数据在序列维度进行切分的并行方法。
                        该方法使用Ring FlashAttention来保障切分后Attention结果的正确性。通过环状通信和迭代更新来得到完整的注意力分数。
                        默认值-1, 表示不启用上下文并行,
                        (`int`, 可选, 默认为 `-1`)
                        (注: 该方法需要修改模型结构, 目前支持LLAMA)
                        (注: 该方法对通信开销较大, 建议只有在序列长度超长时, 如1024k, 时才使用)
                        Context parallelism is a parallel method that segments training data in the sequence dimension.
                        This method uses Ring FlashAttention to ensure the correctness of the Attention result after segmentation. The complete attention score is obtained through ring communication and iterative updates.
  --recompute
                        是否使用重计算训练。可以节省显存。
                        重新计算前向过程以获取梯度，减少中间变量显存.
                        注：需要组网支持 recompute，默认使用 enable_recompute 关键字作为recompute功能开关。
                        (`bool`, 可选, 默认为 `False`)

                        Recompute the forward pass to calculate gradients. Used for saving memory (default: False)

  --refined_recompute
                        精化重新计算参数，用于在GPU显存使用和计算速度之间寻求最佳平衡。
                        此参数允许用户对重新计算过程进行细致控制，以优化资源利用。具体配置示例如下：
                        `"attention_column_ln:-1,attention_row_ln:-1,flash_attn:-1,mlp_column_ln:5,mlp_row_ln:-1"`

                        在配置中，支持的参数包括：
                            `attention_column_ln`
                            `attention_row_ln`
                            `mlp_column_ln`
                            `mlp_row_ln`
                            `flash_attn`

                        每个参数后的数字，即`skip_num`，决定了对应操作跳过重计算的次数。具体解释如下：
                            `skip_num` 为 `-1`：表示在所有阶段均不进行重新计算，从而最大化显存使用。
                            `skip_num` 为 `0`：表示在每个阶段都强制进行重新计算，以最小化显存使用。

                        此外，您还可以将`skip_num`设置为`[1, ..., num_layers]`范围内的任意值。若`skip_num`超出`num_layers`，其行为将等同于设置为`-1`。
                        若配置中省略了某个参数，则系统默认将其设置为`xxx:0`。

                        (类型: `str`, 可选, 默认为: "")

                        Refined recompute parameter for optimizing the balance between GPU memory usage and computational speed.
                        This parameter allows fine-grained control over the recomputation process to optimize resource utilization. An example configuration is as follows:
                        `"attention_column_ln:-1,attention_row_ln:-1,flash_attn:-1,mlp_column_ln:5,mlp_row_ln:-1"`

                        The supported parameters in the configuration include:
                            `attention_column_ln`
                            `attention_row_ln`
                            `mlp_column_ln`
                            `mlp_row_ln`
                            `flash_attn`

                        The number following each parameter, `skip_num`, determines the number of times to bypass recomputation for the specified operation. Specifically:
                            `skip_num of -1`: Indicates no recomputation across all stages, maximizing memory usage.
                            `skip_num of 0`: Enforces recomputation at every stage, minimizing memory usage.

                        Additionally, you can set skip_num to any value within the range `[1, ..., num_layers]`. If `skip_num` exceeds `num_layers`, it will behave as if set to `-1`.
                        If a parameter is omitted from the configuration, it defaults to `xxx:0`.

                        (Type: `str`, optional, default: "")

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

  --max_evaluate_steps
                        如果设置为正数，则表示要执行的评估步骤的总数。
                        （`int`，可选，默认为 -1)

                        If set to a positive number, the total number of evaluation steps to perform. (default: -1)

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
                        优化器名称，默认为adamw，(`str`, 可选，默认为 `adamw`)
                        The optimizer to use. (default: adamw)
                        可能的值为：
                            - `"adamw"`
                            - `"adamw_mini"`

  --report_to
                        日志可视化显示，默认使用visualdl可视化展示。(可选，默认为 None，展示所有)
                        The list of integrations to report the results and
                        logs to. (default: None)

  --resume_from_checkpoint
                        是否从断点重启恢复训练，(可选，默认为 None)
                        The path to a folder with a valid checkpoint for your
                        model. (default: None)

  --unified_checkpoint
                       是否使用unified_checkpoint，开启后训练的checkpoint将存储为新格式。
                       可以支持跨分布式策略重启、动态扩缩容重启。(可选，默认为False)
                       Whether to use unified_checkpoint, enable it to store training checkpoint in a new format.
                       Supporting restart with different distribution strategies and devices，(optional, defaults to False)

  --unified_checkpoint_config
                       与Unified Checkpoint相关的一些优化配置项，以str形式传入配置。
                       支持如下选项:
                           skip_save_model_weight: 当master_weights存在时，跳过保存模型权重。
                           master_weight_compatible: 1. 仅当optimizer需要master_weights时，才进行加载;
                                                     2. 如果checkpoint中不存在master_weights，则将model weight作为master_weights进行加载。
                           remove_master_weight: 是否保存 master weight, 如果checkpoint中不存在master_weights，则将model weight作为master_weights进行加载。
                           async_save: 在保存Checkpoint至磁盘时做异步保存，不影响训练过程，提高训练效率。
                           enable_all_options: 上述参数全部开启。

                       Some additional config of Unified checkpoint, we provide some options to config.
                       Following config is support:
                           skip_save_model_weight, no need to save model weights when the master_weights exist.
                           master_weight_compatible, 1. if the master_weights exist, only load when needed.
                                                     2. if master_weights does not exist, convert model weights to master_weights when needed.
                           remove_master_weight, whether save master weight, if master_weights does not exist, convert model weights to master_weights when needed.
                           async_save, enable asynchronous saving checkpoints to disk.
                           enable_all_options, enable all unified checkpoint optimization configs.

  --ordered_save_group_size
                       选择同时轮流save checkpoint的进程数量。如果设置为0，则不使用轮流save checkpoint功能。

  --skip_memory_metrics
                       是否跳过内存profiler检测。（可选，默认为True，跳过）
                       Whether or not to skip adding of memory profiler reports
                       to metrics.(default:True)

  --flatten_param_grads
                       是否在优化器中使用flatten_param_grads策略，该策略将素有参数摊平后输入Optimizer更新。目前该策略仅在NPU设备上生效。（可选，默认为False）
                       Whether use flatten_param_grads method in optimizer,
                       only used on NPU devices.(default:False)

  --use_expert_parallel
                       Whether to enable MoE (Mixture of Experts) expert parallel training.
                       (default: False)

  --release_grads
                      是否在训练过程每次迭代后对梯度进行释放,减少峰值显存. 可选，默认为False）
                      Whether to reduce peak memory usage by releasing gradients after each iteration. (default: False)

  --ckpt_quant_stage
                      是否开启 unified Checkpoint 压缩, 可选项["O0", "O1", "O2"], 默认为O0）
                        O1: 对 Adam 优化器一/二阶动量进行 Int8 压缩.
                        O2: 对 Adam 优化器一/二阶动量进行 Int4 压缩.
                      Whether use unified Checkpoint compression, choices=["O0", "O1", "O2"]. (default: O0)
                        O1: Compress Adam moment1/moment2 to Int8 dtype.
                        O2: Compress Adam moment1/moment2 to Int4 dtype.

```
