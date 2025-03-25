# PaddleNLP Trainer API

PaddleNLP provides the Trainer training API, which encapsulates general training configurations for the training process, such as:

- Training configurations like optimizer, learning rate scheduling
- Features like multi-GPU, mixed precision, gradient accumulation
- Checkpointing, resuming from checkpoints (dataset, random seed recovery)
- Logging, loss visualization, etc.

Users can input a model and dataset to efficiently and quickly implement tasks like pre-training and fine-tuning using the Trainer API.

## Introduction to Basic Usage of Trainer

Below is a simple example of using the Trainer API for a fine-tuning task, using the Chinese sentiment classification dataset `chnsenticorp` as an example. For more detailed usage, refer to the [CLUE Trainer](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/slm/examples/benchmark/clue/classification/run_clue_classifier_trainer.py) version.

1. Import the necessary headers.
    - Mainly the model, Tokenizer
    - Also the Trainer components
        - `Trainer` is the main entry point for training; users can pass in the model and dataset to start training.
        - `TrainingArguments` contains most of the training parameters needed by the user.
        - `PdArgumentParser` is a tool for outputting user parameters.
```python
from functools import partial
import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.trainer import Trainer, TrainingArguments, PdArgumentParser
```
2. Set up user parameters
    - `PdArgumentParser` can accept multiple parameters similar to `TrainingArguments`. Users can customize the required `ModelArguments`, `DataArguments` and pass them as a tuple to `PdArgumentParser`.
    - These parameters are passed in via `python xxx.py --dataset xx --max_seq_length xx`. All configurable parameters of `TrainingArguments` are detailed later.
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

3. Load the model, tokenizer, and dataset
    - Note that the dataset here needs to output a `dict`. The keys in the `dict` need to correspond to the model's input names.
    - Here, if `labels` are not used by the model, we need to additionally define a `criterion` to compute the final loss.
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

4. Construct a `Trainer` instance for model training.
    - Here, we pass in `model, criterion, args, train_dataset, tokenizer` and other components required for training to construct an instantiated `trainer`.
    - Use the `trainer.train()` interface to start the training process. After training is complete, you can save the model and some logs.
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
For usage of pre-training, refer to the [ERNIE-1.0 Trainer](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/slm/model_zoo/ernie-1.0/run_pretrain_trainer.py) version.


## Advanced Distributed Capabilities of Trainer

**General Distributed Capabilities**
For general distributed capabilities, PaddleNLP primarily supports data parallelism and distributed parameter sharding. These features do not require users to modify the network architecture and can be executed directly on multiple GPUs.

Users can switch the running program to multi-GPU data parallelism using `paddle.distruted.launch --devices "0,1,2,3" train.py`. To utilize the sharding feature and reduce model memory usage, specify the parameter `--sharding "stage2"`. For more sharding configuration options, refer to the parameter introduction section.


**Hybrid Parallel Distributed Capabilities**

PaddlePaddle 4D parallelism includes: data parallel + sharding parallel + tensor parallel + pipeline parallel.

In hybrid parallelism, tensor parallel (TP) and pipeline parallel (PP) support have been primarily added. Currently, PaddleNLP provides TP and PP support for large models such as GPT and Llama, allowing users to employ these strategies.

For related code implementations, refer to the [example](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm) of Llama training.

For network transformation in pipeline parallelism, see [modeling_pp.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/llama/modeling_pp.py).

Once the network is adapted for Tensor Parallel (TP) and Pipeline Parallel (PP), users can enable hybrid parallel training using `--tensor_parallel_degree` and `--pipeline_parallel_degree`.


## Introduction to Trainer Instantiation Parameters
Trainer is a simple yet fully functional Paddle training and evaluation module, optimized for PaddleNLP models.
```text
Parameters:
    model ([`PretrainedModel`] or `paddle.nn.Layer`, *optional*):
        The model to train, evaluate, or use for predictions.
        The [`Trainer`] is optimized for use with PaddleNLP's [`PretrainedModel`]. You can still use
        your own model defined as `paddle.nn.Layer`, as long as they function similarly to PaddleNLP models.

    criterion (`paddle.nn.Layer`, *optional*):
        The model may only output intermediate results like logits. If you want to perform additional computations
        on the model's output, you can add a criterion layer.

    args ([`TrainingArguments`], *optional*):
        The arguments required for training. Will default to an instance of [`TrainingArguments`] with the
        `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.

    data_collator (`DataCollator`, *optional*):
        The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
        default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
        [`DataCollatorWithPadding`] otherwise.

    train_dataset (`paddle.io.Dataset` or `paddle.io.IterableDataset`, *optional*):
        The dataset to use for training. If it is a `datasets.Dataset`, columns not accepted by the
        `model.forward()` method are automatically removed.

    eval_dataset (`paddle.io.Dataset` or `Dict[str, paddle.io.Dataset]`, *optional*):
        The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
        `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
        dataset, prepending the dictionary key to the metric name.

    tokenizer ([`PretrainedTokenizer`], *optional*):
        The tokenizer used to preprocess the data. If provided, it will be used to automatically pad the inputs to
        the maximum length when batching inputs, and it will be saved along with the model to facilitate rerunning
        an interrupted training or reusing the fine-tuned model.

    compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
        The function that will be used to compute metrics at evaluation. Must take an [`EvalPrediction`] and return
        a dictionary mapping strings to metric values.

    callbacks (List of [`TrainerCallback`], *optional*):
        A list of callbacks to customize the training loop. These will be added to the list of default callbacks.
        If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.

    optimizers (`Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler]`, *optional*):
        A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your model
        and a scheduler [`LinearDecayWithWarmup`].

    preprocess_logits_for_metrics (`Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]`, *optional*):
        A function that preprocesses the logits right before caching them at each evaluation step. Must take two
        tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
        by this function will be reflected in the predictions received by `compute_metrics`.
```
## Introduction to TrainingArguments Parameters
It seems there is no content provided for translation. Could you please provide the text you would like translated from Chinese to English?
