# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import Accuracy

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments
from paddlenlp.transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    ElectraForSequenceClassification,
    ElectraTokenizer,
    ErnieForSequenceClassification,
    ErnieTokenizer,
    LinearDecayWithWarmup,
)

FORMAT = "%(asctime)s-%(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "mrpc": AccuracyAndF1,
    "sts-b": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
}

MODEL_CLASSES = {
    "bert": (BertForSequenceClassification, BertTokenizer),
    "electra": (ElectraForSequenceClassification, ElectraTokenizer),
    "ernie": (ErnieForSequenceClassification, ErnieTokenizer),
}


@dataclass
class ModelArguments:
    task_name: str = field(
        default=None,
        metadata={"help": "The namve of the task to train selected in the list: " + ", ".join(METRIC_CLASSES.keys())},
    )
    model_type: str = field(
        default=None,
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())},
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pre-trained model or shortcut name selected in the list: "
            + ", ".join(list(ElectraForSequenceClassification.pretrained_init_configuration.keys()))
        },
    )
    max_seq_length: int = field(
        default=128, metadata={"help": "The maximum total input sequence length after tokenization"}
    )
    warmup_proportion: float = field(default=0.1, metadata={"help": "Linear warmup proportion over total steps."})


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        print(
            "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
            % (
                loss.numpy(),
                res[0],
                res[1],
                res[2],
                res[3],
                res[4],
            ),
            end="",
        )
    elif isinstance(metric, Mcc):
        print("eval loss: %f, mcc: %s, " % (loss.numpy(), res[0]), end="")
    elif isinstance(metric, PearsonAndSpearman):
        print(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
            % (loss.numpy(), res[0], res[1], res[2]),
            end="",
        )
    else:
        print("eval loss: %f, acc: %s, " % (loss.numpy(), res), end="")
    model.train()


def convert_example(example, tokenizer, label_list, max_seq_length=512, is_test=False):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example["labels"]
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if (int(is_test) + len(example)) == 2:
        example = tokenizer(example["sentence"], padding="max_length", max_seq_len=max_seq_length)
    else:
        example = tokenizer(
            example["sentence1"], text_pair=example["sentence2"], max_seq_len=max_seq_length, padding=True
        )

    if not is_test:
        example["labels"] = label

    return example


def do_train():

    training_args, model_args = PdArgumentParser([TrainingArguments, ModelArguments]).parse_args_into_dataclasses()
    training_args: TrainingArguments = training_args
    model_args: ModelArguments = model_args

    model_args.task_name = model_args.task_name.lower()

    train_ds = load_dataset("glue", model_args.task_name, splits="train")
    tokenizer = ElectraTokenizer.from_pretrained(model_args.model_name_or_path)

    trans_func = partial(
        convert_example, tokenizer=tokenizer, label_list=train_ds.label_list, max_seq_length=model_args.max_seq_length
    )
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=training_args.train_batch_size, shuffle=True
    )
    train_data_loader = DataLoader(
        dataset=train_ds, batch_sampler=train_batch_sampler, num_workers=0, return_list=True
    )
    if model_args.task_name == "mnli":
        dev_ds = load_dataset("glue", model_args.task_name, splits=["dev_matched"])
    else:
        dev_ds = load_dataset("glue", model_args.task_name, splits="dev")

    dev_ds = dev_ds.map(trans_func, lazy=True)

    num_classes = 1 if train_ds.label_list is None else len(train_ds.label_list)
    model = ElectraForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_labels=num_classes)

    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
        model = paddle.DataParallel(model)

    num_training_steps = (
        training_args.max_steps
        if training_args.max_steps > 0
        else (len(train_data_loader) * training_args.num_train_epochs)
    )
    warmup = training_args.warmup_steps if training_args.warmup_steps > 0 else training_args.warmup_ratio

    lr_scheduler = LinearDecayWithWarmup(training_args.learning_rate, num_training_steps, warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=training_args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=training_args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    loss_fct = paddle.nn.loss.CrossEntropyLoss() if train_ds.label_list else paddle.nn.loss.MSELoss()

    def compute_metrics(p):
        # Define the metrics of tasks.
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        preds = paddle.to_tensor(preds)
        label = paddle.to_tensor(p.label_ids)

        metric = Accuracy()
        metric.reset()
        result = metric.compute(preds, label)
        metric.update(result)
        accu = metric.accumulate()
        metric.reset()
        return {"accuracy": accu}

    # TODO: use amp
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=model_args.max_seq_length),
        criterion=loss_fct,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=[optimizer, lr_scheduler],
    )

    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    do_train()
