# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
from dataclasses import dataclass, field
from functools import partial

import paddle

from paddlenlp.data import DataCollatorForTokenClassification
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments
from paddlenlp.transformers import (
    GPTChineseTokenizer,
    GPTForTokenClassification,
    LinearDecayWithWarmup,
)

parser = argparse.ArgumentParser()


# yapf: disable
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pre-trained model or shortcut name selected in the list: "}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded." + "Path to pre-trained model or shortcut name selected in the list: " + ", ".join(list(GPTChineseTokenizer.pretrained_init_configuration.keys()))}
    )


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    avg_loss, precision, recall, f1_score = 0, 0, 0, 0
    for batch in data_loader:
        input_ids, length, labels = batch
        logits = model(input_ids)
        loss = loss_fct(logits, labels)
        avg_loss = paddle.mean(loss)
        preds = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(length, preds, labels)
        metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(), num_correct_chunks.numpy())
    precision, recall, f1_score = metric.accumulate()
    print("eval loss: %f, precision: %f, recall: %f, f1: %f" % (avg_loss, precision, recall, f1_score))
    model.train()


def tokenize_and_align_labels(example, tokenizer, no_entity_id, max_seq_len=512):
    labels = example["labels"]
    example = example["tokens"]
    tokenized_input = tokenizer(
        example, is_split_into_words="token", max_seq_len=max_seq_len, return_token_type_ids=False
    )

    tokenized_input["labels"] = labels[: len(tokenized_input["input_ids"])]
    return tokenized_input


def do_train():
    training_args, model_args = PdArgumentParser([TrainingArguments, ModelArguments]).parse_args_into_dataclasses()

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # Create dataset, tokenizer and dataloader.
    train_ds, test_ds = load_dataset("msra_ner", splits=("train", "test"), lazy=False)

    tokenizer = GPTChineseTokenizer.from_pretrained(model_args.model_name_or_path)

    # add pad_token to tokenizer
    tokenizer.add_special_tokens({
        "pad_token": tokenizer.convert_ids_to_tokens(0)
    })

    label_list = train_ds.label_list
    label_num = len(label_list)
    no_entity_id = label_num - 1

    batchify_fn = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        no_entity_id=no_entity_id,
        max_seq_len=model_args.max_seq_length)

    train_ds = train_ds.map(batchify_fn)
    test_ds = test_ds.map(batchify_fn)

    # Define the model netword and its loss
    model = GPTForTokenClassification.from_pretrained(model_args.model_name_or_path,
                                                      num_classes=label_num)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = training_args.max_steps if training_args.max_steps > 0 else len(train_ds) // training_args.train_batch_size * training_args.num_train_epochs

    lr_scheduler = LinearDecayWithWarmup(training_args.learning_rate, num_training_steps,
                                         training_args.warmup_steps)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=training_args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=training_args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    metric = ChunkEvaluator(label_list=label_list)

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,
            max_length=model_args.max_seq_length
        ),
        args=training_args,
        criterion=paddle.nn.loss.CrossEntropyLoss(ignore_index=-100),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=metric,
        optimizers=[optimizer, lr_scheduler]
    )

    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    do_train()
