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
import os
import time
from functools import partial

import paddle
from paddle.io import DataLoader

from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import GPTForTokenClassification, GPTChineseTokenizer
from paddlenlp.data import Stack, Pad, Dict

parser = argparse.ArgumentParser()

# yapf: disable
parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(list(GPTChineseTokenizer.pretrained_init_configuration.keys())))
parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.", )
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu", "xpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
# yapf: enable


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
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            length, preds, labels)
        metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(),
                      num_correct_chunks.numpy())
    precision, recall, f1_score = metric.accumulate()
    print("eval loss: %f, precision: %f, recall: %f, f1: %f" %
          (avg_loss, precision, recall, f1_score))
    model.train()


def tokenize_and_align_labels(example,
                              tokenizer,
                              no_entity_id,
                              max_seq_len=512):
    labels = example['labels']
    example = example['tokens']
    tokenized_input = tokenizer(example,
                                return_length=True,
                                is_split_into_words=True,
                                max_seq_len=max_seq_len,
                                return_token_type_ids=False)

    tokenized_input['labels'] = labels[:len(tokenized_input["input_ids"])]
    return tokenized_input


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # Create dataset, tokenizer and dataloader.
    train_ds, test_ds = load_dataset('msra_ner',
                                     splits=('train', 'test'),
                                     lazy=False)

    tokenizer = GPTChineseTokenizer.from_pretrained(args.model_name_or_path)

    label_list = train_ds.label_list
    label_num = len(label_list)
    no_entity_id = label_num - 1

    trans_func = partial(tokenize_and_align_labels,
                         tokenizer=tokenizer,
                         no_entity_id=no_entity_id,
                         max_seq_len=args.max_seq_length)

    train_ds = train_ds.map(trans_func)

    ignore_label = -100

    batchify_fn = lambda samples, fn=Dict(
        {
            'input_ids': Pad(axis=0, pad_val=0, dtype='int64'),  # input
            'seq_len': Stack(dtype='int64'),  # seq_len
            'labels': Pad(axis=0, pad_val=ignore_label, dtype='int64')  # label
        }): fn(samples)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    train_data_loader = DataLoader(dataset=train_ds,
                                   collate_fn=batchify_fn,
                                   num_workers=0,
                                   batch_sampler=train_batch_sampler,
                                   return_list=True)

    test_ds = test_ds.map(trans_func)

    test_data_loader = DataLoader(dataset=test_ds,
                                  collate_fn=batchify_fn,
                                  num_workers=0,
                                  batch_size=args.batch_size,
                                  return_list=True)

    # Define the model netword and its loss
    model = GPTForTokenClassification.from_pretrained(args.model_name_or_path,
                                                      num_classes=label_num)
    #model = ErnieCtmForTokenClassification.from_pretrained(
    #    args.model_name_or_path, num_classes=label_num)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.num_train_epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_steps)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    metric = ChunkEvaluator(label_list=label_list)

    global_step = 0
    last_step = args.num_train_epochs * len(train_data_loader)
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, _, labels = batch
            logits = model(input_ids)
            loss = loss_fct(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.logging_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, args.logging_steps /
                       (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % args.save_steps == 0 or global_step == last_step:
                if paddle.distributed.get_rank() == 0:
                    evaluate(model, loss_fct, metric, test_data_loader)
                    paddle.save(
                        model.state_dict(),
                        os.path.join(args.output_dir,
                                     "model_%d.pdparams" % global_step))


if __name__ == "__main__":
    args = parser.parse_args()
    do_train(args)
