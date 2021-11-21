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

import argparse
import distutils.util
import logging
import os
import os.path as osp
import random
import re
import time
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import Accuracy, Metric, Precision, Recall
from paddlenlp.data import Dict, Pad, Stack, Tuple
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.transformers import (BertTokenizer, LinearDecayWithWarmup,
                                    VisualBertForQuestionAnswering)
from tqdm import tqdm

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

METRIC_CLASSES = {
    "vqa2": Accuracy,
    "nlvr2": Accuracy,
}

MODEL_CLASSES = {
    "visualbert": (VisualBertForQuestionAnswering, BertTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--bert_model_name",
        default="bert-base-uncased",
        type=str,
        help="Path to bert model or shortcut name selected in the list: " +
        ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input directory where the data will be read from.", )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion"
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.")
    parser.add_argument(
        "--use_amp",
        type=distutils.util.strtobool,
        default=False,
        help="Enable mixed precision training.")
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=2**15,
        help="The value of scale_loss for fp16.")
    args = parser.parse_args()
    return args


# ===================================================================

_num_answers = 10
# ===================================================================

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def tokenize(sentence):
    sentence = sentence.lower()
    sentence = (
        sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s'))
    tokens = SENTENCE_SPLIT_REGEX.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


class VocabDict:
    def __init__(self, vocab_file):
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.num_vocab = len(self.word_list)
        self.UNK_idx = (self.word2idx_dict['<unk>']
                        if '<unk>' in self.word2idx_dict else None)

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_idx is not None:
            return self.UNK_idx
        else:
            raise ValueError('word %s not in dictionary \
                             (while dictionary does not contain <unk>)' % w)

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds


# ===================================================================


def compute_answer_scores(answers, num_of_answers, unk_idx):
    scores = np.zeros((num_of_answers), np.float32)
    for answer in set(answers):
        if answer == unk_idx:
            scores[answer] = 0
        else:
            answer_count = answers.count(answer)
            scores[answer] = min(np.float32(answer_count) * 0.3, 1)
    return scores


# ===================================================================


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


def masked_unk_softmax(x, dim, mask_idx):
    x1 = paddle.nn.functional.softmax(x, axis=dim)
    x1[:, mask_idx] = 0
    x1_sum = paddle.sum(x1, axis=1, keepdim=True)
    y = x1 / x1_sum
    return y


def compute_score_with_logits(logits, labels):
    logits = masked_unk_softmax(logits, 1, 0)
    logits = paddle.argmax(logits, 1)  # argmax
    # one_hots = paddle.zeros_like(labels)
    one_hots = paddle.nn.functional.one_hot(logits, num_classes=labels.shape[1])
    scores = (one_hots * labels)
    return scores


@paddle.no_grad()
def evaluate_vqa(model, data_loader):
    model.eval()
    val_probs = []
    val_labels = []
    val_loss_sum = 0
    val_counter = 0
    # for batch in tqdm(data_loader, desc="Evaluate VQA2 on minival2014"):
    for idx, batch in enumerate(data_loader):
        return_dict = False
        labels = batch[6]
        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "visual_embeds": batch[3],
            "visual_token_type_ids": batch[4],
            "visual_attention_mask": batch[5],
            "labels": batch[6],
            "return_dict": return_dict
        }
        outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        # _batch_acc = paddle.sum(compute_score_with_logits(logits, labels)) / labels.shape[0]
        val_loss_sum += loss.mean() * labels.shape[0]
        val_loss_sum = val_loss_sum.detach().cpu()
        val_counter += labels.shape[0]
        val_probs.append(logits.detach().cpu().numpy())
        val_labels.append(labels.detach().cpu().numpy())
        # if (idx+1) % 200 == 0:
        #     print("eval batch:{}/{} batch_acc: {}, _batch_loss: {}".format(idx+1, len(data_loader), _batch_acc.detach().cpu().numpy(), loss.mean().detach().cpu().numpy()))

    # val_acc_avg /= val_counter
    # print("val_acc_avg", val_acc_avg.detach().cpu().numpy())
    val_loss_avg = val_loss_sum / val_counter
    val_loss_avg = val_loss_avg.detach().cpu().numpy()
    val_probs = np.concatenate(val_probs, 0)
    val_labels = np.concatenate(val_labels, 0)

    val_acc_avg = paddle.sum(
        compute_score_with_logits(
            paddle.to_tensor(val_probs), paddle.to_tensor(
                val_labels))) / val_labels.shape[0]
    val_acc_avg = val_acc_avg.detach().cpu().numpy()
    print("eval loss: %f, acc: %s" % (
        val_loss_avg,
        val_acc_avg, ), )
    model.train()


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in tqdm(data_loader, desc="Evaluate VQA2 on val"):
        return_dict = False
        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "visual_embeds": batch[3],
            "visual_token_type_ids": batch[4],
            "visual_attention_mask": batch[5],
            "labels": batch[6],
            "return_dict": return_dict
        }
        outputs = model(**inputs)
        # loss = loss_fct(logits, labels)
        loss = outputs[0]
        logits = outputs[1]
        correct = metric.compute(logits, inputs["labels"])
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
                res[4], ),
            end='')
    elif isinstance(metric, Mcc):
        print("eval loss: %f, mcc: %s, " % (loss.numpy(), res[0]), end='')
    elif isinstance(metric, PearsonAndSpearman):
        print(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
            % (loss.numpy(), res[0], res[1], res[2]),
            end='')
    else:
        print("eval loss: %f, acc: %s, " % (loss.numpy(), res), end='')
    model.train()


def prepare_train_features_single(example, tokenizer, answer_dict, args):
    image_name = example['image_name']
    image_id = example['image_id']
    question_id = example['question_id']
    feature_path = example['feature_path']
    question_str = example['question_str']
    question_tokens = example['question_tokens']
    # ocr_tokens = example['ocr_tokens']
    answers = example['answers']

    # generate label
    answers_idx = np.zeros((10), np.int32)
    answers_idx.fill(-1)
    answer_scores = np.zeros(answer_dict.num_vocab, np.float32)
    answers_idx[:len(answers)] = (
        [answer_dict.word2idx(ans) for ans in answers])
    ans_idx = ([answer_dict.word2idx(ans) for ans in answers])
    answer_scores = (compute_answer_scores(ans_idx, answer_dict.num_vocab,
                                           answer_dict.UNK_idx))
    label = paddle.to_tensor(
        answer_scores,
        dtype=paddle.float32)  # float tensor for loss computation

    # load image feature
    if "train" in feature_path:
        folder = osp.join(args.input_dir,
                          "data/detectron_fix_100/fc6/vqa/train2014")
    elif "val" in feature_path:
        folder = osp.join(args.input_dir,
                          "data/detectron_fix_100/fc6/vqa/val2014")

    detectron_features = np.load(os.path.join(folder, feature_path))
    visual_embeds = paddle.to_tensor(detectron_features)
    visual_token_type_ids = paddle.zeros(
        visual_embeds.shape[:-1], dtype=paddle.int64)
    visual_attention_mask = paddle.ones(
        visual_embeds.shape[:-1], dtype=paddle.int64)

    # two sentence for training: (a) Question ? (b) [MASK]
    tokens_a = tokenizer.tokenize(" ".join(question_tokens)) + ["?", "[MASK]"]
    tokens_b = None

    #  For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            input_type_ids.append(1)
        tokens.append("[SEP]")
        input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = paddle.to_tensor(input_ids, dtype=paddle.int64)
    token_type_ids = paddle.to_tensor(input_type_ids)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)

    data = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "question_id": question_id,
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
        "label": label,
    }

    return data


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # ===================================================================

    train_ds = load_dataset('vqa2', splits="trainval")
    answers_vqa_fullname = train_ds.vocab_info['filepath']
    answer_dict = VocabDict(answers_vqa_fullname)
    tokenizer = tokenizer_class.from_pretrained(args.bert_model_name)
    trans_func = partial(
        prepare_train_features_single,
        tokenizer=tokenizer,
        answer_dict=answer_dict,
        args=args)

    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=False)

    batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "attention_mask": Pad(axis=0, pad_val=0),
        "visual_embeds": Pad(axis=0),
        "visual_token_type_ids": Pad(axis=0),
        "visual_attention_mask": Pad(axis=0),
        "label": Pad(axis=0), }): fn(samples)
    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    dev_ds = load_dataset('vqa2', splits='minival')
    dev_ds = dev_ds.map(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=args.batch_size, shuffle=False)
    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    # ===================================================================

    num_classes = 3129 if train_ds.label_list == None else len(
        train_ds.label_list)
    model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_train_epochs)
    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    loss_fct = paddle.nn.loss.CrossEntropyLoss(
    ) if train_ds.label_list else paddle.nn.loss.MSELoss()

    metric = metric_class()
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            return_dict = False
            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[1],
                "attention_mask": batch[2],
                "visual_embeds": batch[3],
                "visual_token_type_ids": batch[4],
                "visual_attention_mask": batch[5],
                "labels": batch[6],
                "return_dict": return_dict
            }

            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                outputs = model(**inputs)
                if not return_dict:
                    loss = outputs[0]
                    prediction_logits = outputs[1].cpu().detach().numpy()
                else:
                    loss = outputs['loss']
                    prediction_logits = outputs['prediction_logits'].cpu(
                    ).detach().numpy()
            loss = loss / args.gradient_accumulation_steps

            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                if args.use_amp:
                    scaler.minimize(optimizer, loss)
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

            if global_step % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()

                # evaluate(model, loss_fct, metric, dev_data_loader)
                evaluate_vqa(model, dev_data_loader)
                print("eval done total : %s s" % (time.time() - tic_eval))

                if paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir,
                                              "%s_ft_model_%d.pdparams" %
                                              (args.task_name, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
            if global_step >= num_training_steps:
                return


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_train(args)
