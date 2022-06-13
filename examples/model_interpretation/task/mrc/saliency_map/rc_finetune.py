#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import re
import time
import logging
import json
import sys
from pathlib import Path
from visualdl import LogWriter
import argparse

from paddle.io import DataLoader
from paddlenlp.data import Pad, Stack, Dict
from paddlenlp.transformers.roberta.tokenizer import RobertaTokenizer, RobertaBPETokenizer
import paddle

from squad import DuReaderChecklist
from saliency_map.utils import create_if_not_exists, get_warmup_and_linear_decay
from roberta.modeling import RobertaForQuestionAnswering

sys.path.append('../../..')
from model_interpretation.utils import convert_tokenizer_res_to_old_version

sys.path.remove('../../..')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser('mrc task with roberta')
    parser.add_argument('--from_pretrained',
                        type=str,
                        required=True,
                        help='pretrained model directory or tag')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=128,
                        help='max sentence length, should not greater than 512')
    parser.add_argument(
        '--doc_stride',
        type=int,
        default=128,
        help=
        'When splitting up a long document into chunks, how much stride to take between chunks.'
    )
    parser.add_argument('--bsz', type=int, default=32, help='batchsize')
    parser.add_argument('--epoch', type=int, default=3, help='epoch')
    parser.add_argument('--train_data_dir',
                        type=str,
                        required=True,
                        help='train data file')
    parser.add_argument('--dev_data_dir',
                        type=str,
                        required=True,
                        help='develop data file')
    parser.add_argument(
        '--max_steps',
        type=int,
        required=True,
        help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE')
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--save_dir',
                        type=Path,
                        required=True,
                        help='model output directory')
    parser.add_argument('--init_checkpoint',
                        type=str,
                        default=None,
                        help='checkpoint to warm start from')
    parser.add_argument('--wd',
                        type=float,
                        default=0.01,
                        help='weight decay, aka L2 regularizer')
    parser.add_argument(
        '--use_amp',
        action='store_true',
        help=
        'only activate AMP(auto mixed precision accelatoin) on TensorCore compatible devices'
    )
    parser.add_argument('--language',
                        type=str,
                        required=True,
                        help='language that the model based on')
    args = parser.parse_args()
    return args


def map_fn_DuCheckList_finetune(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    questions = [examples[i]['question'] for i in range(len(examples))]
    contexts = [
        examples[i]['context'] + examples[i]['title']
        for i in range(len(examples))
    ]

    tokenized_examples = tokenizer(questions,
                                   contexts,
                                   stride=args.doc_stride,
                                   max_seq_len=args.max_seq_len)
    tokenized_examples = convert_tokenizer_res_to_old_version(
        tokenized_examples)

    for i, tokenized_example in enumerate(tokenized_examples):

        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_example["input_ids"]  # list(seq)
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']  # list(seq)

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offsets = tokenized_example['offset_mapping']  # list(seq)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']  # int
        if args.language == 'ch':
            answers = examples[sample_index]['answers']  # list
            answer_starts = examples[sample_index]['answer_starts']  # list
        else:
            example = examples[sample_index]
            example['question_len'] = len(example['question'].split())
            example['context_len'] = len(example['context'].split())

            answers = example['answers']  # list
            answer_starts = example['answer_starts']  # list

        # If no answers are given, set the cls_index as answer.
        if len(answer_starts) == 0:
            tokenized_examples[i]["start_positions"] = cls_index
            tokenized_examples[i]["end_positions"] = cls_index
            tokenized_examples[i]['answerable_label'] = 0
        else:
            # Start/end character index of the answer in the text.
            start_char = answer_starts[0]
            end_char = start_char + len(answers[0])
            if args.language == 'en':
                # Start token index of the current span in the text.
                token_start_index = 0
                while not (offsets[token_start_index] == (0, 0)
                           and offsets[token_start_index + 1] == (0, 0)):
                    token_start_index += 1
                token_start_index += 2

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 2
            else:
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 2
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char):
                tokenized_examples[i]["start_positions"] = cls_index
                tokenized_examples[i]["end_positions"] = cls_index
                tokenized_examples[i]['answerable_label'] = 0
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[
                        token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples[i]["start_positions"] = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples[i]["end_positions"] = token_end_index + 1
                tokenized_examples[i]['answerable_label'] = 1

    return tokenized_examples


if __name__ == "__main__":
    args = get_args()

    if args.language == 'ch':
        tokenizer = RobertaTokenizer.from_pretrained(args.from_pretrained)
    else:
        tokenizer = RobertaBPETokenizer.from_pretrained(args.from_pretrained)
    model = RobertaForQuestionAnswering.from_pretrained(args.from_pretrained,
                                                        num_classes=2)

    train_ds = DuReaderChecklist().read(args.train_data_dir)
    dev_ds = DuReaderChecklist().read(args.dev_data_dir)

    train_ds.map(map_fn_DuCheckList_finetune, batched=True)
    dev_ds.map(map_fn_DuCheckList_finetune, batched=True)

    log.debug('train set: %d' % len(train_ds))
    log.debug('dev set: %d' % len(dev_ds))

    train_batch_sampler = paddle.io.DistributedBatchSampler(train_ds,
                                                            batch_size=args.bsz,
                                                            shuffle=True)
    dev_batch_sample = paddle.io.DistributedBatchSampler(dev_ds,
                                                         batch_size=args.bsz,
                                                         shuffle=False)

    batchify_fn = lambda samples, fn=Dict(
        {
            'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),
            'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            'start_positions': Stack(dtype='int64'),
            'end_positions': Stack(dtype='int64'),
            'answerable_label': Stack(dtype='int64')
        }): fn(samples)

    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_sampler=train_batch_sampler,
                                   collate_fn=batchify_fn,
                                   return_list=True)
    dev_data_loader = DataLoader(dataset=dev_ds,
                                 batch_sampler=dev_batch_sample,
                                 collate_fn=batchify_fn,
                                 return_list=True)

    max_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.epoch
    lr_scheduler = paddle.optimizer.lr.LambdaDecay(
        args.lr,
        get_warmup_and_linear_decay(max_steps,
                                    int(args.warmup_proportion * max_steps)))

    param_name_to_exclue_from_weight_decay = re.compile(
        r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')

    opt = paddle.optimizer.AdamW(
        lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.wd,
        apply_decay_param_fun=lambda n:
        not param_name_to_exclue_from_weight_decay.match(n),
        grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0)
        if args.language == 'ch' else None)

    scaler = paddle.amp.GradScaler(enable=args.use_amp)

    with LogWriter(logdir=str(create_if_not_exists(args.save_dir /
                                                   'vdl'))) as log_writer:
        with paddle.amp.auto_cast(enable=args.use_amp):
            max_acc = 0.0
            log.debug('start training...')
            for epoch in range(args.epoch):
                s_time = time.time()
                for step, d in enumerate(train_data_loader, start=1):
                    # input_ids:        paddle.Tensor(bsz, seq)
                    # token_type_ids:   paddle.Tensor(bsz, seq)
                    # start_positions:  paddle.Tensor(bsz)
                    # end_positions:    paddle.Tensor(bsz)
                    # answerable_label:    paddle.Tensor(bsz)
                    input_ids, token_type_ids, start_positions, end_positions, answerable_label = d
                    loss, _, _, _ = model(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          start_pos=start_positions,
                                          end_pos=end_positions,
                                          labels=answerable_label)
                    loss = scaler.scale(loss)
                    loss.backward()
                    scaler.minimize(opt, loss)
                    opt.clear_grad()
                    lr_scheduler.step()

                    if step % 100 == 0:
                        _lr = lr_scheduler.get_lr()
                        time_cost = time.time() - s_time
                        s_time = time.time()
                        if args.use_amp:
                            _l = (loss / scaler._scale).numpy()
                            msg = '[epoch-%d step-%d] train loss %.5f lr %.3e scaling %.3e' % (
                                epoch, step, _l, _lr, scaler._scale.numpy())
                        else:
                            _l = loss.numpy()
                            msg = '[epoch-%d step-%d] train loss %.5f lr %.3e time_cost: %.1fs' % (
                                epoch, step, _l, _lr, time_cost)
                        log.debug(msg)
                        log_writer.add_scalar('loss', _l, step=step)
                        log_writer.add_scalar('lr', _lr, step=step)

                    if step % 1000 == 0:
                        if args.save_dir is not None:
                            paddle.save(model.state_dict(),
                                        os.path.join(args.save_dir, 'ckpt.bin'))
                            log.debug('save model!')

                if args.save_dir is not None:
                    paddle.save(model.state_dict(),
                                os.path.join(args.save_dir, 'ckpt.bin'))
                    log.debug('save model!')
