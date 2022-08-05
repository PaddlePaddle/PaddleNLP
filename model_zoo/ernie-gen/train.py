#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import ast
import time
import argparse
import logging

import paddle
from tqdm import tqdm
import paddle.nn as nn
from paddle.io import DataLoader
from paddlenlp.transformers import ErnieForGeneration, ErnieTokenizer, ErnieTinyTokenizer, BertTokenizer, ElectraTokenizer, RobertaTokenizer, LinearDecayWithWarmup
from paddlenlp.datasets import load_dataset

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import Rouge1, Rouge2
from paddlenlp.utils.log import logger

from encode import convert_example, after_padding
from decode import post_process, beam_search_infilling
from model import StackModel

parser = argparse.ArgumentParser('seq2seq model with ERNIE-GEN')
parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list: " +
    ", ".join(list(ErnieTokenizer.pretrained_init_configuration.keys())))
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help=
    "The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument('--max_encode_len',
                    type=int,
                    default=5,
                    help="The max encoding sentence length")
parser.add_argument('--max_decode_len',
                    type=int,
                    default=5,
                    help="The max decoding sentence length")
parser.add_argument(
    "--batch_size",
    default=8,
    type=int,
    help="Batch size per GPU/CPU for training.",
)
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay",
                    default=0.1,
                    type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon",
                    default=1e-8,
                    type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument(
    "--num_epochs",
    default=3,
    type=int,
    help="Total number of training epochs to perform.",
)
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Linear warmup proportion.")
parser.add_argument("--logging_steps",
                    type=int,
                    default=1,
                    help="Log every X updates steps.")
parser.add_argument("--save_steps",
                    type=int,
                    default=100,
                    help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--device",
    default="gpu",
    type=str,
    choices=["cpu", "gpu", "xpu"],
    help="The device to select to train the model, is must be cpu/gpu/xpu.")
parser.add_argument('--beam_width',
                    type=int,
                    default=1,
                    help="Beam search width.")
parser.add_argument('--noise_prob',
                    type=float,
                    default=0.,
                    help='Probability of token be repalced.')
parser.add_argument(
    '--use_random_noice',
    action='store_true',
    help=
    'If set, replace target tokens with random token from vocabulary, else replace with `[NOISE]`.'
)
parser.add_argument('--label_smooth',
                    type=float,
                    default=0.,
                    help="The soft label smooth rate.")
parser.add_argument('--length_penalty',
                    type=float,
                    default=1.0,
                    help="The length penalty during decoding.")
parser.add_argument('--init_checkpoint',
                    type=str,
                    default=None,
                    help='Checkpoint to warm start from.')
parser.add_argument('--save_dir',
                    type=str,
                    default=None,
                    help='Model output directory.')
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help=
    "If > 0: set total number of training steps to perform. Override num_train_epochs."
)

args = parser.parse_args()


def evaluate(model, data_loader, tokenizer, rouge1, rouge2, attn_id,
             tgt_type_id, args):
    model.eval()

    vocab = tokenizer.vocab
    eos_id = vocab[tokenizer.sep_token]
    sos_id = vocab[tokenizer.cls_token]
    pad_id = vocab[tokenizer.pad_token]
    unk_id = vocab[tokenizer.unk_token]
    vocab_size = len(vocab)
    evaluated_sentences_ids = []
    reference_sentences_ids = []
    logger.info("Evaluating...")
    for data in tqdm(data_loader):
        (src_ids, src_tids, src_pids, _, _, _, _, _, _, _, _,
         raw_tgt_labels) = data  # never use target when infer
        # Use greedy_search_infilling or beam_search_infilling to get predictions
        output_ids = beam_search_infilling(model,
                                           src_ids,
                                           src_tids,
                                           eos_id=eos_id,
                                           sos_id=sos_id,
                                           attn_id=attn_id,
                                           pad_id=pad_id,
                                           unk_id=unk_id,
                                           vocab_size=vocab_size,
                                           max_decode_len=args.max_decode_len,
                                           max_encode_len=args.max_encode_len,
                                           beam_width=args.beam_width,
                                           length_penalty=args.length_penalty,
                                           tgt_type_id=tgt_type_id)

        for ids in output_ids.tolist():
            if eos_id in ids:
                ids = ids[:ids.index(eos_id)]
            evaluated_sentences_ids.append(ids)

        for ids in raw_tgt_labels.numpy().tolist():
            ids = ids[:ids.index(eos_id)]
            reference_sentences_ids.append(ids)

    score1 = rouge1.score(evaluated_sentences_ids, reference_sentences_ids)
    score2 = rouge2.score(evaluated_sentences_ids, reference_sentences_ids)

    logger.info("Rouge-1: %.5f ,Rouge-2: %.5f" % (score1 * 100, score2 * 100))

    evaluated_sentences = []
    reference_sentences = []
    for ids in reference_sentences_ids[:5]:
        reference_sentences.append(''.join(
            map(post_process, vocab.to_tokens(ids))))
    for ids in evaluated_sentences_ids[:5]:
        evaluated_sentences.append(''.join(
            map(post_process, vocab.to_tokens(ids))))
    logger.debug(reference_sentences)
    logger.debug(evaluated_sentences)

    model.train()


def train():
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    model = ErnieForGeneration.from_pretrained(args.model_name_or_path)
    if "ernie-tiny" in args.model_name_or_path:
        tokenizer = ErnieTinyTokenizer.from_pretrained(args.model_name_or_path)
    elif "ernie" in args.model_name_or_path:
        tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)
    elif "roberta" in args.model_name_or_path or "rbt" in args.model_name_or_path:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    elif "electra" in args.model_name_or_path:
        tokenizer = ElectraTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    if args.init_checkpoint:
        model_state = paddle.load(args.init_checkpoint)
        model.set_state_dict(model_state)

    train_dataset, dev_dataset = load_dataset('poetry',
                                              splits=('train', 'dev'),
                                              lazy=False)
    attn_id = tokenizer.vocab[
        '[ATTN]'] if '[ATTN]' in tokenizer.vocab else tokenizer.vocab['[MASK]']
    tgt_type_id = model.sent_emb.weight.shape[0] - 1

    trans_func = convert_example(tokenizer=tokenizer,
                                 attn_id=attn_id,
                                 tgt_type_id=tgt_type_id,
                                 max_encode_len=args.max_encode_len,
                                 max_decode_len=args.max_decode_len,
                                 noise_prob=args.noise_prob,
                                 use_random_noice=args.use_random_noice)

    train_dataset = train_dataset.map(trans_func)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_pids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # src_tids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_pids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # tgt_tids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # attn_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # tgt_labels
    ): after_padding(fn(samples))
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_sampler=train_batch_sampler,
                                   collate_fn=batchify_fn,
                                   num_workers=0,
                                   return_list=True)

    dev_dataset = dev_dataset.map(trans_func)
    dev_data_loader = DataLoader(dataset=dev_dataset,
                                 batch_size=args.batch_size,
                                 collate_fn=batchify_fn,
                                 num_workers=0,
                                 return_list=True)

    label_num = model.word_emb.weight.shape[0]
    train_model = StackModel(model)
    if paddle.distributed.get_world_size() > 1:
        # All 'forward' outputs derived from the module parameters using in DataParallel
        # must participate in the calculation of losses and subsequent gradient calculations.
        # So we use StackModel here to make the model only output loss in its 'forward' function.
        train_model = paddle.DataParallel(train_model)

    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.num_train_epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

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
        grad_clip=nn.ClipGradByGlobalNorm(1.0),
        apply_decay_param_fun=lambda x: x in decay_params)

    rouge1 = Rouge1()
    rouge2 = Rouge2()

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            (src_ids, src_tids, src_pids, tgt_ids, tgt_tids, tgt_pids, attn_ids,
             mask_src_2_src, mask_tgt_2_srctgt, mask_attn_2_srctgtattn,
             tgt_labels, _) = batch
            # import pdb; pdb.set_trace()
            if args.label_smooth > 0.:
                tgt_labels = nn.functional.label_smooth(
                    nn.functional.one_hot(tgt_labels, label_num),
                    epsilon=args.label_smooth)
            tgt_pos = paddle.nonzero(attn_ids == attn_id)
            loss = train_model(src_ids, src_tids, src_pids, tgt_ids, tgt_tids,
                               tgt_pids, attn_ids, mask_src_2_src,
                               mask_tgt_2_srctgt, mask_attn_2_srctgtattn,
                               tgt_labels, tgt_pos)
            if global_step % args.logging_steps == 0:
                if paddle.distributed.get_rank() == 0:
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s, lr: %.3e"
                        % (global_step, epoch, step, loss, args.logging_steps /
                           (time.time() - tic_train), lr_scheduler.get_lr()))
                tic_train = time.time()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % args.save_steps == 0 or global_step == num_training_steps and paddle.distributed.get_rank(
            ) == 0:
                evaluate(model, dev_data_loader, tokenizer, rouge1, rouge2,
                         attn_id, tgt_type_id, args)
                output_dir = os.path.join(args.output_dir,
                                          "model_%d" % global_step)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            if global_step >= num_training_steps:
                return


if __name__ == "__main__":
    train()
