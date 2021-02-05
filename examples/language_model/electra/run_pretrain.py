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
import collections
import itertools
import logging
import os
import io
import random
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, Dataset

from paddlenlp.transformers import ElectraForTotalPretraining, ElectraModel, ElectraPretrainingCriterion
from paddlenlp.transformers import ElectraDiscriminator, ElectraGenerator
from paddlenlp.transformers import ElectraTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {"electra": (ElectraForTotalPretraining, ElectraTokenizer), }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default="electra",
        type=str,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default="electra-small",
        type=str,
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
        help="max length of each sequence")
    parser.add_argument(
        "--mask_prob",
        default=0.15,
        type=float,
        help="the probability of one word to be mask")
    parser.add_argument(
        "--train_batch_size",
        default=96,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--eval_batch_size",
        default=96,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-4,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--num_train_epochs",
        default=4,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=10000,
        type=int,
        help="Linear warmup over warmup_steps.")

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--init_from_ckpt",
        type=bool,
        default=False,
        help="Whether to load model checkpoint. if True, args.model_name_or_path must be dir store ckpt"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--eager_run", type=bool, default=True, help="Use dygraph mode.")
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed + paddle.distributed.get_rank())
    np.random.seed(args.seed + paddle.distributed.get_rank())
    paddle.seed(args.seed + paddle.distributed.get_rank())


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


class BookCorpus(paddle.io.Dataset):
    """
    https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html
    Args:
        data_path (:obj:`str`) : The dataset file path, which contains train.tsv, dev.tsv and test.tsv.
        tokenizer (:obj:`class PretrainedTokenizer`) : The tokenizer to split word and convert word to id.
        max_seq_length (:obj:`int`) : max length for each sequence.
        mode (:obj:`str`, `optional`, defaults to `train`):
            It identifies the dataset mode (train, test or dev).
    """

    def __init__(
            self,
            data_path,
            tokenizer,
            max_seq_length,
            mode='train', ):
        if mode == 'train':
            data_file = 'train.data'
        elif mode == 'test':
            data_file = 'test.data'
        else:
            data_file = 'dev.data'

        self.data_file = os.path.join(data_path, data_file)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.raw_examples = self._read_file(self.data_file)

    def _read_file(self, input_file):
        """
        Reads a text file.

        Args:
            input_file (:obj:`str`) : The file to be read.

        Returns:
            examples (:obj:`list`): All the input data.
        """
        if not os.path.exists(input_file):
            raise RuntimeError("The file {} is not found.".format(input_file))
        else:
            with io.open(input_file, "r", encoding="UTF-8") as f:
                examples = []
                for line in f.read().splitlines():
                    if (len(line) > 0 and not line.isspace()):
                        tokens = self.tokenizer(line)
                        ids = self.tokenizer.convert_tokens_to_ids(tokens)
                        example = self.truncation_ids(ids, self.max_seq_length)
                        examples.append(example)
                return examples

    def truncation_ids(self, ids, max_seq_length):
        if len(ids) <= (max_seq_length - 2):
            return ids
        else:
            return ids[:(max_seq_length - 2)]

    def __getitem__(self, idx):
        return self.raw_examples[idx]

    def __len__(self):
        return len(self.raw_examples)


class DataCollatorForElectra(object):
    """
    pads, gets batch of tensors and preprocesses batches for masked language modeling
    when dataloader num_worker > 0, this collator may trigger some bugs, for safe, be sure dataloader num_worker=0
    """

    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 mlm=True,
                 mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm = True
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        if self.mlm:
            inputs, raw_inputs, labels = self.mask_tokens(examples)
            return inputs, raw_inputs, labels
        else:
            raw_inputs, _ = self.add_special_tokens_and_set_maskprob(
                examples, True, self.max_seq_length)
            raw_inputs = self.tensorize_batch(raw_inputs, "int64")
            inputs = raw_inputs.clone().detach()
            labels = raw_inputs.clone().detach()
            if self.tokenizer.pad_token is not None:
                pad_token_id = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.pad_token)
                labels[labels == pad_token_id] = -100
            return batch, raw_inputs, labels

    def tensorize_batch(self, examples, dtype):
        if isinstance(examples[0], (list, tuple)):
            examples = [paddle.to_tensor(e, dtype=dtype) for e in examples]
        length_of_first = examples[0].shape[0]
        are_tensors_same_length = all(x.shape[0] == length_of_first
                                      for x in examples)
        if are_tensors_same_length:
            return paddle.stack(examples, axis=0)
        else:
            raise ValueError(
                "the tensor in examples not have same shape, please check input examples"
            )

    def add_special_tokens_and_set_maskprob(self, inputs, truncation,
                                            max_seq_length):
        sep_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.sep_token)
        pad_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.pad_token)
        cls_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.cls_token)
        full_inputs = []
        full_maskprob = []
        max_length = 0
        for ids in inputs:
            if len(ids) > max_length:
                max_length = len(ids)
        max_length = min(max_length + 2, max_seq_length)

        for ids in inputs:
            if len(ids) <= (max_length - 2):
                padding_num = max_length - len(ids) - 2
                full_inputs.append([cls_token_id] + ids + [sep_token_id] + (
                    [pad_token_id] * padding_num))
                full_maskprob.append([0] + ([self.mlm_probability] * len(ids)) +
                                     [0] + ([0] * padding_num))
            else:
                if truncation:
                    full_inputs.append([cls_token_id] + ids[:(max_length - 2)] +
                                       [sep_token_id])
                    full_maskprob.append([0] + ([self.mlm_probability] * (
                        max_length - 2)) + [0])
                else:
                    full_inputs.append([cls_token_id] + ids + [sep_token_id])
                    full_maskprob.append([0] + ([self.mlm_probability] * len(
                        ids)) + [0])
        return full_inputs, full_maskprob

    def mask_tokens(self, examples):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "the tokenizer does not have mask_token, please check!")
        mask_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        raw_inputs, probability_matrix = self.add_special_tokens_and_set_maskprob(
            examples, True, self.max_seq_length)
        raw_inputs = self.tensorize_batch(raw_inputs, "int64")
        probability_matrix = self.tensorize_batch(probability_matrix, "float32")
        inputs = raw_inputs.clone()
        labels = raw_inputs.clone()

        total_indices = paddle.bernoulli(probability_matrix).astype("bool")
        unuse_labels = paddle.full(labels.shape, -100).astype("int64")
        labels = paddle.where(total_indices, labels, unuse_labels)

        # 80% MASK
        indices_mask = paddle.bernoulli(paddle.full(labels.shape, 0.8)).astype(
            "bool").logical_and(total_indices)
        masked_inputs = paddle.full(inputs.shape, mask_token_id).astype("int64")
        inputs = paddle.where(indices_mask, masked_inputs, inputs)

        # 10% Random
        indices_random = paddle.bernoulli(paddle.full(
            labels.shape, 0.5)).astype("bool").logical_and(
                total_indices).logical_and(indices_mask.logical_not())
        random_words = paddle.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            shape=labels.shape,
            dtype="int64")
        inputs = paddle.where(indices_random, random_words, inputs)

        # 10% Original
        return inputs, raw_inputs, labels


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      use_gpu=True,
                      data_collator=None):
    """
    Creats dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`):
            Dataset instance.
        mode(obj:`str`, optional, defaults to obj:`train`):
            If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): 
            The sample number of a mini-batch.
        use_gpu(obj:`bool`, optional, defaults to obj:`True`):
            Whether to use gpu to run.

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """

    #print("%s.data has batch_size : %s" % (mode, batch_size))
    if mode == 'train' and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=True)
        dataloader = paddle.io.DataLoader(
            dataset,
            batch_sampler=sampler,
            return_list=True,
            collate_fn=data_collator,
            num_workers=0)
    else:
        shuffle = True if mode == 'train' else False
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        dataloader = paddle.io.DataLoader(
            dataset,
            batch_sampler=sampler,
            return_list=True,
            collate_fn=data_collator,
            num_workers=0)

    return dataloader


def do_train(args):
    paddle.enable_static() if not args.eager_run else None
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    worker_init = WorkerInitObj(args.seed + paddle.distributed.get_rank())

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Loads or initializes a model.
    pretrained_models = list(tokenizer_class.pretrained_init_configuration.keys(
    ))
    if args.model_name_or_path in pretrained_models:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        generator = ElectraGenerator(
            ElectraModel(**model_class.pretrained_init_configuration[
                args.model_name_or_path + "-generator"]))
        discriminator = ElectraDiscriminator(
            ElectraModel(**model_class.pretrained_init_configuration[
                args.model_name_or_path + "-discriminator"]))
        model = model_class(generator, discriminator)
    else:
        if os.path.isdir(args.model_name_or_path) and args.init_from_ckpt:
            # load checkpoint
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
            for file_id, file_name in model_class.resource_files_names.items():
                full_file_name = os.path.join(args.model_name_or_path,
                                              file_name)
                # to be write : load model ckpt file
        else:
            raise ValueError("initialize a model need identifier or the "
                             "path to a directory instead. The supported model "
                             "identifiers are as follows: {}".format(
                                 model_class.pretrained_init_configuration.keys(
                                 )))

    criterion = ElectraPretrainingCriterion(
        getattr(model.generator,
                ElectraGenerator.base_model_prefix).config["vocab_size"],
        model.gen_weight, model.disc_weight)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # Loads dataset.
    tic_load_data = time.time()
    print("start load data : %s" %
          (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    train_dataset = BookCorpus(
        data_path=args.input_dir,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        mode='train')
    print("load data done, total : %s s" % (time.time() - tic_load_data))

    # Reads data and generates mini-batches.
    data_collator = DataCollatorForElectra(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        mlm=True,
        mlm_probability=args.mask_prob)

    train_data_loader = create_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        mode='train',
        use_gpu=True if args.n_gpu else False,
        data_collator=data_collator)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_train_epochs)

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_steps)

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=clip,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    print("start train : %s" %
          (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, raw_input_ids, gen_labels = batch
            gen_logits, disc_logits, disc_labels = model(
                input_ids=input_ids,
                raw_input_ids=raw_input_ids,
                gen_labels=gen_labels)
            loss = criterion(gen_logits, disc_logits, gen_labels, disc_labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_gradients()
            #print("backward done, total %s s" % (time.time() - tic_train))
            #tic_train = time.time()
            if global_step % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if global_step % args.save_steps == 0:
                if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir,
                                              "model_%d.pdparams" % global_step)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    #model_to_save.save_pretrained(output_dir)
                    paddle.save(model.state_dict(),
                                os.path.join(output_dir,
                                             "model_state.pdparams"))
                    tokenizer.save_pretrained(output_dir)
                    paddle.save(optimizer.state_dict(),
                                os.path.join(output_dir, "model_state.pdopt"))


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
