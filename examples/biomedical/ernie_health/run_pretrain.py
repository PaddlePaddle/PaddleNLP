# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import json
import copy
from collections import defaultdict
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, Dataset

from paddlenlp.transformers import ErnieHealthForTotalPretraining, ElectraModel, ErnieHealthPretrainingCriterion
from paddlenlp.transformers import ErnieHealthDiscriminator, ElectraGenerator
from paddlenlp.transformers import ElectraTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup

from utils import PreTokenizer

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "ernie-health": (ErnieHealthForTotalPretraining, ElectraTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        default="ernie-health",
        type=str,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument(
        "--model_name_or_path",
        default="ernie-health-chinese",
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
        default=512,
        type=int,
        help="max length of each sequence")
    parser.add_argument(
        "--train_batch_size",
        default=256,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--eval_batch_size",
        default=256,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=2e-4,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--num_train_epochs",
        default=100,
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
        default=10000,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--init_from_ckpt",
        action="store_true",
        help="Whether to load model checkpoint. if True, args.model_name_or_path must be dir store ckpt or will train from fresh start"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Whether to use float16(Automatic Mixed Precision) to train.")
    parser.add_argument(
        "--eager_run", type=bool, default=True, help="Use dygraph mode.")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu"],
        help="The device to select to train the model, is must be cpu/gpu.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    return args


def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(seed)


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


class MedicalCorpus(paddle.io.Dataset):
    def __init__(self, data_path, tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer
        # Add ids for suffixal chinese tokens in tokenized text, e.g. '##度' in '百度'.
        # It should coincide with the vocab dictionary in preprocess.py.
        suffix_vocab = {}
        for idx, token in enumerate(range(0x4E00, 0x9FA6)):
            suffix_vocab[len(self.tokenizer) + idx] = '##' + chr(token)
        self.tokenizer.added_tokens_decoder.update(suffix_vocab)
        self._samples, self._global_index = self._read_data_files(data_path)

    def _get_data_files(self, data_path):
        # Get all prefix of .npy/.npz files in the current and next-level directories.
        files = [
            os.path.join(data_path, f) for f in os.listdir(data_path)
            if (os.path.isfile(os.path.join(data_path, f)) and "_idx.npz" in
                str(f))
        ]
        files = [x.replace("_idx.npz", "") for x in files]
        return files

    def _read_data_files(self, data_path):
        data_files = self._get_data_files(data_path)
        samples = []
        indexes = []
        for file_id, file_name in enumerate(data_files):

            for suffix in ["_ids.npy", "_idx.npz"]:
                if not os.path.isfile(file_name + suffix):
                    raise ValueError("File Not found, %s" %
                                     (file_name + suffix))

            token_ids = np.load(
                file_name + "_ids.npy", mmap_mode="r", allow_pickle=True)
            samples.append(token_ids)

            split_ids = np.load(file_name + "_idx.npz")
            end_ids = np.cumsum(split_ids["lens"], dtype=np.int64)
            file_ids = np.full(end_ids.shape, file_id)
            split_ids = np.stack([file_ids, end_ids], axis=-1)
            indexes.extend(split_ids)
        indexes = np.stack(indexes, axis=0)
        return samples, indexes

    def __len__(self):
        return len(self._global_index)

    def __getitem__(self, index):
        file_id, end_id = self._global_index[index]
        start_id = 0
        if index > 0:
            pre_file_id, pre_end_id = self._global_index[index - 1]
            if pre_file_id == file_id:
                start_id = pre_end_id
        word_token_ids = self._samples[file_id][start_id:end_id]
        token_ids = []
        is_suffix = np.zeros(word_token_ids.shape)
        for idx, token_id in enumerate(word_token_ids):
            token = self.tokenizer.convert_ids_to_tokens(int(token_id))
            if '##' in token:
                token_id = self.tokenizer.convert_tokens_to_ids(token[-1])
                is_suffix[idx] = 1
            token_ids.append(token_id)

        return token_ids, is_suffix.astype(np.int64)


class DataCollatorForErnieHealth(object):
    def __init__(self, tokenizer, mlm_prob, max_seq_length):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.max_seq_len = max_seq_length
        self._ids = {
            'cls':
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token),
            'sep':
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token),
            'pad':
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
            'mask':
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        }

    def __call__(self, data):
        masked_input_ids_a, input_ids_a, labels_a = self.mask_tokens(data)
        masked_input_ids_b, input_ids_b, labels_b = self.mask_tokens(data)
        masked_input_ids = paddle.concat(
            [masked_input_ids_a, masked_input_ids_b], axis=0).astype('int64')
        input_ids = paddle.concat([input_ids_a, input_ids_b], axis=0)
        labels = paddle.concat([labels_a, labels_b], axis=0)
        return masked_input_ids, input_ids, labels

    def mask_tokens(self, batch_data):

        token_ids = [x[0] for x in batch_data]
        is_suffix = [x[1] for x in batch_data]

        # Create probability matrix where the probability of real tokens is
        # self.mlm_prob, while that of others is zero.
        data = self.add_special_tokens_and_set_maskprob(token_ids, is_suffix)
        token_ids, is_suffix, prob_matrix = data
        token_ids = paddle.to_tensor(
            token_ids, dtype="int64", stop_gradient=True)
        masked_token_ids = token_ids.clone()
        labels = token_ids.clone()

        # Create masks for words, where '百' must be masked if '度' is masked
        # for the word '百度'.
        prob_matrix = prob_matrix * (1 - is_suffix)
        word_mask_index = np.random.binomial(1, prob_matrix).astype("float")
        is_suffix_mask = (is_suffix == 1)
        word_mask_index_tmp = word_mask_index
        while word_mask_index_tmp.sum() > 0:
            word_mask_index_tmp = np.concatenate(
                [
                    np.zeros((word_mask_index.shape[0], 1)),
                    word_mask_index_tmp[:, :-1]
                ],
                axis=1)
            word_mask_index_tmp = word_mask_index_tmp * is_suffix_mask
            word_mask_index += word_mask_index_tmp
        word_mask_index = word_mask_index.astype('bool')
        labels[~word_mask_index] = -100

        # 80% replaced with [MASK].
        token_mask_index = paddle.bernoulli(paddle.full(
            labels.shape, 0.8)).astype('bool').numpy() & word_mask_index
        masked_token_ids[token_mask_index] = self._ids['mask']

        # 10% replaced with random token ids.
        token_random_index = paddle.to_tensor(
            paddle.bernoulli(paddle.full(labels.shape, 0.5)).astype("bool")
            .numpy() & word_mask_index & ~token_mask_index)
        random_tokens = paddle.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            shape=labels.shape,
            dtype='int64')
        masked_token_ids = paddle.where(token_random_index, random_tokens,
                                        masked_token_ids)

        return masked_token_ids, token_ids, labels

    def add_special_tokens_and_set_maskprob(self, token_ids, is_suffix):
        batch_size = len(token_ids)
        batch_token_ids = np.full((batch_size, self.max_seq_len),
                                  self._ids['pad'])
        batch_token_ids[:, 0] = self._ids['cls']
        batch_is_suffix = np.full_like(batch_token_ids, -1)
        prob_matrix = np.zeros_like(batch_token_ids, dtype='float32')

        for idx in range(batch_size):
            if len(token_ids[idx]) > self.max_seq_len - 2:
                token_ids[idx] = token_ids[idx][:self.max_seq_len - 2]
                is_suffix[idx] = is_suffix[idx][:self.max_seq_len - 2]
            seq_len = len(token_ids[idx])
            batch_token_ids[idx, seq_len + 1] = self._ids['sep']
            batch_token_ids[idx, 1:seq_len + 1] = token_ids[idx]
            batch_is_suffix[idx, 1:seq_len + 1] = is_suffix[idx]
            prob_matrix[idx, 1:seq_len + 1] = self.mlm_prob

        return batch_token_ids, batch_is_suffix, prob_matrix


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

    if mode == 'train' and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=False)  #True)
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
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)
    worker_init = WorkerInitObj(args.seed + paddle.distributed.get_rank())

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Loads or initialize a model.
    pretrained_models = list(tokenizer_class.pretrained_init_configuration.keys(
    ))

    if args.model_name_or_path in pretrained_models:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        generator = ElectraGenerator(
            ElectraModel(**model_class.pretrained_init_configuration[
                args.model_name_or_path + "-generator"]))
        discriminator = ErnieHealthDiscriminator(
            ElectraModel(**model_class.pretrained_init_configuration[
                args.model_name_or_path + "-discriminator"]))
        model = model_class(generator, discriminator)
        args.init_from_ckpt = False
    else:
        if os.path.isdir(args.model_name_or_path) and args.init_from_ckpt:
            # Load checkpoint
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
            with open(
                    os.path.join(args.model_name_or_path, "run_states.json"),
                    'r') as f:
                config_dict = json.load(f)
                model_name = config_dict["model_name"]
            if model_name in pretrained_models:
                generator = ElectraGenerator(
                    ElectraModel(**model_class.pretrained_init_configuration[
                        model_name + "-generator"]))
                discriminator = ErnieHealthDiscriminator(
                    ElectraModel(**model_class.pretrained_init_configuration[
                        model_name + "-discriminator"]))
                model = model_class(generator, discriminator)
                model.set_state_dict(
                    paddle.load(
                        os.path.join(args.model_name_or_path,
                                     "model_state.pdparams")))
            else:
                raise ValueError(
                    "initialize a model from ckpt need model_name "
                    "in model_config_file. The supported model_name "
                    "are as follows: {}".format(
                        tokenizer_class.pretrained_init_configuration.keys()))
        else:
            raise ValueError(
                "initialize a model need identifier or the "
                "directory of storing model. if use identifier, the supported model "
                "identifiers are as follows: {}, if use directory, "
                "make sure set init_from_ckpt as True".format(
                    model_class.pretrained_init_configuration.keys()))

    criterion = ErnieHealthPretrainingCriterion(
        getattr(model.generator,
                ElectraGenerator.base_model_prefix).config["vocab_size"],
        model.gen_weight)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # Loads dataset.
    tic_load_data = time.time()
    print("start load data : %s" %
          (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    train_dataset = MedicalCorpus(data_path=args.input_dir, tokenizer=tokenizer)
    print("load data done, total : %s s" % (time.time() - tic_load_data))

    # Reads data and generates mini-batches.
    data_collator = DataCollatorForErnieHealth(
        tokenizer=tokenizer, max_seq_length=args.max_seq_length, mlm_prob=0.15)

    train_data_loader = create_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        mode='train',
        use_gpu=True if args.device in "gpu" else False,
        data_collator=data_collator)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_train_epochs)

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_steps)

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)

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
        grad_clip=clip,
        apply_decay_param_fun=lambda x: x in decay_params)
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    print("start train : %s" %
          (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    trained_global_step = global_step = 0
    t_loss = defaultdict(lambda: paddle.to_tensor([0.0]))
    log_loss = defaultdict(lambda: paddle.to_tensor([0.0]))
    loss_list = defaultdict(list)
    log_list = []
    tic_train = time.time()

    if os.path.isdir(args.model_name_or_path) and args.init_from_ckpt:
        optimizer.set_state_dict(
            paddle.load(
                os.path.join(args.model_name_or_path, "model_state.pdopt")))
        trained_global_step = global_step = config_dict["global_step"]
        if trained_global_step < num_training_steps:
            print(
                "[ start train from checkpoint ] we have already trained %s steps, seeking next step : %s"
                % (trained_global_step, trained_global_step + 1))
        else:
            print(
                "[ start train from checkpoint ] we have already trained %s steps, but total training steps is %s, please check configuration !"
                % (trained_global_step, num_training_steps))
            exit(0)

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            if trained_global_step > 0:
                trained_global_step -= 1
                continue
            global_step += 1
            masked_input_ids, input_ids, gen_labels = batch

            if args.use_amp:
                with paddle.amp.auto_cast():
                    gen_logits, logits_rtd, logits_mts, logits_csp, disc_labels, masks = model(
                        input_ids=masked_input_ids,
                        raw_input_ids=input_ids,
                        gen_labels=gen_labels)
                    loss, gen_loss, rtd_loss, mts_loss, csp_loss = criterion(
                        gen_logits, gen_labels, logits_rtd, logits_mts,
                        logits_csp, disc_labels, masks)

                scaled = scaler.scale(loss)
                scaled.backward()
                t_loss['loss'] += loss.detach()
                t_loss['gen'] += gen_loss.detach()
                t_loss['rtd'] += rtd_loss.detach()
                t_loss['mts'] += mts_loss.detach()
                t_loss['csp'] += csp_loss.detach()
                scaler.minimize(optimizer, scaled)
            else:
                gen_logits, disc_labels, logits_rtd, logits_mts, logits_csp, masks = model(
                    input_ids=masked_input_ids,
                    raw_input_ids=input_ids,
                    gen_labels=gen_labels)
                loss, gen_loss, rtd_loss, mts_loss, csp_loss = criterion(
                    gen_logits, gen_labels, logits_rtd, logits_mts, logits_csp,
                    disc_labels, masks)
                loss.backward()
                t_loss['loss'] += loss.detach()
                t_loss['gen'] += gen_loss.detach()
                t_loss['rtd'] += rtd_loss.detach()
                t_loss['mts'] += mts_loss.detach()
                t_loss['csp'] += csp_loss.detach()
                optimizer.step()

            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % args.logging_steps == 0:
                local_loss = dict(
                    [(k, (t_loss[k] - log_loss[k]) / args.logging_steps)
                     for k in ['loss', 'gen', 'rtd', 'mts', 'csp']])
                if paddle.distributed.get_world_size() > 1:
                    for k in ['loss', 'gen', 'rtd', 'mts', 'csp']:
                        paddle.distributed.all_gather(loss_list[k],
                                                      local_loss[k])
                    if paddle.distributed.get_rank() == 0:
                        tmp_loss = dict(
                            [(k, float((paddle.stack(loss_list[k]).sum() / len(
                                loss_list[k])).numpy()))
                             for k in ['loss', 'gen', 'rtd', 'mts', 'csp']])
                        log_str = (
                            "global step {0:d}/{1:d}, epoch: {2:d}, batch: {3:d}, "
                            "avg_loss: {4:.15f}, generator: {5:.15f}, rtd: {6:.15f}, multi_choice: {7:.15f}, "
                            "seq_contrastive: {8:.15f}, lr: {9:.10f}, speed: {10:.2f} s/it"
                        ).format(global_step, num_training_steps, epoch, step,
                                 tmp_loss['loss'], tmp_loss['gen'],
                                 tmp_loss['rtd'], tmp_loss['mts'],
                                 tmp_loss['csp'],
                                 optimizer.get_lr(),
                                 (time.time() - tic_train) / args.logging_steps)
                        print(log_str)
                        log_list.append(log_str)
                    loss_list = defaultdict(list)
                else:
                    local_loss = dict(
                        [(k, v.numpy()[0]) for k, v in local_loss.items()])
                    log_str = (
                        "global step {0:d}/{1:d}, epoch: {2:d}, batch: {3:d}, "
                        "avg_loss: {4:.15f}, generator: {5:.15f}, rtd: {6:.15f}, multi_choice: {7:.15f}, "
                        "seq_contrastive_loss: {8:.15f}, lr: {9:.10f}, speed: {10:.2f} s/it"
                    ).format(global_step, num_training_steps, epoch, step,
                             local_loss['loss'], local_loss['gen'],
                             local_loss['rtd'], local_loss['mts'],
                             local_loss['csp'],
                             optimizer.get_lr(),
                             (time.time() - tic_train) / args.logging_steps)
                    print(log_str)
                    log_list.append(log_str)
                log_loss = dict(t_loss)
                tic_train = time.time()

            if global_step % args.save_steps == 0:
                if paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir,
                                              "model_%d.pdparams" % global_step)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    config_to_save = copy.deepcopy(
                        model_to_save.discriminator.electra.config)
                    if 'self' in config_to_save:
                        del config_to_save['self']
                    run_states = {
                        "model_name": model_name
                        if args.init_from_ckpt else args.model_name_or_path,
                        "global_step": global_step,
                        "epoch": epoch,
                        "step": step,
                    }
                    with open(
                            os.path.join(output_dir, "model_config.json"),
                            'w') as f:
                        json.dump(config_to_save, f)
                    with open(
                            os.path.join(output_dir, "run_states.json"),
                            'w') as f:
                        json.dump(run_states, f)
                    paddle.save(model.state_dict(),
                                os.path.join(output_dir,
                                             "model_state.pdparams"))
                    tokenizer.save_pretrained(output_dir)
                    paddle.save(optimizer.state_dict(),
                                os.path.join(output_dir, "model_state.pdopt"))
                    if len(log_list) > 0:
                        with open(os.path.join(output_dir, "train.log"),
                                  'w') as f:
                            for log in log_list:
                                if len(log.strip()) > 0:
                                    f.write(log.strip() + '\n')
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
