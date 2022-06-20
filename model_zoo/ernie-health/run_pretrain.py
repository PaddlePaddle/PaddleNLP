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
import os
import io
import random
import time
import json
import copy
from collections import defaultdict

import numpy as np
import paddle
import paddle.distributed as dist
from paddlenlp.transformers import ErnieHealthForTotalPretraining, ElectraModel
from paddlenlp.transformers import ErnieHealthDiscriminator, ElectraGenerator
from paddlenlp.transformers import ElectraTokenizer, ErnieHealthPretrainingCriterion
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from visualdl import LogWriter

from dataset import MedicalCorpus, DataCollatorForErnieHealth, create_dataloader

MODEL_CLASSES = {
    'ernie-health': (ErnieHealthForTotalPretraining, ElectraTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name_or_path',
        default='ernie-health-chinese',
        type=str,
        help='Path to pre-trained model or shortcut name selected in the list: '
        + ', '.join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])),
    )
    parser.add_argument(
        '--input_dir',
        default=None,
        type=str,
        required=True,
        help='The input directory where the data will be read from.',
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        type=str,
        required=True,
        help=
        'The output directory where the model predictions and checkpoints will be written.',
    )
    parser.add_argument('--max_seq_length',
                        default=512,
                        type=int,
                        help='The max length of each sequence')
    parser.add_argument(
        '--mlm_prob',
        default=0.15,
        type=float,
        help='The probability of tokens to be sampled as masks.')
    parser.add_argument(
        '--batch_size',
        default=256,
        type=int,
        help='Batch size per GPU/CPU for training.',
    )
    parser.add_argument('--learning_rate',
                        default=2e-4,
                        type=float,
                        help='The initial learning rate for Adam.')
    parser.add_argument('--weight_decay',
                        default=0.01,
                        type=float,
                        help='Weight decay if we apply some.')
    parser.add_argument('--adam_epsilon',
                        default=1e-8,
                        type=float,
                        help='Epsilon for Adam optimizer.')
    parser.add_argument(
        '--num_epochs',
        default=100,
        type=int,
        help='Total number of training epochs to perform.',
    )
    parser.add_argument(
        '--max_steps',
        default=-1,
        type=int,
        help=
        'If > 0: set total number of training steps to perform. Override num_epochs.',
    )
    parser.add_argument('--warmup_steps',
                        default=10000,
                        type=int,
                        help='Linear warmup over warmup_steps.')
    parser.add_argument('--logging_steps',
                        type=int,
                        default=100,
                        help='Log every X updates steps.')
    parser.add_argument('--save_steps',
                        type=int,
                        default=10000,
                        help='Save checkpoint every X updates steps.')
    parser.add_argument(
        '--init_from_ckpt',
        action='store_true',
        help=
        'Whether to load model checkpoint. if True, args.model_name_or_path must be dir store ckpt or will train from fresh start'
    )
    parser.add_argument(
        '--use_amp',
        action='store_true',
        help='Whether to use float16(Automatic Mixed Precision) to train.')
    parser.add_argument('--eager_run',
                        type=bool,
                        default=True,
                        help='Use dygraph mode.')
    parser.add_argument(
        '--device',
        default='gpu',
        type=str,
        choices=['cpu', 'gpu'],
        help='The device to select to train the model, is must be cpu/gpu.')
    parser.add_argument('--seed',
                        type=int,
                        default=1000,
                        help='random seed for initialization')
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


def do_train(args):
    paddle.enable_static() if not args.eager_run else None
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)
    worker_init = WorkerInitObj(args.seed + paddle.distributed.get_rank())

    model_class, tokenizer_class = MODEL_CLASSES['ernie-health']

    # Loads or initialize a model.
    pretrained_models = list(
        tokenizer_class.pretrained_init_configuration.keys())

    if args.model_name_or_path in pretrained_models:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        generator = ElectraGenerator(
            ElectraModel(**model_class.pretrained_init_configuration[
                args.model_name_or_path + '-generator']))
        discriminator = ErnieHealthDiscriminator(
            ElectraModel(**model_class.pretrained_init_configuration[
                args.model_name_or_path + '-discriminator']))
        model = model_class(generator, discriminator)
        args.init_from_ckpt = False
    else:
        if os.path.isdir(args.model_name_or_path) and args.init_from_ckpt:
            # Load checkpoint
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
            with open(os.path.join(args.model_name_or_path, 'run_states.json'),
                      'r') as f:
                config_dict = json.load(f)
                model_name = config_dict['model_name']
            if model_name in pretrained_models:
                generator = ElectraGenerator(
                    ElectraModel(**model_class.pretrained_init_configuration[
                        model_name + '-generator']))
                discriminator = ErnieHealthDiscriminator(
                    ElectraModel(**model_class.pretrained_init_configuration[
                        model_name + '-discriminator']))
                model = model_class(generator, discriminator)
                model.set_state_dict(
                    paddle.load(
                        os.path.join(args.model_name_or_path,
                                     'model_state.pdparams')))
            else:
                raise ValueError(
                    'initialize a model from ckpt need model_name '
                    'in model_config_file. The supported model_name '
                    'are as follows: {}'.format(
                        tokenizer_class.pretrained_init_configuration.keys()))
        else:
            raise ValueError(
                'initialize a model need identifier or the '
                'directory of storing model. if use identifier, the supported model '
                'identifiers are as follows: {}, if use directory, '
                'make sure set init_from_ckpt as True'.format(
                    model_class.pretrained_init_configuration.keys()))

    criterion = ErnieHealthPretrainingCriterion(
        getattr(model.generator,
                ElectraGenerator.base_model_prefix).config['vocab_size'],
        model.gen_weight)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # Loads dataset.
    tic_load_data = time.time()
    logger.info('start load data : %s' %
                (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

    train_dataset = MedicalCorpus(data_path=args.input_dir, tokenizer=tokenizer)
    logger.info('load data done, total : %s s' % (time.time() - tic_load_data))

    # Reads data and generates mini-batches.
    data_collator = DataCollatorForErnieHealth(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        mlm_prob=args.mlm_prob)

    train_data_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        mode='train',
        use_gpu=True if args.device in 'gpu' else False,
        data_collator=data_collator)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_epochs)
    args.num_epochs = (num_training_steps - 1) // len(train_data_loader) + 1

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_steps)

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ['bias', 'norm'])
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

    logger.info('start train : %s' %
                (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    trained_global_step = global_step = 0
    t_loss = defaultdict(lambda: paddle.to_tensor([0.0]))
    log_loss = defaultdict(lambda: paddle.to_tensor([0.0]))
    loss_list = defaultdict(list)
    log_list = []
    tic_train = time.time()

    if os.path.isdir(args.model_name_or_path) and args.init_from_ckpt:
        optimizer.set_state_dict(
            paddle.load(
                os.path.join(args.model_name_or_path, 'model_state.pdopt')))
        trained_global_step = global_step = config_dict['global_step']
        if trained_global_step < num_training_steps:
            logger.info(
                '[ start train from checkpoint ] we have already trained %s steps, seeking next step : %s'
                % (trained_global_step, trained_global_step + 1))
        else:
            logger.info(
                '[ start train from checkpoint ] we have already trained %s steps, but total training steps is %s, please check configuration !'
                % (trained_global_step, num_training_steps))
            exit(0)

    if paddle.distributed.get_rank() == 0:
        writer = LogWriter(os.path.join(args.output_dir, 'loss_log'))

    for epoch in range(args.num_epochs):
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
                        generator_labels=gen_labels)
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
                gen_logits, logits_rtd, logits_mts, logits_csp, disc_labels, masks = model(
                    input_ids=masked_input_ids,
                    raw_input_ids=input_ids,
                    generator_labels=gen_labels)
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
                local_loss = dict([
                    (k, (t_loss[k] - log_loss[k]) / args.logging_steps)
                    for k in ['loss', 'gen', 'rtd', 'mts', 'csp']
                ])
                if paddle.distributed.get_world_size() > 1:
                    for k in ['loss', 'gen', 'rtd', 'mts', 'csp']:
                        paddle.distributed.all_gather(loss_list[k],
                                                      local_loss[k])
                    if paddle.distributed.get_rank() == 0:
                        tmp_loss = dict([
                            (k,
                             float((paddle.stack(loss_list[k]).sum() /
                                    len(loss_list[k])).numpy()))
                            for k in ['loss', 'gen', 'rtd', 'mts', 'csp']
                        ])
                        log_str = (
                            'global step {0:d}/{1:d}, epoch: {2:d}, batch: {3:d}, '
                            'avg_loss: {4:.15f}, generator: {5:.15f}, rtd: {6:.15f}, multi_choice: {7:.15f}, '
                            'seq_contrastive: {8:.15f}, lr: {9:.10f}, speed: {10:.2f} s/it'
                        ).format(global_step, num_training_steps, epoch, step,
                                 tmp_loss['loss'], tmp_loss['gen'],
                                 tmp_loss['rtd'], tmp_loss['mts'],
                                 tmp_loss['csp'], optimizer.get_lr(),
                                 (time.time() - tic_train) / args.logging_steps)
                        logger.info(log_str)
                        log_list.append(log_str)
                        writer.add_scalar('generator_loss', tmp_loss['gen'],
                                          global_step)
                        writer.add_scalar('rtd_loss', tmp_loss['rtd'] * 50,
                                          global_step)
                        writer.add_scalar('mts_loss', tmp_loss['mts'] * 20,
                                          global_step)
                        writer.add_scalar('csp_loss', tmp_loss['csp'],
                                          global_step)
                        writer.add_scalar('total_loss', tmp_loss['loss'],
                                          global_step)
                        writer.add_scalar('lr', optimizer.get_lr(), global_step)
                    loss_list = defaultdict(list)
                else:
                    local_loss = dict([(k, v.numpy()[0])
                                       for k, v in local_loss.items()])
                    log_str = (
                        'global step {0:d}/{1:d}, epoch: {2:d}, batch: {3:d}, '
                        'avg_loss: {4:.15f}, generator: {5:.15f}, rtd: {6:.15f}, multi_choice: {7:.15f}, '
                        'seq_contrastive_loss: {8:.15f}, lr: {9:.10f}, speed: {10:.2f} s/it'
                    ).format(global_step, num_training_steps, epoch, step,
                             local_loss['loss'], local_loss['gen'],
                             local_loss['rtd'], local_loss['mts'],
                             local_loss['csp'], optimizer.get_lr(),
                             (time.time() - tic_train) / args.logging_steps)
                    logger.info(log_str)
                    log_list.append(log_str)
                    loss_dict = {
                        'generator_loss': local_loss['gen'],
                        'rtd_loss': local_loss['rtd'] * 50,
                        'mts_loss': local_loss['mts'] * 20,
                        'csp_loss': local_loss['csp']
                    }
                    for k, v in loss_dict.items():
                        writer.add_scalar('loss/%s' % k, v, global_step)
                    writer.add_scalar('total_loss', local_loss['loss'],
                                      global_step)
                    writer.add_scalar('lr', optimizer.get_lr(), global_step)
                log_loss = dict(t_loss)
                tic_train = time.time()

            if global_step % args.save_steps == 0:
                if paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir,
                                              'model_%d.pdparams' % global_step)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    config_to_save = copy.deepcopy(
                        model_to_save.discriminator.electra.config)
                    if 'self' in config_to_save:
                        del config_to_save['self']
                    run_states = {
                        'model_name': model_name
                        if args.init_from_ckpt else args.model_name_or_path,
                        'global_step': global_step,
                        'epoch': epoch,
                        'step': step,
                    }
                    with open(os.path.join(output_dir, 'model_config.json'),
                              'w') as f:
                        json.dump(config_to_save, f)
                    with open(os.path.join(output_dir, 'run_states.json'),
                              'w') as f:
                        json.dump(run_states, f)
                    paddle.save(
                        model.state_dict(),
                        os.path.join(output_dir, 'model_state.pdparams'))
                    tokenizer.save_pretrained(output_dir)
                    paddle.save(optimizer.state_dict(),
                                os.path.join(output_dir, 'model_state.pdopt'))
                    if len(log_list) > 0:
                        with open(os.path.join(output_dir, 'train.log'),
                                  'w') as f:
                            for log in log_list:
                                if len(log.strip()) > 0:
                                    f.write(log.strip() + '\n')
            if global_step >= num_training_steps:
                if paddle.distributed.get_rank() == 0:
                    writer.close()
                return


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    do_train(args)
