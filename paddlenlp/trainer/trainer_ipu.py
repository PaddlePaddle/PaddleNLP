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

import os
import logging
import time

import numpy as np
import paddle
from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import Dict, Stack
from paddlenlp.utils.log import logger

class IPUTrainer:
    """
    This is the trainer specially for Graphcore IPU.

    Args:
        args(`argparse.Namespace`): 
            The arguments to tweak for training.
        dataset(`paddle.utils.data.Dataset` or `paddle.utils.data.IterableDataset`, *optional*):
            The dataset to use for training
        exe(`paddle.fluid.executor.Executor`):
            An Executor in Python, specially for IPU
        tensor_list(`list[list, list]`):
            Contain model input tensors (feed list) and output tensors (fetch list).
        program(`list[`paddle.fluid.framework.Program`, `paddle.fluid.framework.Program`]`):
            Contain main program and startup program.
        optimizers(`list[`paddle.optimizer`, `paddlenlp.transformers.optimization`]`):
            Contain the optimizer and the scheduler to use.       
    """
    def __init__(
        self,
        args = None,
        dataset = None,
        exe = None,
        tensor_list = [None, None],
        program = [None, None],
        optimizers=[None, None]
    ):
        self.args = args
        self.dataset = dataset
        self.exe = exe
        self.feed_list, self.fetch_list = tensor_list
        self.main_program, self.startup_program = program
        self.hp_name = None
        self.is_in_train = False
        self.optimizer, self.lr_scheduler = optimizers        
        
        self.compile()
        
        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")


    def create_ipu_strategy(self):
        """
        IPUStrategy is a class which targets a single system with one or more IPUs attached.
        """
        ipu_strategy = paddle.static.IpuStrategy()
        options = {
            'is_training': self.args.is_training,
            'enable_manual_shard': True,
            'enable_pipelining': True,
            'batches_per_step': self.args.batches_per_step,
            'micro_batch_size': self.args.micro_batch_size,
            'loss_scaling': self.args.scale_loss,
            'enable_replicated_graphs': True,
            'replicated_graph_count': self.args.num_replica,
            'num_ipus': self.args.num_ipus * self.args.num_replica,
            'enable_gradient_accumulation': self.args.enable_grad_acc,
            'accumulation_factor': self.args.grad_acc_factor,
            'auto_recomputation': 3,
            'enable_half_partial': True,
            'available_memory_proportion': self.args.available_mem_proportion,
            'enable_stochastic_rounding': True,
            'max_weight_norm': 65504.0,
            'default_prefetch_buffering_depth': 3,
            'rearrange_anchors_on_host': False,
            'enable_fp16': self.args.ipu_enable_fp16,
            'random_seed': self.args.seed,
            'use_no_bias_optimizer': True,
            'enable_prefetch_datastreams': True,
            'enable_outlining': True,
            'subgraph_copying_strategy': 1,  # JustInTime
            'outline_threshold': 10.0,
            'disable_grad_accumulation_tensor_streams': True,
            'schedule_non_weight_update_gradient_consumers_early': True,
            'cache_path': 'paddle_cache',
            'enable_floating_point_checks': False,
            'accl1_type': self.args.accl1_type,
            'accl2_type': self.args.accl2_type,
            'weight_decay_mode': self.args.weight_decay_mode,
        }

        if not self.args.optimizer_state_offchip:
            # Store the tensor in streaming memory or on-chip memory.
            options['location_optimizer'] = {
                'on_chip': 1,
                'use_replicated_tensor_sharding':
                1,
            }
        # options['accumulate_outer_fragment'] = {3: [0]} # Configuration setting for operations in the accumulate outer fragment
        options['convolution_options'] = {"partialsType": "half"}
        options['engine_options'] = {
            "opt.useAutoloader": "true",
            "target.syncReplicasIndependently": "true",
            "exchange.streamBufferOverlap": "hostRearrangeOnly",
        }
        options['enable_engine_caching'] = self.args.enable_engine_caching

        ipu_strategy.set_options(options)
        # enable custom patterns
        ipu_strategy.enable_pattern('DisableAttnDropoutBwdPattern')

        return ipu_strategy


    def compile(self):
        """
        Builds an operator that compiles and runs computation
        """
        logging.info(f'start compiling, please wait some minutes')
        logging.info(
            f'you can run `export POPART_LOG_LEVEL=INFO` before running program to see the compile progress'
        )

        # Create ipu_strategy
        self.ipu_strategy = self.create_ipu_strategy()

        # Compile program for IPU
        self.ipu_compiler = paddle.static.IpuCompiledProgram(
            self.main_program, ipu_strategy=self.ipu_strategy)
        cur_time = time.time()
        self.main_program = self.ipu_compiler.compile(self.feed_list, self.fetch_list)
        time_cost = time.time() - cur_time
        logging.info(f'finish compiling! time cost: {time_cost}')


    def get_train_dataloader(self):
        """
        Returns the training [`~paddle.io.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        # Load the training dataset
        bs = self.args.micro_batch_size * self.args.grad_acc_factor * self.args.batches_per_step * self.args.num_replica
        self.args.batch_size = bs
        if self.args.is_training:
            train_batch_sampler = BatchSampler(
                self.dataset, batch_size=bs, shuffle=self.args.shuffle, drop_last=True)
        else:
            train_batch_sampler = BatchSampler(
                self.dataset, batch_size=bs, shuffle=self.args.shuffle, drop_last=False)

        if self.args.is_training:
            collate_fn = lambda samples, fn=Dict({
                "input_ids": Stack(),
                "token_type_ids": Stack(),
                "position_ids": Stack(),
                "input_mask": Stack(),
                "start_positions": Stack(),
                "end_positions": Stack()
            }): fn(samples)
        else:
            collate_fn = lambda samples, fn=Dict({
                "input_ids": Stack(),
                "token_type_ids": Stack(),
                "position_ids": Stack(),
                "input_mask": Stack()}): fn(samples)

        data_loader = DataLoader(
            dataset=self.dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            return_list=True)
        return data_loader

    def log(self):
        logging.info({
            "global_step": self.global_step,
        })


    def save_model(self):
        self.ipu_compiler._backend.weights_to_host()
        paddle.static.save(self.main_program.org_program,
                        os.path.join(self.args.output_dir,
                                        'step_{}'.format(self.global_step)))


    def train(self):
        main_program = self.main_program

        self.train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            num_train_epoch = float('inf')
        else:
            num_train_epoch = self.args.epochs
        batch_start = time.time()
        self.global_step = 0

        logging.info("Start running")
        epoch = 0
        while(epoch < num_train_epoch):
            for batch in self.train_dataloader:
                self.global_step += 1
                self.read_cost = time.time() - batch_start

                if len(batch) != len(self.feed_list):
                    raise ValueError("Batch size doesn't match with feed size")
                feed = {self.feed_list[i]: batch[i] for i in range(len(self.feed_list))}
                train_start = time.time()
                self.lr_scheduler.step()
                self.loss = self.exe.run(main_program,
                                    feed=feed,
                                    fetch_list=self.fetch_list,
                                    use_program_cache=True)

                self.train_cost = time.time() - train_start
                self.total_cost = time.time() - batch_start
                self.tput = self.args.batch_size / self.total_cost

                if self.global_step % self.args.logging_steps == 0:
                    self.log()

                if self.global_step % self.args.save_steps == 0:
                    self.save_model()
                batch_start = time.time()

                if self.global_step >= self.args.max_steps > 0:
                    return
            epoch += 1          


    def eval(self):
        main_program = self.main_program

        eval_dataloader = self.get_train_dataloader()
        batch_start = time.time()
        self.global_step = 0

        self.all_start_logits = []
        self.all_end_logits = []
        for step, batch in enumerate(eval_dataloader):
            if step % self.args.logging_steps == 0:
                logging.info(f'running step: {step}')

            self.pad_batch(batch)

            feed = {self.feed_list[i]: batch[i] for i in range(len(self.feed_list))}
            start_logits, end_logits = self.exe.run(main_program,
                                               feed=feed,
                                               fetch_list=self.fetch_list)

            start_logits = start_logits.reshape([-1, self.args.seq_len])
            end_logits = end_logits.reshape([-1, self.args.seq_len])
            for idx in range(self.real_len):
                self.all_start_logits.append(start_logits[idx])
                self.all_end_logits.append(end_logits[idx])


    def pad_batch(self, batch):
        self.real_len = np.array(batch[0]).shape[0]
        # padding zeros if needed
        if self.real_len < self.args.batch_size:
            batch = [np.asarray(x) for x in batch]
            pad0 = np.zeros([self.args.batch_size - self.real_len,
                                self.args.seq_len]).astype(batch[0].dtype)
            batch[0] = np.vstack((batch[0], pad0))
            batch[1] = np.vstack((batch[1], pad0))
            batch[2] = np.vstack((batch[2], pad0))
            pad1 = np.zeros(
                [self.args.batch_size - self.real_len, 1, 1, self.args.seq_len]) - 1e3
            pad1 = pad1.astype(batch[3].dtype)
            batch[3] = np.vstack((batch[3], pad1))        
