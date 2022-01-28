#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import paddle
from tqdm import tqdm
from paddle.optimizer import AdamW
from paddle.optimizer.lr import LRScheduler


class LinearScheduleWithWarmup(LRScheduler):
    def __init__(self,
                 learning_rate,
                 warmup_steps,
                 max_train_steps,
                 last_epoch=-1,
                 verbose=False):

        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.max_train_steps = max_train_steps
        super(LinearScheduleWithWarmup, self).__init__(learning_rate,
                                                       last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_percent = self.last_epoch / self.warmup_steps
        else:
            warmup_percent = 1 - self.last_epoch / self.max_train_steps

        return self.learning_rate * warmup_percent


def _create_model_arguments(batch):
    return batch


class Trainer(object):
    def __init__(self,
                 args,
                 model,
                 dataloader,
                 num_train_steps,
                 step_callback=None):
        self.args = args
        self.model = model
        self.dataloader = dataloader
        self.num_train_steps = num_train_steps
        self.step_callback = step_callback

        self.optimizer, self.scheduler = self._create_optimizer(model)
        self.scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        self.wd_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "LayerNorm"])
        ]

    def train(self, is_op=False):
        model = self.model

        epoch = 0
        global_step = 0
        tr_loss = 0.0

        model.train()

        with tqdm(total=self.num_train_steps) as pbar:
            while True:
                for step, batch in enumerate(self.dataloader):
                    with paddle.amp.auto_cast():
                        if not is_op:
                            outputs = model(
                                word_ids=batch[0],
                                word_segment_ids=batch[1],
                                word_attention_mask=batch[2],
                                entity_ids=batch[3],
                                entity_position_ids=batch[4],
                                entity_segment_ids=batch[5],
                                entity_attention_mask=batch[6],
                                start_positions=batch[7],
                                end_positions=batch[8])
                        else:
                            outputs = model(
                                word_ids=batch[0],
                                word_segment_ids=batch[1],
                                word_attention_mask=batch[2],
                                entity_ids=batch[3],
                                entity_position_ids=batch[4],
                                entity_segment_ids=batch[5],
                                entity_attention_mask=batch[6],
                                labels=batch[7])

                    loss = outputs[0]

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    scaled = self.scaler.scale(loss)
                    scaled.backward()
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        self.scaler.minimize(self.optimizer, scaled)
                        self.scheduler.step()
                        self.optimizer.clear_grad()
                        pbar.set_description("epoch: %d loss: %.7f" %
                                             (epoch, loss))
                        pbar.update()
                        global_step += 1

                        if global_step == self.num_train_steps:
                            break
                output_dir = self.args.output_dir

                model.save_pretrained(output_dir)
                if global_step == self.num_train_steps:
                    break
                epoch += 1

        return model, global_step, tr_loss / global_step

    def _create_optimizer(self, model):
        scheduler = self._create_scheduler()
        clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
        return AdamW(
            parameters=model.parameters(),
            grad_clip=clip,
            learning_rate=scheduler,
            beta1=self.args.adam_b1,
            apply_decay_param_fun=lambda x: x in self.wd_params,
            weight_decay=self.args.weight_decay,
            beta2=self.args.adam_b2), scheduler

    def _create_scheduler(self):
        warmup_steps = int(self.num_train_steps * self.args.warmup_proportion)
        return LinearScheduleWithWarmup(self.args.learning_rate, warmup_steps,
                                        self.num_train_steps)
