# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import paddle
from tqdm import tqdm
from paddle.optimizer import AdamW
from paddlenlp.transformers import LinearDecayWithWarmup


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
            if not any(nd in n for nd in ["bias", "norm"])
        ]

    def train(self):
        model = self.model

        epoch = 0
        global_step = 0
        tr_loss = 0.0
        acc = 0.0

        model.train()
        model, self.optimizer = paddle.amp.decorate(models=model,
                                                    optimizers=self.optimizer,
                                                    level='O2',
                                                    master_weight=None,
                                                    save_dtype='float32')

        with tqdm(total=self.num_train_steps) as pbar:
            while True:
                for step, batch in enumerate(self.dataloader):
                    with paddle.amp.auto_cast(enable=True,
                                              custom_white_list=None,
                                              custom_black_list=None,
                                              level='O2'):
                        logits = model(input_ids=batch[0],
                                       token_type_ids=batch[1])

                    loss = paddle.nn.CrossEntropyLoss()(logits,
                                                        batch[2].reshape(
                                                            (-1, )))

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    scaled = self.scaler.scale(loss)
                    scaled.backward()
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        self.scaler.minimize(self.optimizer, scaled)
                        self.scheduler.step()
                        self.optimizer.clear_grad()
                        pbar.set_description(
                            "epoch: {} loss: {} acc: {}".format(
                                epoch, loss.numpy(), acc))
                        pbar.update()
                        global_step += 1

                        if global_step == self.num_train_steps:
                            break
                    if (step + 1) % self.args.eval_step == 0:
                        ac = self.step_callback(model, self.args)
                        if ac > acc:
                            acc = ac
                            model.save_pretrained(self.args.output_dir)

                if global_step == self.num_train_steps:
                    break
                epoch += 1

        return model, global_step, tr_loss / global_step

    def _create_optimizer(self, model):
        scheduler = self._create_scheduler()
        clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
        return AdamW(parameters=model.parameters(),
                     grad_clip=clip,
                     learning_rate=scheduler,
                     beta1=0.9,
                     apply_decay_param_fun=lambda x: x in self.wd_params,
                     weight_decay=self.args.weight_decay,
                     beta2=0.99), scheduler

    def _create_scheduler(self):
        return LinearDecayWithWarmup(self.args.learning_rate,
                                     self.num_train_steps,
                                     self.args.warmup_proportion)
