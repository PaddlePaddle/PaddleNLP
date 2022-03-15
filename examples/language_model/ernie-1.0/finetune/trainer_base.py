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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddle.io import DataLoader


class TrainerBase(object):
    """
    """

    def create_dataloader(self,
                          dataset,
                          mode='train',
                          batch_size=16,
                          batchify_fn=None,
                          trans_fn=None,
                          batched=False):
        """
        """
        if trans_fn:
            dataset = dataset.map(trans_fn, batched=batched)

        shuffle = True if mode == 'train' else False
        if mode == 'train':
            batch_sampler = paddle.io.DistributedBatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            batch_sampler = paddle.io.BatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle)

        return paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)

    def train(self, *args, **kwargs):
        """
        """
        pass

    def eval(self, *args, **kwargs):
        """
        """
        pass

    def prepare_train_config(self):
        """
        """
        if self.args.max_steps > 0:
            self.args.num_training_steps = self.args.max_steps
            self.args.num_train_epochs = math.ceil(
                self.args.num_training_steps / len(self.train_dl))

        else:
            self.args.num_training_steps = len(
                self.train_dl) * self.args.num_train_epochs
            self.args.num_train_epochs = self.args.num_train_epochs

        if self.args.num_training_steps // self.args.valid_steps < self.args.minimum_valid_times:
            exp_step = self.args.num_training_steps / self.args.minimum_valid_times
            exp_step = max(int(exp_step - exp_step % 10), 10)
            logger.info("Set eval step to %d" % exp_step)
            self.args.valid_steps = exp_step

        warmup = self.args.warmup_steps if self.args.warmup_steps > 0 else self.args.warmup_proportion

        self.lr_scheduler = LinearDecayWithWarmup(
            self.args.learning_rate, self.args.num_training_steps, warmup)

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=self.lr_scheduler,
            beta1=0.9,
            beta2=0.999,
            epsilon=self.args.adam_epsilon,
            parameters=self.model.parameters(),
            weight_decay=self.args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm))

    def print_config(self):
        """
        """
        logger.info('{:^40}'.format("Configuration Arguments"))
        logger.info('{:20}:{}'.format("paddle commit id",
                                      paddle.version.commit))
        for arg in vars(self.args):
            logger.info('{:20}:{}'.format(arg, getattr(self.args, arg)))


class Trainer:
    """
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Layer]=None,
            args: TrainingArguments=None,
            data_collator: Optional[DataCollator]=None,
            train_dataset: Optional[Dataset]=None,
            eval_dataset: Optional[Dataset]=None,
            tokenizer: Optional[PreTrainedTokenizerBase]=None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]]=None,
            optimizers: Tuple[paddle.optim.Optimizer, paddle.optim.lr_scheduler.
                              LambdaLR]=(None, None), ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(
                f"No `TrainingArguments` passed, using `output_dir={output_dir}`."
            )
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        if model is None:
            raise RuntimeError(
                "`Trainer` requires either a `model` or `model_init` argument")

        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(
            tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.model_wrapped = model
        self.model = model

        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers

        if args.max_steps > 0:
            logger.info(
                "max_steps is given, it will override any value given in num_train_epochs"
            )

        if train_dataset is not None and not isinstance(
                train_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError(
                "train_dataset does not implement __len__, max_steps has to be specified"
            )

        if args.fp16:
            logger.info(f"Using  half precision backend")

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]]=None,
            trial: Union["optuna.Trial", Dict[str, Any]]=None,
            ignore_keys_for_eval: Optional[List[str]]=None,
            **kwargs, ):
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model_wrapped)
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        for epoch in range(epochs_trained, num_train_epochs):
            step = -1
            for step, inputs in enumerate(epoch_iterator):
                tr_loss_step = self.training_step(model, inputs)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.step()

                self.lr_scheduler.step()
                model.zero_grad()

    def training_step(
            self, model: nn.Layer,
            inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        loss.backward()

        return loss.detach()

    def get_train_dataloader(self):
        pass

    def _get_eval_sampler(self, eval_dataset: Dataset):
        pass

    def get_eval_dataloader(self,
                            eval_dataset: Optional[Dataset]=None) -> DataLoader:
        pass

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        pass

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        pass

    def create_optimizer(self):
        pass

    @staticmethod
    def get_optimizer_cls_and_kwargs(
            args: TrainingArguments) -> Tuple[Any, Any]:
        pass

    def create_scheduler(self,
                         num_training_steps: int,
                         optimizer: paddle.optim.Optimizer=None):
        pass

    def _wrap_model(self, model, training=True):
        pass

    def _prepare_input(
            self, data: Union[paddle.Tensor, Any]) -> Union[paddle.Tensor, Any]:
        pass

    def _prepare_inputs(self, inputs: Dict[str, Union[paddle.Tensor, Any]]
                        ) -> Dict[str, Union[paddle.Tensor, Any]]:
        pass

    def autocast_smart_context_manager(self):
        pass

    def training_step(
            self, model: nn.Layer,
            inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        pass

    def save_model(self,
                   output_dir: Optional[str]=None,
                   _internal_call: bool=False):
        pass

    def _save(self, output_dir: Optional[str]=None, state_dict=None):
        pass

    def _load_optimizer_and_scheduler(self, checkpoint):
        pass

    def evaluate(
            self,
            eval_dataset: Optional[Dataset]=None,
            ignore_keys: Optional[List[str]]=None,
            metric_key_prefix: str="eval", ) -> Dict[str, float]:
        pass

    def predict(self,
                test_dataset: Dataset,
                ignore_keys: Optional[List[str]]=None,
                metric_key_prefix: str="test") -> PredictionOutput:
        pass

    def prediction_step(
            self,
            model: nn.Layer,
            inputs: Dict[str, Union[paddle.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]]=None, ) -> Tuple[Optional[
                paddle.Tensor], Optional[paddle.Tensor], Optional[
                    paddle.Tensor]]:
        pass

    def create_dataloader(self,
                          dataset,
                          mode='train',
                          batch_size=16,
                          batchify_fn=None,
                          trans_fn=None,
                          batched=False):
        """
        """
        if trans_fn:
            dataset = dataset.map(trans_fn, batched=batched)

        shuffle = True if mode == 'train' else False
        if mode == 'train':
            batch_sampler = paddle.io.DistributedBatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            batch_sampler = paddle.io.BatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle)

        return paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)

    def train(self, *args, **kwargs):
        """
        """
        pass

    def eval(self, *args, **kwargs):
        """
        """
        pass

    def prepare_train_config(self):
        """
        """
        if self.args.max_steps > 0:
            self.args.num_training_steps = self.args.max_steps
            self.args.num_train_epochs = math.ceil(
                self.args.num_training_steps / len(self.train_dl))

        else:
            self.args.num_training_steps = len(
                self.train_dl) * self.args.num_train_epochs
            self.args.num_train_epochs = self.args.num_train_epochs

        if self.args.num_training_steps // self.args.valid_steps < self.args.minimum_valid_times:
            exp_step = self.args.num_training_steps / self.args.minimum_valid_times
            exp_step = max(int(exp_step - exp_step % 10), 10)
            logger.info("Set eval step to %d" % exp_step)
            self.args.valid_steps = exp_step

        warmup = self.args.warmup_steps if self.args.warmup_steps > 0 else self.args.warmup_proportion

        self.lr_scheduler = LinearDecayWithWarmup(
            self.args.learning_rate, self.args.num_training_steps, warmup)

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=self.lr_scheduler,
            beta1=0.9,
            beta2=0.999,
            epsilon=self.args.adam_epsilon,
            parameters=self.model.parameters(),
            weight_decay=self.args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm))

    def print_config(self):
        """
        """
        logger.info('{:^40}'.format("Configuration Arguments"))
        logger.info('{:20}:{}'.format("paddle commit id",
                                      paddle.version.commit))
        for arg in vars(self.args):
            logger.info('{:20}:{}'.format(arg, getattr(self.args, arg)))
