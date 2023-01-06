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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset
from visualdl import LogWriter

from paddlenlp.data import DataCollator
from paddlenlp.trainer import Trainer, TrainerCallback

# from paddlenlp.trainer.trainer_utils import EvalLoopOutput, has_length, find_batch_size
from paddlenlp.trainer.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    IterableDatasetShard,
    find_batch_size,
    has_length,
)
from paddlenlp.trainer.training_args import TrainingArguments
from paddlenlp.utils.batch_sampler import (
    DistributedBatchSampler as NlpDistributedBatchSampler,
)
from paddlenlp.trainer.utils.helper import (
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
)
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.utils.log import logger


class MyCallback(TrainerCallback):
    def __init__(self, log_dir="log") -> None:
        super().__init__()

        self._LogWriter = LogWriter
        self.vdl_writer = self._LogWriter(logdir=log_dir)

    def on_step_end(self, args, state, control, **kwargs):
        # print(state)
        # print(control)
        # print(kwargs)
        # lr_scheduler = kwargs['lr_scheduler']
        # optimizer = kwargs["optimizer"]
        # current_learning_rate=[]
        # for param_group in optimizer._param_groups:
        #     for param in param_group['params']:
        #         if param.stop_gradient:
        #             continue
        #         lr1 = lr_scheduler.get_lr() * param.optimize_attr['learning_rate']
        #         lr2 = optimizer._global_learning_rate() * param.optimize_attr['learning_rate']
        #         # print(lr1)
        #         # print(lr2)
        #         # print(param[0].optimize_attr['learning_rate'])
        #         grad_var = param._grad_ivar()
        #         param_and_grad = (param, grad_var)
        #         lr = optimizer._create_param_lr(param_and_grad)
        #         # print(lr.item())
        #         current_learning_rate.append(lr1)
        #         with open('learning_rates_pd_{}.txt'.format(str(lr.place)),'a') as f:
        #             f.write('global_step {} learning rate {} final learning rate {}  \n'.format(state.global_step, lr.item(), lr2.item()))
        #         break
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        print(args)


class BridgeTowerTrainer(Trainer):
    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor] = None,
    ):
        super(BridgeTowerTrainer, self).__init__(
            model,
            criterion,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        # meter_utils.set_metrics(model)

    def compute_loss(self, model, inputs, split, eval_step, return_outputs=False):
        # print(inputs.keys())
        loss_name = "snli"
        # print(eval_step)
        ret = model(inputs, split, loss_name=loss_name, global_step=self.state.global_step, eval_step=eval_step)
        # loss = {k:v for k,v in ret.items() if "loss" in k}
        # logits = {k:v for k,v in ret.items() if "logits" in k}
        loss = ret["snli_loss"]
        outputs = (ret["snli_loss"], ret["snli_logits"])
        # outputs = (loss, logits)

        if self.criterion is not None:
            loss = self.criterion(outputs, ret["snli_labels"])
            outputs = (loss, outputs)
            # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def validation_epoch_end(self):
        # if self.jump_val_first_for_irtr_itm_irc and "irtr_itm_itc" in self.hparams.config["group_name"]:
        #     old_get_recall_metric = self.hparams.config["get_recall_metric"]
        #     self.hparams.config["get_recall_metric"] = False
        #     meter_utils.epoch_wrapup(self, 'val')
        #     self.hparams.config["get_recall_metric"] = old_get_recall_metric
        #     self.jump_val_first_for_irtr_itm_irc = False
        # else:
        #     meter_utils.epoch_wrapup(self, 'val')
        self.model(
            batch=None, split="val", loss_name="", status="validation_epoch_end", global_step=self.state.global_step
        )

    def training_epoch_end(self):
        # meter_utils.epoch_wrapup(self, 'train')
        self.model(
            batch=None, split="train", loss_name="", status="training_epoch_end", global_step=self.state.global_step
        )

    def training_step(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to train.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `paddle.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(
                model,
                inputs,
                "train",
                eval_step=0,
            )
            # # Add multiple loss
            # if isinstance(loss, dict):
            #     loss = sum([v for k, v in loss.items() if "loss" in k])

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        # eval_dataloader = self.get_eval_dataloader(eval_dataset)
        # start_time = time.time()

        # output = self.evaluation_loop(
        #     eval_dataloader,
        #     description="Evaluation",
        #     # No point gathering the predictions if there are no metrics, otherwise we defer to
        #     # self.args.prediction_loss_only
        #     prediction_loss_only=True if self.compute_metrics is None else None,
        #     ignore_keys=ignore_keys,
        #     metric_key_prefix=metric_key_prefix,
        # )

        # total_batch_size = self.args.eval_batch_size * self.args.world_size
        # output.metrics.update(
        #     speed_metrics(
        #         metric_key_prefix,
        #         start_time,
        #         num_samples=output.num_samples,
        #         num_steps=math.ceil(output.num_samples / total_batch_size),
        #     )
        # )

        # self.log(output.metrics)

        # self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        # # set log after validation epoch end
        # self.validation_epoch_end()
        # return output.metrics
        return metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_eval_iters: Optional[int] = -1,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self.model

        if isinstance(dataloader, paddle.io.DataLoader):
            batch_size = dataloader.batch_sampler.batch_size
        elif isinstance(dataloader, paddle.fluid.dataloader.dataloader_iter._DataLoaderIterBase):
            # support for inner dataloader
            batch_size = dataloader._batch_sampler.batch_size
            # alias for inner dataloader
            dataloader.dataset = dataloader._dataset
        else:
            raise ValueError("Only support for paddle.io.DataLoader")

        num_samples = None
        if max_eval_iters > 0:
            # on eval limit steps
            num_samples = batch_size * self.args.world_size * max_eval_iters
            if isinstance(dataloader, paddle.fluid.dataloader.dataloader_iter._DataLoaderIterBase) and isinstance(
                dataloader._batch_sampler, NlpDistributedBatchSampler
            ):
                consumed_samples = (
                    ((self.state.global_step) // args.eval_steps)
                    * max_eval_iters
                    * args.per_device_eval_batch_size
                    * args.world_size
                )
                dataloader._batch_sampler.set_epoch(consumed_samples=consumed_samples)

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            logger.info(f"  Total prediction steps = {len(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Pre device batch size = {batch_size}")
        logger.info(f"  Total Batch size = {batch_size * self.args.world_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        losses = []
        for step, inputs in enumerate(dataloader):
            # if(metric_key_prefix =='train'):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size
            # else:
            #     val_batches = [i for i, n in enumerate(inputs["table_name"]) if "dev" in n]
            #     test_batches = [i for i, n in enumerate(inputs["table_name"]) if "test" in n]
            #     val_batch_size = len(val_batches)
            #     test_batch_size = len(test_batches)

            # Prediction step
            loss, logits, labels = self.prediction_step(
                model,
                inputs,
                prediction_loss_only,
                metric_key_prefix=metric_key_prefix,
                ignore_keys=ignore_keys,
                step=step,
            )
            # Update containers on host
            if loss is not None:
                # losses = self._nested_gather(loss.repeat(batch_size))
                # if isinstance(loss, dict):
                #     for key, loss in loss.items():
                #         if "dev" in key:
                #             val_losses = self._nested_gather(paddle.tile(loss, repeat_times=[val_batch_size, 1]))
                #             losses_hosts[key] = (
                #                 losses
                #                 if key not in losses_hosts
                #                 else paddle.concat((losses_hosts[key], val_losses), axis=0)
                #             )
                #         elif "test" in key:
                #             test_losses = self._nested_gather(paddle.tile(loss, repeat_times=[test_batch_size, 1]))
                #             losses_hosts[key] = (
                #                 losses
                #                 if key not in losses_hosts
                #                 else paddle.concat((losses_hosts[key], test_losses), axis=0)
                #             )
                # else:
                losses = self._nested_gather(paddle.tile(loss, repeat_times=[batch_size, 1]))
                losses_host = losses if losses_host is None else paddle.concat((losses_host, losses), axis=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            if max_eval_iters > 0 and step >= max_eval_iters - 1:
                break
        # Gather all remaining tensors and put them back on the CPU
        # if losses_hosts is not None:
        #     for key, loss in losses_hosts.items():
        #         losses_hosts[key]=losses_hosts[key].cpu().numpy()
        #         all_losses_dicts[key]=losses_hosts[key] if losses_hosts[key] is None else np.concatenate((losses_hosts[key], losses[key]), axis=0)

        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if num_samples is not None:
            pass
        elif has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        model.train()

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # if all_losses_dicts is not None:
        #     for key,loss in all_losses_dicts.items():
        #         metrics[key] = loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        metric_key_prefix: str = "eval",
        step: int = 0,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to evaluate.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with paddle.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(
                        model, inputs, metric_key_prefix, eval_step=step, return_outputs=True
                    )
                    loss = loss.mean().detach()
                # Add multiple loss
                # if isinstance(loss, dict):
                #     loss = {k:v.mean().detach() for k, v in loss.items() if "loss" in k}
                # else:
                #     loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)
        logits = nested_detach(logits)
        if isinstance(logits, (list, tuple)) and len(logits) == 1:
            logits = logits[0]
        return (loss, logits, labels)


class BridgeTowerPreTrainTrainer(BridgeTowerTrainer):
    def compute_loss(self, model, inputs, split, eval_step, return_outputs=False):
        loss_name = "mlm_itm"
        ret = model(inputs, split, loss_name=loss_name, global_step=self.state.global_step, eval_step=eval_step)
        loss = ret["mlm_loss"] + ret["itm_loss"]
        # print(loss)
        outputs = (loss, ret["itm_logits"])

        if self.criterion is not None:
            loss = self.criterion(outputs, ret["item_labels"])
            outputs = (loss, outputs)
            # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
