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
from typing import Any, Callable, Dict, List, Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..data import DataCollator
from ..datasets import MapDataset
from ..losses import RDropLoss
from ..trainer import Trainer, TrainerCallback
from ..trainer.trainer_utils import EvalPrediction, get_scheduler
from ..transformers import PretrainedTokenizer, export_model
from ..utils.log import logger
from .prompt_args import PromptTuningArguments
from .prompt_utils import PromptDataCollatorWithPadding
from .template import AutoTemplate
from .verbalizer import SoftVerbalizer


class PromptTrainer(Trainer):
    """
    PromptTrainer is a feature-complete training and eval loop for PaddleNLP
    on prompt-tuning.
    """

    def __init__(
        self,
        model: nn.Layer,
        tokenizer: PretrainedTokenizer,
        criterion: Optional[nn.Layer] = None,
        args: Optional[PromptTuningArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[MapDataset] = None,
        eval_dataset: Optional[MapDataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(
                "No `TrainingArguments` passed, initialized with " "output_dir={} by default.".format(output_dir)
            )
            args = PromptTuningArguments(output_dir=output_dir)

        if data_collator is None:
            data_collator = PromptDataCollatorWithPadding(tokenizer, padding=True, return_tensors="pd")

        if criterion is None and (args.use_rgl or args.use_rdrop):
            raise Exception("'To use 'use_rgl', 'use_rdrop', 'criterion' must be specified")

        super(PromptTrainer, self).__init__(
            model=model,
            criterion=criterion,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.load_state_dict_from_checkpoint(args.resume_from_checkpoint)

        self.train_dataset = self._map_dataset(self.train_dataset)
        self.eval_dataset = self._map_dataset(self.eval_dataset)

        if self.args.use_rdrop:
            self.rdrop_criterion = RDropLoss()

    def _get_model(self):
        model = self.model
        if isinstance(model, paddle.DataParallel):
            model = model._layers
        return model

    @property
    def template(self):
        return self._get_model().template

    @template.setter
    def template(self, template):
        self._get_model().template = template

    @property
    def verbalizer(self):
        return self._get_model().verbalizer

    @verbalizer.setter
    def verbalizer(self, verbalizer):
        self._get_model().verbalizer = verbalizer

    @property
    def pretrained_model(self):
        self._set_model_attributes(self.model, "plm")

    @pretrained_model.setter
    def pretrained_model(self, model):
        self._set_model_attributes(self.model, "plm", model)

    def _map_dataset(self, dataset: MapDataset):
        if dataset is None:
            return None
        if not isinstance(dataset, MapDataset):
            raise ValueError("Expected `MapDataset` but received {}.".format(type(dataset)))

        def encode_with_template(example):
            return self.template(example)

        return dataset.map(encode_with_template)

    def _prepare_input(self, inputs: Dict):
        return inputs

    def _save(self, output_dir: Optional[str] = None, state_dict: Dict[str, Any] = None):
        super(PromptTrainer, self)._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.template:
            self.template.save(output_dir)
        if self.verbalizer is not None:
            self.verbalizer.save(output_dir)

    def load_state_dict_from_checkpoint(self, resume_from_checkpoint: os.PathLike = None):
        if resume_from_checkpoint is not None:
            self.template = AutoTemplate.load_from(
                resume_from_checkpoint, self.tokenizer, self.args.max_seq_length, self._get_model()
            )
        super(PromptTrainer, self).load_state_dict_from_checkpoint(resume_from_checkpoint)

    def get_test_dataloader(self, test_dataset):
        test_dataset = self._map_dataset(test_dataset)
        return super(PromptTrainer, self).get_test_dataloader(test_dataset)

    def create_optimizer(self, lr_scheduler=None):
        """
        Setup the optimizer for both model and prompt parameters.
        """
        if self.optimizer is None:
            optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            plm_parameters = []
            if not self.args.freeze_plm:
                plm_parameters.extend([p for p in self._get_model().plm.parameters() if not p.stop_gradient])

            ppt_parameters = []
            if self.template is not None:
                ppt_parameters.extend([x for n, x in self.template.named_parameters() if not x.stop_gradient])
            if self.verbalizer is not None:
                if isinstance(self.verbalizer, SoftVerbalizer):
                    if not self.args.freeze_plm:
                        plm_parameters.extend(
                            [p for n, p in self.verbalizer.non_head_parameters() if not p.stop_gradient]
                        )
                    ppt_parameters.extend([p for n, p in self.verbalizer.head_parameters()])
                else:
                    ppt_parameters.extend([p for n, p in self.verbalizer.parameters()])

            decay_parameters = [
                p.name for n, p in self._get_model().named_parameters() if not any(nd in n for nd in ["bias", "norm"])
            ]

            if len(plm_parameters) > 0:
                ppt_lr = self.args.ppt_learning_rate / self.args.learning_rate
                lr = self.lr_scheduler if lr_scheduler is None else lr_scheduler
                if len(ppt_parameters) > 0:
                    params = [
                        {"params": plm_parameters},
                        {
                            "params": ppt_parameters,
                            "learning_rate": ppt_lr,
                            "weight_decay": self.args.ppt_weight_decay,
                            "beta1": self.args.ppt_adam_beta1,
                            "beta2": self.args.ppt_adam_beta2,
                            "epsilon": self.args.ppt_adam_epsilon,
                        },
                    ]
                else:
                    params = plm_parameters
            else:
                args = self.init_num_steps(self.args, len(self.train_dataset))
                warmup = (
                    args.warmup_steps if args.warmup_steps > 0 else int(args.warmup_ratio * args.num_training_steps)
                )
                self.lr_scheduler = get_scheduler(
                    args.lr_scheduler_type,
                    learning_rate=self.args.ppt_learning_rate,
                    num_warmup_steps=warmup,
                    num_training_steps=args.num_training_steps,
                )
                lr = self.lr_scheduler
                params = ppt_parameters

            self.optimizer = optim_cls(
                learning_rate=lr,
                apply_decay_param_fun=lambda x: x in decay_parameters,
                parameters=params,
                weight_decay=self.args.weight_decay,
                grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm),
                **optim_kwargs,
            )

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the total loss for every batch.
        """
        if "labels" not in inputs:
            raise ValueError("Fail to compute loss as `labels` not in {}.".format(inputs))
        labels = inputs["labels"]

        input_dict = inputs.copy()

        if self.criterion is not None:
            # pop labels to move loss computation out of the model
            input_dict.pop("labels")
            logits, hidden_states = model(**input_dict)
            loss = self.criterion(logits, labels)

            if self.args.use_rdrop:
                loss = self._compute_rdrop_loss(model, input_dict, logits, loss)

            if self.args.use_rgl:
                loss += self._compute_rgl_loss(hidden_states, labels)
        else:
            loss, logits, _ = model(**input_dict)

        outputs = (loss, logits)

        return (loss, outputs) if return_outputs else loss

    def _compute_rdrop_loss(self, model, input_dict, outputs, loss):
        re_outputs, _ = model(**input_dict)
        labels = input_dict["labels"]
        ce_loss = (self.criterion(re_outputs, labels) + loss) * 0.5
        kl_loss = self.rdrop_criterion(outputs, re_outputs)
        loss = ce_loss + self.args.alpha_rdrop * kl_loss
        return loss

    def _compute_rgl_loss(self, embeddings, labels, equal_type="raw"):
        """
        Compute the label consistency loss of sentence embeddings per batch.
        Please refer to https://aclanthology.org/2022.findings-naacl.81/
        for more details.
        """

        def _max_equal(x, y):
            return int(paddle.argmax(x, axis=0) == paddle.argmax(y, axis=0))

        def _raw_equal(x, y):
            return int(x == y)

        if equal_type == "raw":
            equals = _raw_equal
        elif equal_type == "max":
            equals = _max_equal
        else:
            raise ValueError("Unsupported equal type {}.".format(equal_type))
        batch_size = embeddings.shape[0]
        loss = 0
        for i in range(batch_size):
            for j in range(batch_size):
                score = F.cosine_similarity(embeddings[i], embeddings[j], axis=0)
                score = score.unsqueeze(0)
                logits = paddle.concat([(1 - score) * 50, (1 + score) * 50], axis=-1)
                label = paddle.to_tensor([equals(labels[i], labels[j])])
                logits = logits.reshape([-1, logits.shape[-1]])
                loss += F.cross_entropy(logits, label.unsqueeze(0))
        loss = loss / (batch_size * (batch_size - 1))
        loss = loss / 100 * self.args.alpha_rgl

        return loss

    def export_model(self, export_path, input_spec=None, export_type="paddle"):
        os.makedirs(export_path, exist_ok=True)
        self.template.save(export_path)
        if self.verbalizer is not None:
            self.verbalizer.save(export_path)
        if input_spec is None:
            input_spec = self.model.get_input_spec()
        export_model(self.model, input_spec, export_path, export_type)
