import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.metrics import f1_score
import numpy as np
import collections
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.metric import Accuracy
from datasets import load_metric

from ..datasets import MapDataset
from ..utils.log import logger
from ..trainer import Trainer, TrainerCallback
from ..trainer.trainer_utils import EvalPrediction

from ..data import DataCollator
from ..transformers.tokenizer_utils import PretrainedTokenizer

from .template import Template
from .verbalizer import Verbalizer
from .prompt_utils import InputFeatures, signature
from .prompt_tokenizer import MLMTokenizerWrapper
from .prompt_args import PromptTuningArguments

__all__ = ["PromptTrainer", "PromptModelForClassification"]

PROMPT_NAME = "prompt.pdparams"


class PromptTrainer(Trainer):
    """
    PromptTrainer is a feature-complete training and eval loop for PaddleNLP
    on prompt-tuning.
    """

    def __init__(self,
                 model: Union[nn.Layer],
                 tokenizer: PretrainedTokenizer,
                 criterion: Union[nn.Layer],
                 args: PromptTuningArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[MapDataset] = None,
                 eval_dataset: Optional[MapDataset] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction],
                                                    Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[paddle.optimizer.Optimizer,
                                   paddle.optimizer.lr.LRScheduler] = (None,
                                                                       None)):

        if args is None:
            output_dir = "tmp_trainer"
            logger.info(
                "No `TrainingArguments` passed, initialized with "\
                "output_dir={} by default.".format(output_dir)
            )
            args = PromptTuningArguments(output_dir=output_dir)

        if data_collator is None:
            data_collator = InputFeatures.collate_fn

        if compute_metrics is None:
            if args.task_type == 'multi-class':
                compute_metrics = self.compute_metrics_multi_class
            elif args.task_type in ['multi-label', 'hierachical']:
                compute_metrics = self.compute_metrics_multi_label

        super().__init__(model=model,
                         criterion=criterion,
                         args=args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         tokenizer=tokenizer,
                         compute_metrics=compute_metrics,
                         callbacks=callbacks,
                         optimizers=optimizers)

        max_seq_length = getattr(args, "max_seq_length", 512)
        if self.plm.__class__.__name__.endswith("MaskedLM"):
            self.tokenizer_wrapper = MLMTokenizerWrapper(
                max_seq_length, tokenizer)
        else:
            raise ValueError("Unsupported pretrained model {}")

        self.train_dataset = self._map_dataset(self.train_dataset)
        self.eval_dataset = self._map_dataset(self.eval_dataset)

    def compute_metrics_multi_class(self, eval_preds):
        metric = Accuracy()
        correct = metric.compute(paddle.to_tensor(eval_preds.predictions),
                                 paddle.to_tensor(eval_preds.label_ids))
        metric.update(correct)
        acc = metric.accumulate()
        return {'accuracy': acc}

    def compute_metrics_multi_label(self, eval_preds):
        probs = F.sigmoid(eval_preds.predictions)
        preds = probs > self.args.cls_threshold
        micro_f1_score = f1_score(y_pred=preds,
                                  y_true=eval_preds.label_ids,
                                  average='micro')
        return micro_f1_score

    @property
    def template(self):
        return self.model.template

    @template.setter
    def template(self, template):
        self.model.template = template

    @property
    def verbalizer(self):
        return getattr(self.model, "verbalizer", None)

    @verbalizer.setter
    def verbalizer(self, verbalizer):
        setattr(self.model, "verbalizer", verbalizer)

    @property
    def plm(self):
        return self.model.plm

    @plm.setter
    def plm(self, model):
        self.model.plm = model

    def _map_dataset(self, dataset):
        if dataset is None:
            return None
        if not isinstance(dataset, MapDataset):
            raise ValueError("Expected `MapDataset` but received {}.".format(
                type(dataset)))
        return dataset.map(self._convert_example)

    def _convert_example(self, example):
        if self.verbalizer is not None and hasattr(self.verbalizer,
                                                   'wrap_one_example'):
            exmaple = self.verbalizer.wrap_one_example(example)
        example = self.template.wrap_one_example(example)
        encoded_inputs = InputFeatures(
            **self.tokenizer_wrapper.tokenize_one_example(example),
            **example[1])
        return encoded_inputs

    def _prepare_input(self, inputs: InputFeatures):
        return inputs

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Return the evaluation [`~paddle.io.DataLoader`].

        Args:
            eval_dataset (`paddlenlp.datasets.MapDataset`):
                Created by `paddlenlp.prompt.load_dataset`,
                where every item is an InputExample object.
        """
        eval_dataset = self._map_dataset(eval_dataset)
        return super().get_eval_dataloader(eval_dataset)

    def get_test_dataloader(self, test_dataset):
        """
        Return the test [`~paddle.io.DataLoader`].

        Args:
            test_dataset (`paddlenlp.datasets.MapDataset`):
                The test dataset created by `paddlenlp.prompt.load_dataset`,
                where every item is an InputExample object.
        """
        test_dataset = self._map_dataset(test_dataset)
        return super().get_test_dataloader(test_dataset)

    def create_optimizer(self, lr_scheduler=None):
        """
        Setup the optimizer for both model and prompt parameters.
        """
        if self.optimizer is None:
            decay_parameters = [
                p.name for n, p in self.model.plm.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ]
            apply_decay_param_fun = lambda x: x in decay_parameters

            optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args)

            parameters = [{'params': self.model.plm.parameters()}]

            ppt_parameters = [
                p for p in self.template.parameters() if not p.stop_gradient
            ]
            if self.verbalizer is not None:
                ppt_parameters.extend([
                    p for p in self.verbalizer.parameters()
                    if not p.stop_gradient
                ])
            ppt_lr = self.args.ppt_learning_rate / self.args.learning_rate
            if len(ppt_parameters) > 0:
                parameters.append({
                    "params": ppt_parameters,
                    "learning_rate": ppt_lr,
                    "weight_decay": self.args.ppt_weight_decay,
                    "beta1": self.args.ppt_adam_beta1,
                    "beta2": self.args.ppt_adam_beta2,
                    "epsilon": self.args.ppt_adam_epsilon
                })

            self.optimizer = optim_cls(
                learning_rate=self.lr_scheduler
                if lr_scheduler is None else lr_scheduler,
                apply_decay_param_fun=apply_decay_param_fun,
                parameters=parameters,
                weight_decay=self.args.weight_decay,
                grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm),
                **optim_kwargs)

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the total loss for every batch. 
        """
        if "labels" not in inputs:
            raise ValueError("Fail to compute loss as there are no labels "\
                             "in {}.".format(inputs))
        labels = inputs["labels"]

        outputs, hidden_states = self.model(**inputs, return_hidden_states=True)

        if self.criterion is not None:
            loss = self.criterion(outputs, labels)
            outputs = (loss, outputs)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.use_rgl:
            loss += self._compute_rgl_loss(hidden_states, labels)

        return (loss, outputs) if return_outputs else loss

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
        bce_criterion = nn.CrossEntropyLoss()
        cos_criterion = nn.CosineSimilarity(axis=0, eps=1e-6)
        batch_size = embeddings.shape[0]
        loss = 0
        for i in range(batch_size):
            for j in range(batch_size):
                score = cos_criterion(embeddings[i], embeddings[j])
                score = score.unsqueeze(0)
                logits = paddle.concat([(1 - score) * 50, (1 + score) * 50],
                                       axis=-1)
                label = paddle.to_tensor([equals(labels[i], labels[j])])
                logits = logits.reshape([-1, logits.shape[-1]])
                loss += bce_criterion(logits, label.unsqueeze(0))
        loss = loss / (batch_size * (batch_size - 1))
        loss = loss / 100 * self.args.alpha_rgl

        return loss


class PromptModelForClassification(nn.Layer):
    """
    PromptModel for classification tasks.
    """

    def __init__(self,
                 model,
                 template,
                 verbalizer=None,
                 freeze_plm: bool = False,
                 freeze_dropout: bool = False):
        super().__init__()
        self.plm = model
        self.template = template
        self.verbalizer = verbalizer
        self.freeze_dropout = freeze_dropout
        if self.verbalizer is not None and hasattr(verbalizer, "process_model"):
            self.plm = self.verbalizer.process_model(self.plm)
        if freeze_dropout:
            self.plm.eval()
        if freeze_plm or freeze_dropout:
            for param in self.plm.parameters():
                param.stop_gradient = True
        self.forward_keys = signature(self.plm.forward)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if self.freeze_dropout:
            self.plm.eval()
        return_hidden_states = kwargs.pop('return_hidden_states', False)
        inputs = InputFeatures(input_ids=input_ids,
                               attention_mask=attention_mask,
                               **kwargs)
        if hasattr(self.template, "process_batch"):
            inputs = self.template.process_batch(inputs)
        model_inputs = {
            k: inputs[k]
            for k in inputs.keys(keep_none=True) if k in self.forward_keys
        }
        outputs = self.plm(**model_inputs)
        hidden_states = outputs
        if hasattr(self.template, "post_process_batch"):
            outputs = self.template.post_process_batch(outputs)
        if self.verbalizer is not None and hasattr(self.verbalizer,
                                                   "process_outputs"):
            outputs = self.verbalizer.process_outputs(outputs, inputs=inputs)

        if return_hidden_states:
            return outputs, hidden_states
        else:
            return outputs

    def prompt_state_dict(self, destination=None):
        """
        Get all parameters and persistable buffers of prompt-related
        layers to save storage when freeze_model is True.
        """
        if destination is None:
            destination = collections.OrderedDict()
        destination["template"] = self.template.state_dict()
        if self.verbalizer is not None:
            destination["verbalizer"] = self.verbalizer.state_dict()
        return destination

    def set_prompt_state_dict(self, state_dict):
        """
        Set prompt-related parameters and persistable buffers from 
        state_dict, a dictionary with keys `template` and `verbalizer`.
        """
        self.template.set_state_dict(state_dict["template"])
        if self.verbalizer is not None:
            self.verbalizer.set_state_dict(state_dict["verbalizer"])

    def plm_state_dict(self):
        """
        Get all parameters and persistable buffers of plm.
        """
        return self.plm.state_dict()

    def set_plm_state_dict(self, state_dict):
        """
        Set parameters and persistable buffers from state_dict for plm.
        """
        self.plm.set_state_dict(state_dict)
