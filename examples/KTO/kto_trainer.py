# KTO Authors: Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext
from operator import itemgetter
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from datasets import Dataset, concatenate_datasets
from kto_config import KTOConfig
from model_base import create_reference_model
from paddle.io import DataLoader, DistributedBatchSampler
from tqdm import tqdm
from utils import (
    DPODataCollatorWithPadding,
    PaddlePartialState,
    disable_dropout_in_model,
    distribute_gather,
    pad_to_length,
)

from paddlenlp.data import DataCollator
from paddlenlp.peft import LoRAModel
from paddlenlp.trainer import Trainer, TrainingArguments
from paddlenlp.trainer.integrations import is_wandb_available
from paddlenlp.trainer.trainer_callback import TrainerCallback
from paddlenlp.trainer.trainer_utils import EvalLoopOutput, has_length
from paddlenlp.transformers import AutoModelForCausalLM, PretrainedModel
from paddlenlp.transformers.tokenizer_utils_base import PretrainedTokenizerBase

if is_wandb_available():
    import wandb


def _get_kl_dataset(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Creates mismatched pairs of prompts and completions for the KL dataset by reversing the order of completions."""
    batch["answer_input_ids"] = batch["answer_input_ids"][::-1]
    batch["answer_attention_mask"] = batch["answer_attention_mask"][::-1]
    return batch


def _tokenize(batch: Dict[str, List[Any]], tokenizer: PretrainedTokenizerBase) -> Dict[str, List[Any]]:
    """Tokenize a batch from a KTO specific dataset."""
    prompt_tokenized = tokenizer(batch["prompt"], add_special_tokens=False)
    prompt_input_ids = prompt_tokenized["input_ids"]
    prompt_attention_mask = prompt_tokenized["attention_mask"]
    prompt_and_completion = [prompt + completion for prompt, completion in zip(batch["prompt"], batch["completion"])]
    # prompt_and_completion = batch["completion"]
    full_tokenized = tokenizer(prompt_and_completion, add_special_tokens=False)
    full_input_ids = full_tokenized["input_ids"]
    full_attention_mask = full_tokenized["attention_mask"]

    answer_input_ids = [f[len(p) :] for f, p in zip(full_input_ids, prompt_input_ids)]
    answer_attention_mask = [f[len(p) :] for f, p in zip(full_attention_mask, prompt_attention_mask)]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = [np.concatenate([p, a]) for p, a in zip(prompt_input_ids, answer_input_ids)]
    # Prepare input tokens for token by token comparison
    full_input_ids = [np.array(f) for f in full_input_ids]
    for full, concat in zip(full_input_ids, full_concat_input_ids):
        if len(full) != len(concat):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = [len(p) for p in prompt_input_ids]

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    for idx, (p, f, r) in enumerate(zip(prompt_input_ids, full_input_ids, response_token_ids_start_idx)):
        if not np.array_equal(p, f[:r]):
            response_token_ids_start_idx[idx] -= 1

    prompt_input_ids = [f[:r] for f, r in zip(full_input_ids, response_token_ids_start_idx)]
    prompt_attention_mask = [f[:r] for f, r in zip(full_attention_mask, response_token_ids_start_idx)]

    for p, m in zip(prompt_input_ids, prompt_attention_mask):
        if len(p) != len(m):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

    answer_input_ids = [f[r:] for f, r in zip(full_input_ids, response_token_ids_start_idx)]
    answer_attention_mask = [f[r:] for f, r in zip(full_attention_mask, response_token_ids_start_idx)]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        answer_input_ids=answer_input_ids,
        answer_attention_mask=answer_attention_mask,
    )


def _process_tokens(example: Dict[str, Any], model: "PretrainedModel" = None, **kwargs) -> Dict:
    """Process tokens of a KTO specific dataset.

    At this stage, we don't convert to Pypaddle tensors yet; we just handle the truncation
    in case the prompt + completion responses is/are too long. First
        we truncate the prompt; if we're still too long, we truncate the completion.

    We also create the labels for the completion responses, which are of length equal to
        the sum of the length of the prompt and the completion response, with
        label_pad_token_id  for the prompt tokens.
    """
    prompt = example["prompt"]
    completion = example["completion"]

    batch = {
        f"{kwargs['prefix']}prompt": prompt,
        f"{kwargs['prefix']}completion": completion,
        f"{kwargs['prefix']}label": example["label"],
    }

    if not kwargs["is_encoder_decoder"]:
        # Check issues below for more details
        #  1. https://github.com/huggingface/trl/issues/907
        #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        #  3. https://github.com/LianjiaTech/BELLE/issues/337

        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")

        if not isinstance(completion, str):
            raise ValueError(f"completion should be an str but got {type(completion)}")

        # keys of format prompt_* refers to just the prompt and answer_* refers to just the answer
        all_tokens = {
            "prompt_input_ids": example["prompt_input_ids"],
            "prompt_attention_mask": example["prompt_attention_mask"],
            "answer_input_ids": example["answer_input_ids"],
            "answer_attention_mask": example["answer_attention_mask"],
        }

        max_length = kwargs["max_length"] - 2
        # if combined sequence is too long (> max_length - 1 for BOS token - 1 for EOS), truncate the prompt
        if len(all_tokens["prompt_input_ids"]) + len(all_tokens["answer_input_ids"]) > max_length:
            for k in ["prompt_input_ids", "prompt_attention_mask"]:
                if kwargs["truncation_mode"] == "keep_start":
                    all_tokens[k] = all_tokens[k][: kwargs["max_prompt_length"]]
                elif kwargs["truncation_mode"] == "keep_end":
                    all_tokens[k] = all_tokens[k][-kwargs["max_prompt_length"] :]
                else:
                    raise ValueError(f"Unknown truncation mode: {kwargs['truncation_mode']}")

        # if that's still too long, truncate the response
        if len(all_tokens["prompt_input_ids"]) + len(all_tokens["answer_input_ids"]) > max_length:
            for k in ["answer_input_ids", "answer_attention_mask"]:
                all_tokens[k] = all_tokens[k][: max_length - kwargs["max_prompt_length"]]

        # for legacy reasons, use the completion_* prefix to now refer to the joint sequence
        batch[f"{kwargs['prefix']}prompt_input_ids"] = [kwargs["tokenizer"].bos_token_id] + all_tokens[
            "prompt_input_ids"
        ]
        batch[f"{kwargs['prefix']}prompt_attention_mask"] = [1] + all_tokens["prompt_attention_mask"]
        batch[f"{kwargs['prefix']}completion_input_ids"] = (
            [kwargs["tokenizer"].bos_token_id]
            + all_tokens["prompt_input_ids"]
            + all_tokens["answer_input_ids"]
            + [kwargs["tokenizer"].eos_token_id]
        )
        batch[f"{kwargs['prefix']}completion_attention_mask"] = (
            [1] + all_tokens["prompt_attention_mask"] + all_tokens["answer_attention_mask"] + [1]
        )

        batch[f"{kwargs['prefix']}completion_labels"] = batch[f"{kwargs['prefix']}completion_input_ids"][:]
        batch[f"{kwargs['prefix']}completion_labels"][: len(batch[f"{kwargs['prefix']}prompt_input_ids"])] = [
            kwargs["label_pad_token_id"]
        ] * len(batch[f"{kwargs['prefix']}prompt_input_ids"])
    else:
        completion_tokens = kwargs["tokenizer"](
            completion, truncation=True, max_length=kwargs["max_completion_length"], add_special_tokens=True
        )
        prompt_tokens = kwargs["tokenizer"](
            prompt, truncation=True, max_length=kwargs["max_prompt_length"], add_special_tokens=True
        )

        batch[f"{kwargs['prefix']}prompt_input_ids"] = prompt_tokens["input_ids"]
        batch[f"{kwargs['prefix']}prompt_attention_mask"] = prompt_tokens["attention_mask"]

        batch[f"{kwargs['prefix']}completion_labels"] = completion_tokens["input_ids"]
        batch[f"{kwargs['prefix']}completion_attention_mask"] = completion_tokens["attention_mask"]
        if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
            batch[f"{kwargs['prefix']}completion_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=paddle.to_tensor(batch["completion_labels"])
            )

    return batch


class KTOTrainer(Trainer):
    r"""
    Initialize KTOTrainer.

    Args:
        model (`transformers.PretrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PretrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        args (`KTOConfig`):
            The arguments to use for training.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PretrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        data_collator (`transformers.DataCollator`, *optional*, defaults to `None`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        model_init (`Callable[[], transformers.PretrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LambdaDecay]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "kto"]

    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer, str] = None,
        ref_model: Optional[Union[PretrainedModel, nn.Layer, str]] = None,
        args: KTOConfig = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PretrainedTokenizerBase] = None,
        data_collator: Optional[DataCollator] = None,
        model_init: Optional[Callable[[], PretrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LambdaDecay] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        if type(args) == TrainingArguments:
            raise ValueError("Please use `KTOConfig` instead TrainingArguments.")

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the KTOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["paddle_dtype"] = (
                model_init_kwargs["paddle_dtype"]
                if model_init_kwargs["paddle_dtype"] in ["auto", None]
                else getattr(paddle, model_init_kwargs["paddle_dtype"])
            )

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the KTOTrainer. But your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            ref_model_init_kwargs["paddle_dtype"] = (
                ref_model_init_kwargs["paddle_dtype"]
                if ref_model_init_kwargs["paddle_dtype"] in ["auto", None]
                else getattr(paddle, ref_model_init_kwargs["paddle_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the KTOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `LoRAModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the KTOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False
        if peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, LoRAModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(args, "recompute_kwargs")

                prepare_model_kwargs = {"use_recompute": args.recompute}

                if _support_gc_kwargs:
                    prepare_model_kwargs["recompute_kwargs"] = args.recompute_kwargs

                # model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

            # get peft model with the given config
            model = LoRAModel(model, peft_config)
            model.mark_only_lora_as_trainable()
            model.print_trainable_parameters()

        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install with `pip install wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        self.is_peft_model = isinstance(model, LoRAModel)

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError(
                "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
            )
        if args.max_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_length` in the KTOTrainer's init"
                " it will be set to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if args.max_length is not None:
            max_length = args.max_length

        if args.max_prompt_length is None:
            warnings.warn(
                "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the KTOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128
        if args.max_prompt_length is not None:
            max_prompt_length = args.max_prompt_length

        max_completion_length = None
        if args.max_completion_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using DPODataCollatorWithPadding with an encoder decoder architecture, you should set `max_completion_length` in the KTOTrainer's init"
                " it will be set to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_completion_length = 128
        if args.max_completion_length is not None and self.is_encoder_decoder:
            max_completion_length = args.max_completion_length

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your KTOConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        # disable dropout in the model and reference model
        disable_dropout_in_model(model)
        if self.ref_model is not None:
            disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_completion_length = max_completion_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        # metric
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # KTO parameter
        self.beta = args.beta
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight

        with PaddlePartialState().local_main_process_first():
            # Shuffle the datasets
            train_dataset = train_dataset.shuffle(seed=args.data_seed)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.shuffle(seed=args.data_seed)
            # Tokenize and prepare the training datasets
            train_dataset = train_dataset.map(
                _tokenize,
                fn_kwargs={"tokenizer": self.tokenizer},
                batched=True,
                desc="Tokenizing train dataset",
            )
            # Get KL datasets
            total_batch_size = (
                max(paddle.device.cuda.device_count(), 1)
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
            )
            if total_batch_size <= 1:
                raise ValueError(
                    "Batch size is 1 (too small). KTO will not work properly because the KL term will be equivalent to the implied reward."
                )
            # create pairs for estimating the KL term by flipping the matched pairs in each batch of size total_batch_size
            # i.e., (x_1, y_1), ..., (x_n, y_n) --> (x_1, y_n), ..., (x_n, y_1) = (x'_1, y'_1), ..., (x'_n, y'_n)
            train_kl_dataset = train_dataset.map(
                _get_kl_dataset, batched=True, batch_size=total_batch_size, desc="Extracting KL train dataset"
            )
            # Prepare the datasets
            fn_kwargs = {
                "prefix": "",
                "is_encoder_decoder": self.is_encoder_decoder,
                "tokenizer": self.tokenizer,
                "max_length": self.max_length,
                "truncation_mode": self.truncation_mode,
                "label_pad_token_id": self.label_pad_token_id,
                "max_prompt_length": self.max_prompt_length,
            }
            train_dataset = train_dataset.map(
                _process_tokens,
                fn_kwargs=fn_kwargs,
                num_proc=args.dataset_num_proc,
                desc="Processing tokenized train dataset",
            )
            fn_kwargs["prefix"] = "KL_"
            train_kl_dataset = train_kl_dataset.map(
                _process_tokens,
                fn_kwargs=fn_kwargs,
                num_proc=args.dataset_num_proc,
                remove_columns=[c for c in train_kl_dataset.column_names if c in train_dataset.column_names],
                desc="Processing tokenized train KL dataset",
            )

            # merge the datasets
            train_dataset = concatenate_datasets([train_dataset, train_kl_dataset], axis=1)

            if eval_dataset is not None:
                # Tokenize
                eval_dataset = eval_dataset.map(
                    _tokenize,
                    fn_kwargs={"tokenizer": self.tokenizer},
                    batched=True,
                    desc="Tokenizing eval dataset",
                )
                # Get KL dataset
                eval_kl_dataset = eval_dataset.map(
                    _get_kl_dataset, batched=True, batch_size=total_batch_size, desc="Extracting eval KL dataset"
                )
                # Process
                fn_kwargs = {
                    "prefix": "",
                    "is_encoder_decoder": self.is_encoder_decoder,
                    "tokenizer": self.tokenizer,
                    "max_length": self.max_length,
                    "truncation_mode": self.truncation_mode,
                    "label_pad_token_id": self.label_pad_token_id,
                    "max_prompt_length": self.max_prompt_length,
                }
                eval_dataset = eval_dataset.map(
                    _process_tokens,
                    fn_kwargs=fn_kwargs,
                    num_proc=args.dataset_num_proc,
                    desc="Processing tokenized eval dataset",
                )
                fn_kwargs["prefix"] = "KL_"
                eval_kl_dataset = eval_kl_dataset.map(
                    _process_tokens,
                    fn_kwargs=fn_kwargs,
                    num_proc=args.dataset_num_proc,
                    remove_columns=[c for c in eval_kl_dataset.column_names if c in eval_dataset.column_names],
                    desc="Processing tokenized eval KL dataset",
                )

                # merge the datasets
                eval_dataset = concatenate_datasets([eval_dataset, eval_kl_dataset], axis=1)

            desirable = train_dataset.filter(
                lambda x: x["label"], num_proc=args.dataset_num_proc, desc="Filtering desirable examples"
            )
            undesirable = train_dataset.filter(
                lambda x: not x["label"], num_proc=args.dataset_num_proc, desc="Filtering undesirable examples"
            )

            if len(desirable) != len(undesirable):
                # The lower and upper bounds come from Eq. (8) of https://arxiv.org/abs/2402.01306
                des_weight_lower_bound = round((len(undesirable) * self.undesirable_weight / len(desirable)) * 1, 2)
                des_weight_upper_bound = round((len(undesirable) * self.undesirable_weight / len(desirable)) * 1.33, 2)
                und_weight_lower_bound = round((len(desirable) * self.desirable_weight / len(undesirable)) / 1.33, 2)
                und_weight_upper_bound = round((len(desirable) * self.desirable_weight / len(undesirable)) / 1, 2)

                des_weight_in_range = des_weight_lower_bound <= self.desirable_weight <= des_weight_upper_bound
                und_weight_in_range = und_weight_lower_bound <= self.undesirable_weight <= und_weight_upper_bound

                if not (des_weight_in_range or und_weight_in_range):
                    warnings.warn(
                        f"""
                        You have different amounts of desirable/positive and undesirable/negative examples but the
                        weights on the desirable and undesirable losses don't seem to be in an ideal range. Based
                        on your data, we recommend EITHER desirable_weight in [{des_weight_lower_bound}, {des_weight_upper_bound}]
                        or undesirable_weight in [{und_weight_lower_bound}, {und_weight_upper_bound}] (but NOT BOTH).
                        See the documentation on how to optimally set these weights.""",
                        UserWarning,
                    )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            # model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            self.ref_model.eval()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~paddle.io.DataLoader`].

        Subclass of paddlenlp.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = DataLoader(self.train_dataset, **dataloader_params)
            reference_completion_logps = []
            reference_KL_logps = []

            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_completion_logp, reference_KL_logp = self.compute_reference_log_probs(padded_batch)
                if dist.is_initialized():
                    reference_completion_logp = distribute_gather(reference_completion_logp)
                reference_completion_logps.append(reference_completion_logp.cpu())
                if dist.is_initialized():
                    reference_KL_logp = distribute_gather(reference_KL_logp)
                reference_KL_logps.append(reference_KL_logp.cpu())

            self.train_dataset = self.train_dataset.add_column(
                name="reference_logps", column=paddle.concat(reference_completion_logps).float().numpy()
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_KL_logps", column=paddle.concat(reference_KL_logps).float().numpy()
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~paddle.io.DataLoader`].

        Subclass of paddlenlp.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`paddle.io.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = DataLoader(eval_dataset, **dataloader_params)

            reference_completion_logps = []
            reference_KL_logps = []

            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_completion_logp, reference_KL_logp = self.compute_reference_log_probs(padded_batch)
                if dist.is_initialized():
                    reference_completion_logp = distribute_gather(reference_completion_logp)

                reference_completion_logps.append(reference_completion_logp.cpu())
                if dist.is_initialized():
                    reference_KL_logp = distribute_gather(reference_KL_logp)
                reference_KL_logps.append(reference_KL_logp.cpu())

            eval_dataset = eval_dataset.add_column(
                name="reference_logps", column=paddle.concat(reference_completion_logps).float().numpy()
            )
            eval_dataset = eval_dataset.add_column(
                name="reference_KL_logps", column=paddle.concat(reference_KL_logps).float().numpy()
            )

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a KTO specific dataset."""
        with paddle.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(
                    self.model
                ).disable_adapter() if self.is_peft_model else nullcontext():
                    if self.is_encoder_decoder:
                        completion_logits = self.model(
                            padded_batch["prompt_input_ids"],
                            attention_mask=padded_batch["prompt_attention_mask"],
                            decoder_input_ids=padded_batch.get("completion_decoder_input_ids"),
                            labels=padded_batch["completion_labels"],
                            return_dict=True,
                        ).logits

                        KL_logits = self.model(
                            padded_batch["KL_prompt_input_ids"],
                            attention_mask=padded_batch["KL_prompt_attention_mask"],
                            decoder_input_ids=padded_batch.get("KL_completion_decoder_input_ids"),
                            labels=padded_batch["KL_completion_labels"],
                            return_dict=True,
                        ).logits
                    else:
                        completion_logits = self.model(
                            padded_batch["completion_input_ids"],
                            attention_mask=padded_batch["completion_attention_mask"],
                            return_dict=True,
                        ).logits

                        KL_logits = self.model(
                            padded_batch["KL_completion_input_ids"],
                            attention_mask=padded_batch["KL_completion_attention_mask"],
                            return_dict=True,
                        ).logits
            else:
                if self.is_encoder_decoder:
                    completion_logits = self.ref_model(
                        padded_batch["prompt_input_ids"],
                        attention_mask=padded_batch["prompt_attention_mask"],
                        decoder_input_ids=padded_batch.get("completion_decoder_input_ids"),
                        labels=padded_batch["completion_labels"],
                        return_dict=True,
                    ).logits

                    KL_logits = self.ref_model(
                        padded_batch["KL_prompt_input_ids"],
                        attention_mask=padded_batch["KL_prompt_attention_mask"],
                        decoder_input_ids=padded_batch.get("KL_completion_decoder_input_ids"),
                        labels=padded_batch["KL_completion_labels"],
                        return_dict=True,
                    ).logits
                else:
                    completion_logits = self.ref_model(
                        padded_batch["completion_input_ids"],
                        attention_mask=padded_batch["completion_attention_mask"],
                        return_dict=True,
                    ).logits

                    KL_logits = self.ref_model(
                        padded_batch["KL_completion_input_ids"],
                        attention_mask=padded_batch["KL_completion_attention_mask"],
                        return_dict=True,
                    ).logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            padded_batch["completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        KL_logps = self.get_batch_logps(
            KL_logits,
            padded_batch["KL_completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        return completion_logps, KL_logps

    @staticmethod
    def get_batch_logps(
        logits: paddle.Tensor,
        labels: paddle.Tensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> paddle.Tensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        else:
            # Fixes end-dec RuntimeError
            labels = labels.clone()

        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0
        per_token_logps = paddle.take_along_axis(
            F.log_softmax(logits, axis=-1), axis=2, indices=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def forward(
        self, model: nn.Layer, batch: Dict[str, Union[List, paddle.Tensor]]
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        if self.is_encoder_decoder:
            with paddle.no_grad():
                KL_logits = model(
                    batch["KL_prompt_input_ids"],
                    attention_mask=batch["KL_prompt_attention_mask"],
                    decoder_input_ids=batch.get("KL_completion_decoder_input_ids"),
                    labels=batch["KL_completion_labels"],
                    return_dict=True,
                ).logits

            completion_logits = model(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                decoder_input_ids=batch.get("completion_decoder_input_ids"),
                labels=batch["completion_labels"],
                return_dict=True,
            ).logits
        else:
            with paddle.no_grad():
                KL_logits = model(
                    batch["KL_completion_input_ids"],
                    attention_mask=batch["KL_completion_attention_mask"],
                    return_dict=True,
                ).logits

            completion_logits = model(
                batch["completion_input_ids"],
                attention_mask=batch["completion_attention_mask"],
                return_dict=True,
            ).logits

        completion_logps = self.get_batch_logps(
            completion_logits,
            batch["completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        KL_logps = self.get_batch_logps(
            KL_logits,
            batch["KL_completion_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        if completion_logps.shape[0] != len(batch["label"]):
            raise ValueError(
                "There is a mismatch between the number of examples in this batch and the number of "
                "examples for which an output sequence was predicted."
            )

        chosen_idx = [i for i in range(completion_logps.shape[0]) if batch["label"][i] is True]
        rejected_idx = [i for i in range(completion_logps.shape[0]) if batch["label"][i] is False]

        chosen_logps = completion_logps[chosen_idx, ...]
        rejected_logps = completion_logps[rejected_idx, ...]

        chosen_logits = completion_logits[chosen_idx, ...]
        rejected_logits = completion_logits[rejected_idx, ...]

        if chosen_logps.shape[0] == 0:
            chosen_logps = paddle.to_tensor([0.0])
            chosen_logits = paddle.to_tensor([0.0])

        if rejected_logps.shape[0] == 0:
            rejected_logits = paddle.to_tensor([0.0])
            rejected_logps = paddle.to_tensor([0.0])

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps)

    def kto_loss(
        self,
        policy_chosen_logps: paddle.Tensor,
        policy_rejected_logps: paddle.Tensor,
        policy_KL_logps: paddle.Tensor,
        reference_chosen_logps: paddle.Tensor,
        reference_rejected_logps: paddle.Tensor,
        reference_KL_logps: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute the KTO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (num(chosen) in batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (num(rejected) in batch_size,)
            policy_KL_logps: Log probabilities of the policy model for the KL responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (num(chosen) in batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (num(rejected) in batch_size,)
            reference_KL_logps: Log probabilities of the reference model for the KL responses. Shape: (batch_size,)

        Returns:
            A tuple of four tensors: (losses, chosen_rewards, rejected_rewards, KL).
            The losses tensor contains the KTO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The KL tensor contains the detached KL divergence estimate between the policy and reference models.
        """
        kl = (policy_KL_logps - reference_KL_logps).mean().detach()
        if dist.is_initialized():
            gathered_kl_score_list = [paddle.zeros_like(kl) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_kl_score_list, kl)
            # ValueError: (InvalidArgument) The axis is expected to be in range of [0, 0), but got 0
            gathered_kl_score_list = [item.unsqueeze(0) for item in gathered_kl_score_list]
            gathered_kl_score = paddle.concat(gathered_kl_score_list, axis=0)
            kl = gathered_kl_score.mean().clip(min=0)
        else:
            kl = kl.mean().clip(min=0)

        if policy_chosen_logps.shape[0] != 0 or reference_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            chosen_losses = paddle.to_tensor([0.0])
            chosen_rewards = paddle.to_tensor([0.0])

        if policy_rejected_logps.shape[0] != 0 or reference_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            rejected_losses = paddle.to_tensor([0.0])
            rejected_rewards = paddle.to_tensor([0.0])

        losses = paddle.concat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses),
            0,
        )

        return losses, chosen_rewards, rejected_rewards, kl

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, paddle.Tensor]],
    ):
        """Compute the KTO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        batch = {k: (v if isinstance(v, paddle.Tensor) else v) for k, v in batch.items()}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_KL_logps,
        ) = self.forward(model, batch)

        # if reference_logps in batch use them, otherwise use the reference model
        if "reference_logps" in batch:
            chosen_idx = [i for i in range(batch["reference_logps"].shape[0]) if batch["label"][i] is True]
            rejected_idx = [i for i in range(batch["reference_logps"].shape[0]) if batch["label"][i] is False]

            reference_chosen_logps = batch["reference_logps"][chosen_idx, ...]
            reference_rejected_logps = batch["reference_logps"][rejected_idx, ...]
            reference_KL_logps = batch["reference_KL_logps"]
        else:
            with paddle.no_grad():
                if self.ref_model is None:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            reference_KL_logps,
                        ) = self.forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        reference_KL_logps,
                    ) = self.forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards, kl = self.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_KL_logps,
        )

        num_chosen = paddle.to_tensor([len(chosen_rewards)])
        num_rejected = paddle.to_tensor([len(rejected_rewards)])

        if dist.is_initialized():
            # TODO test
            all_num_chosen = distribute_gather(num_chosen).sum().item()
            all_num_rejected = distribute_gather(num_rejected).sum().item()
        else:
            all_num_chosen = num_chosen.sum().item()
            all_num_rejected = num_rejected.sum().item()

        if all_num_chosen > 0:
            if dist.is_initialized():
                metrics["rewards/chosen_sum"] = distribute_gather(chosen_rewards.nansum()).nansum().item()
                metrics["logps/chosen_sum"] = distribute_gather(policy_chosen_logps.nansum()).nansum().item()
                metrics["count/chosen"] = all_num_chosen
            else:
                metrics["rewards/chosen_sum"] = chosen_rewards.nansum().item()
                metrics["logps/chosen_sum"] = policy_chosen_logps.nansum().item()
                metrics["count/chosen"] = all_num_chosen

        if all_num_rejected > 0:
            if dist.is_initialized():
                metrics["rewards/rejected_sum"] = distribute_gather(rejected_rewards.nansum()).nansum().item()
                metrics["logps/rejected_sum"] = distribute_gather(policy_rejected_logps.nansum()).nansum().item()
                metrics["count/rejected"] = all_num_rejected
            else:
                metrics["rewards/rejected_sum"] = rejected_rewards.nansum().item()
                metrics["logps/rejected_sum"] = policy_rejected_logps.nansum().item()
                metrics["count/rejected"] = all_num_rejected

        metrics["kl"] = kl.item()

        return losses.nanmean(), metrics

    def compute_loss(
        self,
        model: Union[PretrainedModel, nn.Layer],
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        return_outputs=False,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, Dict[str, paddle.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        compute_loss_context_manager = paddle.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs)

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        # force log the metrics
        rank = paddle.distributed.get_rank()
        is_main_process = rank == 0
        if is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        # return SequentialSampler(self.train_dataset)
        # if self.train_dataset is None or not has_length(self.train_dataset):
        #     return None

        if self.args.world_size <= 1:
            return paddle.io.BatchSampler(
                dataset=self.train_dataset,
                shuffle=False,
                batch_size=self.args.per_device_train_batch_size,
                drop_last=self.args.dataloader_drop_last,
            )

        return DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
            drop_last=self.args.dataloader_drop_last,
        )

    def get_batch_samples(self, model, batch: Dict[str, paddle.Tensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the paddle cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else paddle.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PretrainedModel, nn.Layer],
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = paddle.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        with paddle.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs)

        # force log the metrics
        rank = paddle.distributed.get_rank()
        is_main_process = rank == 0
        if is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["logits/chosen"],
            "eval_logits/rejected": metrics["logits/rejected"],
        }
        logits = tuple(v.unsqueeze(axis=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = paddle.stack(logits).mean(axis=1)
        labels = paddle.zeros(logits.shape[0])

        return (loss.detach(), logits, labels)

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
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            target_indicies = [i for i in range(len(random_batch["kl"])) if random_batch["kl"][i] is False]
            target_batch = {
                "prompt_input_ids": itemgetter(*target_indicies)(random_batch["prompt_input_ids"]),
                "prompt_attention_mask": itemgetter(*target_indicies)(random_batch["prompt_attention_mask"]),
                "prompt": itemgetter(*target_indicies)(random_batch["prompt"]),
            }
            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, target_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                target_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix, max_eval_iters
        )

        return initial_output

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # accumulate average metrics from sums and lengths
        for split in ["chosen", "rejected"]:
            if f"count/{split}" in self._stored_metrics[train_eval]:
                count_sum = paddle.to_tensor(self._stored_metrics[train_eval][f"count/{split}"]).sum().item()
                logs[f"{train_eval}/rewards/{split}"] = (
                    paddle.to_tensor(self._stored_metrics[train_eval][f"rewards/{split}_sum"]).sum().item() / count_sum
                )
                logs[f"{train_eval}/logps/{split}"] = (
                    paddle.to_tensor(self._stored_metrics[train_eval][f"logps/{split}_sum"]).sum().item() / count_sum
                )
                for key in [f"count/{split}", f"rewards/{split}_sum", f"logps/{split}_sum"]:
                    del self._stored_metrics[train_eval][key]
        # calculate reward margin
        if f"{train_eval}/rewards/chosen" in logs and f"{train_eval}/rewards/rejected" in logs:
            logs[f"{train_eval}/rewards/margins"] = (
                logs[f"{train_eval}/rewards/chosen"] - logs[f"{train_eval}/rewards/rejected"]
            )
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[f"{train_eval}/{key}"] = paddle.to_tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)
