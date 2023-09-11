# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.optimizer.lr import LambdaDecay
from rouge import Rouge

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.metrics import BLEU
from paddlenlp.trainer import Trainer
from paddlenlp.utils.log import logger

PREFIX_CHECKPOINT_DIR = "model_state"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\.tp(\d+)" + ".pdparams$")


_hcg = None


def set_hcg(hcg):
    global _hcg
    _hcg = hcg


def get_hcg():
    global _hcg
    return _hcg


def set_seed(seed):
    # NOTE(shenliang03): For parameter init seed:
    # seed: dp/mp_undistributed_paramter/sharding is same; others is different
    # For compute seed(dropout):
    # global seed: only mp group is same.
    # local seed: all groups are different

    hcg = get_hcg()
    if paddle.distributed.get_world_size() > 1:
        # obtain rank message of hybrid parallel

        mp_rank = hcg.get_model_parallel_rank()
        mp_size = hcg.get_model_parallel_world_size()

        pp_rank = hcg.get_stage_id()
        pp_size = hcg.get_pipe_parallel_world_size()

        dp_rank = hcg.get_data_parallel_rank()
        dp_size = hcg.get_data_parallel_world_size()

        sharding_rank = hcg.get_sharding_parallel_rank()
        # sharding_size = hcg.get_sharding_parallel_world_size()
    else:
        mp_rank, mp_size = 0, 1
        pp_rank, pp_size = 0, 1
        dp_rank, dp_size = 0, 1
        sharding_rank, _ = 0, 1

    # NOTE: the commented seeds are set only for precision validation
    # seed += 100 * pp_rank
    random.seed(seed + 100 * pp_rank)
    np.random.seed(seed + 100 * pp_rank)

    # seed = mp_rank +
    #        pp_rank * (mp_size) +
    #        dp_rank * (mp_size * pp_size) +
    #        sharding_rank * (mp_size * pp_size * dp_size)
    # seed offset is order to avoid conflicts with the parameter initialization seed

    seed_offset = seed + 1024 + paddle.distributed.get_world_size()
    global_seed = (
        seed_offset
        + pp_rank * (mp_size)
        + dp_rank * (mp_size * pp_size)
        + sharding_rank * (mp_size * pp_size * dp_size)
    )

    seed_offset += paddle.distributed.get_world_size()
    local_seed = (
        seed_offset
        + mp_rank
        + pp_rank * (mp_size)
        + dp_rank * (mp_size * pp_size)
        + sharding_rank * (mp_size * pp_size * dp_size)
    )

    tracker = get_rng_state_tracker()
    tracker.add("global_seed", global_seed)
    tracker.add("local_seed", local_seed)

    paddle.seed(global_seed)

    logger.info("The global seed is set to {} and local seed is set to {}.".format(global_seed, local_seed))


def create_hcg(strategy, hcg_name="HybridCommunicateGroup"):
    if hcg_name == "HybridCommunicateGroup":
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
    else:
        dist.init_parallel_env()
        hcg = eval("{}".format(hcg_name))(strategy)

    return hcg


def init_dist_env(
    tensor_parallel_degree=1, sharding_parallel_degree=1, pipeline_parallel_degree=1, data_parallel_degree=1, seed=1
):

    strategy = fleet.DistributedStrategy()

    def is_segment_parallel_supported():
        import inspect

        members = [name for (name, date) in inspect.getmembers(fleet.HybridCommunicateGroup)]
        return "get_sep_parallel_world_size" in members

    if tensor_parallel_degree == 1 and sharding_parallel_degree == 1:
        if is_segment_parallel_supported():
            order = ["pp", "dp", "sharding", "sep", "mp"]
        else:
            order = ["pp", "dp", "sharding", "mp"]
    else:
        if is_segment_parallel_supported():
            order = ["dp", "pp", "sharding", "sep", "mp"]
        else:
            order = ["dp", "pp", "sharding", "mp"]

    strategy.hybrid_configs = {
        "dp_degree": data_parallel_degree,
        "mp_degree": tensor_parallel_degree,
        "pp_degree": pipeline_parallel_degree,
        "sharding_degree": sharding_parallel_degree,
        "order": order,
    }

    # TODO(wawltor) The inference parallel do not support the pipeline mode

    """
    if pipeline_parallel_degree > 1:
        if "sequence_parallel" in config.Model:
            if config.Model.sequence_parallel:
                assert config.Global.enable_partial_send_recv is False, (
                    "if config.Distributed.pp_degree > 1 and config.Model.sequence_parallel is True, "
                    "config.Global.enable_partial_send_recv should be set False."
                )

    strategy.pipeline_configs = {
        "accumulate_steps": config.Global.local_batch_size // config.Global.micro_batch_size,
        "micro_batch_size": config.Global.micro_batch_size,
        "enable_partial_send_recv": config.Global.enable_partial_send_recv,
    }
    """

    # set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": seed}

    hcg = create_hcg(strategy)
    set_hcg(hcg)


def convert_example(
    example,
    tokenizer,
    max_source_length,
    max_target_length,
    is_test=False,
):
    """
    Convert an example into necessary features.
    """
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    context = example["context"]
    question = example["question"]
    try:
        answer = example["answers"][0]
    except Exception:
        print(example["context"])
        print(example["question"])
        print(example["answers"])
        print(example["answer_starts"])
        print(example["is_impossible"])

    input_seq = f"answer: {answer} context: {context} </s>"
    output_seq = f"question: {question} </s>"

    outputs = tokenizer(
        output_seq,
        max_length=max_target_length,
        # pad_to_max_seq_len=True,
        truncation_strategy="longest_first",
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    inputs = tokenizer(
        input_seq,
        max_length=max_source_length,
        # pad_to_max_seq_len=True,
        truncation_strategy="longest_first",
        return_attention_mask=False,
        return_length=False,
    )

    final = {}
    for k in outputs.keys():
        final[k] = inputs[k] + outputs[k]
        if k == "input_ids":
            final["labels"] = [tokenizer.pad_token_id] * len(inputs["input_ids"]) + outputs[k]
    if is_test:
        return dict(input_ids=inputs["input_ids"], labels=outputs["input_ids"])

    # shift inputs and labels
    final["input_ids"] = final["input_ids"][:-1]
    final["labels"] = final["labels"][1:]
    return final


def compute_metrics(preds, targets):
    assert len(preds) == len(targets), (
        "The length of pred_responses should be equal to the length of "
        "target_responses. But received {} and {}.".format(len(preds), len(targets))
    )
    rouge = Rouge()
    bleu4 = BLEU(n_size=4)
    scores = []
    for pred, target in zip(preds, targets):
        try:
            score = rouge.get_scores(" ".join(pred), " ".join(target))
            scores.append([score[0]["rouge-1"]["f"], score[0]["rouge-2"]["f"], score[0]["rouge-l"]["f"]])
        except ValueError:
            scores.append([0, 0, 0])
        bleu4.add_inst(pred, [target])
    rouge1 = np.mean([i[0] for i in scores])
    rouge2 = np.mean([i[1] for i in scores])
    rougel = np.mean([i[2] for i in scores])

    rouge1 = round(rouge1, 4)
    rouge2 = round(rouge2, 4)
    rougel = round(rougel, 4)
    bleu4 = round(bleu4.score(), 4)
    return dict(
        rouge1=rouge1,
        rouge2=rouge2,
        rougel=rougel,
        bleu4=bleu4,
    )


class DataCollatorForSupervisedDataset(DataCollatorForSeq2Seq):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, features, return_tensors=None):
        # Deep copy to avoid modifying features in-place
        batch = copy.deepcopy(features)
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in batch] if "labels" in batch[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            # Note(gongenlei): In pipeline, max_label_length = self.max_length
            if self.padding == "max_length" and self.max_length is not None:
                max_label_length = self.max_length
            else:
                max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in batch:
                remainder = [self.tokenizer.pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
            return_attention_mask=self.return_attention_mask,
        )

        return batch


class GPTTrainer(Trainer):
    def __init__(self, do_generation: bool, **kwargs):
        super().__init__(**kwargs)
        self.do_generation = do_generation

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:

        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        elif not self.do_generation:
            loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            # argmax here to avoid gather all logits, which is too memory-consuming.
            # keepdim in order to maintain the same shape as logits
            return (loss, logits.argmax(axis=-1, keepdim=True), labels)

        model.eval()

        preds = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
            max_length=self.args.tgt_length,
            min_length=0,
            use_cache=True,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
            decode_strategy="sampling",
        )[0]
        all_labels = []
        for label in inputs["labels"].numpy():
            label = [x for x in label[label != self.tokenizer.pad_token_id]]
            all_labels.append(label)
        max_label_length = max([len(x) for x in all_labels])
        for index, labels in enumerate(all_labels):
            all_labels[index] = labels + [-100] * (max_label_length - len(labels))

        return (None, paddle.to_tensor(preds), paddle.to_tensor(all_labels))

    def create_scheduler(self, num_training_steps: int):
        num_warmup_steps = (
            self.args.warmup_steps if self.args.warmup_steps > 0 else self.args.warmup_ratio * num_training_steps
        )

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                decay_step_ratio = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
                return 1.0 - (1.0 - self.args.lr_decay_ratio) * decay_step_ratio

        if self.lr_scheduler is None:
            self.lr_scheduler = LambdaDecay(self.args.learning_rate, lr_lambda, last_epoch=-1)
        return self.lr_scheduler

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        if "loss" in logs:
            logs["ppl"] = np.exp(logs["loss"])
        if "eval_loss" in logs:
            logs["eval_ppl"] = np.exp(logs["eval_loss"])

        super(GPTTrainer, self).log(logs, **kwargs)
