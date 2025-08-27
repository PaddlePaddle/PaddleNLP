# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Tuple, Union

import paddle
import requests
from paddle import nn
from paddle.distributed import fleet
from paddle.io import Dataset

from ...data import DataCollator
from ...datasets.rlhf_datasets.protocol import DataProto
from ...trainer.trainer import (
    EvalPrediction,
    TrainerCallback,
    TrainingArguments,
    logger,
)
from ...transformers import PretrainedModel, PretrainedTokenizer
from ..models.ppo_model_utils import create_startend_row_indices
from ..models.reward_utils import extract_solution
from .rl_trainer import RLTrainer
from .trainer_utils import batch_retokenize


class RewardTrainer(RLTrainer):
    trainer_type = "reward"

    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]] = None,
        reward_server: str = None,
    ):
        if args.use_rule_reward:
            self.args = args
            self.tokenizer = tokenizer
        elif args.use_rm_server:
            assert isinstance(model, str), "reward trainer need a str (http://xxx:port) for request"
            self.args = args
            self.tokenizer = tokenizer
            self.model = reward_server
        else:
            assert isinstance(model, PretrainedModel), "reward trainer need a PretrainedModel instance for forward"
            super().__init__(
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

    @paddle.no_grad()
    def compute_reward(
        self,
        batch: DataProto,
        input_ids_tokenizer: PretrainedTokenizer = None,
    ) -> Dict[str, paddle.Tensor]:
        input_ids = batch.batch["input_ids"]
        position_ids = batch.batch["position_ids"]
        label_ids = batch.batch["label_ids"]
        prompt = batch.batch["prompt"]

        if self.args.use_rule_reward:
            prompt_len = prompt.shape[-1]
            if label_ids is None:
                raise ValueError("Rule-based reward needs labels.")
            tgt = input_ids_tokenizer.batch_decode(label_ids, skip_special_tokens=False)
            response = input_ids_tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=False)
            ground_truth = [i.replace(self.tokenizer.pad_token, "") for i in tgt]
            response = [i.replace(self.tokenizer.pad_token, "") for i in response]
            reward_score = self.compute_score(
                solution=response,
                ground_truth=ground_truth,
            )
        elif not self.args.use_rm_server:
            if self.tokenizer is not input_ids_tokenizer:
                # right padding
                reward_tokenize_output = batch_retokenize(
                    input_ids,
                    src_tokenizer=input_ids_tokenizer,
                    dest_tokenizer=self.tokenizer,
                )
                reward_input_ids = reward_tokenize_output["input_ids"]
                reward_position_ids = reward_tokenize_output["position_ids"]
            else:
                reward_input_ids = input_ids
                reward_position_ids = position_ids

            attn_mask_startend_row_indices = create_startend_row_indices(reward_input_ids, self.tokenizer.pad_token_id)
            reward_score = self.model(
                reward_input_ids,
                attention_mask=None,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                position_ids=reward_position_ids,
            )[1]
        else:
            prompt_len = prompt.shape[-1]
            if label_ids is None:
                raise ValueError("Rule-based reward needs labels.")
            src = input_ids_tokenizer.batch_decode(input_ids[:, :prompt_len], skip_special_tokens=False)
            tgt = input_ids_tokenizer.batch_decode(label_ids, skip_special_tokens=False)
            response = input_ids_tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=False)
            reward_score = self.request_reward_server(
                [i.replace(self.tokenizer.pad_token, "") for i in src],
                [i.replace(self.tokenizer.pad_token, "") for i in tgt],
                [i.replace(self.tokenizer.pad_token, "") for i in response],
            )

        reward_score = reward_score.squeeze(axis=-1)

        return reward_score

    def request_reward_server(self, src, tgt, response):
        data = {"src": src, "tgt": tgt, "response": response}
        dtype = self.args.model_dtype

        def post():
            try:
                res = requests.post(self.model, json=data)
                result = json.loads(res.text)
                reward_score = paddle.to_tensor(
                    result["score"], dtype=dtype if not self.args.use_fp32_compute else "float32"
                )
            except Exception as e:
                logger.warning(f"Request reward server failed({e}) and rewards_score will be set zero.")
                reward_score = paddle.zeros(
                    len(response), dtype=dtype if not self.args.use_fp32_compute else "float32"
                )
            return reward_score

        try:
            hcg = fleet.get_hybrid_communicate_group()
            tp_group = hcg.get_model_parallel_group()
            nranks = tp_group.nranks
            tp_rank = hcg.get_model_parallel_rank()
        except Exception:
            nranks = 1
            tp_rank = 0

        if nranks == 1:
            reward_score = post()
        else:
            if tp_rank == 0:
                reward_score = post()
            else:
                reward_score = paddle.empty(
                    shape=[len(response)],
                    dtype=dtype if not self.args.use_fp32_compute else "float32",
                )
            paddle.distributed.barrier(tp_group)
            paddle.distributed.broadcast(reward_score, src=tp_group.ranks[0], group=tp_group)

        reward_score = reward_score.unsqueeze(-1)
        return reward_score

    def compute_score(self, solution, ground_truth, method="strict", format_score=0.0, score=1.0):
        """The scoring function for GSM8k.

        Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
        Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

        Args:
            solution: the solution text
            ground_truth: the ground truth
            method: the method to extract the solution, choices are 'strict' and 'flexible'
            format_score: the score for the format
            score: the score for the correct answer
        """
        reward_tensor = paddle.zeros((len(solution), 1), dtype=paddle.float32)
        for i in range(len(solution)):
            answer = extract_solution(solution_str=solution[i], method=method)
            if answer is None:
                reward_tensor[i] = 0
            else:
                if answer == ground_truth[i]:
                    reward_tensor[i] = score
                else:
                    reward_tensor[i] = format_score
        return reward_tensor
