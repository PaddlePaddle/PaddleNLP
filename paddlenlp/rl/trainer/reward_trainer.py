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
from ...trainer.trainer import (
    EvalPrediction,
    TrainerCallback,
    TrainingArguments,
    logger,
)
from ...transformers import PretrainedModel, PretrainedTokenizer
from ..models.ppo_model_utils import create_startend_row_indices
from .rl_trainer import RLTrainer
from .trainer_utils import batch_retokenize


class RewardTrainer(RLTrainer):
    """Reward trainer"""

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
        reward_server: Optional[str] = None,
    ):
        """
        Initialize the RewardTrainer class.

        This class extends the functionality of the RLTrainer class by adding support for reward functions. It allows
        users to train models based on reward functions instead of loss values.

        Args:
            model (Union[PretrainedModel, nn.Layer], optional): The model to be trained. Can be either a
                PretrainedModel instance or a custom nn.Layer object. Defaults to None.
            criterion (nn.Layer, optional): The criterion used for calculating losses during training. Defaults to
                                            None.
            args (TrainingArguments, optional): The arguments used for configuring the training process. Defaults
            to None.
            data_collator (Optional[DataCollator], optional): The collator used for preparing batches of data.
                                                            Defaults to None.
            train_dataset (Optional[Dataset], optional): The dataset used for training. Defaults to None.
            eval_dataset (Union[Dataset, Dict[str, Dataset]], optional): The evaluation dataset(s). Can be either
                a single Dataset instance or a dictionary mapping string keys to Dataset instances. Defaults to
                None.
            tokenizer (Optional[PretrainedTokenizer], optional): The tokenizer used for processing the input text.
                Defaults to None.
            compute_metrics (Optional[Callable[[EvalPrediction], Dict]], optional): The function used for computing
                metrics during evaluation. Defaults to None.
            callbacks (Optional[List[TrainerCallback]], optional): A list of TrainerCallback objects that will be
                called during the training process. Defaults to None.
            optimizers (Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler], optional): A tuple
                containing the optimizer and learning rate scheduler used for training. Defaults to (None, None).
            preprocess_logits_for_metrics (Optional[Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]],
                optional): A function used for preprocessing logits before passing them to the `compute_metrics`
                function. Defaults to None.
            reward_server (Optional[str], optional): The URL of the reward server. Must be in the format
                'http://xxx:port'. Defaults to None.
        """
        if args.use_rm_server:
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
        input_ids: paddle.Tensor,
        position_ids: paddle.Tensor = None,
        input_ids_tokenizer: PretrainedTokenizer = None,
        label_ids: paddle.Tensor = None,
        **kwargs,
    ) -> Dict[str, paddle.Tensor]:
        """
        Compute the reward function value for training the model. If using an RM server, `label_ids` must be provided.
        If `input_ids_tokenizer` is not the current `tokenizer`, the input will be retokenized.

        Args:
            input_ids (paddle.Tensor, shape [B, L]): The IDs of the input sequences, including prompt and response
                                                    parts.
            position_ids (paddle.Tensor, optional, shape [B, L], defaults to None): The position IDs for each token in
                                                                                    the input sequences.
            input_ids_tokenizer (PretrainedTokenizer, optional, defaults to None): The tokenizer used to process the
                                                                                    input sequences.
            label_ids (paddle.Tensor, optional, shape [B, L], defaults to None): The label IDs, required only when
                                                                                using an RM server.
            **kwargs (dict, optional): Other optional parameters, including `prompt` (string), defaults to None.

        Returns:
            Dict[str, paddle.Tensor]: A dictionary containing the following key:
                - rewards (paddle.Tensor, shape [B]): The values of the reward function.

        Raises:
            ValueError: If using an RM server and `label_ids` are not provided.
        """
        if not self.args.use_rm_server:
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
            prompt_len = kwargs["prompt"].shape[-1]
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
        # if self.args.rl_algorithm in ["grpo", "reinforce_plus_plus"]:
        #     return {"rewards": reward_score}

    def request_reward_server(self, src, tgt, response):
        """
        Request the reward server to get the score for the response. If the request fails, the score will be set to
        zero.

        Args:
            src (str): The source language text.
            tgt (str): The target language text.
            response (List[str]): A list of user responses.

        Returns:
            paddle.Tensor: A tensor of shape [batch_size, 1] containing the score for each sample.
        """
        data = {"src": src, "tgt": tgt, "response": response}
        dtype = self.args.model_dtype

        def post():
            try:
                res = requests.post(self.model, json=data)
                result = json.loads(res.text)
                reward_score = paddle.to_tensor(
                    result["score"],
                    dtype=dtype if not self.args.use_fp32_compute else "float32",
                )
            except Exception as e:
                logger.warning(f"Request reward server failed({e}) and rewards_score will be set zero.")
                reward_score = paddle.zeros(
                    len(response),
                    dtype=dtype if not self.args.use_fp32_compute else "float32",
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

        return reward_score.unsqueeze(-1)
