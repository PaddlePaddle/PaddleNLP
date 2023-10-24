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

from typing import Dict

import paddle
import paddle.nn.functional as F

from paddlenlp.trainer import Trainer
from paddlenlp.transformers.score_model_utils import ScoreModelOutput


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        better_input_ids = inputs["better_input_ids"]
        worse_input_ids = inputs["worse_input_ids"]
        better_attention_mask = inputs["better_attention_mask"]
        worse_attention_mask = inputs["worse_attention_mask"]

        assert better_input_ids.shape[0] == worse_input_ids.shape[0], "batch size mismatch!"
        batch_size = better_input_ids.shape[0]

        output: ScoreModelOutput = model(
            paddle.concat([better_input_ids, worse_input_ids], axis=0),
            attention_mask=paddle.concat([better_attention_mask, worse_attention_mask], axis=0),
        )
        if isinstance(output, dict):
            scores = output.scores  # size = (2 * B, L, 1)
            end_scores = output.end_scores  # size = (2 * B, 1)
        else:
            scores, end_scores = output

        # size = (B, L)
        higher_rewards, lower_rewards = scores.squeeze(dim=-1).chunk(chunks=2, axis=0)
        # size = (B,)
        higher_end_rewards, lower_end_rewards = end_scores.squeeze(axis=-1).chunk(chunks=2, axis=0)

        if self.args.loss_type == "token-wise":
            losses = []
            for i in range(batch_size):
                assert not paddle.all(
                    paddle.equal(better_input_ids[i], worse_input_ids[i]),
                ).item(), "The better and worse answers are the same!"
                higher_end_index = better_attention_mask[i].nonzero()[-1]
                lower_end_index = worse_attention_mask[i].nonzero()[-1]
                end_index = max(higher_end_index, lower_end_index)

                diverge_index = (better_input_ids[i] != worse_input_ids[i]).nonzero()[0]
                assert 0 <= diverge_index <= end_index, "diverge index is out of range!"

                # size = (L,)
                higher_truncated_rewards = higher_rewards[i, diverge_index : end_index + 1]
                lower_truncated_rewards = lower_rewards[i, diverge_index : end_index + 1]

                losses.append(
                    -F.log_sigmoid(higher_truncated_rewards - lower_truncated_rewards).mean(),
                )

                if self.args.regularization > 0.0:
                    losses[-1] = losses[-1] + self.args.regularization * (
                        paddle.square(lower_truncated_rewards).mean() + paddle.square(higher_truncated_rewards).mean()
                    )

            loss = paddle.stack(losses).mean()  # size = ()
            # print("="*20, "loss:", loss)
            # print("=" * 20, "higher_end_rewards:", higher_end_rewards)
            # print("=" * 20, "lower_end_rewards:", lower_end_rewards)
        elif self.args.loss_type == "sequence-wise":
            loss = -F.logsigmoid(higher_end_rewards - lower_end_rewards).mean()

            if self.args.regularization > 0.0:
                loss = loss + self.args.regularization * (
                    paddle.square(lower_end_rewards).mean() + paddle.square(higher_end_rewards).mean()
                )
        else:
            raise ValueError(f"Unknown loss type: {self.args.loss_type}")

        accuracy = (higher_end_rewards > lower_end_rewards).float().mean()  # size = ()
        # hack for accuracy track in training
        self._acc = accuracy.detach()
        return {
            "loss": loss,  # size = ()
            "higher_end_rewards": higher_end_rewards,  # size = (B,)
            "lower_end_rewards": lower_end_rewards,  # size = (B,)
            "accuracy": accuracy,  # size = ()
        }

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        self._total_acc_scalar += self._tr_acc_scalar

        super(RewardTrainer, self).log(logs, **kwargs)
