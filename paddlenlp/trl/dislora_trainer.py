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

# dislora_trainer.py

import paddle

from .sft_trainer import SFTTrainer


class DisLoRATrainer(SFTTrainer):
    """
    A specialized SFTTrainer that incorporates DisLoRA's orthogonal constraint loss.

    This trainer extends the base SFTTrainer by overriding the compute_loss method
    to add an orthogonal regularization term, which is a key component of the DisLoRA
    method.
    """

    def _calc_ortho_loss(self, model):
        """Calculate the orthogonal constraint loss of DisLoRA"""

        ortho_loss = 0.0
        den = 0

        for name, param in model.named_parameters():
            if "Direc_Ur" in name and "weight" in name:
                u = param
                iu = paddle.eye(u.shape[0], dtype=u.dtype)
                u_loss = paddle.norm(u @ u.T - iu, p="fro")
                ortho_loss += u_loss
                den += 1

            elif "Direc_Vhr" in name and "weight" in name:
                vh = param
                ivh = paddle.eye(vh.shape[1], dtype=vh.dtype)
                vh_loss = paddle.norm(vh.T @ vh - ivh, p="fro")
                ortho_loss += vh_loss
                den += 1

        if den > 0:
            return ortho_loss / den
        else:
            return None

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to add DisLoRA orthogonal regularization"""

        result = super().compute_loss(model, inputs, return_outputs=False)

        if isinstance(result, tuple):
            loss = result[0]
            outputs = result[1] if len(result) > 1 else None
        else:
            loss = result
            outputs = None

        if isinstance(loss, tuple):
            loss = loss[0]

        if hasattr(self.args, "dislora_ortho_lambda") and self.args.dislora_ortho_lambda > 0:
            ortho_loss = self._calc_ortho_loss(model)

            if ortho_loss is not None and loss is not None:

                if loss.numel() > 1:
                    loss = loss.mean()
                if ortho_loss.numel() > 1:
                    ortho_loss = ortho_loss.mean()

                if abs(self.args.dislora_ortho_lambda - 1.0) < 1e-6:

                    with paddle.no_grad():
                        ratio = ortho_loss / (loss + 1e-8)
                        alpha_task = paddle.exp(-ratio) / (paddle.exp(-ratio) + paddle.exp(-1 / ratio))
                        alpha_ortho = 1.0 - alpha_task

                    loss = alpha_task * loss + alpha_ortho * ortho_loss
                else:

                    loss = loss + self.args.dislora_ortho_lambda * ortho_loss

        return (loss, outputs) if return_outputs else loss
