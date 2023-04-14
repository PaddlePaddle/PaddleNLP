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
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddle.optimizer.lr import LambdaDecay

from paddlenlp.metrics import BLEU, Rouge1, Rouge2, RougeL
from paddlenlp.trainer import Trainer


class BloomTrainer(Trainer):
    def __init__(self, do_generation, data_args, **kwargs):
        super().__init__(**kwargs)
        self.do_generation = do_generation
        self.data_args = data_args

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:

        if not self.do_generation:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        model.eval()
        with paddle.no_grad():
            tokens = model.generate(
                input_ids=inputs["input_ids"],
                max_length=self.data_args.tgt_length,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                decode_strategy="sampling",
                top_k=1,
            )[0]
        all_preds = []
        for pred_tokens in tokens:
            all_preds.append(pred_tokens[pred_tokens != self.tokenizer.pad_token_id].tolist())
        max_pred_length = max([len(x) for x in all_preds])
        for index, preds in enumerate(all_preds):
            all_preds[index] = preds + [-100] * (max_pred_length - len(preds))

        all_labels = []
        for label in inputs["labels"].numpy():
            label = [x for x in label[label != self.tokenizer.pad_token_id]]
            all_labels.append(label)
        max_label_length = max([len(x) for x in all_labels])
        for index, labels in enumerate(all_labels):
            all_labels[index] = labels + [-100] * (max_label_length - len(labels))

        return (None, paddle.to_tensor(all_preds), paddle.to_tensor(all_labels))

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


def compute_metrics(predictions, references):
    assert len(predictions) == len(references), (
        "The length of pred_responses should be equal to the length of "
        "target_responses. But received {} and {}.".format(len(predictions), len(references))
    )
    rouge1 = Rouge1()
    rouge2 = Rouge2()
    rougel = RougeL()
    bleu4 = BLEU(n_size=4)

    # for pred in predictions:

    rouge1_score = rouge1.score(predictions, references)
    rouge2_score = rouge2.score(predictions, references)
    for pred, ref in zip(predictions, references):
        rougel.add_inst(pred, [ref])
        bleu4.add_inst(pred, [ref])
    return {
        "rouge1": rouge1_score,
        "rouge2": rouge2_score,
        "rougel": rougel.score(),
        "bleu4": bleu4.score(),
    }
