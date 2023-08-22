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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn

from paddlenlp.trainer import Trainer


class GLMTrainer(Trainer):
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

        if not self.do_generation:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        model.eval()
        with paddle.no_grad():
            tokens = model.generate(
                input_ids=inputs["input_ids"],
                position_ids=inputs["position_ids"],
                attention_mask=inputs["attention_mask"],
                decode_strategy="sampling",
                top_k=1,
                repetition_penalty=2.0,
                bos_token_id=self.tokenizer.sop_token_id,
                eos_token_id=self.tokenizer.eop_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]
            all_preds = []
            for pred_tokens in tokens:
                all_preds.append(pred_tokens[pred_tokens != self.tokenizer.pad_token_id].tolist())
            max_pred_length = max([len(x) for x in all_preds])
            for index, preds in enumerate(all_preds):
                all_preds[index] = preds + [-100] * (max_pred_length - len(preds))

            all_labels = []
            for label, mask in zip(inputs["labels"].numpy(), inputs["loss_mask"].numpy()):
                label = label[mask.astype("bool")]
                label = [x for x in label[label != self.tokenizer.pad_token_id]]
                all_labels.append(label)
            max_label_length = max([len(x) for x in all_labels])
            for index, labels in enumerate(all_labels):
                all_labels[index] = labels + [-100] * (max_label_length - len(labels))

        return (None, paddle.to_tensor(all_preds), paddle.to_tensor(all_labels))

    def log(self, logs: Dict[str, float], **kwargs) -> None:

        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 4)

        if "eval_loss" in logs:
            logs["eval_ppl"] = np.exp(logs["eval_loss"])
        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs, **kwargs)
