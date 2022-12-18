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

from src.modules.objectives import compute_itm, compute_mlm, compute_snli, compute_vqa

from paddlenlp.trainer import Trainer


class BridgeTowerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if "vqa_scores" in inputs:
            ret = compute_vqa(model, inputs, "train")
            # loss = self.criterion(outputs, labels)
            # return ret["vqa_loss"]
            return (ret["vqa_loss"], ret["vqa_logits"]) if return_outputs else ret["vqa_loss"]
        else:
            ret = compute_snli(model, inputs, "train")
            # breakpoint()
            loss = ret["snli_loss"]
            outputs = (ret["snli_loss"], ret["snli_logits"])

        labels = None
        if self.criterion is not None:
            loss = self.criterion(outputs, labels)
            outputs = (loss, outputs)
            # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class BridgeTowerPreTrainTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        ret_mlm = compute_mlm(model, inputs, "train")
        # breakpoint()
        ret_itm = compute_itm(model, inputs, "train")

        loss = ret_mlm["mlm_loss"] + ret_itm["itm_loss"]
        outputs = (loss, ret_mlm["mlm_logits"])

        labels = None
        if self.criterion is not None:
            loss = self.criterion(outputs, labels)
            outputs = (loss, outputs)
            # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
