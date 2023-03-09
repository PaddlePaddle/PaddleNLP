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

import re

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LoRALinear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # Actual trainable parameters
        if r > 0:
            self.lora_A = self.create_parameter(
                shape=[in_features, r],
                dtype=self._dtype,
                is_bias=False,
            )
            self.lora_B = self.create_parameter(
                shape=[r, out_features],
                dtype=self._dtype,
                is_bias=False,
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.stop_gradient = True
            self.bias.stop_gradient = True

    def train(self):
        super().train()
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
                self.weight.set_value(new_weight)
            self.merged = False

    def eval(self):
        super().eval()
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
                self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
            if self.r > 0:
                result += (self.lora_dropout(input) @ self.lora_A @ self.lora_B) * self.scaling
            return result
        else:
            return F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


# TODO (this is tmp API. will formalize before release)
def _find_and_replace_module(model, module_name, r, lora_alpha):
    parent_module = model
    attribute_chain = module_name.split(".")
    for name in attribute_chain[:-1]:
        parent_module = getattr(parent_module, name)
    module = getattr(parent_module, attribute_chain[-1])
    lora_module = LoRALinear(
        in_features=module.weight.shape[0],
        out_features=module.weight.shape[1],
        r=r,
        lora_alpha=lora_alpha,
    )
    setattr(parent_module, attribute_chain[-1], lora_module)


# TODO (this is tmp API. will formalize before release)
def get_lora_model(model, lora_config):
    target_modules = lora_config["target_modules"]
    for target_module in target_modules:
        for i in model.named_sublayers():
            module_name = i[0]
            if re.fullmatch(target_module, module_name):
                _find_and_replace_module(model, module_name, lora_config["r"], lora_config["lora_alpha"])
    return model
