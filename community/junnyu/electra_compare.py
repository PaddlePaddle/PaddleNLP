# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers import ElectraDiscriminator, ElectraForMaskedLM, ElectraTokenizer
from transformers.models.electra.modeling_electra import ElectraForPreTraining, ElectraForMaskedLM as PTElectraForMaskedLM
import paddle
import torch
import numpy as np


def compare(a, b):
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    meandif = np.abs(a - b).mean()
    maxdif = np.abs(a - b).max()
    print("mean dif:", meandif)
    print("max dif:", maxdif)


def compare_discriminator(
        path="MODEL/hfl-chinese-electra-180g-base-discriminator"):
    pdmodel = ElectraDiscriminator.from_pretrained(path)
    ptmodel = ElectraForPreTraining.from_pretrained(path).cuda()
    tokenizer = ElectraTokenizer.from_pretrained(path)
    pdmodel.eval()
    ptmodel.eval()
    text = "欢迎使用paddlenlp！"
    pdinputs = {
        k: paddle.to_tensor(
            v, dtype="int64").unsqueeze(0)
        for k, v in tokenizer(text).items()
    }
    ptinputs = {
        k: torch.tensor(
            v, dtype=torch.long).unsqueeze(0).cuda()
        for k, v in tokenizer(text).items()
    }
    with paddle.no_grad():
        pd_logits = pdmodel(**pdinputs)

    with torch.no_grad():
        pt_logits = ptmodel(**ptinputs).logits

    compare(pd_logits, pt_logits)


def compare_generator(path="MODEL/hfl-chinese-legal-electra-small-generator"):
    pdmodel = ElectraForMaskedLM.from_pretrained(path)
    ptmodel = PTElectraForMaskedLM.from_pretrained(path).cuda()
    tokenizer = ElectraTokenizer.from_pretrained(path)
    pdmodel.eval()
    ptmodel.eval()
    text = "欢迎使用paddlenlp！"
    pdinputs = {
        k: paddle.to_tensor(
            v, dtype="int64").unsqueeze(0)
        for k, v in tokenizer(text).items()
    }
    ptinputs = {
        k: torch.tensor(
            v, dtype=torch.long).unsqueeze(0).cuda()
        for k, v in tokenizer(text).items()
    }
    with paddle.no_grad():
        pd_prediction_scores = pdmodel(**pdinputs)

    with torch.no_grad():
        pt_logits = ptmodel(**ptinputs).logits

    compare(pd_prediction_scores, pt_logits)


if __name__ == "__main__":
    compare_discriminator(
        path="MODEL/hfl-chinese-electra-180g-base-discriminator")
    # # mean dif: 3.1698835e-06
    # # max dif: 1.335144e-05
    compare_discriminator(
        path="MODEL/hfl-chinese-electra-180g-small-ex-discriminator")
    # mean dif: 3.7930229e-06
    # max dif: 1.04904175e-05
    compare_generator(path="MODEL/hfl-chinese-legal-electra-small-generator")
    # mean dif: 6.6151397e-06
    # max dif: 9.346008e-05
