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

import paddle
import torch
import numpy as np
import paddlenlp.transformers as ppnlp
import transformers as hgnlp


def compare(a, b):
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    meandif = np.abs(a - b).mean()
    maxdif = np.abs(a - b).max()
    print("mean dif:", meandif)
    print("max dif:", maxdif)


def compare_discriminator(
        path="junnyu/hfl-chinese-electra-180g-base-discriminator"):
    pdmodel = ppnlp.ElectraDiscriminator.from_pretrained(path)
    ptmodel = ppnlp.ElectraForPreTraining.from_pretrained(path).cuda()
    tokenizer = ppnlp.ElectraTokenizer.from_pretrained(path)
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


def compare_generator():
    text = "本院经审查认为，本案[MASK]民间借贷纠纷申请再审案件，应重点审查二审判决是否存在错误的情形。"
    # ppnlp
    path = "junnyu/hfl-chinese-legal-electra-small-generator"
    model = ppnlp.ElectraForMaskedLM.from_pretrained(path)
    tokenizer = ppnlp.ElectraTokenizer.from_pretrained(path)
    model.eval()
    tokens = ["[CLS]"]
    text_list = text.split("[MASK]")
    for i, t in enumerate(text_list):
        tokens.extend(tokenizer.tokenize(t))
        if i == len(text_list) - 1:
            tokens.extend(["[SEP]"])
        else:
            tokens.extend(["[MASK]"])

    input_ids_list = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = paddle.to_tensor([input_ids_list])
    with paddle.no_grad():
        pd_outputs = model(input_ids)[0]
    pd_outputs_sentence = "paddle: "
    for i, id in enumerate(input_ids_list):
        if id == tokenizer.convert_tokens_to_ids(["[MASK]"])[0]:
            scores, index = paddle.nn.functional.softmax(pd_outputs[i],
                                                         -1).topk(5)
            tokens = tokenizer.convert_ids_to_tokens(index.tolist())
            outputs = []
            for score, tk in zip(scores.tolist(), tokens):
                outputs.append(f"{tk}={score}")
            pd_outputs_sentence += "[" + "||".join(outputs) + "]" + " "
        else:
            pd_outputs_sentence += "".join(
                tokenizer.convert_ids_to_tokens(
                    [id], skip_special_tokens=True)) + " "

    print(pd_outputs_sentence)

    # transformers
    path = "hfl/chinese-legal-electra-small-generator"
    config = hgnlp.ElectraConfig.from_pretrained(path)
    config.hidden_size = 64
    config.intermediate_size = 256
    config.num_attention_heads = 1
    model = hgnlp.ElectraForMaskedLM.from_pretrained(path, config=config)
    tokenizer = hgnlp.ElectraTokenizer.from_pretrained(path)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        pt_outputs = model(**inputs).logits[0]
    pt_outputs_sentence = "pytorch: "
    for i, id in enumerate(inputs["input_ids"][0].tolist()):
        if id == tokenizer.convert_tokens_to_ids(["[MASK]"])[0]:
            scores, index = torch.nn.functional.softmax(pt_outputs[i],
                                                        -1).topk(5)
            tokens = tokenizer.convert_ids_to_tokens(index.tolist())
            outputs = []
            for score, tk in zip(scores.tolist(), tokens):
                outputs.append(f"{tk}={score}")
            pt_outputs_sentence += "[" + "||".join(outputs) + "]" + " "
        else:
            pt_outputs_sentence += "".join(
                tokenizer.convert_ids_to_tokens(
                    [id], skip_special_tokens=True)) + " "

    print(pt_outputs_sentence)


if __name__ == "__main__":
    compare_discriminator(
        path="junnyu/hfl-chinese-electra-180g-base-discriminator")
    # # mean dif: 3.1698835e-06
    # # max dif: 1.335144e-05
    compare_discriminator(
        path="junnyu/hfl-chinese-electra-180g-small-ex-discriminator")
    # mean dif: 3.7930229e-06
    # max dif: 1.04904175e-05
    compare_generator()
    # paddle:  本 院 经 审 查 认 为 ， 本 案 [因=0.27444931864738464||经=0.18613006174564362||系=0.09408623725175858||的=0.07536833733320236||就=0.033634234219789505] 民 间 借 贷 纠 纷 申 请 再 审 案 件 ， 应 重 点 审 查 二 审 判 决 是 否 存 在 错 误 的 情 形 。
    # pytorch:  本 院 经 审 查 认 为 ， 本 案 [因=0.2744344472885132||经=0.1861187219619751||系=0.09407979995012283||的=0.07537488639354706||就=0.03363779932260513] 民 间 借 贷 纠 纷 申 请 再 审 案 件 ， 应 重 点 审 查 二 审 判 决 是 否 存 在 错 误 的 情 形 。  
