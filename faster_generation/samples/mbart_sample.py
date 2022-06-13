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
from paddlenlp.transformers import MBartForConditionalGeneration, MBartTokenizer

model_name = "mbart-large-50-one-to-many-mmt"

tokenizer = MBartTokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name,
                                                      src_lang="en_XX")
model.eval()


def postprocess_response(seq, bos_idx, eos_idx):
    """Post-process the decoded sequence."""
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1] if idx != bos_idx and idx != eos_idx
    ]
    res = tokenizer.convert_ids_to_string(seq)
    return res


bos_id = tokenizer.lang_code_to_id["zh_CN"]
eos_id = model.mbart.config["eos_token_id"]

inputs = "PaddleNLP is a powerful NLP library with Awesome pre-trained models and easy-to-use interface, supporting wide-range of NLP tasks from research to industrial applications."
input_ids = tokenizer(inputs)["input_ids"]
input_ids = paddle.to_tensor(input_ids, dtype='int64').unsqueeze(0)

outputs, _ = model.generate(input_ids=input_ids,
                            forced_bos_token_id=bos_id,
                            decode_strategy="beam_search",
                            num_beams=4,
                            max_length=50,
                            use_faster=True)

result = postprocess_response(outputs[0].numpy().tolist(), bos_id, eos_id)

print("Model input:", inputs)
print("Result:", result)
# PaddleNLP是一个强大的NLP库,具有超乎寻常的预训练模型和易于使用的接口,支持从研究到工业应用的广泛的NLP任务。
