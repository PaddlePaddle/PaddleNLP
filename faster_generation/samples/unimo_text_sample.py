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

from paddlenlp.transformers import UNIMOLMHeadModel, UNIMOTokenizer

model_name = 'unimo-text-1.0-lcsts-new'

model = UNIMOLMHeadModel.from_pretrained(model_name)
model.eval()
tokenizer = UNIMOTokenizer.from_pretrained(model_name)


def postprocess_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.mask_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    return tokens


inputs = "深度学习是人工智能的核心技术领域。百度飞桨作为中国首个自主研发、功能丰富、开源开放的产业级深度学习平台,将从多层次技术产品、产业AI人才培养和强大的生态资源支持三方面全面护航企业实现快速AI转型升级。"

inputs_ids = tokenizer.gen_encode(inputs,
                                  add_start_token_for_decoding=True,
                                  return_tensors=True,
                                  is_split_into_words=False)

outputs, _ = model.generate(input_ids=inputs_ids['input_ids'],
                            token_type_ids=inputs_ids['token_type_ids'],
                            position_ids=inputs_ids['position_ids'],
                            attention_mask=inputs_ids['attention_mask'],
                            max_length=64,
                            decode_strategy='beam_search',
                            num_beams=2,
                            use_faster=True)

result = postprocess_response(outputs[0].numpy(), tokenizer)
result = "".join(result)

print("Model input:", inputs)
print("Result:", result)
# 百度飞桨：深度学习助力企业转型升级
