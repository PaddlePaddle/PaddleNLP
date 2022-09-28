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

from paddlenlp.transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer

model_name = 'plato-mini'

tokenizer = UnifiedTransformerTokenizer.from_pretrained(model_name)
model = UnifiedTransformerLMHeadModel.from_pretrained(model_name)
model.eval()


def postprocess_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.sep_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    return tokens


inputs = '你好啊，你今年多大了'

inputs_ids = tokenizer.dialogue_encode(inputs,
                                       add_start_token_as_response=True,
                                       return_tensors=True,
                                       is_split_into_words=False)

outputs, _ = model.generate(input_ids=inputs_ids['input_ids'],
                            token_type_ids=inputs_ids['token_type_ids'],
                            position_ids=inputs_ids['position_ids'],
                            attention_mask=inputs_ids['attention_mask'],
                            max_length=64,
                            decode_strategy='sampling',
                            top_k=5,
                            use_faster=True)

result = postprocess_response(outputs[0].numpy(), tokenizer)
result = "".join(result)

print("Model input:", inputs)
print("Result:", result)
# 我今年23岁了,你今年多大了?
