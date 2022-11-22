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

import fast_tokenizer
from fast_tokenizer import ErnieFastTokenizer, models

fast_tokenizer.set_thread_num(1)
vocab = models.WordPiece.read_file("ernie_vocab.txt")
fast_tokenizer = ErnieFastTokenizer(vocab)
output = fast_tokenizer.encode("我爱中国")
print("ids: ", output.ids)
print("type_ids: ", output.type_ids)
print("tokens: ", output.tokens)
print("offsets: ", output.offsets)
print("attention_mask: ", output.attention_mask)
