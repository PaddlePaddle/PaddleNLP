# -*- coding: UTF-8 -*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import time

import tensorflow as tf
import tensorflow_text as tf_text

import paddle
import paddlenlp
from paddlenlp.transformers import BertTokenizer
from paddlenlp.experimental import FasterTokenizer
from paddlenlp.experimental import to_tensor

from transformers import AutoTokenizer

max_seq_length = 128
batch_size = 32
epochs = 10
steps = 100
total_tokens = epochs * steps * max_seq_length

text = '小说是文学的一种样式，一般描写人物故事，塑造多种多样的人物形象，但亦有例外。它是拥有不完整布局、发展及主题的文学作品。而对话是不是具有鲜明的个性，每个人物说的没有独特的语言风格，是衡量小说水准的一个重要标准。与其他文学样式相比，小说的容量较大，它可以细致的展现人物性格和命运，可以表现错综复杂的矛盾冲突，同时还可以描述人物所处的社会生活环境。小说一词，最早见于《庄子·外物》：“饰小说以干县令，其于大达亦远矣。”这里所说的小说，是指琐碎的言谈、小的道理，与现时所说的小说相差甚远。文学中，小说通常指长篇小说、中篇、短篇小说和诗的形式。小说是文学的一种样式，一般描写人物故事，塑造多种多样的人物形象，但亦有例外。它是拥有不完整布局、发展及主题的文学作品。而对话是不是具有鲜明的个性，每个人物说的没有独特的语言风格，是衡量小说水准的一个重要标准。与其他文学样式相比，小说的容量较大，它可以细致的展现人物性格和命运，可以表现错综复杂的矛盾冲突，同时还可以描述人物所处的社会生活环境。小说一词，最早见于《庄子·外物》：“饰小说以干县令，其于大达亦远矣。”这里所说的小说，是指琐碎的言谈、小的道理，与现时所说的小说相差甚远。文学中'
data = [text[:max_seq_length]] * steps

# BERT Tokenizer using PaddleNLP FasterTokenizer
pp_tokenizer = FasterTokenizer.from_pretrained("bert-base-chinese")

batches = [
    to_tensor(data[idx:idx + batch_size])
    for idx in range(0, len(data), batch_size)
]

for batch_data in batches:
    input_ids, token_type_ids = pp_tokenizer(
        text=batch_data, max_seq_len=max_seq_length)

start = time.time()
for _ in range(epochs):
    for batch_data in batches:
        input_ids, token_type_ids = pp_tokenizer(
            batch_data, max_seq_len=max_seq_length)
end = time.time()
print("The throughput of paddle FasterTokenizer: %.5f tokens/s" %
      (total_tokens / (end - start)))

hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=True)

batches = [
    data[idx:idx + batch_size] for idx in range(0, len(data), batch_size)
]
for batch_data in batches:
    encoded_inputs = hf_tokenizer(batch_data)

# BERT Tokenizer using HuggingFace AutoTokenizer
start = time.time()
for _ in range(epochs):
    for batch_data in batches:
        encoded_inputs = hf_tokenizer(
            batch_data)  #, padding=True, truncation=True)
end = time.time()
print("The throughput of huggingface FastTokenizer: %.5f tokens/s" %
      (total_tokens / (end - start)))

# BERT Tokenizer using PaddleNLP BertTokenizer
py_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
for batch_data in batches:
    encoded_inputs = py_tokenizer(batch_data)

start = time.time()
for _ in range(epochs):
    for batch_data in batches:
        encoded_inputs = py_tokenizer(batch_data)
end = time.time()
print("The throughput of paddle BertTokenizer: %.5f tokens/s" % (total_tokens /
                                                                 (end - start)))

# BERT Tokenizer using HuggingFace AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-chinese", use_fast=False)

batches = [
    data[idx:idx + batch_size] for idx in range(0, len(data), batch_size)
]
for batch_data in batches:
    encoded_inputs = hf_tokenizer(batch_data)

start = time.time()
for _ in range(epochs):
    for batch_data in batches:
        encoded_inputs = hf_tokenizer(
            batch_data)  #, padding=True, truncation=True)
end = time.time()
print("The throughput of huggingface python tokenizer: %.5f tokens/s" %
      (total_tokens / (end - start)))

# BERT Tokenizer using TensorFlow Text
vocab_list = list(py_tokenizer.vocab.token_to_idx.keys())
lookup_table = tf.lookup.StaticVocabularyTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=vocab_list,
        key_dtype=tf.string,
        values=tf.range(
            tf.size(
                vocab_list, out_type=tf.int64), dtype=tf.int64),
        value_dtype=tf.int64),
    num_oov_buckets=1)
batches = [
    data[idx:idx + batch_size] for idx in range(0, len(data), batch_size)
]
tf_tokenizer = tf_text.BertTokenizer(lookup_table)

for batch_data in batches:
    input_ids = tf_tokenizer.tokenize(batch_data)

start = time.time()
for _ in range(epochs):
    for batch_data in batches:
        input_ids = tf_tokenizer.tokenize(batch_data)
end = time.time()
print("The throughput of tensorflow text BertTokenizer: %.5f tokens/s" %
      (total_tokens / (end - start)))
