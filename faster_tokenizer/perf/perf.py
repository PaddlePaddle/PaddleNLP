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
import argparse
import time

import tensorflow as tf
import tensorflow_text as tf_text

import paddle
import paddlenlp
from paddlenlp.transformers import BertTokenizer
from paddlenlp.experimental import FasterTokenizer
from paddlenlp.experimental import to_tensor

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()

# yapf: disable
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size for tokenization.")
parser.add_argument("--epochs", default=10, type=int, help="Total number of tokenization epochs to perform.")
parser.add_argument("--num_samples", default=100, type=int, help="The number of samples to be tokenized")
# yapf: enable
args = parser.parse_args()

max_seq_length = args.max_seq_length
batch_size = args.batch_size
epochs = args.epochs
num_samples = args.num_samples
total_tokens = epochs * num_samples * max_seq_length

text = '在世界几大古代文明中，中华文明源远流长、从未中断，至今仍充满蓬勃生机与旺盛生命力，这在人类历史上是了不起的奇迹。'   \
       '本固根深、一脉相承的历史文化是铸就这一奇迹的重要基础。先秦时期是中华文化的创生期，奠定了此后几千年中华文化发展的'   \
       '基础。考古发现证实，早期中华文明的形成经历了从“满天星斗”到“月明星稀”再到“多元一体”的过程。在这个过程中，不同地域、' \
       '不同人群的文化交流交融，中华民族最早的大家庭逐渐成形，国家由此诞生，“大同”社会理想和“天下为公，选贤与能，讲信修睦”' \
       '的价值追求逐渐深入人心。在早期国家形成过程中，我们的先人积累了初步的国家治理经验，包括经济、政治、军事、法律、文化'  \
       '等各个方面，最终以典章、思想的形式进行总结和传承。流传至今的夏商西周国家治理经验、春秋战国诸子百家思想，是先秦时期'  \
       '历史文化的集中反映。秦汉至宋元时期是中华文化的发展期，中华传统文化在这个时期走向成熟并迈向新的高峰。中央集权制度的'  \
       '形成、郡县制度的推广、官僚制度的健全，推动中国传统社会形成国家治理的基本形态，为中国传统社会的长期延续和发展提供了'  \
       '坚实的制度和文化支撑，贯穿其中的价值主线是对“大一统”的坚定追求。与此同时，民为邦本的民本思想、以文化人的文治主张、'  \
       '协和万邦的天下观等，也在实践中得到丰富和完善。在追求“大一统”的历史中，民族精神世代相传，民族英雄史不绝书。'

data = [text[:max_seq_length]] * num_samples

# BERT Tokenizer using PaddleNLP FasterTokenizer
pp_tokenizer = FasterTokenizer.from_pretrained("bert-base-chinese")

batches = [
    to_tensor(data[idx:idx + batch_size])
    for idx in range(0, len(data), batch_size)
]

for batch_data in batches:
    input_ids, token_type_ids = pp_tokenizer(text=batch_data,
                                             max_seq_len=max_seq_length)

start = time.time()
for _ in range(epochs):
    for batch_data in batches:
        input_ids, token_type_ids = pp_tokenizer(batch_data,
                                                 max_seq_len=max_seq_length)
end = time.time()

print("The throughput of paddle FasterTokenizer: {:,.2f} tokens/s".format(
    (total_tokens / (end - start))))

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
print("The throughput of huggingface FasterTokenizer: {:,.2f} tokens/s".format(
    (total_tokens / (end - start))))

# BERT Tokenizer using PaddleNLP BertTokenizer
py_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
for batch_data in batches:
    encoded_inputs = py_tokenizer(batch_data)

start = time.time()
for _ in range(epochs):
    for batch_data in batches:
        encoded_inputs = py_tokenizer(batch_data)
end = time.time()
print("The throughput of paddle BertTokenizer: {:,.2f} tokens/s".format(
    (total_tokens / (end - start))))

# BERT Tokenizer using HuggingFace AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",
                                             use_fast=False)

for batch_data in batches:
    encoded_inputs = hf_tokenizer(batch_data)

start = time.time()
for _ in range(epochs):
    for batch_data in batches:
        encoded_inputs = hf_tokenizer(
            batch_data)  #, padding=True, truncation=True)
end = time.time()
print("The throughput of huggingface python tokenizer: {:,.2f} tokens/s".format(
    (total_tokens / (end - start))))

# BERT Tokenizer using TensorFlow Text
vocab_list = list(py_tokenizer.vocab.token_to_idx.keys())
lookup_table = tf.lookup.StaticVocabularyTable(
    tf.lookup.KeyValueTensorInitializer(keys=vocab_list,
                                        key_dtype=tf.string,
                                        values=tf.range(tf.size(
                                            vocab_list, out_type=tf.int64),
                                                        dtype=tf.int64),
                                        value_dtype=tf.int64),
    num_oov_buckets=1)

tf_tokenizer = tf_text.BertTokenizer(lookup_table)

for batch_data in batches:
    input_ids = tf_tokenizer.tokenize(batch_data)

start = time.time()
for _ in range(epochs):
    for batch_data in batches:
        input_ids = tf_tokenizer.tokenize(batch_data)
end = time.time()
print(
    "The throughput of TensorFlow Text BertTokenizer: {:,.2f} tokens/s".format(
        (total_tokens / (end - start))))
