import io
import os
import time

import paddle
import paddlenlp
from paddlenlp.transformers import BertTokenizer
from paddlenlp.experimental import FasterTokenizer
from paddlenlp.utils.downloader import get_path_from_url
from paddlenlp.experimental import to_tensor

from transformers import AutoTokenizer

max_seq_length = 128
batch_size = 32

text = '小说是文学的一种样式，一般描写人物故事，塑造多种多样的人物形象，但亦有例外。它是拥有不完整布局、发展及主题的文学作品。而对话是不是具有鲜明的个性，每个人物说的没有独特的语言风格，是衡量小说水准的一个重要标准。与其他文学样式相比，小说的容量较大，它可以细致的展现人物性格和命运，可以表现错综复杂的矛盾冲突，同时还可以描述人物所处的社会生活环境。小说一词，最早见于《庄子·外物》：“饰小说以干县令，其于大达亦远矣。”这里所说的小说，是指琐碎的言谈、小的道理，与现时所说的小说相差甚远。文学中，小说通常指长篇小说、中篇、短篇小说和诗的形式。小说是文学的一种样式，一般描写人物故事，塑造多种多样的人物形象，但亦有例外。它是拥有不完整布局、发展及主题的文学作品。而对话是不是具有鲜明的个性，每个人物说的没有独特的语言风格，是衡量小说水准的一个重要标准。与其他文学样式相比，小说的容量较大，它可以细致的展现人物性格和命运，可以表现错综复杂的矛盾冲突，同时还可以描述人物所处的社会生活环境。小说一词，最早见于《庄子·外物》：“饰小说以干县令，其于大达亦远矣。”这里所说的小说，是指琐碎的言谈、小的道理，与现时所说的小说相差甚远。文学中'
data = [text[:max_seq_length]] * 100

pp_tokenizer = FasterTokenizer.from_pretrained("bert-base-chinese")

batches = [
    to_tensor(data[idx:idx + batch_size])
    for idx in range(0, len(data), batch_size)
]

for batch_data in batches:
    input_ids, token_type_ids = pp_tokenizer(
        text=batch_data, max_seq_len=max_seq_length)

start = time.time()
for _ in range(10):
    for batch_data in batches:
        input_ids, token_type_ids = pp_tokenizer(
            batch_data, max_seq_len=max_seq_length)
end = time.time()
print("pp_tokenizer: %.5f" % (end - start))

hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=True)

batches = [
    data[idx:idx + batch_size] for idx in range(0, len(data), batch_size)
]
for batch_data in batches:
    encoded_inputs = hf_tokenizer(batch_data)

start = time.time()
for _ in range(10):
    for batch_data in batches:
        encoded_inputs = hf_tokenizer(
            batch_data)  #, padding=True, truncation=True)
end = time.time()
print("hf_tokenizer: %.5f" % (end - start))

py_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
for batch_data in batches:
    encoded_inputs = py_tokenizer(batch_data)

start = time.time()
for _ in range(10):
    for batch_data in batches:
        encoded_inputs = py_tokenizer(batch_data)
end = time.time()
print("py_tokenizer: %.5f" % (end - start))

hf_tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-chinese", use_fast=False)

batches = [
    data[idx:idx + batch_size] for idx in range(0, len(data), batch_size)
]
for batch_data in batches:
    encoded_inputs = hf_tokenizer(batch_data)

start = time.time()
for _ in range(10):
    for batch_data in batches:
        encoded_inputs = hf_tokenizer(
            batch_data)  #, padding=True, truncation=True)
end = time.time()
print("hf_py_tokenizer: %.5f" % (end - start))
