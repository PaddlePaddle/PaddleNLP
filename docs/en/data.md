# PaddleNLP Data API

This module provides common APIs for constructing effective data processing pipelines in NLP tasks.

## API List

| API                             | Description                                  |
| ------------------------------- | :------------------------------------------- |
| `paddlenlp.data.Stack`          | Stack N input data with the same shape to build a batch |
| `paddlenlp.data.Pad`            | Stack N input data to build a batch, each input will be padded to the maximum length among the N inputs |
| `paddlenlp.data.Tuple`          | Wrap multiple batchify functions together to form a tuple |
| `paddlenlp.data.Dict`           | Wrap multiple batchify functions together to form a dict |
| `paddlenlp.data.SamplerHelper`  | Build iterable sampler for `Dataloader`      |
| `paddlenlp.data.Vocab`          | Map between text tokens and IDs              |
| `paddlenlp.data.JiebaTokenizer` | Jieba tokenizer                              |

## API Usage

The above APIs are all used to assist in building `DataLoader`. The three important initialization parameters of `DataLoader` are `dataset`, `batch_sampler`, and `collate_fn`.

`paddlenlp.data.Vocab` and `paddlenlp.data.JiebaTokenizer` are used when constructing `dataset` to handle the mapping between text tokens and IDs.

`paddlenlp.data.SamplerHelper` is used to build an iterable `batch_sampler`.

`paddlenlp.data.Stack`, `paddlenlp.data.Pad`, `paddlenlp.data.Tuple`, and `paddlenlp.data.Dict` are used to build the `collate_fn` function that generates mini-batches.

### Data Preprocessing

#### `paddlenlp.data.Vocab`

The `paddlenlp.data.Vocab` class is a vocabulary that includes a series of methods for mapping between text tokens and IDs. It supports building vocabulary from files, dictionaries, json, and other sources.
```python
from paddlenlp.data import Vocab
# Build from file
vocab1 = Vocab.load_vocabulary(vocab_file_path)
# Build from dictionary
# dic = {'unk':0, 'pad':1, 'bos':2, 'eos':3, ...}
vocab2 = Vocab.from_dict(dic)
# Build from json (usually for restoring previously saved Vocab objects from json_str or json files)
# json_str method
json_str = vocab1.to_json()
vocab3 = Vocab.from_json(json_str)
# json file method
vocab1.to_json(json_file_path)
vocab4 = Vocab.from_json(json_file_path)
```

#### `paddlenlp.data.JiebaTokenizer`

`paddlenlp.data.JiebaTokenizer` requires initializing with a `paddlenlp.data.Vocab` class, containing the `cut` tokenization method and the `encode` method for converting plain text sentences to ids.

```python
from paddlenlp.data import Vocab, JiebaTokenizer
# Vocabulary file path (download the vocabulary file first when running the sample)
# wget https://bj.bcebos.com/paddlenlp/data/senta_word_dict.txt
vocab_file_path = './senta_word_dict.txt'
# Build vocabulary
vocab = Vocab.load_vocabulary(
    vocab_file_path,
    unk_token='[UNK]',
    pad_token='[PAD]')
tokenizer = JiebaTokenizer(vocab)
tokens = tokenizer.cut('I love you, China') # ['I love you', 'China']
ids = tokenizer.encode('I love you, China') # [1170578, 575565]
```

### Building `Sampler`

#### `paddlenlp.data.SamplerHelper`

`paddlenlp.data.SamplerHelper` serves to build iterable samplers for `DataLoader`, containing methods like `shuffle`, `sort`, `batch`, `shard`, etc., providing flexible usage for users.
```python
from paddlenlp.data import SamplerHelper
from paddle.io import Dataset

class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.data = [
            [[1, 2, 3, 4], [1]],
            [[5, 6, 7], [0]],
            [[8, 9], [1]],
        ]

    def __getitem__(self, index):
        data = self.data[index][0]
        label = self.data[index][1]
        return data, label

    def __len__(self):
        return len(self.data)

dataset = MyDataset()
# SamplerHelper returns an iterable of data indices, generated indices: [0, 1, 2]
sampler = SamplerHelper(dataset)
# `shuffle()` randomly shuffles the index order, generated indices: [0, 2, 1]
sampler = sampler.shuffle()
# sort() arranges indices based on specified key within buffer_size samples
# Example: sort by length of the first field in ascending order, generated indices: [2, 0, 1]
key = (lambda x, data_source: len(data_source[x][0]))
sampler = sampler.sort(key=key, buffer_size=2)
# batch() creates mini-batches according to batch_size, generated indices: [[2, 0], [1]]
sampler = sampler.batch(batch_size=2)
# shard() splits dataset for multi-GPU training, current GPU indices: [[2, 0]]
sampler = sampler.shard(num_replicas=2)
```

### Constructing `collate_fn`

#### `paddlenlp.data.Stack`

`paddlenlp.data.Stack` is used to create batches. Its inputs must have identical shapes, and the output is a batch formed by stacking these inputs.

```python
from paddlenlp.data import Stack
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
c = [5, 6, 7, 8]
result = Stack()([a, b, c])
"""
[[1, 2, 3, 4],
 [3, 4, 5, 6],
 [5, 6, 7, 8]]
"""
```

#### `paddlenlp.data.Pad`

`paddlenlp.data.Pad` is used to create batches. It first pads all input data to the maximum length, then stacks them to form batch data.

```python
from paddlenlp.data import Pad
a = [1, 2]
b = [3, 4, 5]
c = [6, 7, 8, 9]
result = Pad(pad_val=0)([a, b, c])
"""
[[1, 2, 0, 0],
 [3, 4, 5, 0],
 [6, 7, 8, 9]]
"""
```
```python
from paddlenlp.data import Pad
a = [1, 2, 3, 4]
b = [5, 6, 7]
c = [8, 9]
result = Pad(pad_val=0)([a, b, c])
"""
[[1, 2, 3, 4],
 [5, 6, 7, 0],
 [8, 9, 0, 0]]
"""
```

#### `paddlenlp.data.Tuple`

`paddlenlp.data.Tuple` wraps multiple batch functions together into a tuple.

```python
from paddlenlp.data import Stack, Pad, Tuple
data = [
        [[1, 2, 3, 4], [1]],
        [[5, 6, 7], [0]],
        [[8, 9], [1]],
       ]
batchify_fn = Tuple(Pad(pad_val=0), Stack())
ids, label = batchify_fn(data)
"""
ids:
[[1, 2, 3, 4],
 [5, 6, 7, 0],
 [8, 9, 0, 0]]
label: [[1], [0], [1]]
"""
```

#### `paddlenlp.data.Dict`

`paddlenlp.data.Dict` wraps multiple batch functions together into a dictionary.

```python
from paddlenlp.data import Stack, Pad, Dict
data = [
        {'labels':[1], 'token_ids':[1, 2, 3, 4]},
        {'labels':[0], 'token_ids':[5, 6, 7]},
        {'labels':[1], 'token_ids':[8, 9]},
       ]
batchify_fn = Dict({'token_ids':Pad(pad_val=0), 'labels':Stack()})
ids, label = batchify_fn(data)
"""
ids:
[[1, 2, 3, 4],
 [5, 6, 7, 0],
 [8, 9, 0, 0]]
label: [[1], [0], [1]]
"""
```

### Comprehensive Example
```python
from paddlenlp.data import Vocab, JiebaTokenizer, Stack, Pad, Tuple, SamplerHelper
from paddlenlp.datasets import load_dataset
from paddle.io import DataLoader

# Vocabulary file path, example program needs to download vocabulary file first
# wget https://bj.bcebos.com/paddlenlp/data/senta_word_dict.txt
vocab_file_path = './senta_word_dict.txt'
# Build vocabulary
vocab = Vocab.load_vocabulary(
    vocab_file_path,
    unk_token='[UNK]',
    pad_token='[PAD]')
# Initialize tokenizer
tokenizer = JiebaTokenizer(vocab)

def convert_example(example):
    text, label = example['text'], example['label']
    ids = tokenizer.encode(text)
    label = [label]
    return ids, label

dataset = load_dataset('chnsenticorp', splits='train')
dataset = dataset.map(convert_example, lazy=True)

pad_id = vocab.token_to_idx[vocab.pad_token]
batchify_fn = Tuple(
    Pad(axis=0, pad_val=pad_id),  # ids
    Stack(dtype='int64')  # label
)

batch_sampler = SamplerHelper(dataset).shuffle().batch(batch_size=16)
data_loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=batchify_fn,
    return_list=True)

# Test dataset
for batch in data_loader:
    ids, label = batch
    print(ids.shape, label.shape)
    print(ids)
    print(label)
    break
```
