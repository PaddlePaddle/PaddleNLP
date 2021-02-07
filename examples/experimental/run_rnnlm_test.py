from paddlenlp.datasets.experimental import PTB
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.data import Stack, Tuple, Pad, Dict, Vocab
import numpy as np
from functools import partial
from paddle.io import DataLoader

train_ds, valid_ds, test_ds = PTB().read_datasets('train', 'valid', 'test')

train_examples = [train_ds[i]['sentence'].split() for i in range(len(train_ds))]
vocab = Vocab.build_vocab(train_examples, eos_token='</eos>')

batch_size = 8
num_steps = 35


def group_texts(examples):
    concat_examples = []
    for example in examples:
        concat_examples += example['sentence'].split() + ['</eos>']

    concat_examples = vocab.to_indices(concat_examples)

    max_seq_len = len(concat_examples) // batch_size
    reshaped_examples = np.asarray(
        concat_examples[0:batch_size * max_seq_len], dtype='int64').reshape(
            (batch_size, max_seq_len))
    encoded_examples = []
    for i in range(max_seq_len // num_steps):
        encoded_examples.append(
            (np.copy(reshaped_examples[:, i * num_steps:(i + 1) * num_steps]),
             np.copy(reshaped_examples[:, i * num_steps + 1:(i + 1) * num_steps
                                       + 1])))

    return encoded_examples


train_ds.map(group_texts)

print(len(train_ds))
print('========================================')

train_data_loader = DataLoader(
    dataset=train_ds, batch_size=None, num_workers=0, return_list=True)

for batch in train_data_loader:
    print(batch[0])
    print(batch[1])
    break
