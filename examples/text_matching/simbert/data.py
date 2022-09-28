import paddle
import numpy as np

from paddlenlp.datasets import MapDataset


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=batchify_fn,
                                return_list=True)


def read_text_pair(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 2:
                continue
            yield {'query': data[0], 'title': data[1]}


def convert_example(example, tokenizer, max_seq_length=512, phase="train"):

    query, title = example['query'], example['title']

    query_encoded_inputs = tokenizer(text=query, max_seq_len=max_seq_length)
    query_input_ids = query_encoded_inputs["input_ids"]
    query_token_type_ids = query_encoded_inputs["token_type_ids"]
    title_encoded_inputs = tokenizer(text=title, max_seq_len=max_seq_length)

    title_input_ids = title_encoded_inputs["input_ids"]
    title_token_type_ids = title_encoded_inputs["token_type_ids"]

    return query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids
