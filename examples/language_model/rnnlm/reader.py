import numpy as np

import paddle

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Vocab


def create_data_loader(batch_size, num_steps, data_path=None):
    train_ds, valid_ds, test_ds = load_dataset('ptb',
                                               splits=('train', 'valid',
                                                       'test'))

    train_examples = [
        train_ds[i]['sentence'].split() for i in range(len(train_ds))
    ]
    vocab = Vocab.build_vocab(train_examples, eos_token='</eos>')

    # Because the sentences in PTB dataset might be consecutive, we need to concatenate
    # all texts from our dataset and fold them into chunks while the number of rows is
    # equal to batch size. For example:
    #
    #   Sentence1: we're talking about years ago before anyone heard of asbestos having
    #              any questionable properties.
    #   Sentence2: there is no asbestos in our products now.
    #   Batch_size: 5
    #   Grouped_text: [["we're", "talking", "about", "years"],
    #                  ["ago", "before", "anyone", "heard"],
    #                  ["of", "asbestos", "having", "any"],
    #                  ["questionable", "properties", "there", "is"],
    #                  ["no", "asbestos", "in", "our"]]
    #
    def group_texts(examples):
        concat_examples = []
        for example in examples:
            concat_examples += example['sentence'].split() + ['</eos>']

        concat_examples = vocab.to_indices(concat_examples)

        max_seq_len = len(concat_examples) // batch_size
        reshaped_examples = np.asarray(concat_examples[0:batch_size *
                                                       max_seq_len],
                                       dtype='int64').reshape(
                                           (batch_size, max_seq_len))
        encoded_examples = []
        for i in range(max_seq_len // num_steps):
            encoded_examples.append(
                (np.copy(reshaped_examples[:,
                                           i * num_steps:(i + 1) * num_steps]),
                 np.copy(reshaped_examples[:, i * num_steps +
                                           1:(i + 1) * num_steps + 1])))

        return encoded_examples

    train_ds.map(group_texts, batched=True)
    valid_ds.map(group_texts, batched=True)
    test_ds.map(group_texts, batched=True)

    train_loader = paddle.io.DataLoader(train_ds,
                                        return_list=True,
                                        batch_size=None)
    valid_loader = paddle.io.DataLoader(valid_ds,
                                        return_list=True,
                                        batch_size=None)
    test_loader = paddle.io.DataLoader(test_ds,
                                       return_list=True,
                                       batch_size=None)
    return train_loader, valid_loader, test_loader, len(vocab)
