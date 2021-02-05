import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader

import re
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models.keyedvectors import KeyedVectors


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(
        open(
            positive_data_file, 'r', encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(
        open(
            negative_data_file, 'r', encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = list(map(lambda x: x.split(), x_text))
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.array(positive_labels + negative_labels)
    return [x_text, y]


class Word2VecBoWTextClassification(nn.Layer):
    def __init__(self, batch_size, word_embedding_dim, sent_embedding_dim,
                 num_labels):
        super(Word2VecBoWTextClassification, self).__init__()

        self._fc1 = nn.Linear(word_embedding_dim, sent_embedding_dim)
        self._fc2 = nn.Linear(sent_embedding_dim, num_labels)
        self._dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        word_emb, seq_lens = inputs

        # [batch_size, word_embedding_dim]
        sent_emb = self.average_word_embedding(word_emb, seq_lens)

        # [batch_size, sent_embedding_dim]
        dense = self._fc1(sent_emb)
        dense = self._dropout(dense)

        # [batch_size, num_labels]
        out = self._fc2(dense)
        return out

    def average_word_embedding(self, word_emb, seq_lens):
        """
        Parameters:
            word_emb: It is a Tensor with shape `[batch_size, max_seq_len, word_embedding_dim]`.
            seq_lens: It is a Tensor with shape `[batch_size]`.
        """
        seq_lens = paddle.unsqueeze(seq_lens, axis=-1)
        seq_lens = paddle.cast(seq_lens, dtype=word_emb.dtype)

        # [batch_size, word_embedding_dim]
        sent_emb = paddle.sum(word_emb, axis=1)
        # [batch_size, word_embedding_dim]
        sent_emb = sent_emb / seq_lens
        return sent_emb


class SentencePolarityDatasetV1(Dataset):
    def __init__(self, x, y, gensim_model, max_seq_len):
        super(SentencePolarityDatasetV1, self).__init__()

        self._text = list(zip(x, y))
        self._gensim_model = gensim_model
        self._vector_size = gensim_model.vector_size
        self._max_seq_len = max_seq_len
        self._data = self.convert_to_ids()

    def convert_to_ids(self):
        data = []
        for sentence, label in self._text:
            sentence = sentence[:self._max_seq_len]
            ids = np.zeros([len(sentence), self._vector_size], dtype=np.float32)
            for i, word in enumerate(sentence):
                if word in self._gensim_model:
                    ids[i] = self._gensim_model[word]
                else:
                    ids[i] = np.random.uniform(-0.25, 0.25, self._vector_size)
            data.append([ids, label])
        return data

    def __getitem__(self, idx):
        ids = np.copy(self._data[idx][0])
        label = self._data[idx][1]
        return (ids, label)

    def __len__(self):
        return len(self._data)


def generate_batch(batch):
    batch_ids, batch_label = zip(*batch)
    max_len = max([ids.shape[0] for ids in batch_ids])
    new_batch_ids = np.zeros(
        [len(batch_ids), max_len, batch_ids[0].shape[1]], dtype=np.float32)
    new_batch_label = []
    new_batch_seq_len = []
    for i, (ids, label) in enumerate(zip(batch_ids, batch_label)):
        seq_len = ids.shape[0]
        new_batch_ids[i, :seq_len, :] = ids
        new_batch_label.append(label)
        new_batch_seq_len.append(seq_len)
    return new_batch_ids, new_batch_label, new_batch_seq_len


def train_base():
    paddle.disable_static()

    batch_size = 64
    max_seq_len = 256
    epochs = 20
    lr = 0.001
    weight_decay = 0.0001
    sent_embedding_dim = 64
    num_labels = 2

    pos_file = './sentence-polarity-dataset-v1/rt-polarity.pos'
    neg_file = './sentence-polarity-dataset-v1/rt-polarity.neg'
    x_text, y = load_data_and_labels(pos_file, neg_file)
    x_train, x_test, y_train, y_test = train_test_split(
        x_text, y, test_size=0.1, random_state=1)

    pretrained_word2vec_file = './sentence-polarity-dataset-v1/GoogleNews-vectors-negative300.bin'
    #gensim_model = KeyedVectors.load_word2vec_format(pretrained_word2vec_file, binary=True, limit=300000)
    gensim_model = KeyedVectors.load_word2vec_format(
        pretrained_word2vec_file, binary=True)
    print('\nLoaded word2vec from %s\n' % pretrained_word2vec_file)

    train_dataset = SentencePolarityDatasetV1(x_train, y_train, gensim_model,
                                              max_seq_len)
    test_dataset = SentencePolarityDatasetV1(x_test, y_test, gensim_model,
                                             max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        return_list=True,
        shuffle=True,
        collate_fn=lambda batch: generate_batch(batch))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        return_list=True,
        shuffle=False,
        collate_fn=lambda batch: generate_batch(batch))

    model = Word2VecBoWTextClassification(batch_size, gensim_model.vector_size,
                                          sent_embedding_dim, num_labels)
    model.train()

    adam = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=lr,
        weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        for step, batch_data in enumerate(train_loader, start=1):
            ids, label, seq_lens = batch_data

            output = model((ids, seq_lens))
            loss = criterion(output, label)
            loss.backward()
            adam.step()
            adam.clear_grad()

            if step % 10 == 0:
                print('step %d, loss %.4f' % (step, loss.numpy()[0]))

    acc = test(model, test_loader)
    print('\ntest acc %.4f\n' % acc)


def test(model, test_loader):
    correct = num = 0
    model.eval()
    with paddle.no_grad():
        for batch_data in test_loader:
            ids, label, seq_lens = batch_data

            # [batch_size, 2]
            output = model((ids, seq_lens))

            num += label.shape[0]
            predict = paddle.argmax(output, axis=1)
            label = paddle.cast(label, dtype=predict.dtype)
            correct += paddle.sum(
                paddle.cast(
                    predict == label, dtype='int64')).numpy()[0]
    model.train()
    return correct * 1.0 / num


if __name__ == '__main__':
    train_base()
