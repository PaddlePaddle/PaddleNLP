import glob
import json
import math
import os
import copy
import itertools

import pandas as pd
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp as nlp
from ..datasets import MapDataset
from ..data import Stack, Pad, Tuple, Vocab, JiebaTokenizer
from ..utils.downloader import get_path_from_url
from ..utils.env import MODEL_HOME
from .task import Task

URLS = {
    "senta_vocab": "https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt",
    "senta_bow":
    "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/senta/senta_bow.pdparams",
    "senta_lstm":
    "https://paddlenlp.bj.bcebos.com/taskflow/sentiment_analysis/senta/senta_lstm.pdparams"
}


class BoWModel(nn.Layer):
    """
    This class implements the Bag of Words Classification Network model to classify texts.
    At a high level, the model starts by embedding the tokens and running them through
    a word embedding. Then, we encode these epresentations with a `BoWEncoder`.
    Lastly, we take the output of the encoder to create a final representation,
    which is passed through some feed-forward layers to output a logits (`output_layer`).

    """

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 hidden_size=128,
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx)
        self.bow_encoder = nlp.seq2vec.BoWEncoder(emb_dim)
        self.fc1 = nn.Linear(self.bow_encoder.get_output_dim(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len=None):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)

        # Shape: (batch_size, embedding_dim)
        summed = self.bow_encoder(embedded_text)
        encoded_text = paddle.tanh(summed)

        # Shape: (batch_size, hidden_size)
        fc1_out = paddle.tanh(self.fc1(encoded_text))
        # Shape: (batch_size, fc_hidden_size)
        fc2_out = paddle.tanh(self.fc2(fc1_out))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc2_out)
        return logits


class LSTMModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=198,
                 direction='forward',
                 lstm_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx)
        self.lstm_encoder = nlp.seq2vec.LSTMEncoder(
            emb_dim,
            lstm_hidden_size,
            num_layers=lstm_layers,
            direction=direction,
            dropout=dropout_rate,
            pooling_type=pooling_type)
        self.fc = nn.Linear(self.lstm_encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens, num_directions*lstm_hidden_size)
        # num_directions = 2 if direction is 'bidirect'
        # if not, num_directions = 1
        text_repr = self.lstm_encoder(embedded_text, sequence_length=seq_len)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits


class SentaTask(Task):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._model, self._tokenizer, _ = self._construct_model_tokenizer(
            model, **self.kwargs)
        self._label_map = {0: 'negative', 1: 'positive'}

    def _download_resoures(self, save_dir, name, filename):
        default_root = os.path.join(MODEL_HOME, save_dir)
        fullname = os.path.join(default_root, filename)
        url = URLS[name]
        if not os.path.exists(fullname):
            get_path_from_url(url, default_root)
        return fullname

    def _construct_model_tokenizer(self, model, **kwargs):
        # Download the vocab from the url 
        full_name = self._download_resoures("senta", "senta_vocab",
                                            "senta_word_dict.txt")
        vocab = Vocab.load_vocabulary(
            full_name, unk_token='[UNK]', pad_token='[PAD]')

        vocab_size = len(vocab)
        pad_token_id = vocab.to_indices('[PAD]')
        num_classes = 2

        # Construct the tokenizer form the JiebaToeknizer
        tokenizer = JiebaTokenizer(vocab)

        # Select the senta network for the inference
        network = "bow"
        if 'network' in kwargs:
            network = kwargs['network']
        if network == "bow":
            model = BoWModel(vocab_size, num_classes, padding_idx=pad_token_id)
            model_full_name = self._download_resoures("senta", "senta_bow",
                                                      "senta_bow.pdparams")
        elif network == "lstm":
            model = LSTMModel(
                vocab_size,
                num_classes,
                direction='forward',
                padding_idx=pad_token_id,
                pooling_type='max')
            model_full_name = self._download_resoures("senta", "senta_lstm",
                                                      "senta_lstm.pdparams")
        else:
            raise ValueError(
                "Unknown network: %s, it must be one of bow, lstm, bilstm, cnn, gru, bigru, rnn, birnn and bilstm_attn."
                % network)

        # Load the model parameter for the predict
        state_dict = paddle.load(model_full_name)
        model.set_dict(state_dict)
        return model, tokenizer, kwargs

    def _text_tokenize(self, inputs, padding=True, add_special_tokens=True):
        inputs = inputs[0]
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, str) and not isinstance(inputs, list):
            raise TypeError(
                f"Bad inputs, input text should be str or list of str, {type(inputs)} found!"
            )
        infer_data = []
        for i in range(0, len(inputs)):
            ids = self._tokenizer.encode(inputs[i])
            lens = len(ids)
            infer_data.append([ids, lens])
        infer_ds = MapDataset(infer_data)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.vocab.token_to_idx.get('[PAD]', 0)),  # input_ids
            Stack(dtype='int64'),  # seq_len
        ): fn(samples)

        batch_size = self.kwargs[
            'batch_size'] if 'batch_size' in self.kwargs else 1
        num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0
        infer_data_loader = paddle.io.DataLoader(
            infer_ds,
            collate_fn=batchify_fn,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
            return_list=True)
        outputs = {}
        outputs['text'] = inputs
        outputs['data_loader'] = infer_data_loader
        return outputs

    def _run_model(self, inputs):
        results = []
        with paddle.no_grad():
            for batch in inputs['data_loader']:
                input_ids, seq_len = batch
                logits = self._model(input_ids, seq_len)
                probs = F.softmax(logits, axis=1)
                idx = paddle.argmax(probs, axis=1).numpy()
                idx = idx.tolist()
                labels = [self._label_map[i] for i in idx]
                results.extend(labels)
        inputs['result'] = results
        return inputs

    def _postprocess(self, inputs):
        final_results = []
        for text, label in zip(inputs['text'], inputs['result']):
            result = {}
            result['text'] = text
            result['label'] = label
            final_results.append(result)
        return final_results
