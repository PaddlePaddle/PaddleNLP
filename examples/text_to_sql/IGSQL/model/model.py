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
""" Class for the Sequence to sequence model for ATIS."""

import os

import paddle

from . import model_utils
from . import bert_utils

from data_util.vocabulary import DEL_TOK, UNK_TOK

from .embedder import Embedder
from .token_predictor import construct_token_predictor

import numpy as np

from data_util.atis_vocab import ATISVocabulary

import pickle


def get_token_indices(token, index_to_token):
    """ Maps from a gold token (string) to a list of indices.

    Args:
        token (`string`): String to look up.
        index_to_token (`list`): Ordered list of tokens.

    Returns:
        `list`: Representing the indices of the token in the probability
            distribution.
    """
    if token in index_to_token:
        if len(set(index_to_token)) == len(index_to_token):  # no duplicates
            return [index_to_token.index(token)]
        else:
            indices = []
            for index, other_token in enumerate(index_to_token):
                if token == other_token:
                    indices.append(index)
            assert len(indices) == len(set(indices))
            return indices
    else:
        return [index_to_token.index(UNK_TOK)]


def flatten_utterances(utterances):
    """ Gets a flat sequence from a sequence of utterances.

    Args:
        utterances (`list`): Utterances to concatenate.

    Returns:
        `list`: Representing the flattened sequence with separating
            delimiter tokens.
    """
    sequence = []
    for i, utterance in enumerate(utterances):
        sequence.extend(utterance)
        if i < len(utterances) - 1:
            sequence.append(DEL_TOK)

    return sequence


def encode_snippets_with_states(snippets, states):
    """ Encodes snippets by using previous query states instead.

    Args:
        snippets (`list`): Input snippets.
        states (`list`): Previous hidden states to use.
    """
    for snippet in snippets:
        snippet.set_embedding(
            paddle.concat([states[snippet.startpos], states[snippet.endpos]],
                          axis=0))
    return snippets


def load_word_embeddings(input_vocabulary, output_vocabulary,
                         output_vocabulary_schema, params):
    print(output_vocabulary.inorder_tokens)
    print()

    if params.reload_embedding == 1:
        input_vocabulary_embeddings = np.load(params.data_directory +
                                              "/input_embeddings.npy")
        output_vocabulary_embeddings = np.load(params.data_directory +
                                               "/ouput_embeddings.npy")
        output_vocabulary_schema_embeddings = np.load(
            params.data_directory + "/output_schema_embeddings.npy")
        input_embedding_size = 300
        return input_vocabulary_embeddings, output_vocabulary_embeddings, output_vocabulary_schema_embeddings, input_embedding_size

    def read_glove_embedding(embedding_filename, embedding_size):
        glove_embeddings = {}

        with open(embedding_filename) as f:
            cnt = 1
            for line in f:
                cnt += 1
                if params.debug or not params.train:
                    if cnt == 1000:
                        print('Read 1000 word embeddings')
                        break
                l_split = line.split()
                word = " ".join(l_split[0:len(l_split) - embedding_size])
                embedding = np.array([
                    float(val)
                    for val in l_split[len(l_split) - embedding_size:]
                ])
                glove_embeddings[word] = embedding

        return glove_embeddings

    print('Loading Glove Embedding from', params.embedding_filename)
    glove_embedding_size = 300
    glove_embeddings = read_glove_embedding(params.embedding_filename,
                                            glove_embedding_size)
    print('Done')

    input_embedding_size = glove_embedding_size

    def create_word_embeddings(vocab):

        vocabulary_embeddings = np.zeros((len(vocab), glove_embedding_size),
                                         dtype=np.float32)
        vocabulary_tokens = vocab.inorder_tokens

        glove_oov = 0
        para_oov = 0
        for token in vocabulary_tokens:
            token_id = vocab.token_to_id(token)
            if token in glove_embeddings:
                vocabulary_embeddings[
                    token_id][:glove_embedding_size] = glove_embeddings[token]
            else:
                glove_oov += 1

        print('Glove OOV:', glove_oov, 'Para OOV', para_oov, 'Total',
              len(vocab))

        return vocabulary_embeddings

    input_vocabulary_embeddings = create_word_embeddings(input_vocabulary)
    output_vocabulary_embeddings = create_word_embeddings(output_vocabulary)
    output_vocabulary_schema_embeddings = None
    if output_vocabulary_schema:
        output_vocabulary_schema_embeddings = create_word_embeddings(
            output_vocabulary_schema)

    np.save(params.data_directory + "/input_embeddings",
            input_vocabulary_embeddings)
    np.save(params.data_directory + "/ouput_embeddings",
            output_vocabulary_embeddings)
    np.save(params.data_directory + "/output_schema_embeddings",
            output_vocabulary_schema_embeddings)

    return input_vocabulary_embeddings, output_vocabulary_embeddings, output_vocabulary_schema_embeddings, input_embedding_size


class ATISModel(paddle.nn.Layer):
    """ Sequence-to-sequence model for predicting a SQL query given an utterance
        and an interaction prefix.
    """

    def __init__(self, params, input_vocabulary, output_vocabulary,
                 output_vocabulary_schema, anonymizer):
        super().__init__()

        self.params = params

        self.dropout = 0.

        if params.use_bert:
            self.model_bert, self.tokenizer, self.bert_config = bert_utils.get_bert(
                params)

        if 'atis' not in params.data_directory:
            if params.use_bert:
                input_vocabulary_embeddings, output_vocabulary_embeddings, output_vocabulary_schema_embeddings, input_embedding_size = load_word_embeddings(
                    input_vocabulary, output_vocabulary,
                    output_vocabulary_schema, params)

                # Create the output embeddings
                self.output_embedder = Embedder(
                    params.output_embedding_size,
                    name="output-embedding",
                    initializer=output_vocabulary_embeddings,
                    vocabulary=output_vocabulary,
                    anonymizer=anonymizer,
                    freeze=False)
                self.column_name_token_embedder = None

        # Create the encoder
        encoder_input_size = params.input_embedding_size
        encoder_output_size = params.encoder_state_size
        if params.use_bert:
            encoder_input_size = self.bert_config["hidden_size"]

        if params.discourse_level_lstm:
            encoder_input_size += params.encoder_state_size // 2

        self.utterance_encoder = paddle.nn.LSTM(
            encoder_input_size,
            encoder_output_size // 2,
            num_layers=params.encoder_num_layers,
            direction='bidirect')

        # Positional embedder for utterances
        attention_key_size = params.encoder_state_size
        self.schema_attention_key_size = attention_key_size
        if params.state_positional_embeddings:
            attention_key_size += params.positional_embedding_size
            self.positional_embedder = Embedder(
                params.positional_embedding_size,
                name="positional-embedding",
                num_tokens=params.maximum_utterances)

        self.utterance_attention_key_size = attention_key_size

        # Create the discourse-level LSTM parameters
        if params.discourse_level_lstm:
            self.discourse_lstms = paddle.nn.LSTMCell(
                params.encoder_state_size, params.encoder_state_size // 2)

            initial_discourse_state = self.create_parameter(
                [params.encoder_state_size // 2],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Uniform(low=-0.1,
                                                                  high=0.1))
            self.add_parameter("initial_discourse_state",
                               initial_discourse_state)

        # Snippet encoder
        final_snippet_size = 0

        # Previous query Encoder
        if params.use_previous_query:
            self.query_encoder = paddle.nn.LSTM(
                params.output_embedding_size,
                params.encoder_state_size // 2,
                num_layers=params.encoder_num_layers,
                direction='bidirect')

        self.final_snippet_size = final_snippet_size

    def _initialize_discourse_states(self):
        discourse_state = self.initial_discourse_state

        hidden_size = self.discourse_lstms.weight_hh.shape[1]

        h_0 = paddle.zeros([1, hidden_size])
        c_0 = paddle.zeros([1, hidden_size])

        return discourse_state, (h_0, c_0)

    def _add_positional_embeddings(self,
                                   hidden_states,
                                   utterances,
                                   group=False):
        grouped_states = []

        start_index = 0
        for utterance in utterances:
            grouped_states.append(hidden_states[start_index:start_index +
                                                len(utterance)])
            start_index += len(utterance)
        assert len(hidden_states) == sum([
            len(seq) for seq in grouped_states
        ]) == sum([len(utterance) for utterance in utterances])

        new_states = []
        flat_sequence = []

        num_utterances_to_keep = min(self.params.maximum_utterances,
                                     len(utterances))
        for i, (states, utterance) in enumerate(
                zip(grouped_states[-num_utterances_to_keep:],
                    utterances[-num_utterances_to_keep:])):
            positional_sequence = []
            index = num_utterances_to_keep - i - 1

            for state in states:
                positional_sequence.append(
                    paddle.concat(
                        [state, self.positional_embedder(index)], axis=0))

            assert len(positional_sequence) == len(utterance), \
                "Expected utterance and state sequence length to be the same, " \
                + "but they were " + str(len(utterance)) \
                + " and " + str(len(positional_sequence))

            if group:
                new_states.append(positional_sequence)
            else:
                new_states.extend(positional_sequence)
            flat_sequence.extend(utterance)

        return new_states, flat_sequence

    def build_optim(self):
        params_trainer = []
        params_bert_trainer = []
        for name, param in self.named_parameters():
            if not param.stop_gradient:
                if self.params.all_in_one_trainer:
                    param.name = name
                    params_trainer.append(param)
                else:
                    if 'model_bert' in name:
                        params_bert_trainer.append(param)
                    else:
                        params_trainer.append(param)
        clip = paddle.nn.ClipGradByNorm(clip_norm=self.params.clip)

        if self.params.scheduler:
            self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(
                learning_rate=self.params.initial_learning_rate,
                mode='min',
            )
            self.trainer = paddle.optimizer.Adam(parameters=params_trainer,
                                                 learning_rate=self.scheduler,
                                                 grad_clip=clip)
        else:
            self.trainer = paddle.optimizer.Adam(parameters=params_trainer,
                                                 learning_rate=1.0,
                                                 grad_clip=clip)
        if self.params.fine_tune_bert:
            if self.params.scheduler:
                self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(
                    learning_rate=self.params.initial_learning_rate,
                    mode='min',
                )
                self.bert_trainer = paddle.optimizer.Adam(
                    parameters=params_bert_trainer,
                    learning_rate=self.scheduler,
                    grad_clip=clip)
            else:
                yyy = 1.0
                self.bert_trainer = paddle.optimizer.Adam(
                    parameters=params_bert_trainer,
                    learning_rate=1.0,
                    grad_clip=clip)

    def set_dropout(self, value):
        """ Sets the dropout to a specified value.

        Args:
            value (`float`): Value to set dropout to.
        """
        self.dropout = value

    def set_learning_rate(self, value):
        """ Sets the learning rate for the trainer.

        Args:
            value (`float`): The new learning rate.
        """
        #return
        for param_group in self.trainer._parameter_list:
            if self.params.all_in_one_trainer:
                if 'model_bert' in param_group.name:
                    param_group.optimize_attr['learning_rate'] = value * 0.01
                else:
                    param_group.optimize_attr['learning_rate'] = value
            else:
                param_group.optimize_attr['learning_rate'] = value

        if self.params.use_bert:
            if not self.params.all_in_one_trainer:
                for param_group in self.bert_trainer._parameter_list:
                    param_group.optimize_attr['learning_rate'] = value * 0.01

    def save(self, filename):
        """ Saves the model to the specified filename.

        Args:
            filename (`str`): The filename to save to.
        """
        paddle.save(self.state_dict(), filename)

    def load(self, filename):
        """ Loads saved parameters into the parameter collection.

        Args:
            filename (`str`): Name of file containing parameters.
        """
        self.load_dict(paddle.load(filename))
        print("Loaded model from file " + filename)
