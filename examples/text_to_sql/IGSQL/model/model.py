""" Class for the Sequence to sequence model for ATIS."""

import os

# import torch
# import torch.nn.functional as F

import paddle

from . import torch_utils
from . import utils_ernie

from data_util.vocabulary import DEL_TOK, UNK_TOK

# from .encoder import Encoder
from .embedder import Embedder
from .token_predictor import construct_token_predictor

import numpy as np

from data_util.atis_vocab import ATISVocabulary

import pickle


def get_token_indices(token, index_to_token):
    """ Maps from a gold token (string) to a list of indices.

    Inputs:
        token (string): String to look up.
        index_to_token (list of tokens): Ordered list of tokens.

    Returns:
        list of int, representing the indices of the token in the probability
            distribution.
    """
    if token in index_to_token:
        if len(set(index_to_token)) == len(index_to_token):  # no duplicates
            return [index_to_token.index(token)]
        else:
            indices = []
            # print(f"token:{token}")
            # print(f"index_to_token:{index_to_token}")
            for index, other_token in enumerate(index_to_token):
                # print(f"token:{token}==other_token:{other_token}")
                if token == other_token:
                    indices.append(index)
            assert len(indices) == len(set(indices))
            return indices
    else:
        return [index_to_token.index(UNK_TOK)]


def flatten_utterances(utterances):
    """ Gets a flat sequence from a sequence of utterances.

    Inputs:
        utterances (list of list of str): Utterances to concatenate.

    Returns:
        list of str, representing the flattened sequence with separating
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

    Inputs:
        snippets (list of Snippet): Input snippets.
        states (list of dy.Expression): Previous hidden states to use.
        TODO: should this by dy.Expression or vector values?
    """
    for snippet in snippets:
        snippet.set_embedding(
            paddle.concat(
                [states[snippet.startpos], states[snippet.endpos]], axis=0))
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

        vocabulary_embeddings = np.zeros(
            (len(vocab), glove_embedding_size), dtype=np.float32)
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
            self.model_ernie, self.tokenizer, self.ernie_config = utils_ernie.get_ernie(
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
        #     else:
        #         input_vocabulary_embeddings, output_vocabulary_embeddings, output_vocabulary_schema_embeddings, input_embedding_size = load_word_embeddings(input_vocabulary, output_vocabulary, output_vocabulary_schema, params)

        #         params.input_embedding_size = input_embedding_size
        #         self.params.input_embedding_size = input_embedding_size

        #         # Create the input embeddings
        #         self.input_embedder = Embedder(params.input_embedding_size,
        #                                        name="input-embedding",
        #                                        initializer=input_vocabulary_embeddings,
        #                                        vocabulary=input_vocabulary,
        #                                        anonymizer=anonymizer,
        #                                        freeze=params.freeze)

        #         # Create the output embeddings
        #         self.output_embedder = Embedder(params.output_embedding_size,
        #                                         name="output-embedding",
        #                                         initializer=output_vocabulary_embeddings,
        #                                         vocabulary=output_vocabulary,
        #                                         anonymizer=anonymizer,
        #                                         freeze=False)

        #         self.column_name_token_embedder = Embedder(params.input_embedding_size,
        #                                         name="schema-embedding",
        #                                         initializer=output_vocabulary_schema_embeddings,
        #                                         vocabulary=output_vocabulary_schema,
        #                                         anonymizer=anonymizer,
        #                                         freeze=params.freeze)
        # else:
        #     # Create the input embeddings
        #     self.input_embedder = Embedder(params.input_embedding_size,
        #                                    name="input-embedding",
        #                                    vocabulary=input_vocabulary,
        #                                    anonymizer=anonymizer,
        #                                    freeze=False)

        #     # Create the output embeddings
        #     self.output_embedder = Embedder(params.output_embedding_size,
        #                                     name="output-embedding",
        #                                     vocabulary=output_vocabulary,
        #                                     anonymizer=anonymizer,
        #                                     freeze=False)

        #     self.column_name_token_embedder = None

        # Create the encoder
        encoder_input_size = params.input_embedding_size
        encoder_output_size = params.encoder_state_size
        if params.use_bert:
            encoder_input_size = self.ernie_config["hidden_size"]

        if params.discourse_level_lstm:
            encoder_input_size += params.encoder_state_size // 2

        # self.utterance_encoder = Encoder(params.encoder_num_layers, encoder_input_size, encoder_output_size)
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
            # self.discourse_lstms = torch_utils.create_multilayer_lstm_params(1, params.encoder_state_size, params.encoder_state_size / 2, "LSTM-t")
            self.discourse_lstms = paddle.nn.LSTMCell(
                params.encoder_state_size, params.encoder_state_size // 2)

            initial_discourse_state = self.create_parameter(
                [params.encoder_state_size // 2],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Uniform(
                    low=-0.1, high=0.1))
            self.add_parameter("initial_discourse_state",
                               initial_discourse_state)

            # self.initial_discourse_state = torch_utils.add_params(tuple([params.encoder_state_size // 2]))

        # Snippet encoder
        final_snippet_size = 0
        # if params.use_snippets and not params.previous_decoder_snippet_encoding:
        #     snippet_encoding_size = int(params.encoder_state_size / 2)
        #     final_snippet_size = params.encoder_state_size
        #     if params.snippet_age_embedding:
        #         snippet_encoding_size -= int(
        #             params.snippet_age_embedding_size / 4)
        #         self.snippet_age_embedder = Embedder(
        #             params.snippet_age_embedding_size,
        #             name="snippet-age-embedding",
        #             num_tokens=params.max_snippet_age_embedding)
        #         final_snippet_size = params.encoder_state_size + params.snippet_age_embedding_size / 2

        #     self.snippet_encoder = Encoder(params.snippet_num_layers,
        #                                    params.output_embedding_size,
        #                                    snippet_encoding_size)

        # Previous query Encoder
        if params.use_previous_query:
            self.query_encoder = paddle.nn.LSTM(
                params.output_embedding_size,
                params.encoder_state_size // 2,
                num_layers=params.encoder_num_layers,
                direction='bidirect')
            # Encoder(params.encoder_num_layers, params.output_embedding_size, params.encoder_state_size)

        self.final_snippet_size = final_snippet_size

# def _encode_snippets(self, previous_query, snippets, input_schema):
#     """ Computes a single vector representation for each snippet.

#     Inputs:
#         previous_query (list of str): Previous query in the interaction.
#         snippets (list of Snippet): Snippets extracted from the previous

#     Returns:
#         list of Snippets, where the embedding is set to a vector.
#     """
#     startpoints = [snippet.startpos for snippet in snippets]
#     endpoints = [snippet.endpos for snippet in snippets]
#     assert len(startpoints) == 0 or min(startpoints) >= 0
#     if input_schema:
#         assert len(endpoints) == 0 or max(endpoints) <= len(previous_query)
#     else:
#         assert len(endpoints) == 0 or max(endpoints) < len(previous_query)

#     snippet_embedder = lambda query_token: self.get_query_token_embedding(query_token, input_schema)
#     if previous_query and snippets:
#         _, previous_outputs = self.snippet_encoder(
#             previous_query, snippet_embedder, dropout_amount=self.dropout)
#         assert len(previous_outputs) == len(previous_query)

#         for snippet in snippets:
#             if input_schema:
#                 embedding = paddle.concat([previous_outputs[snippet.startpos],previous_outputs[snippet.endpos-1]], axis=0)
#             else:
#                 embedding = paddle.concat([previous_outputs[snippet.startpos],previous_outputs[snippet.endpos]], axis=0)
#             if self.params.snippet_age_embedding:
#                 embedding = paddle.concat([embedding, self.snippet_age_embedder(min(snippet.age, self.params.max_snippet_age_embedding - 1))], axis=0)
#             snippet.set_embedding(embedding)

#     return snippets

    def _initialize_discourse_states(self):
        discourse_state = self.initial_discourse_state

        # discourse_lstm_states = []
        # for lstm in self.discourse_lstms:
        hidden_size = self.discourse_lstms.weight_hh.shape[1]
        # if paddle.get_device()=='cpu':
        # h_0 = torch.cuda.FloatTensor(1,hidden_size).fill_(0)
        # c_0 = torch.cuda.FloatTensor(1,hidden_size).fill_(0)

        h_0 = paddle.zeros([1, hidden_size])
        c_0 = paddle.zeros([1, hidden_size])

        # else:
        #     # h_0 = torch.zeros(1,hidden_size)
        #     # c_0 = torch.zeros(1,hidden_size)

        # discourse_lstm_states.append((h_0, c_0))

        return discourse_state, (h_0, c_0)

    def _add_positional_embeddings(self, hidden_states, utterances,
                                   group=False):
        grouped_states = []

        start_index = 0
        for utterance in utterances:
            grouped_states.append(hidden_states[start_index:start_index + len(
                utterance)])
            start_index += len(utterance)
        assert len(hidden_states) == sum(
            [len(seq) for seq in grouped_states]) == sum(
                [len(utterance) for utterance in utterances])

        new_states = []
        flat_sequence = []

        num_utterances_to_keep = min(self.params.maximum_utterances,
                                     len(utterances))
        for i, (states, utterance) in enumerate(
                zip(grouped_states[-num_utterances_to_keep:], utterances[
                    -num_utterances_to_keep:])):
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
                    if 'model_ernie' in name:
                        params_bert_trainer.append(param)
                    else:
                        params_trainer.append(param)
        clip = paddle.nn.ClipGradByNorm(clip_norm=self.params.clip)

        if self.params.scheduler:
            self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(
                learning_rate=self.params.initial_learning_rate,
                mode='min', )
            self.trainer = paddle.optimizer.Adam(
                parameters=params_trainer,
                learning_rate=self.scheduler,
                grad_clip=clip)
        else:
            #print('decoder warmup scheduler')
            sss = 1.0  #paddle.optimizer.lr.LinearWarmup(self.params.initial_learning_rate, 100, 0, self.params.initial_learning_rate)
            self.trainer = paddle.optimizer.Adam(
                parameters=params_trainer, learning_rate=sss
            )  #self.params.initial_learning_rate,grad_clip=clip)
        if self.params.fine_tune_bert:
            if self.params.scheduler:
                self.scheduler = paddle.optimizer.lr.ReduceOnPlateau(
                    learning_rate=self.params.initial_learning_rate,
                    mode='min', )
                self.bert_trainer = paddle.optimizer.Adam(
                    parameters=params_bert_trainer,
                    learning_rate=self.scheduler,
                    grad_clip=clip)
            else:
                # print("bert warmup scheduler")
                yyy = 1.0  #paddle.optimizer.lr.LinearWarmup(self.params.initial_learning_rate*0.01, 100, 0, self.params.initial_learning_rate*0.01)
                self.bert_trainer = paddle.optimizer.Adam(
                    parameters=params_bert_trainer,
                    learning_rate=yyy)  #)self.params.lr_bert,grad_clip=clip)

    def set_dropout(self, value):
        """ Sets the dropout to a specified value.

        Inputs:
            value (float): Value to set dropout to.
        """
        self.dropout = value

    def set_learning_rate(self, value):
        """ Sets the learning rate for the trainer.

        Inputs:
            value (float): The new learning rate.
        """
        #return
        for param_group in self.trainer._parameter_list:
            if self.params.all_in_one_trainer:
                if 'model_ernie' in param_group.name:
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

        Inputs:
            filename (str): The filename to save to.
        """
        paddle.save(self.state_dict(), filename)

    def load(self, filename):
        """ Loads saved parameters into the parameter collection.

        Inputs:
            filename (str): Name of file containing parameters.
        """
        self.load_dict(paddle.load(filename))
        print("Loaded model from file " + filename)
