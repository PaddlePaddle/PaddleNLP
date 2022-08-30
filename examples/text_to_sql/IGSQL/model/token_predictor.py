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
"""Predicts a token."""

from collections import namedtuple

import paddle
import paddle.nn.functional as F

from . import model_utils

from .attention import Attention, AttentionResult


class PredictionInput(
        namedtuple('PredictionInput', ('decoder_state', 'input_hidden_states',
                                       'snippets', 'input_sequence'))):
    """ Inputs to the token predictor. """
    __slots__ = ()


class PredictionInputWithSchema(
        namedtuple('PredictionInputWithSchema',
                   ('decoder_state', 'input_hidden_states', 'schema_states',
                    'snippets', 'input_sequence', 'previous_queries',
                    'previous_query_states', 'input_schema'))):
    """ Inputs to the token predictor. """
    __slots__ = ()


class TokenPrediction(
        namedtuple(
            'TokenPrediction',
            ('scores', 'aligned_tokens', 'utterance_attention_results',
             'schema_attention_results', 'query_attention_results',
             'copy_switch', 'query_scores', 'query_tokens', 'decoder_state'))):
    """A token prediction."""
    __slots__ = ()


def score_schema_tokens(input_schema, schema_states, scorer):
    # schema_states: emd_dim x num_tokens
    scores = paddle.t(paddle.mm(paddle.t(scorer),
                                schema_states))  # num_tokens x 1
    if scores.shape[0] != len(input_schema):
        raise ValueError("Got " + str(scores.shape[0]) + " scores for " +
                         str(len(input_schema)) + " schema tokens")
    return scores, input_schema.column_names_surface_form


def score_query_tokens(previous_query, previous_query_states, scorer):
    scores = paddle.t(paddle.mm(paddle.t(scorer),
                                previous_query_states))  # num_tokens x 1
    if scores.shape[0] != len(previous_query):
        raise ValueError("Got " + str(scores.shape[0]) + " scores for " +
                         str(len(previous_query)) + " query tokens")
    return scores, previous_query


class TokenPredictor(paddle.nn.Layer):
    """ Predicts a token given a (decoder) state.

    Attributes:
        vocabulary (`Vocabulary`): A vocabulary object for the output.
        attention_module (`Attention`): An attention module.
        state_transformation_weights (`Parameter`): Transforms the input state
            before predicting a token.
        vocabulary_weights (`Parameter`): Final layer weights.
        vocabulary_biases (`Parameter`): Final layer biases.
    """

    def __init__(self, params, vocabulary, attention_key_size):
        super().__init__()
        self.params = params
        self.vocabulary = vocabulary
        self.attention_module = Attention(params.decoder_state_size,
                                          attention_key_size,
                                          attention_key_size)

        bias_initializer = paddle.nn.initializer.Uniform(low=-0.1, high=0.1)

        _initializer = paddle.nn.initializer.XavierUniform()

        state_transform_weights = paddle.ParamAttr(initializer=_initializer)

        vocabulary_weights = paddle.ParamAttr(initializer=_initializer)

        vocabulary_biases = paddle.ParamAttr(initializer=bias_initializer)

        self.state_transform_Linear = paddle.nn.Linear(
            in_features=params.decoder_state_size + attention_key_size,
            out_features=params.decoder_state_size,
            weight_attr=state_transform_weights,
            bias_attr=False)

        self.vocabulary_Linear = paddle.nn.Linear(
            in_features=params.decoder_state_size,
            out_features=len(vocabulary),
            weight_attr=state_transform_weights,
            bias_attr=vocabulary_biases)

    def _get_intermediate_state(self, state, dropout_amount=0.):
        intermediate_state = paddle.tanh(self.state_transform_Linear(state))
        return F.dropout(intermediate_state, dropout_amount)

    def _score_vocabulary_tokens(self, state):
        scores = paddle.t(self.vocabulary_Linear(state))

        if scores.shape[0] != len(self.vocabulary.inorder_tokens):
            raise ValueError("Got " + str(scores.shape[0]) + " scores for " +
                             str(len(self.vocabulary.inorder_tokens)) +
                             " vocabulary items")

        return scores, self.vocabulary.inorder_tokens

    def forward(self, prediction_input, dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states

        attention_results = self.attention_module(decoder_state,
                                                  input_hidden_states)

        state_and_attn = paddle.concat(
            [decoder_state, attention_results.vector], axis=0)

        intermediate_state = self._get_intermediate_state(
            state_and_attn, dropout_amount=dropout_amount)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(
            intermediate_state)

        return TokenPrediction(vocab_scores, vocab_tokens, attention_results,
                               decoder_state)


class SchemaTokenPredictor(TokenPredictor):
    """ Token predictor that also predicts snippets.

    Attributes:
        snippet_weights (`Parameter`): Weights for scoring snippets against some
            state.
    """

    def __init__(self, params, vocabulary, utterance_attention_key_size,
                 schema_attention_key_size, snippet_size):
        TokenPredictor.__init__(self, params, vocabulary,
                                utterance_attention_key_size)

        _initializer = paddle.nn.initializer.XavierUniform()

        if params.use_schema_attention:
            self.utterance_attention_module = self.attention_module
            self.schema_attention_module = Attention(params.decoder_state_size,
                                                     schema_attention_key_size,
                                                     schema_attention_key_size)

        if self.params.use_query_attention:
            self.query_attention_module = Attention(params.decoder_state_size,
                                                    params.encoder_state_size,
                                                    params.encoder_state_size)

            self.start_query_attention_vector = self.create_parameter(
                [params.encoder_state_size],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Uniform(low=-0.1,
                                                                  high=0.1))

        state_transform_weights = paddle.ParamAttr(initializer=_initializer)
        if params.use_schema_attention and self.params.use_query_attention:
            self.state_transform_Linear = paddle.nn.Linear(
                in_features=params.decoder_state_size +
                utterance_attention_key_size + schema_attention_key_size +
                params.encoder_state_size,
                out_features=params.decoder_state_size,
                weight_attr=state_transform_weights,
                bias_attr=False)
        elif params.use_schema_attention:
            self.state_transform_Linear = paddle.nn.Linear(
                in_features=params.decoder_state_size +
                utterance_attention_key_size + schema_attention_key_size,
                out_features=params.decoder_state_size,
                weight_attr=state_transform_weights,
                bias_attr=False)

        schema_token_weights = paddle.ParamAttr(initializer=_initializer)
        self.schema_token_Linear = paddle.nn.Linear(
            in_features=params.decoder_state_size,
            out_features=schema_attention_key_size,
            weight_attr=schema_token_weights,
            bias_attr=False)

        if self.params.use_previous_query:

            query_token_weights = paddle.ParamAttr(initializer=_initializer)
            self.query_token_Linear = paddle.nn.Linear(
                in_features=params.decoder_state_size,
                out_features=self.params.encoder_state_size,
                weight_attr=query_token_weights,
                bias_attr=False)

        if self.params.use_copy_switch:
            state2copyswitch_transform_weights = paddle.ParamAttr(
                initializer=_initializer)
            if self.params.use_query_attention:
                self.state2copyswitch_transform_Linear = paddle.nn.Linear(
                    in_features=params.decoder_state_size +
                    utterance_attention_key_size + schema_attention_key_size +
                    params.encoder_state_size,
                    out_features=1,
                    weight_attr=state2copyswitch_transform_weights,
                    bias_attr=False)
            else:
                self.state2copyswitch_transform_Linear = paddle.nn.Linear(
                    in_features=params.decoder_state_size +
                    utterance_attention_key_size + schema_attention_key_size,
                    out_features=1,
                    weight_attr=state2copyswitch_transform_weights,
                    bias_attr=False)

        state2copy_transform_weights = paddle.ParamAttr(
            initializer=_initializer)
        self.state2copy_transform_Linear = paddle.nn.Linear(
            in_features=params.decoder_state_size,
            out_features=3,
            weight_attr=state2copy_transform_weights,
            bias_attr=False)

    def _get_schema_token_scorer(self, state):
        scorer = paddle.t(self.schema_token_Linear(state))
        return scorer

    def _get_query_token_scorer(self, state):
        scorer = paddle.t(self.query_token_Linear(state))
        return scorer

    def _get_copy_switch(self, state):
        copy_switch = F.sigmoid(self.state2copyswitch_transform_Linear(state))
        return copy_switch.squeeze()

    def forward(self, prediction_input, dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states
        snippets = prediction_input.snippets

        input_schema = prediction_input.input_schema
        schema_states = prediction_input.schema_states

        if self.params.use_schema_attention:
            schema_attention_results = self.schema_attention_module(
                decoder_state, schema_states)
            utterance_attention_results = self.utterance_attention_module(
                decoder_state, input_hidden_states)
        else:
            utterance_attention_results = self.attention_module(
                decoder_state, input_hidden_states)
            schema_attention_results = None

        query_attention_results = None
        if self.params.use_query_attention:
            previous_query_states = prediction_input.previous_query_states
            if len(previous_query_states) > 0:
                query_attention_results = self.query_attention_module(
                    decoder_state, previous_query_states[-1])
            else:
                query_attention_results = self.start_query_attention_vector
                query_attention_results = AttentionResult(
                    None, None, query_attention_results)

        if self.params.use_schema_attention and self.params.use_query_attention:
            state_and_attn = paddle.concat([
                decoder_state, utterance_attention_results.vector,
                schema_attention_results.vector, query_attention_results.vector
            ],
                                           axis=0)
        elif self.params.use_schema_attention:
            state_and_attn = paddle.concat([
                decoder_state, utterance_attention_results.vector,
                schema_attention_results.vector
            ],
                                           axis=0)
        else:
            state_and_attn = paddle.concat(
                [decoder_state, utterance_attention_results.vector], axis=0)

        intermediate_state = self._get_intermediate_state(
            state_and_attn, dropout_amount=dropout_amount)
        copy_score = F.sigmoid(
            self.state2copy_transform_Linear(intermediate_state).squeeze(0))

        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(
            intermediate_state)

        final_scores = vocab_scores
        aligned_tokens = []
        aligned_tokens.extend(vocab_tokens)

        schema_states = paddle.stack(schema_states, axis=1)
        schema_scores, schema_tokens = score_schema_tokens(
            input_schema, schema_states,
            self._get_schema_token_scorer(intermediate_state))

        final_scores = paddle.concat(
            [copy_score[0] * final_scores, copy_score[1] * schema_scores],
            axis=0)
        aligned_tokens.extend(schema_tokens)

        # Previous Queries
        previous_queries = prediction_input.previous_queries
        previous_query_states = prediction_input.previous_query_states

        copy_switch = None
        query_scores = None
        query_tokens = None
        if self.params.use_previous_query and len(previous_queries) > 0:
            if self.params.use_copy_switch:
                copy_switch = self._get_copy_switch(state_and_attn)
            for turn, (previous_query, previous_query_state) in enumerate(
                    zip(previous_queries, previous_query_states)):

                assert len(previous_query) == len(previous_query_state)
                previous_query_state = paddle.stack(previous_query_state,
                                                    axis=1)
                query_scores, query_tokens = score_query_tokens(
                    previous_query, previous_query_state,
                    self._get_query_token_scorer(intermediate_state))
                query_scores = query_scores.squeeze()

        if query_scores is not None:
            final_scores = paddle.concat(
                [final_scores, copy_score[2] * query_scores], axis=0)
            aligned_tokens += query_tokens

        return TokenPrediction(final_scores, aligned_tokens,
                               utterance_attention_results,
                               schema_attention_results,
                               query_attention_results, copy_switch,
                               query_scores, query_tokens, decoder_state)


class AnonymizationTokenPredictor(TokenPredictor):
    """ Token predictor that also predicts anonymization tokens.

    Attributes:
        anonymizer (`Anonymizer`): The anonymization object.

    """

    def __init__(self, params, vocabulary, attention_key_size, anonymizer):
        TokenPredictor.__init__(self, params, vocabulary, attention_key_size)
        if not anonymizer:
            raise ValueError("Expected an anonymizer, but was None")
        self.anonymizer = anonymizer

    def _score_anonymized_tokens(self, input_sequence, attention_scores):
        scores = []
        tokens = []
        for i, token in enumerate(input_sequence):
            if self.anonymizer.is_anon_tok(token):
                scores.append(attention_scores[i])
                tokens.append(token)

        if len(scores) > 0:
            if len(scores) != len(tokens):
                raise ValueError("Got " + str(len(scores)) + " scores for " +
                                 str(len(tokens)) + " anonymized tokens")

            anonymized_scores = paddle.concat(scores, axis=0)
            if anonymized_scores.dim() == 1:
                anonymized_scores = anonymized_scores.unsqueeze(1)
            return anonymized_scores, tokens
        else:
            return None, []

    def forward(self, prediction_input, dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states
        input_sequence = prediction_input.input_sequence
        assert input_sequence

        attention_results = self.attention_module(decoder_state,
                                                  input_hidden_states)

        state_and_attn = paddle.concat(
            [decoder_state, attention_results.vector], axis=0)

        intermediate_state = self._get_intermediate_state(
            state_and_attn, dropout_amount=dropout_amount)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(
            intermediate_state)

        final_scores = vocab_scores
        aligned_tokens = []
        aligned_tokens.extend(vocab_tokens)

        anonymized_scores, anonymized_tokens = self._score_anonymized_tokens(
            input_sequence, attention_results.scores)

        if anonymized_scores:
            final_scores = paddle.concat([final_scores, anonymized_scores],
                                         axis=0)
            aligned_tokens.extend(anonymized_tokens)

        final_scores = final_scores.squeeze()

        return TokenPrediction(final_scores, aligned_tokens, attention_results,
                               None, None, None, None, None, decoder_state)


def construct_token_predictor(params,
                              vocabulary,
                              utterance_attention_key_size,
                              schema_attention_key_size,
                              snippet_size,
                              anonymizer=None):
    """ Constructs a token predictor given the parameters.

    Args:
        params (`dict`): Contains the command line parameters/hyperparameters.
        vocabulary (`Vocabulary`): Vocabulary object for output generation.
        attention_key_size (`int`): The size of the attention keys.
        anonymizer (`Anonymizer`): An anonymization object.
    """

    if not anonymizer and not params.previous_decoder_snippet_encoding:
        print('using SchemaTokenPredictor')
        return SchemaTokenPredictor(params, vocabulary,
                                    utterance_attention_key_size,
                                    schema_attention_key_size, snippet_size)
    elif params.use_snippets and anonymizer and not params.previous_decoder_snippet_encoding:
        print('using SnippetAnonymizationTokenPredictor')
        return SnippetAnonymizationTokenPredictor(params, vocabulary,
                                                  utterance_attention_key_size,
                                                  snippet_size, anonymizer)
    else:
        print('Unknown token_predictor')
        exit()
