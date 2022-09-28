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
""" Decoder for the SQL generation problem."""

from collections import namedtuple
import numpy as np

import paddle
import paddle.nn.functional as F

from . import model_utils

from .token_predictor import PredictionInput, PredictionInputWithSchema
import data_util.snippets as snippet_handler
from . import embedder
from data_util.vocabulary import EOS_TOK, UNK_TOK

np.random.seed(0)


def flatten_distribution(distribution_map, probabilities):
    """ Flattens a probability distribution given a map of "unique" values.
        All values in distribution_map with the same value should get the sum
        of the probabilities.

        Arguments:
            distribution_map (`list`): List of values to get the probability for.
            probabilities (`np.ndarray`): Probabilities corresponding to the values in
                distribution_map.

        Returns:
            `list`: `np.ndarray` of the same size where probabilities for duplicates
                in distribution_map are given the sum of the probabilities in probabilities.
    """
    assert len(distribution_map) == len(probabilities)
    if len(distribution_map) != len(set(distribution_map)):
        idx_first_dup = 0
        seen_set = set()
        for i, tok in enumerate(distribution_map):
            if tok in seen_set:
                idx_first_dup = i
                break
            seen_set.add(tok)
        new_dist_map = distribution_map[:idx_first_dup] + list(
            set(distribution_map) - set(distribution_map[:idx_first_dup]))
        assert len(new_dist_map) == len(set(new_dist_map))
        new_probs = np.array(
            probabilities[:idx_first_dup] \
            + [0. for _ in range(len(set(distribution_map)) \
                                 - idx_first_dup)])
        assert len(new_probs) == len(new_dist_map)

        for i, token_name in enumerate(distribution_map[idx_first_dup:]):
            if token_name not in new_dist_map:
                new_dist_map.append(token_name)

            new_index = new_dist_map.index(token_name)
            new_probs[new_index] += probabilities[i + idx_first_dup]
        new_probs = new_probs.tolist()
    else:
        new_dist_map = distribution_map
        new_probs = probabilities

    assert len(new_dist_map) == len(new_probs)

    return new_dist_map, new_probs


class SQLPrediction(
        namedtuple('SQLPrediction',
                   ('predictions', 'sequence', 'probability'))):
    """Contains prediction for a sequence."""
    __slots__ = ()

    def __str__(self):
        return str(self.probability) + "\t" + " ".join(self.sequence)


class SequencePredictorWithSchema(paddle.nn.Layer):
    """ Predicts a sequence.

    Attributes:
        lstms (list of dy.RNNBuilder): The RNN used.
        token_predictor (TokenPredictor): Used to actually predict tokens.
    """

    def __init__(self, params, input_size, output_embedder,
                 column_name_token_embedder, token_predictor):
        super().__init__()

        self.lstmCell = paddle.nn.LSTMCell(input_size,
                                           params.decoder_state_size)

        self.token_predictor = token_predictor
        self.output_embedder = output_embedder
        self.column_name_token_embedder = column_name_token_embedder

        start_token_embedding = self.create_parameter(
            [params.output_embedding_size],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Uniform(low=-0.1,
                                                              high=0.1))
        self.add_parameter("start_token_embedding", start_token_embedding)

        self.input_size = input_size
        self.params = params

    def _initialize_decoder_lstm(self, encoder_state):
        decoder_lstm_states = []

        # check which one is h_0, which is c_0
        c_0 = encoder_state[1].reshape([1, -1])
        h_0 = encoder_state[0].reshape([1, -1])

        decoder_lstm_states.append((h_0, c_0))
        return decoder_lstm_states

    def get_output_token_embedding(self, output_token, input_schema, snippets):
        if self.params.use_snippets and snippet_handler.is_snippet(
                output_token):
            output_token_embedding = embedder.bow_snippets(
                output_token, snippets, self.output_embedder, input_schema)
        else:
            if input_schema:
                assert self.output_embedder.in_vocabulary(
                    output_token) or input_schema.in_vocabulary(
                        output_token, surface_form=True)
                # 经过
                if self.output_embedder.in_vocabulary(output_token):
                    output_token_embedding = self.output_embedder(output_token)
                else:
                    output_token_embedding = input_schema.column_name_embedder(
                        output_token, surface_form=True)
            else:
                output_token_embedding = self.output_embedder(output_token)
        return output_token_embedding

    def get_decoder_input(self, output_token_embedding, prediction):
        if self.params.use_schema_attention and self.params.use_query_attention:
            decoder_input = paddle.concat([
                output_token_embedding,
                prediction.utterance_attention_results.vector,
                prediction.schema_attention_results.vector,
                prediction.query_attention_results.vector
            ],
                                          axis=0)
        elif self.params.use_schema_attention:
            decoder_input = paddle.concat([
                output_token_embedding,
                prediction.utterance_attention_results.vector,
                prediction.schema_attention_results.vector
            ],
                                          axis=0)
        else:
            decoder_input = paddle.concat([
                output_token_embedding,
                prediction.utterance_attention_results.vector
            ],
                                          axis=0)
        return decoder_input

    def forward(self,
                final_encoder_state,
                encoder_states,
                schema_states,
                max_generation_length,
                snippets=None,
                gold_sequence=None,
                input_sequence=None,
                previous_queries=None,
                previous_query_states=None,
                input_schema=None,
                dropout_amount=0.):
        """ Generates a sequence. """
        index = 0

        context_vector_size = self.input_size - self.params.output_embedding_size

        # Decoder states: just the initialized decoder.
        # Current input to decoder: phi(start_token) ; zeros the size of the
        # context vector
        predictions = []
        sequence = []
        probability = 1.

        decoder_states = self._initialize_decoder_lstm(final_encoder_state)[0]

        decoder_input = paddle.concat(
            [self.start_token_embedding,
             paddle.zeros([context_vector_size])],
            axis=0)
        continue_generating = True
        while continue_generating:
            if len(sequence) == 0 or sequence[-1] != EOS_TOK:

                decoder_state, decoder_states = self.lstmCell(
                    decoder_input.unsqueeze(0), decoder_states)
                decoder_state = decoder_state.squeeze()

                prediction_input = PredictionInputWithSchema(
                    decoder_state=decoder_state,
                    input_hidden_states=encoder_states,
                    schema_states=schema_states,
                    snippets=snippets,
                    input_sequence=input_sequence,
                    previous_queries=previous_queries,
                    previous_query_states=previous_query_states,
                    input_schema=input_schema)

                prediction = self.token_predictor(prediction_input,
                                                  dropout_amount=dropout_amount)

                predictions.append(prediction)
                # 经过
                if gold_sequence:
                    output_token = gold_sequence[index]

                    output_token_embedding = self.get_output_token_embedding(
                        output_token, input_schema, snippets)

                    decoder_input = self.get_decoder_input(
                        output_token_embedding, prediction)

                    sequence.append(gold_sequence[index])

                    if index >= len(gold_sequence) - 1:
                        continue_generating = False
                else:
                    assert prediction.scores.dim() == 1
                    probabilities = F.softmax(prediction.scores,
                                              axis=0).cpu().numpy().tolist()

                    distribution_map = prediction.aligned_tokens
                    assert len(probabilities) == len(distribution_map)

                    if self.params.use_previous_query and self.params.use_copy_switch and len(
                            previous_queries) > 0:
                        assert prediction.query_scores.dim() == 1
                        query_token_probabilities = F.softmax(
                            prediction.query_scores,
                            axis=0).cpu().data.numpy().tolist()

                        query_token_distribution_map = prediction.query_tokens

                        assert len(query_token_probabilities) == len(
                            query_token_distribution_map)

                        copy_switch = prediction.copy_switch.cpu().data.numpy()

                        # Merge the two
                        probabilities = ((np.array(probabilities) *
                                          (1 - copy_switch)).tolist() +
                                         (np.array(query_token_probabilities) *
                                          copy_switch).tolist())
                        distribution_map = distribution_map + query_token_distribution_map
                        assert len(probabilities) == len(distribution_map)

                    # Get a new probabilities and distribution_map consolidating duplicates
                    distribution_map, probabilities = flatten_distribution(
                        distribution_map, probabilities)

                    # Modify the probability distribution so that the UNK token can never be produced
                    probabilities[distribution_map.index(UNK_TOK)] = 0.
                    argmax_index = int(np.argmax(probabilities))

                    argmax_token = distribution_map[argmax_index]
                    sequence.append(argmax_token)

                    output_token_embedding = self.get_output_token_embedding(
                        argmax_token, input_schema, snippets)

                    decoder_input = self.get_decoder_input(
                        output_token_embedding, prediction)

                    probability *= probabilities[argmax_index]

                    continue_generating = False
                    if index < max_generation_length and argmax_token != EOS_TOK:
                        continue_generating = True

            index += 1

        return SQLPrediction(predictions, sequence, probability)
