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
"""Contains various utility functions for Dynet models."""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np


def compute_loss(gold_seq,
                 scores,
                 index_to_token_maps,
                 gold_tok_to_id,
                 noise=0.00000001):
    """ Computes the loss of a gold sequence given scores.

    Args:
        gold_seq (`list`): A sequence of gold tokens.
        scores (`list`): Expressions representing the scores of
            potential output tokens for each token in gold_seq.
        index_to_token_maps (`list`): Maps from index in the
            sequence to a dictionary mapping from a string to a set of integers.
        gold_tok_to_id (`func`): Maps from the gold token
            and some lookup function to the indices in the probability distribution
            where the gold token occurs.
        noise (`float`, optional): The amount of noise to add to the loss.

    Returns:
        `Tensor`: representing the sum of losses over the sequence.
    """
    assert len(gold_seq) == len(scores) == len(index_to_token_maps)

    losses = []
    predicted_sql = []
    for i, gold_tok in enumerate(gold_seq):
        score = scores[i]
        token_map = index_to_token_maps[i]

        gold_indices = gold_tok_to_id(gold_tok, token_map)

        assert len(gold_indices) > 0
        noise_i = noise
        '''
        if len(gold_indices) == 1:
            noise_i = 0
            '''

        probdist = score

        prob_of_tok = paddle.sum(
            paddle.index_select(probdist, paddle.to_tensor(gold_indices)))

        if prob_of_tok < noise_i:
            prob_of_tok = prob_of_tok + noise_i
        elif prob_of_tok > 1 - noise_i:
            prob_of_tok = prob_of_tok - noise_i
        losses.append(-paddle.log(prob_of_tok))

    return paddle.sum(paddle.stack(losses))


def get_seq_from_scores(scores, index_to_token_maps):
    """Gets the argmax sequence from a set of scores.

    Args:
        scores (`list`): Sequences of output scores.
        index_to_token_maps (`list`): For each output token, maps
            the index in the probability distribution to a string.

    Returns:
        `list`: Representing the argmax sequence.
    """
    seq = []
    for score, tok_map in zip(scores, index_to_token_maps):
        # score_numpy_list = score.cpu().detach().numpy()
        score_numpy_list = score.cpu().numpy()
        assert score.shape[0] == len(tok_map) == len(list(score_numpy_list))
        seq.append(tok_map[np.argmax(score_numpy_list)])
    return seq


def per_token_accuracy(gold_seq, pred_seq):
    """ Returns the per-token accuracy comparing two strings (recall).

    Args:
        gold_seq (`list`): A list of gold tokens.
        pred_seq (`list`): A list of predicted tokens.

    Returns:
        `float`: Representing the accuracy.
    """
    num_correct = 0
    for i, gold_token in enumerate(gold_seq):
        if i < len(pred_seq) and pred_seq[i] == gold_token:
            num_correct += 1

    return float(num_correct) / len(gold_seq)


def forward_one_multilayer(rnns, lstm_input, layer_states, dropout_amount=0.):
    """ Goes forward for one multilayer RNN cell step.

    Args:
        lstm_input (`Tensor`): Some input to the step.
        layer_states (`list`): The states of each layer in the cell.
        dropout_amount (`float`, optional): The amount of dropout to apply, in
            between the layers.

    Returns:
        (`list` , `list`), `Tensor`, (`list`): Representing (each layer's cell memory, 
        each layer's cell hidden state), the final hidden state, and (each layer's updated RNNState).
    """
    num_layers = len(layer_states)
    new_states = []
    cell_states = []
    hidden_states = []
    state = lstm_input
    for i in range(num_layers):
        layer_h, new_state = rnns[i](paddle.unsqueeze(state, 0),
                                     layer_states[i])
        new_states.append(new_state)

        layer_h = layer_h.squeeze()
        layer_c = new_state[1].squeeze()

        state = layer_h
        if i < num_layers - 1:
            # p stands for probability of an element to be zeroed. i.e. p=1 means switch off all activations.
            state = F.dropout(state, p=dropout_amount)

        cell_states.append(layer_c)
        hidden_states.append(layer_h)

    return (cell_states, hidden_states), state, new_states


def encode_sequence(sequence, rnns, embedder, dropout_amount=0.):
    """ Encodes a sequence given RNN cells and an embedding function.

    Args:
        seq (`list`): The sequence to encode.
        rnns (`list`): The RNNs to use.
        emb_fn (`func`): Function that embeds strings to
            word vectors.
        size (`int`): The size of the RNN.
        dropout_amount (`float`, optional): The amount of dropout to apply.

    Returns:
        (`list`, `list`), `list`: The first pair is the (final cell memories, final cell states) 
        of all layers, and the second list is a list of the final layer's cell
        state for all tokens in the sequence.
    """

    batch_size = 1
    layer_states = []
    for rnn in rnns:
        hidden_size = rnn.weight_hh.shape[1]

        h_0 = paddle.zeros([batch_size, hidden_size])
        c_0 = paddle.zeros([batch_size, hidden_size])

        layer_states.append((h_0, c_0))

    outputs = []
    for token in sequence:
        rnn_input = embedder(token)
        (cell_states,
         hidden_states), output, layer_states = forward_one_multilayer(
             rnns, rnn_input, layer_states, dropout_amount)
        outputs.append(output)

    return (cell_states, hidden_states), outputs


def mask_fill(input, mask, value):
    return input * paddle.cast(paddle.logical_not(
        mask), input.dtype) + paddle.cast(mask, input.dtype) * value


def LSTM_output_transfer(utterance_states, final_utterance_state):

    if len(utterance_states) != 0:
        utterance_states = utterance_states.squeeze(0)
        utterance_states = paddle.split(utterance_states,
                                        utterance_states.shape[0])
        for idx in range(len(utterance_states)):
            utterance_states[idx] = utterance_states[idx].squeeze(0)

    if len(final_utterance_state) != 0:
        (hidden_state, cell_memory) = final_utterance_state
        hidden_states = paddle.concat([hidden_state[0], hidden_state[1]],
                                      axis=-1).squeeze(0)
        cell_memories = paddle.concat([cell_memory[0], cell_memory[1]],
                                      axis=-1).squeeze(0)
        final_utterance_state = (hidden_states.squeeze(0),
                                 cell_memories.squeeze(0))
    return utterance_states, final_utterance_state
