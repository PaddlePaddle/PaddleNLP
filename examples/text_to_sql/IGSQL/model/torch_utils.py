"""Contains various utility functions for Dynet models."""

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

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

    Inputs:
        gold_seq (list of str): A sequence of gold tokens.
        scores (list of dy.Expression): Expressions representing the scores of
            potential output tokens for each token in gold_seq.
        index_to_token_maps (list of dict str->list of int): Maps from index in the
            sequence to a dictionary mapping from a string to a set of integers.
        gold_tok_to_id (lambda (str, str)->list of int): Maps from the gold token
            and some lookup function to the indices in the probability distribution
            where the gold token occurs.
        noise (float, optional): The amount of noise to add to the loss.

    Returns:
        dy.Expression representing the sum of losses over the sequence.
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

        # predicted_sql.append(token_map[score.argmax()])

        if prob_of_tok < noise_i:
            prob_of_tok = prob_of_tok + noise_i
        elif prob_of_tok > 1 - noise_i:
            prob_of_tok = prob_of_tok - noise_i
        # print(f"prob_of_tok:{prob_of_tok}")
        losses.append(-paddle.log(prob_of_tok))
    # print(f"gold_seq: {gold_seq}")
    # print(f"pred_sql: {predicted_sql}")
    # print('-'*200)
    return paddle.sum(paddle.stack(losses))


def get_seq_from_scores(scores, index_to_token_maps):
    """Gets the argmax sequence from a set of scores.

    Inputs:
        scores (list of dy.Expression): Sequences of output scores.
        index_to_token_maps (list of list of str): For each output token, maps
            the index in the probability distribution to a string.

    Returns:
        list of str, representing the argmax sequence.
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

    Inputs:
        gold_seq (list of str): A list of gold tokens.
        pred_seq (list of str): A list of predicted tokens.

    Returns:
        float, representing the accuracy.
    """
    num_correct = 0
    for i, gold_token in enumerate(gold_seq):
        if i < len(pred_seq) and pred_seq[i] == gold_token:
            num_correct += 1

    return float(num_correct) / len(gold_seq)


def forward_one_multilayer(rnns, lstm_input, layer_states, dropout_amount=0.):
    """ Goes forward for one multilayer RNN cell step.

    Inputs:
        lstm_input (dy.Expression): Some input to the step.
        layer_states (list of dy.RNNState): The states of each layer in the cell.
        dropout_amount (float, optional): The amount of dropout to apply, in
            between the layers.

    Returns:
        (list of dy.Expression, list of dy.Expression), dy.Expression, (list of dy.RNNSTate),
        representing (each layer's cell memory, each layer's cell hidden state),
        the final hidden state, and (each layer's updated RNNState).
    """
    num_layers = len(layer_states)
    new_states = []
    cell_states = []
    hidden_states = []
    state = lstm_input
    for i in range(num_layers):
        # view as (1, input_size)
        ###问题
        layer_h, new_state = rnns[i](paddle.unsqueeze(state, 0),
                                     layer_states[i])
        ###
        new_states.append(new_state)

        layer_h = layer_h.squeeze()
        layer_c = new_state[1].squeeze()

        state = layer_h
        if i < num_layers - 1:
            # In both Dynet and Pytorch
            # p stands for probability of an element to be zeroed. i.e. p=1 means switch off all activations.
            state = F.dropout(state, p=dropout_amount)

        cell_states.append(layer_c)
        hidden_states.append(layer_h)

    return (cell_states, hidden_states), state, new_states


def encode_sequence(sequence, rnns, embedder, dropout_amount=0.):
    """ Encodes a sequence given RNN cells and an embedding function.

    Inputs:
        seq (list of str): The sequence to encode.
        rnns (list of dy._RNNBuilder): The RNNs to use.
        emb_fn (dict str->dy.Expression): Function that embeds strings to
            word vectors.
        size (int): The size of the RNN.
        dropout_amount (float, optional): The amount of dropout to apply.

    Returns:
        (list of dy.Expression, list of dy.Expression), list of dy.Expression,
        where the first pair is the (final cell memories, final cell states) of
        all layers, and the second list is a list of the final layer's cell
        state for all tokens in the sequence.
    """

    batch_size = 1
    layer_states = []
    for rnn in rnns:
        hidden_size = rnn.weight_hh.shape[1]

        # h_0 of shape (batch, hidden_size)
        # c_0 of shape (batch, hidden_size)
        # if rnn.weight_hh.is_cuda:
        #     # h_0 = torch.cuda.FloatTensor(batch_size,hidden_size).fill_(0)
        #     # c_0 = torch.cuda.FloatTensor(batch_size,hidden_size).fill_(0)           
        # else:
        h_0 = paddle.zeros([batch_size, hidden_size])
        c_0 = paddle.zeros([batch_size, hidden_size])

        layer_states.append((h_0, c_0))

    outputs = []
    for token in sequence:
        rnn_input = embedder(token)
        #问题
        (cell_states,
         hidden_states), output, layer_states = forward_one_multilayer(
             rnns, rnn_input, layer_states, dropout_amount)
        outputs.append(output)

        # if sum(paddle.sum(paddle.cast(paddle.isnan(output),'float32')).numpy())!=0:
        #     print(f"output:{output}")

    # is_nan=[paddle.cast(paddle.isnan(outputs[i]),'float32') for i in range(len(outputs))]
    # if sum([paddle.sum(i).numpy() for i in is_nan])!=0:
    #     print(f"outputs:{outputs}")

    return (cell_states, hidden_states), outputs


# def create_multilayer_lstm_params(num_layers, in_size, state_size, name=""):
#     """ Adds a multilayer LSTM to the model parameters.

#     Inputs:
#         num_layers (int): Number of layers to create.
#         in_size (int): The input size to the first layer.
#         state_size (int): The size of the states.
#         model (dy.ParameterCollection): The parameter collection for the model.
#         name (str, optional): The name of the multilayer LSTM.
#     """
#     lstm_layers = []
#     for i in range(num_layers):
#         layer_name = name + "-" + str(i)
#         print("LSTM " + layer_name + ": " + str(in_size) + " x " + str(state_size) + "; default Dynet initialization of hidden weights")
#         lstm_layer = paddle.nn.LSTMCell(input_size=int(in_size), hidden_size=int(state_size), bias_ih_attr=True, bias_hh_attr=True)
#         lstm_layers.append(lstm_layer)
#         in_size = state_size
#     return paddle.nn.LayerList(lstm_layers)


def add_params(size, name="", type=1):
    #     """ Adds parameters to the model.

    #     Inputs:
    #         model (dy.ParameterCollection): The parameter collection for the model.
    #         size (tuple of int): The size to create.
    #         name (str, optional): The name of the parameters.
    #     """
    #     if len(size) == 1:
    #         print("vector " + name + ": " + str(size[0]) + "; uniform in [-0.1, 0.1]")
    #     else:
    #         print("matrix " + name + ": " + str(size[0]) + " x " + str(size[1]) + "; uniform in [-0.1, 0.1]")

    #     # size_int = tuple([int(ss) for ss in size])
    #     size_int = [int(ss) for ss in size]
    #     #return torch.nn.Parameter(torch.empty(size_int).uniform_(-0.1, 0.1))
    #     if len(size) == 1:
    #         # return torch.nn.Parameter(torch.empty(size_int).uniform_(-0.1, 0.1))
    #         _initializer = paddle.nn.initializer.Uniform(low=-0.1, high=0.1)
    #     else:
    #         # tmp_ret = paddle.empty(size_int)
    #         _initializer = paddle.nn.initializer.XavierUniform()

    #     return paddle.static.create_parameter(shape=size_int,dtype='float32',default_initializer=_initializer)      
    return paddle.to_tensor(
        np.random.uniform(
            low=-0.1, high=0.1, size=size).astype(np.float32),
        stop_gradient=False)


def mask_fill(input, mask, value):
    """Fill value to input according to mask
    
    Args:
        input: input matrix
        mask: mask matrix
        value: Fill value

    Returns:
        output

    >>> input
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
    >>> mask
    [
        [True, True, False],
        [True, False, False]
    ]
    >>> mask_fill(input, mask, 0)
    [
        [nan, nan, inf],
        [nan, inf, inf]
    ]
    """
    return input * paddle.cast(paddle.logical_not(mask),
                               input.dtype) + paddle.cast(mask,
                                                          input.dtype) * value


def LSTM_output_transfer(utterance_states, final_utterance_state):

    if len(utterance_states) != 0:
        utterance_states = utterance_states.squeeze(0)
        utterance_states = paddle.split(utterance_states,
                                        utterance_states.shape[0])
        for idx in range(len(utterance_states)):
            utterance_states[idx] = utterance_states[idx].squeeze(0)

    if len(final_utterance_state) != 0:
        (hidden_state, cell_memory) = final_utterance_state
        hidden_states = paddle.concat(
            [hidden_state[0], hidden_state[1]], axis=-1).squeeze(0)
        cell_memories = paddle.concat(
            [cell_memory[0], cell_memory[1]], axis=-1).squeeze(0)
        final_utterance_state = (hidden_states.squeeze(0),
                                 cell_memories.squeeze(0))
    return utterance_states, final_utterance_state
