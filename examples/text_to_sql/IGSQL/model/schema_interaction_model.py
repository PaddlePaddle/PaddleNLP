""" Class for the Sequence to sequence model for ATIS."""

import numpy as np
# import torch
# import torch.nn.functional as F

import paddle
import paddle.nn.functional as F

from copy import deepcopy

from . import torch_utils

import data_util.snippets as snippet_handler
import data_util.sql_util
import data_util.vocabulary as vocab
from data_util.vocabulary import EOS_TOK, UNK_TOK
import data_util.tokenizers

from .token_predictor import construct_token_predictor
from .attention import Attention
from .model import ATISModel, encode_snippets_with_states, get_token_indices
from data_util.utterance import ANON_INPUT_KEY

# from .encoder import Encoder
from .decoder import SequencePredictorWithSchema

from . import utils_ernie

import data_util.atis_batch

np.random.seed(0)


### 问题
class GraphNN(paddle.nn.Layer):
    def __init__(self, params):
        super(GraphNN, self).__init__()
        self.params = params

        weight_attr_final_fc = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())
        bias_attr_final_fc = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.0))

        weight_attr_fc = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())
        bias_attr_fc = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.0))

        weight_attr_qfc = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())
        bias_attr_qfc = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.0))

        self.final_fc = paddle.nn.Linear(
            params.encoder_state_size,
            params.encoder_state_size,
            weight_attr=weight_attr_final_fc,
            bias_attr=bias_attr_final_fc)
        self.fc = paddle.nn.Linear(
            params.encoder_state_size,
            params.encoder_state_size,
            weight_attr=weight_attr_fc,
            bias_attr=bias_attr_fc)
        self.qfc = paddle.nn.Linear(
            params.encoder_state_size,
            params.encoder_state_size,
            weight_attr=weight_attr_qfc,
            bias_attr=bias_attr_qfc)
        self.dropout = paddle.nn.Dropout(0.1)
        # self.weight_init()
        self.leakyReLU = paddle.nn.LeakyReLU(0.2)
        self.elu = paddle.nn.ELU()
        self.relu = paddle.nn.ReLU()

    # def weight_init(self):
    #     for m in self.sublayers():
    #         if isinstance(m, paddle.nn.Linear):
    #             # torch.nn.init.xavier_uniform_(m.weight)
    #             # torch.nn.init.constant_(m.bias, 0.)
    #             m.weight=paddle.static.create_parameter(shape=[m.weight.shape[0],m.weight.shape[1]], dtype='float32',default_initializer=)
    #             m.bias=paddle.static.create_parameter(shape=m.bias.shape, dtype='float32',default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, x, adj_matrix, previous_x=None):
        # x: [len_tokens, d]
        # adj_matrix: [len_tokens, len_tokens]
        len_tokens = x.shape[0]
        if previous_x is not None:
            x_new = self.leakyReLU(
                self.fc(paddle.concat(
                    [previous_x, x], axis=0))).unsqueeze(0)
        else:
            x_new = self.leakyReLU(
                self.fc(x).unsqueeze(0))  # [1, len_tokens, d]
        q = self.leakyReLU(self.qfc(x_new))
        x_ = paddle.concat(paddle.split(x_new, 3, axis=2), axis=0)
        q_ = paddle.concat(paddle.split(q, 3, axis=2), axis=0)
        outputs = paddle.matmul(q_, x_.transpose(
            [0, 2, 1])) / 10.0  # [3, len_tokens, len_tokens]
        tmp_adj_matrix = (adj_matrix == 0).expand(
            shape=[3, adj_matrix.shape[0], adj_matrix.shape[1]])
        outputs = torch_utils.mask_fill(
            input=outputs, mask=tmp_adj_matrix, value=-1e9)
        outputs = self.dropout(F.softmax(outputs, axis=-1))
        outputs = paddle.matmul(outputs, x_)
        outputs = paddle.concat(paddle.split(outputs, 3, axis=0), axis=2)
        if previous_x is not None:
            outputs = paddle.split(outputs, 2, axis=1)[1]
        outputs = x.unsqueeze(0) + outputs
        ret = x + self.dropout(
            self.leakyReLU(self.final_fc(outputs).squeeze(0)))
        return ret


LIMITED_INTERACTIONS = {
    "raw/atis2/12-1.1/ATIS2/TEXT/TRAIN/SRI/QS0/1": 22,
    "raw/atis3/17-1.1/ATIS3/SP_TRN/MIT/8K7/5": 14,
    "raw/atis2/12-1.1/ATIS2/TEXT/TEST/NOV92/770/5": -1
}

END_OF_INTERACTION = {"quit", "exit", "done"}


class SchemaInteractionATISModel(ATISModel):
    """ Interaction ATIS model, where an interaction is processed all at once.
    """

    def __init__(self, params, input_vocabulary, output_vocabulary,
                 output_vocabulary_schema, anonymizer):
        ATISModel.__init__(self, params, input_vocabulary, output_vocabulary,
                           output_vocabulary_schema, anonymizer)

        if self.params.use_schema_encoder:
            # Create the schema encoder
            schema_encoder_num_layer = 1
            schema_encoder_input_size = params.input_embedding_size
            schema_encoder_state_size = params.encoder_state_size
            if params.use_bert:
                schema_encoder_input_size = self.ernie_config["hidden_size"]

            self.schema_encoder = paddle.nn.LSTM(
                schema_encoder_input_size,
                schema_encoder_state_size // 2,
                num_layers=params.encoder_num_layers,
                direction='bidirect')
            # Encoder(schema_encoder_num_layer, schema_encoder_input_size, schema_encoder_state_size)

        # self-attention
        if self.params.use_schema_self_attention:
            self.schema2schema_attention_module = Attention(
                self.schema_attention_key_size, self.schema_attention_key_size,
                self.schema_attention_key_size)

        # utterance level attention
        if self.params.use_utterance_attention:
            self.utterance_attention_module = Attention(
                self.params.encoder_state_size, self.params.encoder_state_size,
                self.params.encoder_state_size)

        # Use attention module between input_hidden_states and schema_states
        # schema_states: self.schema_attention_key_size x len(schema)
        # input_hidden_states: self.utterance_attention_key_size x len(input)
        if params.use_encoder_attention:
            self.utterance2schema_attention_module = Attention(
                self.schema_attention_key_size,
                self.utterance_attention_key_size,
                self.utterance_attention_key_size)
            self.schema2utterance_attention_module = Attention(
                self.utterance_attention_key_size,
                self.schema_attention_key_size, self.schema_attention_key_size)

            new_attention_key_size = self.schema_attention_key_size + self.utterance_attention_key_size
            self.schema_attention_key_size = new_attention_key_size
            self.utterance_attention_key_size = new_attention_key_size

            # if self.params.use_schema_encoder_2:
            #     self.schema_encoder_2 = Encoder(schema_encoder_num_layer, self.schema_attention_key_size, self.schema_attention_key_size)
            #     self.utterance_encoder_2 = Encoder(params.encoder_num_layers, self.utterance_attention_key_size, self.utterance_attention_key_size)

        self.token_predictor = construct_token_predictor(
            params, output_vocabulary, self.utterance_attention_key_size,
            self.schema_attention_key_size, self.final_snippet_size, anonymizer)

        # Use schema_attention in decoder
        if params.use_schema_attention and params.use_query_attention:
            decoder_input_size = params.output_embedding_size + self.utterance_attention_key_size + self.schema_attention_key_size + params.encoder_state_size
        elif params.use_schema_attention:
            decoder_input_size = params.output_embedding_size + self.utterance_attention_key_size + self.schema_attention_key_size
        else:
            decoder_input_size = params.output_embedding_size + self.utterance_attention_key_size

        self.decoder = SequencePredictorWithSchema(
            params, decoder_input_size, self.output_embedder,
            self.column_name_token_embedder, self.token_predictor)

        if params.gnn_layer_number:
            self.gnn_history = paddle.nn.LayerList(
                [GraphNN(params) for _ in range(2 * params.gnn_layer_number)])
            self.gnn = paddle.nn.LayerList(
                [GraphNN(params) for _ in range(params.gnn_layer_number)])

        # fix_utter_h_temp = self.create_parameter([300],dtype='float32',default_initializer=paddle.nn.initializer.Uniform(low=-0.1, high=0.1))
        # self.add_parameter("fix_utter_h_temp", fix_utter_h_temp)

        # fix_utter_c_temp = self.create_parameter([300],dtype='float32',default_initializer=paddle.nn.initializer.Uniform(low=-0.1, high=0.1))
        # self.add_parameter("fix_utter_c_temp", fix_utter_c_temp)

        # if self.params.use_previous_query:
        #     start_query_attention_vector = self.create_parameter([3,params.encoder_state_size],dtype='float32',default_initializer=paddle.nn.initializer.XavierUniform())
        #     self.add_parameter("start_query_attention_vector", start_query_attention_vector)

        self.fix_utter_h_temp = torch_utils.add_params(([300]))
        self.fix_utter_c_temp = torch_utils.add_params(([300]))
        if self.params.fix_use_previous_query:
            if self.params.use_previous_query:
                self.start_query_attention_vector = torch_utils.add_params(
                    (3, params.encoder_state_size))

        # self.utterance_linear=paddle.nn.Linear(768,300)

    def predict_turn(self,
                     utterance_final_state,
                     input_hidden_states,
                     schema_states,
                     max_generation_length,
                     gold_query=None,
                     snippets=None,
                     input_sequence=None,
                     previous_queries=None,
                     previous_query_states=None,
                     input_schema=None,
                     feed_gold_tokens=False,
                     training=False):
        """ Gets a prediction for a single turn -- calls decoder and updates loss, etc.

        TODO:  this can probably be split into two methods, one that just predicts
            and another that computes the loss.
        """
        predicted_sequence = []
        fed_sequence = []
        loss = None
        token_accuracy = 0.
        # 经过
        if self.params.use_encoder_attention:
            schema_attention = self.utterance2schema_attention_module(
                paddle.stack(
                    schema_states, axis=0),
                input_hidden_states).vector  # input_value_size x len(schema)
            utterance_attention = self.schema2utterance_attention_module(
                paddle.stack(
                    input_hidden_states, axis=0),
                schema_states).vector  # schema_value_size x len(input)

            if schema_attention.dim() == 1:
                schema_attention = schema_attention.unsqueeze(1)
            if utterance_attention.dim() == 1:
                utterance_attention = utterance_attention.unsqueeze(1)

            new_schema_states = paddle.concat(
                [paddle.stack(
                    schema_states, axis=1), schema_attention],
                axis=0)  # (input_value_size+schema_value_size) x len(schema)
            schema_states = list(
                paddle.split(
                    new_schema_states,
                    num_or_sections=new_schema_states.shape[1],
                    axis=1))
            schema_states = [
                schema_state.squeeze() for schema_state in schema_states
            ]

            new_input_hidden_states = paddle.concat(
                [
                    paddle.stack(
                        input_hidden_states, axis=1), utterance_attention
                ],
                axis=0)  # (input_value_size+schema_value_size) x len(input)
            input_hidden_states = list(
                paddle.split(
                    new_input_hidden_states,
                    num_or_sections=new_input_hidden_states.shape[1],
                    axis=1))
            input_hidden_states = [
                input_hidden_state.squeeze()
                for input_hidden_state in input_hidden_states
            ]

            # bi-lstm over schema_states and input_hidden_states (embedder is an identify function)
            # if self.params.use_schema_encoder_2:
            #     final_schema_state, schema_states = self.schema_encoder_2(schema_states, lambda x: x, dropout_amount=self.dropout)
            #     final_utterance_state, input_hidden_states = self.utterance_encoder_2(input_hidden_states, lambda x: x, dropout_amount=self.dropout)

        if feed_gold_tokens:
            decoder_results = self.decoder(
                utterance_final_state,
                input_hidden_states,
                schema_states,
                max_generation_length,
                gold_sequence=gold_query,
                input_sequence=input_sequence,
                previous_queries=previous_queries,
                previous_query_states=previous_query_states,
                input_schema=input_schema,
                snippets=snippets,
                dropout_amount=self.dropout)

            all_scores = []
            all_alignments = []
            for prediction in decoder_results.predictions:
                scores = F.softmax(prediction.scores, axis=0)
                alignments = prediction.aligned_tokens
                if self.params.use_previous_query and self.params.use_copy_switch and len(
                        previous_queries) > 0:
                    query_scores = F.softmax(prediction.query_scores, axis=0)
                    copy_switch = prediction.copy_switch
                    scores = paddle.concat(
                        [
                            scores * (1 - copy_switch),
                            query_scores * copy_switch
                        ],
                        axis=0)
                    alignments = alignments + prediction.query_tokens

                all_scores.append(scores)
                all_alignments.append(alignments)

            # Compute the loss
            gold_sequence = gold_query

            loss = torch_utils.compute_loss(gold_sequence, all_scores,
                                            all_alignments, get_token_indices)
            if not training:
                predicted_sequence = torch_utils.get_seq_from_scores(
                    all_scores, all_alignments)
                token_accuracy = torch_utils.per_token_accuracy(
                    gold_sequence, predicted_sequence)
            fed_sequence = gold_sequence
        else:
            decoder_results = self.decoder(
                utterance_final_state,
                input_hidden_states,
                schema_states,
                max_generation_length,
                input_sequence=input_sequence,
                previous_queries=previous_queries,
                previous_query_states=previous_query_states,
                input_schema=input_schema,
                snippets=snippets,
                dropout_amount=self.dropout)
            predicted_sequence = decoder_results.sequence
            fed_sequence = predicted_sequence

        decoder_states = [
            pred.decoder_state for pred in decoder_results.predictions
        ]

        # fed_sequence contains EOS, which we don't need when encoding snippets.
        # also ignore the first state, as it contains the BEG encoding.

        for token, state in zip(fed_sequence[:-1], decoder_states[1:]):
            if snippet_handler.is_snippet(token):
                snippet_length = 0
                for snippet in snippets:
                    if snippet.name == token:
                        snippet_length = len(snippet.sequence)
                        break
                assert snippet_length > 0
                decoder_states.extend([state for _ in range(snippet_length)])
            else:
                decoder_states.append(state)

        return (predicted_sequence, loss, token_accuracy, decoder_states,
                decoder_results)

    def encode_schema_bow_simple(self, input_schema):
        schema_states = []
        for column_name in input_schema.column_names_embedder_input:
            schema_states.append(
                input_schema.column_name_embedder_bow(
                    column_name,
                    surface_form=False,
                    column_name_token_embedder=self.column_name_token_embedder))
        input_schema.set_column_name_embeddings(schema_states)
        return schema_states

    def encode_schema_self_attention(self, schema_states):
        schema_self_attention = self.schema2schema_attention_module(
            paddle.stack(
                schema_states, axis=0), schema_states).vector
        if schema_self_attention.dim() == 1:
            schema_self_attention = schema_self_attention.unsqueeze(1)
        residual_schema_states = list(
            paddle.split(
                schema_self_attention,
                num_or_sections=schema_self_attention.shape[1],
                axis=1))
        residual_schema_states = [
            schema_state.squeeze() for schema_state in residual_schema_states
        ]

        new_schema_states = [
            schema_state + residual_schema_state
            for schema_state, residual_schema_state in zip(
                schema_states, residual_schema_states)
        ]

        return new_schema_states

    def encode_schema(self, input_schema, dropout=False):
        schema_states = []
        for column_name_embedder_input in input_schema.column_names_embedder_input:
            tokens = column_name_embedder_input.split()

            # if dropout:
            #   schema_states_one, final_schema_state_one = self.schema_encoder(tokens, self.column_name_token_embedder, dropout_amount=self.dropout)
            # else:
            #   schema_states_one, final_schema_state_one = self.schema_encoder(tokens, self.column_name_token_embedder)

            # self.schema_encoder.dropout = self.dropout
            schema_states_one, final_schema_state_one = self.schema_encoder(
                paddle.stack(tokens).unsqueeze(0))
            schema_states_one, final_schema_state_one = torch_utils.LSTM_output_transfer(
                schema_states_one, final_schema_state_one)

            # final_schema_state_one: 1 means hidden_states instead of cell_memories, -1 means last layer
            schema_states.append(final_schema_state_one[0])

        input_schema.set_column_name_embeddings(schema_states)

        # self-attention over schema_states
        if self.params.use_schema_self_attention:
            schema_states = self.encode_schema_self_attention(schema_states)

        return schema_states

    def get_ernie_encoding(self, input_sequence, input_schema, discourse_state,
                           dropout):
        utterance_states, schema_token_states = utils_ernie.get_ernie_encoding(
            self.ernie_config,
            self.model_ernie,
            self.tokenizer,
            input_sequence,
            input_schema,
            ernie_input_version=self.params.bert_input_version,
            num_out_layers_n=1,
            num_out_layers_h=1)

        if self.params.discourse_level_lstm:
            utterance_token_embedder = lambda x: paddle.concat([x, discourse_state], axis=0)
            for idx in range(len(utterance_states)):
                # print(utterance_states[idx].shape, discourse_state.shape)
                utterance_states[idx] = utterance_token_embedder(
                    utterance_states[idx])
        # else:
        #     utterance_token_embedder = lambda x: x

        # if dropout:
        #     utterance_states, final_utterance_state = self.utterance_encoder(
        #         utterance_states,
        #         utterance_token_embedder,
        #         dropout_amount=self.dropout)
        # else:
        # self.utterance_encoder.dropout = self.dropout
        utterance_states, final_utterance_state = self.utterance_encoder(
            paddle.stack(utterance_states).unsqueeze(0))
        utterance_states, final_utterance_state = torch_utils.LSTM_output_transfer(
            utterance_states, final_utterance_state)

        # utterance_states = paddle.stack(utterance_states)
        # utterance_states = self.utterance_linear(utterance_states)
        # final_utterance_state=(paddle.sum(utterance_states,axis=0),paddle.sum(utterance_states,axis=0))
        # utterance_states = paddle.split(utterance_states,utterance_states.shape[0])
        # for idx in range(len(utterance_states)):
        #     utterance_states[idx]=utterance_states[idx].squeeze(0)

        schema_states = []
        for schema_token_states1 in schema_token_states:
            # if dropout:
            #     schema_states_one, final_schema_state_one = self.schema_encoder(schema_token_states1, lambda x: x, dropout_amount=self.dropout)
            # else:
            #     schema_states_one, final_schema_state_one = self.schema_encoder(schema_token_states1, lambda x: x)

            # self.schema_encoder.dropout = self.dropout
            schema_states_one, final_schema_state_one = self.schema_encoder(
                paddle.stack(schema_token_states1).unsqueeze(0))
            schema_states_one, final_schema_state_one = torch_utils.LSTM_output_transfer(
                schema_states_one, final_schema_state_one)

            # final_schema_state_one: 1 means hidden_states instead of cell_memories, -1 means last layer
            #schema_states.append(final_schema_state_one[1][-1])
            schema_states.append(
                sum(schema_states_one) / len(schema_states_one))

        input_schema.set_column_name_embeddings(schema_states)

        # self-attention over schema_states
        if self.params.use_schema_self_attention:
            schema_states = self.encode_schema_self_attention(schema_states)

        return final_utterance_state, utterance_states, schema_states

    def get_query_token_embedding(self, output_token, input_schema):
        if input_schema:
            if not (self.output_embedder.in_vocabulary(output_token) or
                    input_schema.in_vocabulary(
                        output_token, surface_form=True)):
                output_token = 'value'
            if self.output_embedder.in_vocabulary(output_token):
                output_token_embedding = self.output_embedder(output_token)
            else:
                output_token_embedding = input_schema.column_name_embedder(
                    output_token, surface_form=True)
        else:
            output_token_embedding = self.output_embedder(output_token)
        return output_token_embedding

    def get_utterance_attention(self, final_utterance_states_c,
                                final_utterance_states_h, final_utterance_state,
                                num_utterances_to_keep):
        if self.params.fix_use_utterance_attention:
            if len(final_utterance_states_c) == 0:
                final_utterance_states_c.append(self.fix_utter_c_temp)
            if len(final_utterance_states_h) == 0:
                final_utterance_states_h.append(self.fix_utter_h_temp)

        # self-attention between utterance_states            
        final_utterance_states_h.append(final_utterance_state[0])
        final_utterance_states_c.append(final_utterance_state[1])
        if self.params.fix_use_utterance_attention:
            final_utterance_states_c = [
                final_utterance_states_c[0]
            ] + final_utterance_states_c[-num_utterances_to_keep:]
            final_utterance_states_h = [
                final_utterance_states_h[0]
            ] + final_utterance_states_h[-num_utterances_to_keep:]
        else:
            final_utterance_states_c = final_utterance_states_c[
                -num_utterances_to_keep:]
            final_utterance_states_h = final_utterance_states_h[
                -num_utterances_to_keep:]

        # final_utterance_states_c_temp=final_utterance_states_c
        # final_utterance_states_h_temp=final_utterance_states_h

        # temp1=final_utterance_state

        # print(f"weight:{paddle.max(self.utterance_attention_module.query_linear.weight)} {paddle.min(self.utterance_attention_module.query_linear.weight)} {paddle.mean(self.utterance_attention_module.query_linear.weight)}")
        # print(f"bias:{paddle.max(self.utterance_attention_module.query_linear.bias)} {paddle.min(self.utterance_attention_module.query_linear.bias)} {paddle.mean(self.utterance_attention_module.query_linear.bias)}")

        # attention_result, _ = attention(final_utterance_states_c[-1], final_utterance_states_c)

        attention_result = self.utterance_attention_module(
            final_utterance_states_c[-1], final_utterance_states_c)
        final_utterance_state_attention_c = final_utterance_states_c[
            -1] + attention_result.vector.squeeze()

        # temp2=final_utterance_state_attention_c 
        # attention_result, _ = attention(final_utterance_states_h[-1], final_utterance_states_h)

        attention_result = self.utterance_attention_module(
            final_utterance_states_h[-1], final_utterance_states_h)
        final_utterance_state_attention_h = final_utterance_states_h[
            -1] + attention_result.vector.squeeze()

        # temp3=final_utterance_state_attention_h

        final_utterance_state = (final_utterance_state_attention_h,
                                 final_utterance_state_attention_c)

        # is_nan=[paddle.cast(paddle.isnan(final_utterance_state[i][j]),'float32') for i in range(len(final_utterance_state)) for j in range(len(final_utterance_state[i]))]
        # if sum([paddle.sum(i).numpy() for i in is_nan])!=0:
        #     print(f"utterance_attention:final_utterance_state:{final_utterance_state}")

        return final_utterance_states_c, final_utterance_states_h, final_utterance_state

    def get_previous_queries(self, previous_queries, previous_query_states,
                             previous_query, input_schema):
        # print(f"num_queries_to_keep:{num_queries_to_keep}")
        # print(f"previous_query:{previous_query}")

        query_token_embedder = lambda query_token: self.get_query_token_embedding(query_token, input_schema)

        previous_query_embedding = []
        if self.params.fix_use_previous_query:
            if len(previous_query) > 0:
                for output_token in previous_query:
                    previous_query_embedding.append(
                        query_token_embedder(output_token))
                previous_query_embedding = paddle.stack(
                    previous_query_embedding)
            else:
                previous_query = ['_UNK', '_UNK', '_UNK']
                previous_query_embedding = self.start_query_attention_vector
        else:
            for output_token in previous_query:
                previous_query_embedding.append(
                    query_token_embedder(output_token))
            previous_query_embedding = paddle.stack(previous_query_embedding)
        # print("previous_query_states_temp:")
        # print(previous_query_embedding)

        previous_queries.append(previous_query)
        num_queries_to_keep = min(self.params.maximum_queries,
                                  len(previous_queries))
        previous_queries = previous_queries[-num_queries_to_keep:]

        # self.query_encoder.dropout=self.dropout
        previous_outputs, _ = self.query_encoder(
            previous_query_embedding.unsqueeze(0))
        previous_outputs, _ = torch_utils.LSTM_output_transfer(previous_outputs,
                                                               _)
        # print(self.query_encoder._all_weights)

        # previous_outputs=previous_query_embedding

        assert len(previous_outputs) == len(previous_query)
        previous_query_states.append(previous_outputs)
        previous_query_states = previous_query_states[-num_queries_to_keep:]

        # is_nan=[paddle.cast(paddle.isnan(previous_query_states[i][j]),'float32') for i in range(len(previous_query_states)) for j in range(len(previous_query_states[i]))]
        # if sum([paddle.sum(i).numpy() for i in is_nan])!=0:
        #     print(f"previous_query_states:{previous_query_states}")

        return previous_queries, previous_query_states

    def get_adj_matrix(self, inner, foreign_keys, num_col):
        ret = np.eye(num_col)
        all_keys = inner + foreign_keys
        for ele in all_keys:
            ret[ele[0]][ele[1]] = 1
            ret[ele[1]][ele[0]] = 1
        return ret

    def get_adj_utterance_matrix(self, inner, foreign_keys, num_col):
        ret = np.eye(2 * num_col)
        #ret = np.pad(ret, ((num_col, 0), (num_col, 0)), 'constant', constant_values=(0, 0))
        all_keys = inner + foreign_keys
        for i in range(num_col):
            ret[i][num_col + i] = 1
            ret[num_col + i][i] = 1
        for ele in all_keys:

            # self graph connect
            ret[ele[0]][ele[1]] = 1
            ret[ele[1]][ele[0]] = 1
            ret[num_col + ele[0]][num_col + ele[1]] = 1
            ret[num_col + ele[1]][num_col + ele[0]] = 1

            ret[ele[0]][num_col + ele[1]] = 1  # useless?
            ret[num_col + ele[1]][ele[0]] = 1
            ret[num_col + ele[0]][ele[1]] = 1  # useless?
            ret[ele[1]][num_col + ele[0]] = 1
        ret = ret.dot(ret)
        return ret

    def train_step(self,
                   interaction,
                   max_generation_length,
                   snippet_alignment_probability=1.,
                   db2id=None,
                   id2db=None,
                   step=None):
        """ Trains the interaction-level model on a single interaction.

        Inputs:
            interaction (Interaction): The interaction to train on.
            learning_rate (float): Learning rate to use.
            snippet_keep_age (int): Age of oldest snippets to use.
            snippet_alignment_probability (float): The probability that a snippet will
                be used in constructing the gold sequence.
        """
        # assert self.params.discourse_level_lstm

        losses = []
        total_gold_tokens = 0

        input_hidden_states = []
        input_sequences = []

        final_utterance_states_c = []
        final_utterance_states_h = []

        previous_query_states = []
        previous_queries = []

        decoder_states = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states(
            )
        discourse_states = []

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []

        if input_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(input_schema)

        # Get the intra-turn graph and cross-turn graph
        inner = []
        for i, ele in enumerate(
                interaction.interaction.schema.column_names_surface_form):
            for j in range(i + 1,
                           len(interaction.interaction.schema.
                               column_names_surface_form)):
                if ele.split(
                        '.'
                )[0] == interaction.interaction.schema.column_names_surface_form[
                        j].split('.')[0]:
                    inner.append([i, j])
        adjacent_matrix = self.get_adj_matrix(
            inner, input_schema.table_schema['foreign_keys'],
            input_schema.num_col)
        adjacent_matrix_cross = self.get_adj_utterance_matrix(
            inner, input_schema.table_schema['foreign_keys'],
            input_schema.num_col)
        adjacent_matrix = paddle.to_tensor(adjacent_matrix)
        adjacent_matrix_cross = paddle.to_tensor(adjacent_matrix_cross)

        previous_schema_states = paddle.zeros(
            [input_schema.num_col, self.params.encoder_state_size])

        for utterance_index, utterance in enumerate(interaction.gold_utterances(
        )):

            # if len(utterance.interaction.utterances)<=1:
            #     continue

            # if utterance_index!=0 and not self.params.fix_use_utterance_attention:
            #     continue

            if interaction.identifier in LIMITED_INTERACTIONS and utterance_index > LIMITED_INTERACTIONS[
                    interaction.identifier]:
                break

            input_sequence = utterance.input_sequence()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            # Get the gold query: reconstruct if the alignment probability is less than one
            if snippet_alignment_probability < 1.:
                gold_query = sql_util.add_snippets_to_query(
                    available_snippets,
                    utterance.contained_entities(),
                    utterance.anonymized_gold_query(),
                    prob_align=snippet_alignment_probability
                ) + [vocab.EOS_TOK]
            else:
                gold_query = utterance.gold_query()
            # print(f"gold_query:{gold_query}")
            # Encode the utterance, and update the discourse-level states
            # if not self.params.use_bert:
            #     if self.params.discourse_level_lstm:
            #         utterance_token_embedder = lambda token: paddle.concat([self.input_embedder(token), discourse_state], axis=0)
            #     else:
            #         utterance_token_embedder = self.input_embedder
            #     final_utterance_state, utterance_states = self.utterance_encoder(
            #         input_sequence,
            #         utterance_token_embedder,
            #         dropout_amount=self.dropout)
            # else:

            final_utterance_state, utterance_states, schema_states = self.get_ernie_encoding(
                input_sequence, input_schema, discourse_state, dropout=True)

            # temp1=final_utterance_state

            schema_states = paddle.stack(schema_states, axis=0)
            for i in range(self.params.gnn_layer_number):
                schema_states = self.gnn_history[2 * i](schema_states,
                                                        adjacent_matrix_cross,
                                                        previous_schema_states)
                schema_states = self.gnn_history[2 * i + 1](
                    schema_states, adjacent_matrix_cross,
                    previous_schema_states)
                schema_states = self.gnn[i](schema_states, adjacent_matrix)
            previous_schema_states = schema_states
            #schema_states = schema_states_ori + schema_states
            schema_states_ls = paddle.split(
                schema_states, schema_states.shape[0], axis=0)
            schema_states = [ele.squeeze(0) for ele in schema_states_ls]

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            num_utterances_to_keep = min(self.params.maximum_utterances,
                                         len(input_sequences))

            # final_utterance_state[1][0] is the first layer's hidden states at the last time step (concat forward lstm and backward lstm)
            # 经过   
            # print("*"*100)
            # print(f"utterance_index:{utterance_index}")
            # print(f"discourse_state:{discourse_state}")
            # print(f"final_utterance_state[1][0]:{final_utterance_state[1][0]}")
            # print("-"*100)
            if self.params.discourse_level_lstm:
                # _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states, self.dropout)
                discourse_state, discourse_lstm_states = self.discourse_lstms(
                    final_utterance_state[0].unsqueeze(0),
                    discourse_lstm_states)
                discourse_state = discourse_state.squeeze()

                # is_nan=paddle.cast(paddle.isnan(discourse_state),'float32')
                # if sum(paddle.sum(is_nan).numpy())!=0:
                #     print(f"discourse_state:{discourse_state}")
            # if sum(list(paddle.cast(paddle.isnan(discourse_state),'float32')))!=0:
            # print(f"utterance_index:{utterance_index}")
            # print(f"discourse_state:{discourse_state}")
            # print("*"*100)
            # 经过
            if self.params.use_utterance_attention:
                # if utterance_index==0:
                #     final_utterance_states_h.append(final_utterance_state[0][0])
                #     final_utterance_states_c.append(final_utterance_state[1][0])
                #     final_utterance_state=([final_utterance_state[0][0]+final_utterance_state[0][0]],[final_utterance_state[1][0]+final_utterance_state[1][0]])
                # else:
                final_utterance_states_c, final_utterance_states_h, final_utterance_state = self.get_utterance_attention(
                    final_utterance_states_c, final_utterance_states_h,
                    final_utterance_state, num_utterances_to_keep)

            # temp2=final_utterance_state

            # 经过
            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(
                    input_hidden_states, input_sequences)
            # else:
            #     flat_sequence = []
            #     for utt in input_sequences[-num_utterances_to_keep:]:
            #         flat_sequence.extend(utt)

            snippets = None
            # if self.params.use_snippets:
            #     if self.params.previous_decoder_snippet_encoding:
            #         snippets = encode_snippets_with_states(available_snippets, decoder_states)
            #     else:
            #         snippets = self._encode_snippets(previous_query, available_snippets, input_schema)

            if self.params.use_previous_query:
                if self.params.fix_use_previous_query:
                    previous_queries, previous_query_states = self.get_previous_queries(
                        previous_queries, previous_query_states, previous_query,
                        input_schema)
                else:
                    if len(previous_query) > 0:
                        previous_queries, previous_query_states = self.get_previous_queries(
                            previous_queries, previous_query_states,
                            previous_query, input_schema)
            # is_nan=[paddle.cast(paddle.isnan(final_utterance_state[i][j]),'float32') for i in range(len(final_utterance_state)) for j in range(len(final_utterance_state[i]))]
            # if sum([paddle.sum(i).numpy() for i in is_nan])!=0:
            #     print(f"utterance_index:{utterance_index}")
            #     print(f"final_utterance_state:{final_utterance_state}")

            # 测试decoder
            # final_utterance_state = (paddle.to_tensor(np.random.random([300]),dtype='float32'),paddle.to_tensor(np.random.random([300]),dtype='float32'))

            # utterance_states = paddle.to_tensor(np.random.random([len(utterance_states),utterance_states[0].shape[0]]),dtype='float32')
            # utterance_states = paddle.split(utterance_states,utterance_states.shape[0])
            # for idx in range(len(utterance_states)):
            #     utterance_states[idx]=utterance_states[idx].squeeze(0)

            # schema_states = paddle.to_tensor(np.random.random([len(schema_states),schema_states[0].shape[0]]),dtype='float32')
            # schema_states = paddle.split(schema_states,schema_states.shape[0])
            # for idx in range(len(schema_states)):
            #     schema_states[idx]=schema_states[idx].squeeze(0)

            if len(gold_query) <= max_generation_length and len(
                    previous_query) <= max_generation_length:
                prediction = self.predict_turn(
                    final_utterance_state,
                    utterance_states,
                    schema_states,
                    max_generation_length,
                    gold_query=gold_query,
                    snippets=snippets,
                    input_sequence=flat_sequence,
                    previous_queries=previous_queries,
                    previous_query_states=previous_query_states,
                    input_schema=input_schema,
                    feed_gold_tokens=True,
                    training=True)
                loss = prediction[1]
                decoder_states = prediction[3]
                total_gold_tokens += len(gold_query)
                losses.append(loss)
            else:
                # Break if previous decoder snippet encoding -- because the previous
                # sequence was too long to run the decoder.
                if self.params.previous_decoder_snippet_encoding:
                    break
                continue

            #torch.cuda.empty_cache()

        if losses:
            average_loss = paddle.sum(paddle.stack(losses)) / total_gold_tokens
            #average_loss = torch.sum(torch.stack(losses))
            # average_loss.item(),
            print(f"total_gold_tokens:{total_gold_tokens}, step:{step}")
            print(f"LOSS:{float(average_loss.numpy())}")
            if paddle.sum(paddle.cast(paddle.isinf(average_loss),
                                      'int32')) == paddle.ones([1]):
                self.save("./inf_checkpoint")

            # Renormalize so the effect is normalized by the batch size.
            normalized_loss = average_loss
            if self.params.reweight_batch:
                normalized_loss = len(losses) * average_loss / float(
                    self.params.batch_size)
                #normalized_loss = average_loss / (len(losses) * float(self.params.batch_size))

            normalized_loss.backward()

            # print("output_embedder:")
            # output_embedder_grad=self.output_embedder.token_embedding_matrix.weight.grad
            # print(f"max:{np.max(output_embedder_grad)} min:{np.min(output_embedder_grad)} mean:{np.mean(output_embedder_grad)}")
            # print("query_encoder:")
            # query_encoder_hh_grad=self.query_encoder.weight_hh_l0.grad
            # query_encoder_hh_reverse_grad=self.query_encoder.weight_hh_l0_reverse.grad
            # query_encoder_ih_grad=self.query_encoder.weight_ih_l0.grad
            # query_encoder_ih_reverse_grad=self.query_encoder.weight_ih_l0_reverse.grad
            # print(f"max:{np.max(query_encoder_hh_grad)} {np.max(query_encoder_hh_reverse_grad)} {np.max(query_encoder_ih_grad)} {np.max(query_encoder_ih_reverse_grad)}")
            # print(f"min:{np.min(query_encoder_hh_grad)} {np.min(query_encoder_hh_reverse_grad)} {np.min(query_encoder_ih_grad)} {np.min(query_encoder_ih_reverse_grad)}")
            # print(f"mean:{np.mean(query_encoder_hh_grad)} {np.mean(query_encoder_hh_reverse_grad)} {np.mean(query_encoder_ih_grad)} {np.mean(query_encoder_ih_reverse_grad)}")
            # print("utterance_attention_module:query_linear:query_weights")
            # query_linear_grad=self.utterance_attention_module.query_linear.weight.grad
            # print(f"max:{np.max(query_linear_grad)} min:{np.min(query_linear_grad)} mean:{np.mean(query_linear_grad)}")
            # print(f"final_utterance_states_c[0]:")
            # print(f"grad: max:{np.max(final_utterance_states_c[0].grad)} min:{np.min(final_utterance_states_c[0].grad)} mean:{np.mean(final_utterance_states_c[0].grad)}")
            # print(f"max:{paddle.max(final_utterance_states_c[0])} min:{paddle.min(final_utterance_states_c[0])} mean:{paddle.mean(final_utterance_states_c[0])}")

            # torch.nn.utils.clip_grad_norm_(self.parameters(), self.params.clip)

            if step <= self.params.warmup_step:
                self.set_learning_rate(step / self.params.warmup_step *
                                       self.params.initial_learning_rate)
            step += 1

            #model_dict = {name: param for name, param in self.named_parameters()}
            #print('decoder.lstmCell.weight_ih: %.8f'%model_dict['decoder.lstmCell.weight_ih'].sum().numpy())
            #print('model_ernie.embeddings.word_embeddings.weight.grad: %.8f'%model_dict['model_ernie.embeddings.word_embeddings.weight'].grad.sum().numpy())
            #print('model_ernie.embeddings.word_embeddings.weight: %.8f'% model_dict['model_ernie.embeddings.word_embeddings.weight'].sum().numpy())

            self.trainer.step()
            if self.params.fine_tune_bert:
                self.bert_trainer.step()
                self.bert_trainer.clear_grad()
            self.trainer.clear_grad()
            #model_dict = {name: param for name, param in self.named_parameters()}
            #print('decoder.lstmCell.weight_ih: %.8f'%model_dict['decoder.lstmCell.weight_ih'].sum().numpy())
            #print('model_ernie.embeddings.word_embeddings.weight.grad: %.8f'%model_dict['model_ernie.embeddings.word_embeddings.weight'].grad.sum().numpy())
            #print('model_ernie.embeddings.word_embeddings.weight: %.8f'% model_dict['model_ernie.embeddings.word_embeddings.weight'].sum().numpy())

            loss_scalar = float(normalized_loss.numpy())
            #assert torch.isnan(normalized_loss).item() == 0
            isNan = sum(
                paddle.cast(paddle.isnan(normalized_loss), 'float32').numpy()
                .tolist()) == 0
            if paddle.isnan(normalized_loss):
                print("nan error but keep running")
                # print(f"utterance_index:{utterance_index}")
                # print(f"original_gold_query:{utterance.interaction.utterances[0].original_gold_query}")
                # print(f"loss:{loss}")
                # print(f"schema_states:{schema_states}")
                # print(f"utterance_states:{utterance_states}")
                # print(f"decoder_states:{decoder_states}")
                # print(f"final_utterance_state:{final_utterance_state}")
                # print(f"previous_query_states:{previous_query_states}")
                # print(f"previous_queries:{previous_queries}")
                # print(f"flat_sequence:{flat_sequence}")
                # print(f"input_schema:{input_schema}")
                # print(f"input_sequence:{input_sequence}")
                # print(f"discourse_state:{discourse_state}")
            assert isNan

        else:
            loss_scalar = 0.

        return loss_scalar, step

    def predict_with_predicted_queries(self,
                                       interaction,
                                       max_generation_length,
                                       syntax_restrict=True):
        """ Predicts an interaction, using the predicted queries to get snippets."""
        # assert self.params.discourse_level_lstm

        syntax_restrict = False

        predictions = []

        input_hidden_states = []
        input_sequences = []

        final_utterance_states_c = []
        final_utterance_states_h = []

        previous_query_states = []
        previous_queries = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states(
            )
        discourse_states = []

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []

        # Get the intra-turn graph and cross-turn graph
        inner = []
        for i, ele in enumerate(
                interaction.interaction.schema.column_names_surface_form):
            for j in range(i + 1,
                           len(interaction.interaction.schema.
                               column_names_surface_form)):
                if ele.split(
                        '.'
                )[0] == interaction.interaction.schema.column_names_surface_form[
                        j].split('.')[0]:
                    inner.append([i, j])
        adjacent_matrix = self.get_adj_matrix(
            inner, input_schema.table_schema['foreign_keys'],
            input_schema.num_col)
        adjacent_matrix_cross = self.get_adj_utterance_matrix(
            inner, input_schema.table_schema['foreign_keys'],
            input_schema.num_col)
        # adjacent_matrix = torch.Tensor(adjacent_matrix)
        # adjacent_matrix_cross = torch.Tensor(adjacent_matrix_cross)
        # previous_schema_states = torch.zeros(input_schema.num_col, self.params.encoder_state_size)

        adjacent_matrix = paddle.to_tensor(adjacent_matrix)
        adjacent_matrix_cross = paddle.to_tensor(adjacent_matrix_cross)
        previous_schema_states = paddle.zeros(
            [input_schema.num_col, self.params.encoder_state_size])

        if input_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(input_schema)

        interaction.start_interaction()

        while not interaction.done():
            utterance = interaction.next_utterance()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            input_sequence = utterance.input_sequence()

            if not self.params.use_bert:
                if self.params.discourse_level_lstm:
                    utterance_token_embedder = lambda token: paddle.concat([self.input_embedder(token), discourse_state], axis=0)
                else:
                    utterance_token_embedder = self.input_embedder
                final_utterance_state, utterance_states = self.utterance_encoder(
                    input_sequence, utterance_token_embedder)
            else:
                final_utterance_state, utterance_states, schema_states = self.get_bert_encoding(
                    input_sequence,
                    input_schema,
                    discourse_state,
                    dropout=False)

            schema_states = paddle.stack(schema_states, axis=0)
            for i in range(self.params.gnn_layer_number):
                schema_states = self.gnn_history[2 * i](schema_states,
                                                        adjacent_matrix_cross,
                                                        previous_schema_states)
                schema_states = self.gnn_history[2 * i + 1](
                    schema_states, adjacent_matrix_cross,
                    previous_schema_states)
                schema_states = self.gnn[i](schema_states, adjacent_matrix)
            previous_schema_states = schema_states
            #schema_states = schema_states_ori + schema_states
            schema_states_ls = paddle.split(
                schema_states, schema_states.shape[0], axis=0)
            schema_states = [ele.squeeze(0) for ele in schema_states_ls]

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            num_utterances_to_keep = min(self.params.maximum_utterances,
                                         len(input_sequences))

            if self.params.discourse_level_lstm:
                # _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states)
                discourse_state, discourse_lstm_states = self.discourse_lstms(
                    final_utterance_state[0].unsqueeze(0),
                    discourse_lstm_states)

            if self.params.use_utterance_attention:
                final_utterance_states_c, final_utterance_states_h, final_utterance_state = self.get_utterance_attention(
                    final_utterance_states_c, final_utterance_states_h,
                    final_utterance_state, num_utterances_to_keep)

            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(
                    input_hidden_states, input_sequences)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)

            snippets = None
            if self.params.use_snippets:
                snippets = self._encode_snippets(
                    previous_query, available_snippets, input_schema)

            if self.params.use_previous_query and len(previous_query) > 0:
                previous_queries, previous_query_states = self.get_previous_queries(
                    previous_queries, previous_query_states, previous_query,
                    input_schema)

            results = self.predict_turn(
                final_utterance_state,
                utterance_states,
                schema_states,
                max_generation_length,
                input_sequence=flat_sequence,
                previous_queries=previous_queries,
                previous_query_states=previous_query_states,
                input_schema=input_schema,
                snippets=snippets)

            predicted_sequence = results[0]
            predictions.append(results)

            # Update things necessary for using predicted queries
            anonymized_sequence = utterance.remove_snippets(predicted_sequence)
            if EOS_TOK in anonymized_sequence:
                anonymized_sequence = anonymized_sequence[:-1]  # Remove _EOS
            else:
                anonymized_sequence = ['select', '*', 'from', 't1']

            if not syntax_restrict:
                utterance.set_predicted_query(
                    interaction.remove_snippets(predicted_sequence))
                if input_schema:
                    # on SParC
                    interaction.add_utterance(
                        utterance,
                        anonymized_sequence,
                        previous_snippets=utterance.snippets(),
                        simple=True)
                else:
                    # on ATIS
                    interaction.add_utterance(
                        utterance,
                        anonymized_sequence,
                        previous_snippets=utterance.snippets(),
                        simple=False)
            else:
                utterance.set_predicted_query(utterance.previous_query())
                interaction.add_utterance(
                    utterance,
                    utterance.previous_query(),
                    previous_snippets=utterance.snippets())

        return predictions

    def predict_with_gold_queries(self,
                                  interaction,
                                  max_generation_length,
                                  feed_gold_query=False):
        """ Predicts SQL queries for an interaction.

        Inputs:
            interaction (Interaction): Interaction to predict for.
            feed_gold_query (bool): Whether or not to feed the gold token to the
                generation step.
        """
        # assert self.params.discourse_level_lstm

        predictions = []

        input_hidden_states = []
        input_sequences = []

        final_utterance_states_c = []
        final_utterance_states_h = []

        previous_query_states = []
        previous_queries = []

        decoder_states = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states(
            )
        discourse_states = []

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []
        if input_schema and not self.params.use_bert:
            schema_states = self.encode_schema_bow_simple(input_schema)

        # Get the intra-turn graph and cross-turn graph
        inner = []
        for i, ele in enumerate(
                interaction.interaction.schema.column_names_surface_form):
            for j in range(i + 1,
                           len(interaction.interaction.schema.
                               column_names_surface_form)):
                if ele.split(
                        '.'
                )[0] == interaction.interaction.schema.column_names_surface_form[
                        j].split('.')[0]:
                    inner.append([i, j])
        adjacent_matrix = self.get_adj_matrix(
            inner, input_schema.table_schema['foreign_keys'],
            input_schema.num_col)
        adjacent_matrix_cross = self.get_adj_utterance_matrix(
            inner, input_schema.table_schema['foreign_keys'],
            input_schema.num_col)
        # adjacent_matrix = torch.Tensor(adjacent_matrix)
        # adjacent_matrix_cross = torch.Tensor(adjacent_matrix_cross)
        # previous_schema_states = torch.zeros(input_schema.num_col, self.params.encoder_state_size)

        adjacent_matrix = paddle.to_tensor(adjacent_matrix)
        adjacent_matrix_cross = paddle.to_tensor(adjacent_matrix_cross)
        previous_schema_states = paddle.zeros(
            [input_schema.num_col, self.params.encoder_state_size])

        for utterance_index, utterance in enumerate(interaction.gold_utterances(
        )):

            # if len(utterance.interaction.utterances)!=1:
            #     continue

            input_sequence = utterance.input_sequence()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            # Encode the utterance, and update the discourse-level states
            # if not self.params.use_bert:
            #     if self.params.discourse_level_lstm:
            #         utterance_token_embedder = lambda token: paddle.concat([self.input_embedder(token), discourse_state], axis=0)
            #     else:
            #         utterance_token_embedder = self.input_embedder
            #     final_utterance_state, utterance_states = self.utterance_encoder(
            #         input_sequence,
            #         utterance_token_embedder,
            #         dropout_amount=self.dropout)
            # else:
            final_utterance_state, utterance_states, schema_states = self.get_ernie_encoding(
                input_sequence, input_schema, discourse_state, dropout=True)

            schema_states = paddle.stack(schema_states, axis=0)
            for i in range(self.params.gnn_layer_number):
                schema_states = self.gnn_history[2 * i](schema_states,
                                                        adjacent_matrix_cross,
                                                        previous_schema_states)
                schema_states = self.gnn_history[2 * i + 1](
                    schema_states, adjacent_matrix_cross,
                    previous_schema_states)
                schema_states = self.gnn[i](schema_states, adjacent_matrix)
            previous_schema_states = schema_states
            #schema_states = schema_states_ori + schema_states
            schema_states_ls = paddle.split(
                schema_states, schema_states.shape[0], axis=0)
            schema_states = [ele.squeeze(0) for ele in schema_states_ls]

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            num_utterances_to_keep = min(self.params.maximum_utterances,
                                         len(input_sequences))

            if self.params.discourse_level_lstm:
                # _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states, self.dropout)
                discourse_state, discourse_lstm_states = self.discourse_lstms(
                    final_utterance_state[0].unsqueeze(0),
                    discourse_lstm_states)
                discourse_state = discourse_state.squeeze()

            if self.params.use_utterance_attention:
                final_utterance_states_c, final_utterance_states_h, final_utterance_state = self.get_utterance_attention(
                    final_utterance_states_c, final_utterance_states_h,
                    final_utterance_state, num_utterances_to_keep)

            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(
                    input_hidden_states, input_sequences)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)

            snippets = None
            # if self.params.use_snippets:
            #     if self.params.previous_decoder_snippet_encoding:
            #         snippets = encode_snippets_with_states(available_snippets, decoder_states)
            #     else:
            #         snippets = self._encode_snippets(previous_query, available_snippets, input_schema)

            if self.params.use_previous_query and len(previous_query) > 0:
                previous_queries, previous_query_states = self.get_previous_queries(
                    previous_queries, previous_query_states, previous_query,
                    input_schema)

            # flat_sequence.append(utterance_index)

            prediction = self.predict_turn(
                final_utterance_state,
                utterance_states,
                schema_states,
                max_generation_length,
                gold_query=utterance.gold_query(),
                snippets=snippets,
                input_sequence=flat_sequence,
                previous_queries=previous_queries,
                previous_query_states=previous_query_states,
                input_schema=input_schema,
                feed_gold_tokens=feed_gold_query)

            decoder_states = prediction[3]
            predictions.append(prediction)

        return predictions
