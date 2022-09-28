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
# modified from https://github.com/naver/sqlova

import os, json
import random as rd
from copy import deepcopy

import paddle
import paddle.nn as nn

from paddlenlp.transformers import BertModel, BertPretrainedModel, BertTokenizer


def get_bert(params):
    model_bert = BertModel.from_pretrained("bert-base-uncased")
    bert_config = BertPretrainedModel.pretrained_init_configuration[
        "bert-base-uncased"]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return model_bert, tokenizer, bert_config


def generate_inputs(tokenizer, nlu1_tok, hds1):
    tokens = []
    segment_ids = []
    sent_ids = []

    t_to_tt_idx_hds1 = []

    tokens.append("[CLS]")
    i_st_nlu = len(tokens)  # to use it later

    segment_ids.append(0)
    sent_ids.append(0)
    for token in nlu1_tok:
        tokens.append(token)
        segment_ids.append(0)
        sent_ids.append(0)
    i_ed_nlu = len(tokens)
    tokens.append("[SEP]")
    segment_ids.append(0)
    sent_ids.append(0)

    i_hds = []
    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        t_to_tt_idx_hds11 = []
        sub_tok = []
        for sub_tok1 in hds11.split():
            t_to_tt_idx_hds11.append(len(sub_tok))
            sub_tok += tokenizer.tokenize(sub_tok1)
        t_to_tt_idx_hds1.append(t_to_tt_idx_hds11)
        tokens += sub_tok

        i_ed_hd = len(tokens)
        i_hds.append((i_st_hd, i_ed_hd))
        segment_ids += [1] * len(sub_tok)
        sent_ids += [1] * len(sub_tok)
        if i < len(hds1) - 1:
            tokens.append("[SEP]")
            segment_ids.append(0)
            sent_ids.append(1)
        elif i == len(hds1) - 1:
            tokens.append("[SEP]")
            segment_ids.append(1)
            sent_ids.append(1)
        else:
            raise EnvironmentError

    i_nlu = (i_st_nlu, i_ed_nlu)

    return tokens, sent_ids, segment_ids, i_nlu, i_hds, t_to_tt_idx_hds1


def gen_l_hpu(i_hds):
    """
    # Treat columns as if it is a batch of natural language utterance with batch-size = # of columns * # of batch_size
    i_hds = [(17, 18), (19, 21), (22, 23), (24, 25), (26, 29), (30, 34)])
    """
    l_hpu = []
    for i_hds1 in i_hds:
        for i_hds11 in i_hds1:
            l_hpu.append(i_hds11[1] - i_hds11[0])

    return l_hpu


def get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length):
    """
    Here, input is toknized further by WordPiece (WP) tokenizer and fed into BERT.
    """

    l_n = []
    l_hs = []  # The length of columns for each batch

    input_ids = []
    tokens = []
    segment_ids = []
    sent_ids = []
    input_mask = []

    i_nlu = []  # index to retreive the position of contextual vector later.
    i_hds = []

    doc_tokens = []
    nlu_tt = []

    t_to_tt_idx = []
    tt_to_t_idx = []

    t_to_tt_idx_hds = []

    for b, nlu_t1 in enumerate(nlu_t):
        hds1 = hds[b]
        l_hs.append(len(hds1))

        # 1. 2nd tokenization using WordPiece
        tt_to_t_idx1 = [
        ]  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
        t_to_tt_idx1 = [
        ]  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        nlu_tt1 = [
        ]  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        for (i, token) in enumerate(nlu_t1):
            t_to_tt_idx1.append(
                len(nlu_tt1)
            )  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tt_to_t_idx1.append(i)
                nlu_tt1.append(
                    sub_token
                )  # all_doc_tokens are further tokenized using WordPiece tokenizer
        nlu_tt.append(nlu_tt1)
        tt_to_t_idx.append(tt_to_t_idx1)
        t_to_tt_idx.append(t_to_tt_idx1)

        l_n.append(len(nlu_tt1))

        # [CLS] nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP]
        # 2. Generate BERT inputs & indices.
        tokens1, sent_ids1, segment_ids1, i_nlu1, i_hds1, t_to_tt_idx_hds1 = generate_inputs(
            tokenizer, nlu_tt1, hds1)

        assert len(t_to_tt_idx_hds1) == len(hds1)

        t_to_tt_idx_hds.append(t_to_tt_idx_hds1)

        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        # 3. Zero-pad up to the sequence length.
        if len(nlu_t) == 1:
            max_seq_length = len(input_ids1)
        while len(input_ids1) < max_seq_length:
            input_ids1.append(0)
            input_mask1.append(0)
            segment_ids1.append(0)
            sent_ids1.append(0)

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length
        assert len(sent_ids1) == max_seq_length

        input_ids.append(input_ids1)
        tokens.append(tokens1)
        segment_ids.append(segment_ids1)
        sent_ids.append(sent_ids1)

        input_mask.append(input_mask1)

        i_nlu.append(i_nlu1)
        i_hds.append(i_hds1)

    # Convert to tensor
    all_input_ids = paddle.to_tensor(input_ids, dtype="int64")
    all_input_mask = paddle.to_tensor(input_mask, dtype="int64")
    all_segment_ids = paddle.to_tensor(segment_ids, dtype="int64")
    all_sent_ids = paddle.to_tensor(sent_ids, dtype="int64")

    # 4. Generate BERT output.
    all_encoder_layer, pooled_output = model_bert(
        all_input_ids,
        token_type_ids=all_segment_ids,
        output_hidden_states=True)

    # 5. generate l_hpu from i_hds
    l_hpu = gen_l_hpu(i_hds)

    assert len(set(l_n)) == 1 and len(set(i_nlu)) == 1
    assert l_n[0] == i_nlu[0][1] - i_nlu[0][0]

    return all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, \
           l_n, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx, t_to_tt_idx_hds


def get_wemb_n(i_nlu, l_n, hS, num_hidden_layers, all_encoder_layer,
               num_out_layers_n):
    """
    Get the representation of each tokens.
    """
    bS = len(l_n)
    l_n_max = max(l_n)
    wemb_n = []
    for b in range(bS):
        # [B, max_len, dim]
        # Fill zero for non-exist part.
        l_n1 = l_n[b]
        i_nlu1 = i_nlu[b]
        for i_noln in range(num_out_layers_n):
            i_layer = num_hidden_layers - 1 - i_noln
            st = i_noln * hS
            ed = (i_noln + 1) * hS
            tmp = all_encoder_layer[i_layer][
                b, i_nlu1[0]:i_nlu1[1], :].unsqueeze(0)
            pad_right = l_n_max - (i_nlu1[1] - i_nlu1[0])
            pad_tmp = paddle.nn.functional.pad(tmp, [0, pad_right],
                                               data_format='NLC').squeeze(0)
            wemb_n.append(pad_tmp)
    wemb_n = paddle.stack(wemb_n)
    return wemb_n


def get_wemb_h(i_hds, l_hpu, l_hs, hS, num_hidden_layers, all_encoder_layer,
               num_out_layers_h):
    """
    As if
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]
    """
    bS = len(l_hs)
    l_hpu_max = max(l_hpu)
    num_of_all_hds = sum(l_hs)
    wemb_h = []
    b_pu = -1

    for b, i_hds1 in enumerate(i_hds):
        for b1, i_hds11 in enumerate(i_hds1):
            b_pu += 1
            for i_nolh in range(num_out_layers_h):
                i_layer = num_hidden_layers - 1 - i_nolh
                st = i_nolh * hS
                ed = (i_nolh + 1) * hS
                tmp = all_encoder_layer[i_layer][
                    b, i_hds11[0]:i_hds11[1], :].unsqueeze(0)
                pad_right = l_hpu_max - (i_hds11[1] - i_hds11[0])
                pad_tmp = paddle.nn.functional.pad(tmp, [0, pad_right],
                                                   data_format='NLC').squeeze(0)
                wemb_h.append(pad_tmp)
    wemb_h = paddle.stack(wemb_h)
    return wemb_h


def get_wemb_bert(bert_config,
                  model_bert,
                  tokenizer,
                  nlu_t,
                  hds,
                  max_seq_length,
                  num_out_layers_n=1,
                  num_out_layers_h=1):

    # get contextual output of all tokens from bert
    all_encoder_layer, pooled_output, tokens, i_nlu, i_hds,\
    l_n, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx, t_to_tt_idx_hds = get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length)
    # all_encoder_layer: BERT outputs from all layers.
    # pooled_output: output of [CLS] vec.
    # tokens: BERT intput tokens
    # i_nlu: start and end indices of question in tokens
    # i_hds: start and end indices of headers

    # get the wemb
    wemb_n = get_wemb_n(i_nlu, l_n, bert_config['hidden_size'],
                        bert_config['num_hidden_layers'], all_encoder_layer,
                        num_out_layers_n)

    wemb_h = get_wemb_h(i_hds, l_hpu, l_hs, bert_config['hidden_size'],
                        bert_config['num_hidden_layers'], all_encoder_layer,
                        num_out_layers_h)

    return wemb_n, wemb_h, l_n, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx, t_to_tt_idx_hds


def prepare_input(tokenizer, input_sequence, input_schema, max_seq_length):
    nlu_t = []
    hds = []

    nlu_t1 = input_sequence
    all_hds = input_schema.column_names_embedder_input

    nlu_tt1 = []
    for (i, token) in enumerate(nlu_t1):
        nlu_tt1 += tokenizer.tokenize(token)

    current_hds1 = []
    for hds1 in all_hds:
        new_hds1 = current_hds1 + [hds1]
        tokens1, _, segment_ids1, i_nlu1, i_hds1, t_to_tt_idx_hds1 = generate_inputs(
            tokenizer, nlu_tt1, new_hds1)
        if len(segment_ids1) > max_seq_length:
            nlu_t.append(nlu_t1)
            hds.append(current_hds1)
            current_hds1 = [hds1]
        else:
            current_hds1 = new_hds1

    if len(current_hds1) > 0:
        nlu_t.append(nlu_t1)
        hds.append(current_hds1)

    return nlu_t, hds


def prepare_input_v2(tokenizer, input_sequence, input_schema):
    nlu_t = []
    hds = []
    max_seq_length = 0

    nlu_t1 = input_sequence
    all_hds = input_schema.column_names_embedder_input

    nlu_tt1 = []
    for (i, token) in enumerate(nlu_t1):
        nlu_tt1 += tokenizer.tokenize(token)

    current_hds1 = []
    current_table = ''
    for hds1 in all_hds:
        hds1_table = hds1.split('.')[0].strip()
        if hds1_table == current_table:
            current_hds1.append(hds1)
        else:
            tokens1, segment_ids1, i_nlu1, i_hds1, t_to_tt_idx_hds1 = generate_inputs(
                tokenizer, nlu_tt1, current_hds1)
            max_seq_length = max(max_seq_length, len(segment_ids1))

            nlu_t.append(nlu_t1)
            hds.append(current_hds1)
            current_hds1 = [hds1]
            current_table = hds1_table

    if len(current_hds1) > 0:
        tokens1, segment_ids1, i_nlu1, i_hds1, t_to_tt_idx_hds1 = generate_inputs(
            tokenizer, nlu_tt1, current_hds1)
        max_seq_length = max(max_seq_length, len(segment_ids1))
        nlu_t.append(nlu_t1)
        hds.append(current_hds1)

    return nlu_t, hds, max_seq_length


def get_bert_encoding(bert_config,
                      model_bert,
                      tokenizer,
                      input_sequence,
                      input_schema,
                      bert_input_version='v1',
                      max_seq_length=512,
                      num_out_layers_n=1,
                      num_out_layers_h=1):
    if bert_input_version == 'v1':
        nlu_t, hds = prepare_input(tokenizer, input_sequence, input_schema,
                                   max_seq_length)
    elif bert_input_version == 'v2':
        nlu_t, hds, max_seq_length = prepare_input_v2(tokenizer, input_sequence,
                                                      input_schema)

    wemb_n, wemb_h, l_n, l_hpu, l_hs, nlu_tt, t_to_tt_idx, tt_to_t_idx, t_to_tt_idx_hds = get_wemb_bert(
        bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
        num_out_layers_n, num_out_layers_h)

    t_to_tt_idx = t_to_tt_idx[0]
    assert len(t_to_tt_idx) == len(input_sequence)
    assert sum(len(t_to_tt_idx_hds1)
               for t_to_tt_idx_hds1 in t_to_tt_idx_hds) == len(
                   input_schema.column_names_embedder_input)

    assert list(wemb_h.shape)[0] == len(
        input_schema.column_names_embedder_input)

    utterance_states = []
    for i in range(len(t_to_tt_idx)):
        start = t_to_tt_idx[i]
        if i == len(t_to_tt_idx) - 1:
            end = l_n[0]
        else:
            end = t_to_tt_idx[i + 1]
        utterance_states.append(
            paddle.mean(wemb_n[:, start:end, :], axis=[0, 1]))
    assert len(utterance_states) == len(input_sequence)

    schema_token_states = []
    cnt = -1
    for t_to_tt_idx_hds1 in t_to_tt_idx_hds:
        for t_to_tt_idx_hds11 in t_to_tt_idx_hds1:
            cnt += 1
            schema_token_states1 = []
            for i in range(len(t_to_tt_idx_hds11)):
                start = t_to_tt_idx_hds11[i]
                if i == len(t_to_tt_idx_hds11) - 1:
                    end = l_hpu[cnt]
                else:
                    end = t_to_tt_idx_hds11[i + 1]
                schema_token_states1.append(
                    paddle.mean(wemb_h[cnt, start:end, :], axis=0))
            assert len(schema_token_states1) == len(
                input_schema.column_names_embedder_input[cnt].split())
            schema_token_states.append(schema_token_states1)

    assert len(schema_token_states) == len(
        input_schema.column_names_embedder_input)

    return utterance_states, schema_token_states
