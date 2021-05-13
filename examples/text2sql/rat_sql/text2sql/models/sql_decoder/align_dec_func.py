#!/usr/bin/env python3
# -*- coding:utf-8 -*-
##########################################################
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved #
##########################################################
"""decoder function for align
adopt from rat-sql spider_dec_func.py, and change to paddle.

Filname: align_dec_func.py
Authors: ZhangAo(@baidu.com)
Date: 2021-01-27 11:31:04
"""

import sys
import os
import traceback
import logging

import numpy as np
import paddle


def compute_align_loss(model, desc_enc, example):
    """model: a nl2code decoder"""
    # find relevant columns
    root_node = example.tree
    rel_cols = list(
        reversed([
            val
            for val in model.ast_wrapper.find_all_descendants_of_type(root_node,
                                                                      'column')
        ]))
    rel_tabs = list(
        reversed([
            val
            for val in model.ast_wrapper.find_all_descendants_of_type(root_node,
                                                                      'table')
        ]))
    rel_vals = np.abs(
        list(
            reversed([
                val
                for val in model.ast_wrapper.find_all_descendants_of_type(
                    root_node, 'value')
            ])))

    rel_cols_t = paddle.to_tensor(sorted(list(set(rel_cols))), dtype='int64')
    rel_tabs_t = paddle.to_tensor(sorted(list(set(rel_tabs))), dtype='int64')
    rel_vals_t = paddle.to_tensor(sorted(list(set(rel_vals))), dtype='int64')

    mc_att_on_rel_col = desc_enc.m2c_align_mat.index_select(rel_cols_t, axis=1)
    mc_max_rel_att = mc_att_on_rel_col.max(axis=0)
    mc_max_rel_att = mc_max_rel_att.clip(min=1e-9)

    mt_att_on_rel_tab = desc_enc.m2t_align_mat.index_select(rel_tabs_t, axis=1)
    mt_max_rel_att = mt_att_on_rel_tab.max(axis=0)
    mt_max_rel_att = mt_max_rel_att.clip(min=1e-9)

    mv_att_on_rel_val = desc_enc.m2v_align_mat.index_select(rel_vals_t, axis=1)
    mv_max_rel_att = mv_att_on_rel_val.max(axis=0)
    mv_max_rel_att = mv_max_rel_att.clip(min=1e-9)

    #c_num = desc_enc.m2c_align_mat.shape[1]
    #un_rel_cols_t = paddle.to_tensor(sorted(list(set(range(c_num)) - set(rel_cols))), dtype='int64')
    #mc_att_on_unrel_col = desc_enc.m2c_align_mat.index_select(un_rel_cols_t, axis=1)
    #mc_max_unrel_att = mc_att_on_unrel_col.max(axis=0)
    #mc_max_unrel_att = mc_max_unrel_att.clip(min=1e-9)
    #mc_margin = paddle.log(mc_max_unrel_att).mean() - paddle.log(mc_max_rel_att).mean()

    #t_num = desc_enc.m2t_align_mat.shape[1]
    #if t_num > len(set(rel_tabs)):
    #    un_rel_tabs_t = paddle.to_tensor(sorted(list(set(range(t_num)) - set(rel_tabs))), dtype='int64')
    #    mt_att_on_unrel_tab = desc_enc.m2t_align_mat.index_select(un_rel_tabs_t, axis=1)
    #    mt_max_unrel_att = mt_att_on_unrel_tab.max(axis=0)
    #    mt_max_unrel_att = mt_max_unrel_att.clip(min=1e-9)
    #    mt_margin = paddle.log(mt_max_unrel_att).mean() - paddle.log(mt_max_rel_att).mean()
    #else:
    #    mt_margin = paddle.to_tensor([0.0])

    value_loss_weight = 2.0
    align_loss = - paddle.log(mc_max_rel_att).mean() \
                 - paddle.log(mt_max_rel_att).mean() \
                 - value_loss_weight * paddle.log(mv_max_rel_att).mean()
    return align_loss


def compute_pointer_with_align(model, node_type, prev_state, prev_action_emb,
                               parent_h, parent_action_emb, desc_enc):
    """compute_pointer_with_align"""
    new_state, attention_weights = model._update_state(
        node_type, prev_state, prev_action_emb, parent_h, parent_action_emb,
        desc_enc)
    # output shape: batch (=1) x emb_size
    output = new_state[0]
    memory_pointer_logits = model.pointers[node_type](output, desc_enc.memory)
    memory_pointer_probs = paddle.nn.functional.softmax(
        memory_pointer_logits, axis=1)
    # pointer_logits shape: batch (=1) x num choices
    if node_type == "column":
        pointer_probs = paddle.mm(memory_pointer_probs, desc_enc.m2c_align_mat)
    elif node_type == 'table':
        pointer_probs = paddle.mm(memory_pointer_probs, desc_enc.m2t_align_mat)
    else:  # value
        pointer_probs = paddle.mm(memory_pointer_probs, desc_enc.m2v_align_mat)
    pointer_probs = pointer_probs.clip(min=1e-9)
    pointer_logits = paddle.log(pointer_probs)
    return output, new_state, pointer_logits, attention_weights


if __name__ == "__main__":
    """run some simple test cases"""
    pass
