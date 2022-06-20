# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unicodedata
import random
import copy

import numpy as np
import paddle
from paddlenlp.data import Pad


def decode(s_arc, s_rel, mask, tree=True):
    """Decode function"""
    mask = mask.numpy()
    lens = np.sum(mask, -1)
    # Prevent self-loops
    arc_preds = paddle.argmax(s_arc, axis=-1).numpy()
    bad = [not istree(seq[:i + 1]) for i, seq in zip(lens, arc_preds)]
    if tree and any(bad):
        arc_preds[bad] = eisner(s_arc.numpy()[bad], mask[bad])
    arc_preds = paddle.to_tensor(arc_preds)
    rel_preds = paddle.argmax(s_rel, axis=-1)
    rel_preds = index_sample(rel_preds, paddle.unsqueeze(arc_preds, axis=-1))
    rel_preds = paddle.squeeze(rel_preds, axis=-1)
    return arc_preds, rel_preds


def pad_sequence(sequences, padding_value=0, fix_len=None):
    """Fill sequences(np.ndarray) into a fixed-length matrix."""
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])
    if fix_len is not None:
        assert fix_len >= max_len, "fix_len is too small."
        max_len = fix_len
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = np.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return out_tensor


def pad_sequence_paddle(inputs, lens, pad_index=0):
    sequences = []
    idx = 0
    for l in lens:
        sequences.append(np.array(inputs[idx:idx + l]))
        idx += l
    outputs = Pad(pad_val=pad_index)(sequences)
    output_tensor = paddle.to_tensor(outputs)
    return output_tensor


def fill_diagonal(x, value, offset=0, dim1=0, dim2=1):
    """Fill value into the diagoanl of x that offset is ${offset} and the coordinate system is (dim1, dim2)."""
    strides = x.strides
    shape = x.shape
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
    assert 0 <= dim1 < dim2 <= 2
    assert len(x.shape) == 3
    assert shape[dim1] == shape[dim2]

    dim_sum = dim1 + dim2
    dim3 = 3 - dim_sum
    if offset >= 0:
        diagonal = np.lib.stride_tricks.as_strided(
            x[:, offset:] if dim_sum == 1 else x[:, :, offset:],
            shape=(shape[dim3], shape[dim1] - offset),
            strides=(strides[dim3], strides[dim1] + strides[dim2]))
    else:
        diagonal = np.lib.stride_tricks.as_strided(
            x[-offset:, :] if dim_sum in [1, 2] else x[:, -offset:],
            shape=(shape[dim3], shape[dim1] + offset),
            strides=(strides[dim3], strides[dim1] + strides[dim2]))

    diagonal[...] = value
    return x


def backtrack(p_i, p_c, heads, i, j, complete):
    """Backtrack the position matrix of eisner to generate the tree"""
    if i == j:
        return
    if complete:
        r = p_c[i, j]
        backtrack(p_i, p_c, heads, i, r, False)
        backtrack(p_i, p_c, heads, r, j, True)
    else:
        r, heads[j] = p_i[i, j], i
        i, j = sorted((i, j))
        backtrack(p_i, p_c, heads, i, r, True)
        backtrack(p_i, p_c, heads, j, r + 1, True)


def stripe(x, n, w, offset=(0, 0), dim=1):
    """
    Returns a diagonal stripe of the tensor.

    Args:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.

    Example:
    >>> x = np.arange(25).reshape(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    """
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)
    strides = x.strides
    m = strides[0] + strides[1]
    k = strides[1] if dim == 1 else strides[0]
    return np.lib.stride_tricks.as_strided(x[offset[0]:, offset[1]:],
                                           shape=[n, w] + list(x.shape[2:]),
                                           strides=[m, k] + list(strides[2:]))


def flat_words(words, pad_index=0):
    mask = words != pad_index
    lens = paddle.sum(paddle.cast(mask, "int64"), axis=-1)
    position = paddle.cumsum(lens + paddle.cast(
        (lens == 0), "int64"), axis=1) - 1
    select = paddle.nonzero(mask)
    words = paddle.gather_nd(words, select)
    lens = paddle.sum(lens, axis=-1)
    words = pad_sequence_paddle(words, lens, pad_index)
    max_len = words.shape[1]
    position = mask_fill(position, position >= max_len, max_len - 1)
    return words, position


def index_sample(x, index):
    """
    Select input value according to index
    
    Arags：
        input: input matrix
        index: index matrix
    Returns:
        output
    >>> input
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
    >>> index
    [
        [1, 2],
        [0, 1]
    ]
    >>> index_sample(input, index)
    [
        [2, 3],
        [4, 5]
    ]
    """
    x_s = x.shape
    dim = len(index.shape) - 1
    assert x_s[:dim] == index.shape[:dim]

    if len(x_s) == 3 and dim == 1:
        r_x = paddle.reshape(x, shape=[-1, x_s[1], x_s[-1]])
    else:
        r_x = paddle.reshape(x, shape=[-1, x_s[-1]])

    index = paddle.reshape(index, shape=[len(r_x), -1, 1])
    # Generate arange index, shape like index
    arr_index = paddle.arange(start=0, end=len(index), dtype=index.dtype)
    arr_index = paddle.unsqueeze(arr_index, axis=[1, 2])
    arr_index = paddle.expand(arr_index, index.shape)
    # Genrate new index
    new_index = paddle.concat((arr_index, index), -1)
    new_index = paddle.reshape(new_index, (-1, 2))
    # Get output
    out = paddle.gather_nd(r_x, new_index)
    if len(x_s) == 3 and dim == 2:
        out = paddle.reshape(out, shape=[x_s[0], x_s[1], -1])
    else:
        out = paddle.reshape(out, shape=[x_s[0], -1])
    return out


def mask_fill(input, mask, value):
    """
    Fill value to input according to mask
    
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
        [1, 2, 0],
        [4, 0, 0]
    ]
    """
    return input * paddle.logical_not(mask) + paddle.cast(mask,
                                                          input.dtype) * value


def kmeans(x, k):
    """
    kmeans algorithm, put sentence id into k buckets according to sentence length
    
    Args:
        x: list, sentence length
        k: int, k clusters

    Returns:
        centroids: list, center point of k clusters
        clusters: list(tuple), k clusters
    """
    x = np.array(x, dtype=np.float32)
    # Count the frequency of each datapoint
    d, indices, f = np.unique(x, return_inverse=True, return_counts=True)
    # Calculate the sum of the values of the same datapoints
    total = d * f
    # Initialize k centroids randomly
    c, old = d[np.random.permutation(len(d))[:k]], None
    # Assign labels to each datapoint based on centroids
    dists_abs = np.absolute(d[..., np.newaxis] - c)
    dists, y = dists_abs.min(axis=-1), dists_abs.argmin(axis=-1)
    # The number of clusters must not be greater than that of datapoints
    k = min(len(d), k)

    while old is None or not np.equal(c, old).all():
        # If an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster
        # and move that the empty one
        for i in range(k):
            if not np.equal(y, i).any():
                # mask.shape=[k, n]
                mask = y == np.arange(k)[..., np.newaxis]
                lens = mask.sum(axis=-1)
                biggest = mask[lens.argmax()].nonzero()[0]
                farthest = dists[biggest].argmax()
                y[biggest[farthest]] = i
        mask = y == np.arange(k)[..., np.newaxis]
        # Update the centroids
        c, old = (total * mask).sum(-1) / (f * mask).sum(-1), c
        # Re-assign all datapoints to clusters
        dists_abs = np.absolute(d[..., np.newaxis] - c)
        dists, y = dists_abs.min(axis=-1), dists_abs.argmin(axis=-1)
    # Assign all datapoints to the new-generated clusters without considering the empty ones
    y, assigned = y[indices], np.unique(y).tolist()
    # Get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # Map all values of datapoints to buckets
    clusters = [np.equal(y, i).nonzero()[0].tolist() for i in assigned]

    return centroids, clusters


def eisner(scores, mask):
    """Eisner algorithm is a general dynamic programming decoding algorithm for bilexical grammar.

    Args：
        scores: Adjacency matrix，shape=(batch, seq_len, seq_len)
        mask: mask matrix，shape=(batch, sql_len)

    Returns:
        output，shape=(batch, seq_len)，the index of the parent node corresponding to the token in the query

    """
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    scores = scores.transpose(2, 1, 0)
    # Score for incomplete span
    s_i = np.full_like(scores, float('-inf'))
    # Score for complete span
    s_c = np.full_like(scores, float('-inf'))
    # Incompelte span position for backtrack
    p_i = np.zeros((seq_len, seq_len, batch_size), dtype=np.int64)
    # Compelte span position for backtrack
    p_c = np.zeros((seq_len, seq_len, batch_size), dtype=np.int64)
    # Set 0 to s_c.diagonal
    s_c = fill_diagonal(s_c, 0)
    # Contiguous
    s_c = np.ascontiguousarray(s_c)
    s_i = np.ascontiguousarray(s_i)
    for w in range(1, seq_len):
        n = seq_len - w
        starts = np.arange(n, dtype=np.int64)[np.newaxis, :]
        # ilr = C(i->r) + C(j->r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # Shape: (batch_size, n, w)
        ilr = ilr.transpose(2, 0, 1)
        # scores.diagonal(-w).shape:[batch, n]
        il = ilr + scores.diagonal(-w)[..., np.newaxis]
        # I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
        il_span, il_path = il.max(-1), il.argmax(-1)
        s_i = fill_diagonal(s_i, il_span, offset=-w)
        p_i = fill_diagonal(p_i, il_path + starts, offset=-w)

        ir = ilr + scores.diagonal(w)[..., np.newaxis]
        # I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
        ir_span, ir_path = ir.max(-1), ir.argmax(-1)
        s_i = fill_diagonal(s_i, ir_span, offset=w)
        p_i = fill_diagonal(p_i, ir_path + starts, offset=w)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl = cl.transpose(2, 0, 1)
        cl_span, cl_path = cl.max(-1), cl.argmax(-1)
        s_c = fill_diagonal(s_c, cl_span, offset=-w)
        p_c = fill_diagonal(p_c, cl_path + starts, offset=-w)

        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr = cr.transpose(2, 0, 1)
        cr_span, cr_path = cr.max(-1), cr.argmax(-1)
        s_c = fill_diagonal(s_c, cr_span, offset=w)
        s_c[0, w][np.not_equal(lens, w)] = float('-inf')
        p_c = fill_diagonal(p_c, cr_path + starts + 1, offset=w)

    predicts = []
    p_c = p_c.transpose(2, 0, 1)
    p_i = p_i.transpose(2, 0, 1)
    for i, length in enumerate(lens.tolist()):
        heads = np.ones(length + 1, dtype=np.int64)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        predicts.append(heads)

    return pad_sequence(predicts, fix_len=seq_len)


class Node:
    """Node class"""

    def __init__(self, id=None, parent=None):
        self.lefts = []
        self.rights = []
        self.id = int(id)
        self.parent = parent if parent is None else int(parent)


class DepTree:
    """
    DepTree class, used to check whether the prediction result is a project Tree.
    A projective tree means that you can project the tree without crossing arcs.
    """

    def __init__(self, sentence):
        # set root head to -1
        sentence = copy.deepcopy(sentence)
        sentence[0] = -1
        self.sentence = sentence
        self.build_tree()
        self.visit = [False] * len(sentence)

    def build_tree(self):
        """Build the tree"""
        self.nodes = [
            Node(index, p_index) for index, p_index in enumerate(self.sentence)
        ]
        # set root
        self.root = self.nodes[0]
        for node in self.nodes[1:]:
            self.add(self.nodes[node.parent], node)

    def add(self, parent, child):
        """Add a child node"""
        if parent.id is None or child.id is None:
            raise Exception("id is None")
        if parent.id < child.id:
            parent.rights = sorted(parent.rights + [child.id])
        else:
            parent.lefts = sorted(parent.lefts + [child.id])

    def judge_legal(self):
        """Determine whether it is a project tree"""
        target_seq = list(range(len(self.nodes)))
        if len(self.root.lefts + self.root.rights) != 1:
            return False
        cur_seq = self.inorder_traversal(self.root)
        if target_seq != cur_seq:
            return False
        else:
            return True

    def inorder_traversal(self, node):
        """Inorder traversal"""
        if self.visit[node.id]:
            return []
        self.visit[node.id] = True
        lf_list = []
        rf_list = []
        for ln in node.lefts:
            lf_list += self.inorder_traversal(self.nodes[ln])
        for rn in node.rights:
            rf_list += self.inorder_traversal(self.nodes[rn])

        return lf_list + [node.id] + rf_list


def istree(sequence):
    """Is the sequence a project tree"""
    return DepTree(sequence).judge_legal()
