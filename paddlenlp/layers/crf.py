# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.nn as nn
from paddlenlp.layers import sequence_mask

__all__ = ['LinearChainCrf', 'LinearChainCrfLoss', 'ViterbiDecoder']


class LinearChainCrf(nn.Layer):
    """
    LinearChainCrf is a linear chain Conditional Random Field layer, it can implement sequential dependencies in the predictions.
    Therefore, it can take context into account whereas a classifier predicts a label for a single sample without considering "neighboring" samples.
    See https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers for reference.

    Args:
        batch_size (int): The batch size.
        num_labels (int): The label number.
        crf_lr (float): The crf layer learning rate.
        with_start_stop_tag (bool): If set to True, the start tag and stop tag will be considered, the transitions params will be a tensor with shape `[num_labels+2, num_labels+2]`.
                                Else, the transitions params will be a tensor with shape `[num_labels, num_labels]`.
    """

    def __init__(self, num_labels, crf_lr=0.1, with_start_stop_tag=True):
        super(LinearChainCrf, self).__init__()
        if with_start_stop_tag:
            self.num_tags = num_labels + 2  # Additional [START] and [STOP]
            self.start_idx = int(self.num_tags - 1)
            self.stop_idx = int(self.num_tags - 2)
        else:
            self.num_tags = num_labels

        self.transitions = self.create_parameter(
            attr=paddle.ParamAttr(learning_rate=crf_lr),
            shape=[self.num_tags, self.num_tags],
            dtype='float32')
        self.with_start_stop_tag = with_start_stop_tag

        self._initial_alpha = None
        self._start_tensor = None
        self._stop_tensor = None
        self._batch_index = None
        self._seq_index = None
        self._batch_seq_index = None

    def _initialize_alpha(self, batch_size):
        # alpha accumulate the path value to get the different next tag
        if self._initial_alpha is None or batch_size > self._initial_alpha.shape[
                0]:
            # Initialized by a small value.
            initial_alpha = paddle.full(
                (batch_size, self.num_tags - 1),
                dtype='float32',
                fill_value=-10000.)
            # alpha_start fill_value = 0. > -10000., means the first one step START gets the most score.
            alpha_start = paddle.full(
                (batch_size, 1), dtype='float32', fill_value=0.)
            self._initial_alpha = paddle.concat(
                [initial_alpha, alpha_start], axis=1)
        return self._initial_alpha[:batch_size, :]

    def forward(self, inputs, lengths):
        """
        Computes the normalization in a linear-chain CRF. See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
        $$ F = logZ(x) = log\\sum_y exp(score(x,y)) $$
        $$ score(x,y) = \\sum_i Emit(x_i,y_i) + Trans(y_{i-1}, y_i) $$
        mark $$ p(y_i) = Emit(x_i,y_i), T(y_{i-1}, y_i)=Trans(y_{i-1}, y_i) $$
        then we can get
        $$ F(1) = log\\sum_{y1} exp(p(y_1) + T([START], y1)) $$
        $$ F(2) = log\\sum_{y1}\\sum_{y2} exp(p(y_1) + T([START], y1) + p(y_2) + T(y_1,y_2)) =  log\\sum_{y2} exp(F(1) + p(y_2) + T(y_1,y_2)) $$
        $$ F(...) = ... $$
        A recursive formula.

        Args:
            inputs (Tensor): The input tensor with shape `[batch_size, sequence_length, num_tags]`.
            lengths (Tensor): The input length with shape `[batch_size]`.

        Returns:
            Tensor: The normalizers tensor, with shape `[batch_size]`.
        """
        batch_size, seq_len, n_labels = inputs.shape
        inputs_t_exp = inputs.transpose([1, 0, 2]).unsqueeze(-1).expand(
            [seq_len, batch_size, n_labels, n_labels])
        # trans_exp: batch_size, num_tags, num_tags
        trans_exp = self.transitions.unsqueeze(0).expand(
            [batch_size, n_labels, n_labels])

        all_alpha = []
        if self.with_start_stop_tag:
            alpha = self._initialize_alpha(batch_size)

        for i, input_exp in enumerate(inputs_t_exp):
            # input_exp: batch_size, num_tags, num_tags
            # alpha_exp: batch_size, num_tags, num_tags
            if i == 0 and not self.with_start_stop_tag:
                mat = input_exp
            else:
                alpha_exp = alpha.unsqueeze(1).expand(
                    [batch_size, n_labels, n_labels])
                # F(n) = logsumexp(F(n-1) + p(y_n) + T(y_{n-1}, y_n))
                mat = input_exp + trans_exp + alpha_exp
            alpha = paddle.logsumexp(mat, 2)
            all_alpha.append(alpha)

        # Get the valid alpha
        all_alpha = paddle.stack(all_alpha).transpose([1, 0, 2])
        batch_index = self._get_batch_index(batch_size)
        last_index = lengths - 1
        idxs = paddle.stack([batch_index, last_index], axis=1)
        alpha = paddle.gather_nd(all_alpha, idxs)

        if self.with_start_stop_tag:
            # The last one step
            alpha += self.transitions[self.stop_idx].unsqueeze(0)
        norm_score = paddle.logsumexp(alpha, 1)
        return norm_score

    def gold_score(self, inputs, labels, lengths):
        """
        Computes the unnormalized score for a tag sequence.
        $$ score(x,y) = \\sum_i Emit(x_i,y_i) + Trans(y_{i-1}, y_i) $$

        Args:
            inputs (Tensor): The input tensor with shape `[batch_size, sequence_length, num_tags]`.
            labels (Tensor) : The label tensor with shape `[batch_size, sequence_length]`
            lengths (Tensor): The input length with shape `[batch_size]`.

        Returns:
            Tensor: The unnormalized sequence scores tensor, with shape `[batch_size]`.
        """
        return self._point_score(inputs, labels, lengths) + self._trans_score(
            labels, lengths)

    def _point_score(self, inputs, labels, lengths):
        batch_size, seq_len, n_labels = inputs.shape
        # Get the true label logit value
        flattened_inputs = inputs.reshape([-1])
        offsets = paddle.unsqueeze(
            self._get_batch_index(batch_size) * seq_len * n_labels, 1)
        offsets += paddle.unsqueeze(self._get_seq_index(seq_len) * n_labels, 0)
        flattened_tag_indices = paddle.reshape(offsets + labels, [-1])

        scores = paddle.gather(flattened_inputs, flattened_tag_indices).reshape(
            [batch_size, seq_len])

        mask = paddle.cast(
            sequence_mask(
                self._get_batch_seq_index(batch_size, seq_len), lengths),
            'float32')
        mask = mask[:, :seq_len]

        mask_scores = scores * mask
        score = paddle.sum(mask_scores, 1)
        return score

    def _trans_score(self, labels, lengths):
        batch_size, seq_len = labels.shape

        if self.with_start_stop_tag:
            # Add START and STOP on either side of the labels
            start_tensor, stop_tensor = self._get_start_stop_tensor(batch_size)
            labels_ext = paddle.concat(
                [start_tensor, labels, stop_tensor], axis=1)
            mask = paddle.cast(
                sequence_mask(
                    self._get_batch_seq_index(batch_size, seq_len),
                    lengths + 1), 'int32')
            pad_stop = paddle.full(
                (batch_size, seq_len + 2),
                dtype='int64',
                fill_value=self.stop_idx)
            labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        else:
            mask = paddle.cast(
                sequence_mask(
                    self._get_batch_seq_index(batch_size, seq_len), lengths),
                'int32')
            labels_ext = labels

        start_tag_indices = labels_ext[:, :-1]
        stop_tag_indices = labels_ext[:, 1:]

        # Encode the indices in a flattened representation.
        transition_indices = start_tag_indices * self.num_tags + stop_tag_indices
        flattened_transition_indices = transition_indices.reshape([-1])
        flattened_transition_params = self.transitions.reshape([-1])

        scores = paddle.gather(
            flattened_transition_params,
            flattened_transition_indices).reshape([batch_size, -1])
        mask_scores = scores * mask[:, 1:]

        # Accumulate the transition score
        score = paddle.sum(mask_scores, 1)

        return score

    def _get_start_stop_tensor(self, batch_size):
        if self._start_tensor is None or self._stop_tensor is None or batch_size != self._start_tensor.shape[
                0]:
            self._start_tensor = paddle.full(
                (batch_size, 1), dtype='int64', fill_value=self.start_idx)
            self._stop_tensor = paddle.full(
                (batch_size, 1), dtype='int64', fill_value=self.stop_idx)
        return self._start_tensor, self._stop_tensor

    def _get_batch_index(self, batch_size):
        if self._batch_index is None or batch_size != self._batch_index.shape[
                0]:
            self._batch_index = paddle.arange(end=batch_size, dtype="int64")
        return self._batch_index

    def _get_seq_index(self, length):
        if self._seq_index is None or length > self._seq_index.shape[0]:
            self._seq_index = paddle.arange(end=length, dtype="int64")
        return self._seq_index[:length]

    def _get_batch_seq_index(self, batch_size, length):
        if self._batch_seq_index is None or length + 2 > self._batch_seq_index.shape[
                1] or batch_size > self._batch_seq_index.shape[0]:
            self._batch_seq_index = paddle.cumsum(
                paddle.ones([batch_size, length + 2], "int64"), axis=1) - 1
        if self.with_start_stop_tag:
            return self._batch_seq_index[:batch_size, :length + 2]
        else:
            return self._batch_seq_index[:batch_size, :length]


class LinearChainCrfLoss(nn.Layer):
    """The negative log-likelihood for linear chain Conditional Random Field (CRF).

    let $$ Z(x) = \\sum_{y'}exp(score(x,y')) $$, means the sum of all path scores,
    then we have $$ loss = -logp(y|x) = -log(exp(score(x,y))/Z(x)) = -score(x,y) + logZ(x) $$

    Args:
        crf (LinearChainCrf): The LinearChainCrf network.
    """

    def __init__(self, crf):
        super(LinearChainCrfLoss, self).__init__()
        self.crf = crf
        if isinstance(crf, paddle.fluid.framework.ParamBase):
            raise ValueError(
                "From paddlenlp >= 2.0.0b4, the first param of LinearChainCrfLoss shoule be a LinearChainCrf object. For input parameter 'crf.transitions', you can remove '.transitions' to 'crf'"
            )

    def forward(self, inputs, lengths, predictions, labels):
        # Note: When closing to convergence, the loss could be a small negative number. This may caused by underflow when calculating exp in logsumexp.
        #       We add relu here to avoid negative loss. In theory, the crf loss must be greater than or equal to 0, relu will not impact on it.
        return nn.functional.relu(
            self.crf.forward(inputs, lengths) - self.crf.gold_score(
                inputs, labels, lengths))


class ViterbiDecoder(nn.Layer):
    """ ViterbiDecoder can decode the highest scoring sequence of tags, it should only be used at test time.

    Args:
        transitions (tensor): The transition matrix with shape `[num_tags, num_tags]`.
        with_start_stop_tag (bool): If set to True, the last row and the last column of transitions will be considered as start tag,
                                the the penultimate row and the penultimate column of transitions will be considered as stop tag.
                                Else, all the rows and columns will be considered as the real tag.
    """

    def __init__(self, transitions, with_start_stop_tag=True):
        super(ViterbiDecoder, self).__init__()
        self.transitions = transitions
        self.with_start_stop_tag = with_start_stop_tag
        # If consider start and stop, -1 should be START and -2 should be STOP.
        if with_start_stop_tag:
            self.start_idx = -1
            self.stop_idx = -2
        self.num_tags = transitions.shape[0]

        self._initial_alpha = None
        self._index = None
        self._batch_index = None

    def _initialize_alpha(self, batch_size):
        # alpha accumulate the path value to get the different next tag
        if self._initial_alpha is None or batch_size > self._initial_alpha.shape[
                0]:
            # Initialized by a small value.
            initial_alpha = paddle.full(
                (batch_size, self.num_tags - 1),
                dtype='float32',
                fill_value=-10000.)
            # alpha_start fill_value = 0. > -10000., means the first one step START gets the most score.
            alpha_start = paddle.full(
                (batch_size, 1), dtype='float32', fill_value=0.)
            self._initial_alpha = paddle.concat(
                [initial_alpha, alpha_start], axis=1)
        return self._initial_alpha[:batch_size, :]

    def forward(self, inputs, lengths):
        """
        Decode the highest scoring sequence of tags.

        Args:
            inputs:  The unary emission tensor with shape `[batch_size, sequence_length, num_tags]`.
            length: The input length tensor with shape `[batch_size]`, storing real length of each sequence for correctness.
        Returns:
            scores: The scores tensor containing the score for the Viterbi sequence, with shape `[batch_size]`.
            paths: The paths tensor containing the highest scoring tag indices, with shape `[batch_size, sequence_length`].
        """
        batch_size, seq_len, n_labels = inputs.shape
        inputs_t = inputs.transpose([1, 0, 2])
        trans_exp = self.transitions.unsqueeze(0).expand(
            [batch_size, n_labels, n_labels])

        all_alpha = []
        historys = []

        if self.with_start_stop_tag:
            alpha = self._initialize_alpha(batch_size)
        else:
            alpha = paddle.zeros((batch_size, self.num_tags), dtype='float32')

        for i, logit in enumerate(inputs_t):
            alpha_exp = alpha.unsqueeze(1).expand(
                [batch_size, n_labels, n_labels])
            # alpha_trn_sum: batch_size, n_labels, n_labels
            alpha_trn_sum = alpha_exp + trans_exp
            # alpha_max: batch_size, n_labels
            # We don't include the emission scores here because the max does not depend on them (we add them in below)
            alpha_max = alpha_trn_sum.max(2)
            if i == 0:
                # if self.with_start_stop_tag, the first antecedent tag must be START, drop it.
                # else, the first label has not antecedent tag, pass it.
                pass
            else:
                alpha_argmax = alpha_trn_sum.argmax(2)
                historys.append(alpha_argmax)
            # Now add the emission scores
            alpha = alpha_max + logit
            all_alpha.append(alpha)

        # Get the valid alpha
        all_alpha = paddle.stack(all_alpha).transpose([1, 0, 2])
        batch_index = self._get_batch_index(batch_size)
        last_index = lengths - 1
        idxs = paddle.stack([batch_index, last_index], axis=1)
        alpha = paddle.gather_nd(all_alpha, idxs)

        if self.with_start_stop_tag:
            # The last one step
            alpha += self.transitions[self.stop_idx].unsqueeze(0).expand_as(
                alpha)
        scores, last_ids = alpha.max(1), alpha.argmax(1).numpy().tolist()
        # Trace back the best path
        # historys: seq_len, batch_size, n_labels
        historys = paddle.stack(historys).numpy()
        lengths_np = lengths.numpy()
        batch_path = []
        max_len = 0
        for batch_id in range(batch_size):
            best_last_tag = last_ids[batch_id]
            path = [best_last_tag]
            for hist in reversed(historys[:lengths_np[batch_id]]):
                # hist: batch_size, n_labels
                best_last_tag = hist[batch_id][best_last_tag]
                path.append(best_last_tag)
            path.reverse()
            max_len = max(max_len, len(path))
            # Pad to the max sequence length, so that the ChunkEvaluator can compute it
            batch_path.append(path)
        batch_path = [path + [0] * (max_len - len(path)) for path in batch_path]
        batch_path = paddle.to_tensor(batch_path)
        return scores, batch_path

    def _get_batch_index(self, batch_size):
        if self._batch_index is None or batch_size != self._batch_index.shape[
                0]:
            self._batch_index = paddle.arange(end=batch_size, dtype="int64")
        return self._batch_index
