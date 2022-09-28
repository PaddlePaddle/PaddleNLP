import re
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

global_dtype = paddle.get_default_dtype()


def sample_logits(embedding, bias, labels, inputs, sampler):
    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
    n_sample = neg_samples.shape[0]
    b1, b2 = labels.shape[0], labels.shape[1]
    all_ids = paddle.concat([paddle.reshape(labels, shape=[-1]), neg_samples])
    all_w = embedding(all_ids)
    true_w = paddle.reshape(all_w[:-n_sample], shape=[b1, b2, -1])
    sample_w = paddle.reshape(all_w[-n_sample:], shape=[n_sample, -1])

    all_b = paddle.gather(bias, all_ids)
    true_b = paddle.reshape(all_b[:-n_sample], shape=[b1, b2])
    sample_b = all_b[-n_sample:]

    hit = paddle.cast((labels.unsqueeze([2]) == neg_samples),
                      dtype=global_dtype).detach()
    true_logits = paddle.sum(true_w * inputs, axis=-1) + true_b - true_log_probs
    sample_logits = paddle.transpose(
        paddle.matmul(sample_w, paddle.transpose(inputs, [0, 2, 1])),
        [0, 2, 1]) + sample_b - samp_log_probs
    sample_logits = sample_logits - 1e30 * hit
    logits = paddle.concat([true_logits.unsqueeze([2]), sample_logits], -1)

    return logits


class ProjAdaptiveSoftmax(nn.Layer):
    """
    Combine projection and logsoftmax. 
    """

    def __init__(self,
                 n_token,
                 d_embed,
                 d_proj,
                 cutoffs,
                 div_val=1,
                 keep_order=False):
        super(ProjAdaptiveSoftmax, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.num_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.num_clusters

        if self.num_clusters > 0:
            self.cluster_weight = paddle.create_parameter(
                shape=[self.num_clusters, self.d_embed],
                dtype=global_dtype,
                default_initializer=paddle.nn.initializer.Normal(mean=0.0,
                                                                 std=0.01))
            self.cluster_bias = paddle.create_parameter(
                shape=[self.num_clusters],
                dtype=global_dtype,
                is_bias=True,
                default_initializer=paddle.nn.initializer.Constant(0.0))

        self.out_layers_weight = nn.ParameterList()
        self.out_layers_bias = nn.ParameterList()
        self.out_projs = nn.ParameterList()

        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(
                        paddle.create_parameter(
                            shape=[d_proj, d_embed],
                            dtype=global_dtype,
                            default_initializer=paddle.nn.initializer.Normal(
                                mean=0.0, std=0.01)))
                else:
                    self.out_projs.append(None)

            self.out_layers_weight.append(
                paddle.create_parameter(
                    shape=[n_token, d_embed],
                    dtype=global_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0.0)))
            self.out_layers_bias.append(
                paddle.create_parameter(
                    shape=[n_token],
                    dtype=global_dtype,
                    is_bias=True,
                    default_initializer=paddle.nn.initializer.Constant(0.0)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)

                self.out_projs.append(
                    paddle.create_parameter(
                        shape=[d_proj, d_emb_i],
                        dtype=global_dtype,
                        default_initializer=paddle.nn.initializer.Normal(
                            mean=0.0, std=0.01)))

                self.out_layers_weight.append(
                    paddle.create_parameter(
                        shape=[r_idx - l_idx, d_emb_i],
                        dtype=global_dtype,
                        default_initializer=paddle.nn.initializer.Uniform(
                            low=-(r_idx - l_idx)**(-1.0 / 2.0),
                            high=(r_idx - l_idx)**(-1.0 / 2.0))))
                self.out_layers_bias.append(
                    paddle.create_parameter(
                        shape=[r_idx - l_idx],
                        dtype=global_dtype,
                        is_bias=True,
                        default_initializer=paddle.nn.initializer.Uniform(
                            low=-(r_idx - l_idx)**(-1.0 / 2.0),
                            high=(r_idx - l_idx)**(-1.0 / 2.0))))

        self.keep_order = keep_order

    def _compute_logits(self, hidden, weight, bias, proj=None):
        if proj is None:
            logit = F.linear(hidden, weight.t(), bias=bias)
        else:
            proj_hid = F.linear(hidden, proj)
            logit = F.linear(proj_hid, weight.t(), bias=bias)

        return logit

    def forward(self, hidden, target, keep_order=False):
        assert (hidden.shape[0] == target.shape[0])

        if self.num_clusters == 0:
            logit = self._compute_logits(hidden, self.out_layers_weight[0],
                                         self.out_layers_bias[0],
                                         self.out_projs[0])
            nll = -paddle.log(F.softmax(logit, axis=-1))
            idx = paddle.concat([
                paddle.arange(0, nll.shape[0]).unsqueeze([1]),
                target.unsqueeze(1)
            ],
                                axis=1)
            nll = paddle.gather_nd(nll, idx)
        else:
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers_weight[0][l_idx:r_idx]
                    bias_i = self.out_layers_bias[0][l_idx:r_idx]
                else:
                    weight_i = self.out_layers_weight[i]
                    bias_i = self.out_layers_bias[i]

                if i == 0:
                    weight_i = paddle.concat([weight_i, self.cluster_weight],
                                             axis=0)
                    bias_i = paddle.concat([bias_i, self.cluster_bias], axis=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[
                0], self.out_projs[0]

            head_logit = self._compute_logits(hidden, head_weight, head_bias,
                                              head_proj)
            head_logprob = paddle.log(F.softmax(head_logit, axis=-1))

            nll = paddle.zeros_like(target, dtype=hidden.dtype)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = paddle.cast(
                    target >= l_idx,
                    dtype=paddle.get_default_dtype()) * paddle.cast(
                        target < r_idx, dtype="int64")
                indices_i = paddle.nonzero(mask_i).squeeze([1])

                if paddle.numel(indices_i) == 0:
                    continue
                target_i = paddle.gather(target, indices_i, axis=0) - l_idx
                head_logprob_i = paddle.gather(head_logprob, indices_i, axis=0)
                if i == 0:
                    target_i_idx = paddle.concat([
                        paddle.arange(0, head_logprob_i.shape[0]).unsqueeze([1
                                                                             ]),
                        target_i.unsqueeze([1])
                    ],
                                                 axis=1)
                    logprob_i = head_logprob_i.gather_nd(target_i_idx)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[
                        i], self.out_projs[i].weight if self.out_projs[
                            i] is not None else None

                    hidden_i = paddle.gather(hidden, indices_i, axis=0)

                    tail_logit_i = self._compute_logits(hidden_i, weight_i,
                                                        bias_i, proj_i)
                    tail_logprob_i = paddle.log(F.softmax(tail_logit_i,
                                                          axis=-1))

                    target_i_idx = paddle.concat([
                        paddle.arange(0, tail_logprob_i.shape[0]).unsqueeze([1
                                                                             ]),
                        target_i.unsqueeze([1])
                    ],
                                                 axis=1)
                    logprob_i = tail_logprob_i.gather_nd(target_i_idx)

                    logprob_i = head_logprob_i[:, -i] + logprob_i

                if self.keep_order or keep_order:
                    nll = paddle.scatter(nll, indices_i, -logprob_i)
                else:
                    index = paddle.arange(offset, offset + logprob_i.shape[0],
                                          1)
                    nll = paddle.scatter(nll, index, -logprob_i)

                offset += logprob_i.shape[0]

        return nll


class LogUniformSampler(object):

    def __init__(self, range_max, n_sample):
        with paddle.no_grad():
            self.range_max = range_max
            log_indices = paddle.log(
                paddle.arange(1., range_max + 2., 1., dtype=global_dtype))
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]

            self.log_q = paddle.cast(paddle.log(
                paddle.exp(-(
                    paddle.log1p(-paddle.cast(self.dist, dtype=global_dtype)) *
                    2 * n_sample)) - 1),
                                     dtype=global_dtype)

        self.n_sample = n_sample

    def sample(self, labels):
        n_sample = self.n_sample
        n_tries = 2 * n_sample
        batch_size = labels.shape[0]

        with paddle.no_grad():
            neg_samples = paddle.unique(
                paddle.multinomial(self.dist, n_tries, replacement=True))
            true_log_probs = paddle.gather(self.log_q, labels.flatten())
            true_log_probs = paddle.reshape(true_log_probs,
                                            shape=[batch_size, -1])
            samp_log_probs = paddle.gather(self.log_q, neg_samples)
            return true_log_probs, samp_log_probs, neg_samples


class PositionEmbedding(nn.Layer):

    def __init__(self, emb_dim):
        super(PositionEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.inv_freq = 1.0 / (10000.0**(
            paddle.arange(0.0, emb_dim, 2.0, dtype=global_dtype) / emb_dim))

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = paddle.matmul(pos_seq.unsqueeze([1]),
                                     self.inv_freq.unsqueeze([0]))
        pos_emb = paddle.concat(
            [paddle.sin(sinusoid_inp),
             paddle.cos(sinusoid_inp)], axis=-1)

        if bsz is not None:
            pos_emb = pos_emb.unsqueeze([0]).expand([bsz, -1, -1])
            pos_emb.stop_gradient = True
            return pos_emb
        else:
            pos_emb = pos_emb.unsqueeze([0])
            pos_emb.stop_gradient = True
            return pos_emb


class PositionwiseFFN(nn.Layer):

    def __init__(self, d_model, d_inner, dropout, normalize_before=False):
        super(PositionwiseFFN, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model,
                      d_inner,
                      weight_attr=paddle.nn.initializer.Normal(mean=0.0,
                                                               std=0.01),
                      bias_attr=paddle.nn.initializer.Constant(0.0)), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner,
                      d_model,
                      weight_attr=paddle.nn.initializer.Normal(mean=0.0,
                                                               std=0.01),
                      bias_attr=paddle.nn.initializer.Constant(0.0)),
            nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(
            d_model,
            weight_attr=paddle.nn.initializer.Normal(mean=1.0, std=0.01),
            bias_attr=paddle.nn.initializer.Constant(0.0))
        self.normalize_before = normalize_before

    def forward(self, inp):
        if self.normalize_before:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)
        return output


class MultiHeadAttn(nn.Layer):

    def __init__(self,
                 n_head,
                 d_model,
                 d_head,
                 dropout,
                 attn_dropout=0,
                 normalize_before=False):
        super(MultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head

        self.q_proj = nn.Linear(d_model,
                                n_head * d_head,
                                weight_attr=paddle.nn.initializer.Normal(
                                    mean=0.0, std=0.01),
                                bias_attr=False)
        self.kv_proj = nn.Linear(d_model,
                                 2 * n_head * d_head,
                                 weight_attr=paddle.nn.initializer.Normal(
                                     mean=0.0, std=0.01),
                                 bias_attr=False)
        self.drop = nn.Dropout(p=dropout)
        self.attn_drop = nn.Dropout(p=attn_dropout)
        self.o_proj = nn.Linear(n_head * d_head,
                                d_model,
                                weight_attr=paddle.nn.initializer.Normal(
                                    mean=0.0, std=0.01),
                                bias_attr=False)
        self.layer_norm = nn.LayerNorm(
            d_model,
            weight_attr=paddle.nn.initializer.Normal(mean=1.0, std=0.01),
            bias_attr=paddle.nn.initializer.Constant(0.0))

        self.scale = 1 / (d_head**0.5)
        self.normalize_before = normalize_before

    def forward(self, h, attn_mask=None, mems=None):
        if mems is not None:
            c = paddle.concat([mems, h], axis=1)
        else:
            c = h

        if self.normalize_before:
            c = self.layer_norm(c)

        head_q = self.q_proj(h)
        head_k, head_v = paddle.chunk(self.kv_proj(c), chunks=2, axis=-1)

        head_q = paddle.reshape(
            head_q, shape=[h.shape[0], h.shape[1], self.n_head, self.d_head])
        head_k = paddle.reshape(
            head_k, shape=[c.shape[0], c.shape[1], self.n_head, self.d_head])
        head_v = paddle.reshape(
            head_v, shape=[c.shape[0], c.shape[1], self.n_head, self.d_head])

        attn_score = paddle.einsum('bind,bjnd->bnij', head_q, head_k)
        attn_score = attn_score * self.scale
        if attn_mask is not None:
            attn_score = attn_score - float('inf') * attn_mask

        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.attn_drop(attn_prob)

        attn_vec = paddle.einsum('bnij,bjnd->bind', attn_prob, head_v)
        attn_vec = paddle.reshape(attn_vec,
                                  shape=[
                                      attn_vec.shape[0], attn_vec.shape[1],
                                      self.n_head * self.d_head
                                  ])

        attn_out = self.o_proj(attn_vec)
        attn_out = self.drop(attn_out)
        if self.normalize_before:
            output = h + attn_out
        else:
            output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(nn.Layer):

    def __init__(self,
                 n_head,
                 d_model,
                 d_head,
                 dropout,
                 attn_dropout=0,
                 tgt_len=None,
                 ext_len=None,
                 mem_len=None,
                 normalize_before=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_proj = nn.Linear(d_model,
                                  3 * n_head * d_head,
                                  weight_attr=paddle.nn.initializer.Normal(
                                      mean=0.0, std=0.01),
                                  bias_attr=False)

        self.drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.o_proj = nn.Linear(n_head * d_head,
                                d_model,
                                weight_attr=paddle.nn.initializer.Normal(
                                    mean=0.0, std=0.01),
                                bias_attr=False)

        self.layer_norm = nn.LayerNorm(
            d_model,
            weight_attr=paddle.nn.initializer.Normal(mean=1.0, std=0.01),
            bias_attr=paddle.nn.initializer.Constant(0.0))

        self.scale = 1 / (d_head**0.5)

        self.normalize_before = normalize_before

    def _rel_shift(self, x, zero_triu=False):
        x_shape = x.shape
        zero_pad = paddle.zeros([x_shape[0], x_shape[1], x_shape[2], 1],
                                dtype=x.dtype)
        x_padded = paddle.concat([zero_pad, x], axis=-1)

        x_padded = paddle.reshape(
            x_padded,
            shape=[x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2]])

        x = paddle.reshape(x_padded[:, :, 1:, :], shape=x_shape)

        if zero_triu:
            ones = paddle.ones([x_shape[2], x_shape[3]])
            x = x * paddle.tril(
                ones, diagonal=x_shape[3] - x_shape[2]).unsqueeze([2, 3])

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):

    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_proj = nn.Linear(self.d_model,
                                self.n_head * self.d_head,
                                weight_attr=paddle.nn.initializer.Normal(
                                    mean=0.0, std=0.01),
                                bias_attr=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.shape[1], r.shape[1], w.shape[0]

        if mems is not None:
            cat = paddle.concat([mems, w], axis=1)
            if self.normalize_before:
                w_heads = self.qkv_proj(self.layer_norm(cat))
            else:
                w_heads = self.qkv_proj(cat)
            r_head_k = self.r_proj(r)

            w_head_q, w_head_k, w_head_v = paddle.chunk(w_heads,
                                                        chunks=3,
                                                        axis=-1)

            w_head_q = w_head_q[:, -qlen:, :]
        else:
            if self.normalize_before:
                w_heads = self.qkv_proj(self.layer_norm(w))
            else:
                w_heads = self.qkv_proj(w)
            r_head_k = self.r_proj(r)

            w_head_q, w_head_k, w_head_v = paddle.chunk(w_heads,
                                                        chunks=3,
                                                        axis=-1)

        klen = w_head_k.shape[1]

        w_head_q = paddle.reshape(w_head_q,
                                  shape=[bsz, qlen, self.n_head, self.d_head])
        w_head_k = paddle.reshape(w_head_k,
                                  shape=[bsz, klen, self.n_head, self.d_head])
        w_head_v = paddle.reshape(w_head_v,
                                  shape=[bsz, klen, self.n_head, self.d_head])

        r_head_k = paddle.reshape(r_head_k,
                                  shape=[bsz, rlen, self.n_head, self.d_head])

        rw_head_q = w_head_q + r_w_bias

        AC = paddle.einsum('bind,bjnd->bnij', rw_head_q, w_head_k)
        rr_head_q = w_head_q + r_r_bias

        BD = paddle.einsum('bind,bjnd->bnij', rr_head_q, r_head_k)
        BD = self._rel_shift(BD)

        attn_score = AC + BD
        attn_score = attn_score * self.scale

        if attn_mask is not None:
            attn_score = attn_score - 1e30 * attn_mask

        attn_prob = F.softmax(attn_score, axis=-1)
        attn_prob = self.attn_drop(attn_prob)

        attn_vec = paddle.einsum('bnij,bjnd->bind', attn_prob, w_head_v)

        attn_vec = paddle.reshape(attn_vec,
                                  shape=[
                                      attn_vec.shape[0], attn_vec.shape[1],
                                      self.n_head * self.d_head
                                  ])

        attn_out = self.o_proj(attn_vec)
        attn_out = self.drop(attn_out)

        if self.normalize_before:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)

        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):

    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        qlen, bsz = w.shape[1], w.shape[0]

        if mems is not None:
            cat = paddle.concat([mems, w], 1)
            if self.normalize_before:
                w_heads = self.qkv_proj(self.layer_norm(cat))
            else:
                w_heads = self.qkv_proj(cat)
            w_head_q, w_head_k, w_head_v = paddle.chunk(w_heads,
                                                        chunks=3,
                                                        axis=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.normalize_before:
                w_heads = self.qkv_proj(self.layer_norm(w))
            else:
                w_heads = self.qkv_proj(w)
            w_head_q, w_head_k, w_head_v = paddle.chunk(w_heads,
                                                        chunks=3,
                                                        axis=-1)

        klen = w_head_k.shape[1]

        w_head_q = paddle.reshape(w_head_q,
                                  shape=[
                                      w_head_q.shape[0], w_head_q.shape[1],
                                      self.n_head, self.d_head
                                  ])
        w_head_k = paddle.reshape(w_head_k,
                                  shape=[
                                      w_head_k.shape[0], w_head_k.shape[1],
                                      self.n_head, self.d_head
                                  ])
        w_head_v = paddle.reshape(w_head_v,
                                  shape=[
                                      w_head_v.shape[0], w_head_v.shape[1],
                                      self.n_head, self.d_head
                                  ])

        if klen > r_emb.shape[0]:
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.shape[0], -1, -1)
            r_emb = paddle.concat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.shape[0], -1)
            r_bias = paddle.concat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        rw_head_q = w_head_q + r_w_bias.unsqueeze([0])

        AC = paddle.einsum('bind,bjnd->bnij', rw_head_q, w_head_k)
        r_emb = r_emb.unsqueeze([0]).expand([bsz, -1, -1, -1])
        B_ = paddle.einsum('bind,bjnd->bnij', w_head_q, r_emb)
        D_ = r_bias.unsqueeze([0, 2])
        BD = self._rel_shift(B_ + D_)

        attn_score = AC + BD
        attn_score = attn_score * self.scale

        if attn_mask is not None:
            attn_score = attn_score - float('inf') * attn_mask

        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.attn_drop(attn_prob)

        attn_vec = paddle.einsum('bnij,bjnd->bind', attn_prob, w_head_v)

        attn_vec = paddle.reshape(attn_vec,
                                  shape=[
                                      attn_vec.shape[0], attn_vec.shape[1],
                                      self.n_head * self.d_head
                                  ])

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.normalize_before:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)

        return output


class DecoderLayer(nn.Layer):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout,
                                      **kwargs)
        self.pos_ff = PositionwiseFFN(
            d_model,
            d_inner,
            dropout,
            normalize_before=kwargs.get('normalize_before'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)

        return output


class RelLearnableDecoderLayer(nn.Layer):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head,
                                                  dropout, **kwargs)
        self.pos_ff = PositionwiseFFN(
            d_model,
            d_inner,
            dropout,
            normalize_before=kwargs.get('normalize_before'))

    def forward(self,
                dec_inp,
                r_emb,
                r_w_bias,
                r_bias,
                dec_attn_mask=None,
                mems=None):

        output = self.dec_attn(dec_inp,
                               r_emb,
                               r_w_bias,
                               r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayer(nn.Layer):

    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFFN(
            d_model,
            d_inner,
            dropout,
            normalize_before=kwargs.get('normalize_before'))

    def forward(self,
                dec_inp,
                r,
                r_w_bias,
                r_r_bias,
                dec_attn_mask=None,
                mems=None):
        output = self.dec_attn(dec_inp,
                               r,
                               r_w_bias,
                               r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(nn.Layer):

    def __init__(self,
                 n_token,
                 d_embed,
                 d_proj,
                 cutoffs,
                 div_val=1,
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj**0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.LayerList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token,
                             d_embed,
                             sparse=sample_softmax > 0,
                             weight_attr=paddle.nn.initializer.Normal(
                                 mean=0.0, std=0.01)))
            if d_proj != d_embed:
                self.emb_projs.append(
                    paddle.create_parameter(
                        shape=[d_embed, d_proj],
                        dtype=global_dtype,
                        default_initializer=paddle.nn.initializer.Normal(
                            mean=0.0, std=0.01)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.emb_layers.append(
                    nn.Embedding(r_idx - l_idx,
                                 d_emb_i,
                                 weight_attr=paddle.nn.initializer.Normal(
                                     mean=0.0, std=0.01)))
                self.emb_projs.append(
                    paddle.create_parameter(
                        shape=[d_emb_i, d_proj],
                        dtype=global_dtype,
                        default_initializer=paddle.nn.initializer.Normal(
                            mean=0.0, std=0.01)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            inp_flat = paddle.reshape(inp, shape=[-1])
            emb_flat = paddle.zeros([inp_flat.shape[0], self.d_proj],
                                    dtype=global_dtype)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = paddle.nonzero(mask_i).squeeze([1])

                if indices_i.numel() == 0:
                    continue

                inp_i = paddle.gather(inp_flat, indices_i, axis=0) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat = paddle.scatter(emb_flat, indices_i, emb_i)

            embed = paddle.reshape(emb_flat,
                                   shape=inp.shape.append(self.d_proj))

        embed = embed * self.emb_scale

        return embed


class MemTransformerLM(nn.Layer):

    def __init__(self,
                 n_token,
                 n_layer,
                 n_head,
                 d_model,
                 d_head,
                 d_inner,
                 dropout,
                 attn_dropout,
                 tie_weight=True,
                 d_embed=None,
                 div_val=1,
                 tie_projs=[False],
                 normalize_before=False,
                 tgt_len=None,
                 ext_len=None,
                 mem_len=None,
                 cutoffs=[],
                 adapt_inp=False,
                 same_length=False,
                 attn_type=0,
                 clamp_len=-1,
                 sample_softmax=-1):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token,
                                          d_embed,
                                          d_model,
                                          cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.LayerList()
        if attn_type == 0:
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head,
                        d_model,
                        d_head,
                        d_inner,
                        dropout,
                        tgt_len=tgt_len,
                        ext_len=ext_len,
                        mem_len=mem_len,
                        attn_dropout=attn_dropout,
                        normalize_before=normalize_before))
        elif attn_type == 1:
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(n_head,
                                             d_model,
                                             d_head,
                                             d_inner,
                                             dropout,
                                             tgt_len=tgt_len,
                                             ext_len=ext_len,
                                             mem_len=mem_len,
                                             attn_dropout=attn_dropout,
                                             normalize_before=normalize_before))
        elif attn_type in [2, 3]:
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(n_head,
                                 d_model,
                                 d_head,
                                 d_inner,
                                 dropout,
                                 attn_dropout=attn_dropout,
                                 normalize_before=normalize_before))

        self.sample_softmax = sample_softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(
                d_model,
                n_token,
                weight_attr=paddle.nn.initializer.Normal(mean=0.0, std=0.01),
                bias_attr=paddle.nn.initializer.Constant(0.0))
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)
        else:
            self.crit = ProjAdaptiveSoftmax(n_token,
                                            d_embed,
                                            d_model,
                                            cutoffs,
                                            div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers_weight)):
                    self.crit.out_layers_weight[i] = self.word_emb.emb_layers[
                        i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:
            self.pos_emb = PositionEmbedding(self.d_model)
            self.r_w_bias = paddle.create_parameter(
                shape=[self.n_head, self.d_head],
                dtype=global_dtype,
                default_initializer=paddle.nn.initializer.Normal(mean=0.0,
                                                                 std=0.01))
            self.r_r_bias = paddle.create_parameter(
                shape=[self.n_head, self.d_head],
                dtype=global_dtype,
                default_initializer=paddle.nn.initializer.Normal(mean=0.0,
                                                                 std=0.01))
        elif self.attn_type == 1:
            self.r_emb = paddle.create_parameter(
                shape=[self.n_layer, self.max_klen, self.n_head, self.d_head],
                dtype=global_dtype,
                default_initializer=paddle.nn.initializer.Normal(mean=0.0,
                                                                 std=0.01))
            self.r_w_bias = paddle.create_parameter(
                shape=[self.n_layer, self.n_head, self.d_head],
                dtype=global_dtype,
                default_initializer=paddle.nn.initializer.Normal(mean=0.0,
                                                                 std=0.01))
            self.r_bias = paddle.create_parameter(
                shape=[self.n_layer, self.max_klen, self.n_head],
                dtype=global_dtype,
                default_initializer=paddle.nn.initializer.Normal(mean=0.0,
                                                                 std=0.01))
        elif self.attn_type == 2:
            self.pos_emb = PositionEmbedding(self.d_model)
        elif self.attn_type == 3:
            self.r_emb = paddle.create_parameter(
                shape=[self.n_layer, self.max_klen, self.n_head, self.d_head],
                dtype=global_dtype,
                default_initializer=paddle.nn.initializer.Normal(mean=0.0,
                                                                 std=0.01))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self, batch_size, d_model):
        if self.mem_len > 0:
            mems = []
            for _ in range(self.n_layer + 1):
                empty = paddle.empty(shape=[batch_size, 0, d_model],
                                     dtype=global_dtype)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        if mems is None: return None

        assert len(hids) == len(
            mems), "length of hids and length of mems must be the same. "

        with paddle.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = paddle.concat([mems[i], hids[i]], axis=1)
                new_mems.append(cat[:, beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inputs, mems=None):
        bsz, qlen = dec_inputs.shape

        word_emb = self.word_emb(dec_inputs)

        mlen = mems[0].shape[1] if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = paddle.ones(shape=[qlen, klen], dtype=word_emb.dtype)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (paddle.triu(all_ones, diagonal=1 + mlen) +
                             paddle.tril(all_ones, -mask_shift_len)).unsqueeze(
                                 [0, 1])
        else:
            dec_attn_mask = paddle.ones(shape=[qlen, klen],
                                        dtype=word_emb.dtype)
            dec_attn_mask = paddle.triu(dec_attn_mask,
                                        diagonal=1 + mlen).unsqueeze([0, 1])

        hids = []
        if self.attn_type == 0:
            pos_seq = paddle.arange(klen - 1, -1, -1.0, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                # TODO: clamp and clip
                pos_seq = paddle.clip(pos_seq, max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq, bsz)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out,
                                 pos_emb,
                                 self.r_w_bias,
                                 self.r_r_bias,
                                 dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 1:
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out,
                                 r_emb,
                                 self.r_w_bias[i],
                                 r_bias,
                                 dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2:
            pos_seq = paddle.arange(klen - 1, -1, -1.0, dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq = paddle.clip(pos_seq, max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq, bsz)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out,
                                 dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(
                            mlen - cur_size, -1, -1)
                        cur_emb = paddle.concat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out,
                                 dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, *mems):
        if not mems:
            batch_size = data.shape[0]
            mems = self.init_mems(batch_size, self.d_model)

        hidden, new_mems = self._forward(data, mems=mems)

        # TODO(FrostML): use getitem.
        tgt_len = target.shape[1]
        pred_hid = paddle.slice(hidden, [1], [-tgt_len], [hidden.shape[1]])
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight, "tie_weight must be True if sample_softmax > 0"
            logit = sample_logits(self.word_emb, self.out_layer.bias, target,
                                  pred_hid, self.sampler)
            loss = -paddle.log(F.softmax(logit, axis=-1))[:, :, 0]
        else:
            loss = self.crit(
                paddle.reshape(pred_hid, shape=[-1, pred_hid.shape[-1]]),
                paddle.reshape(target, shape=[-1]))

        if new_mems is None:
            return [loss.mean()]
        else:
            return [loss.mean()] + new_mems
