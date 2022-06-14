import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid.layers.utils import map_structure

__all__ = [
    "position_encoding_init", "WordEmbedding", "PositionalEmbedding",
    "CrossEntropyCriterion", "TransformerDecodeCell",
    "TransformerBeamSearchDecoder", "TransformerModel", "InferTransformerModel"
]


def position_encoding_init(n_position, d_pos_vec, dtype="float64"):
    """
    Generates the initial values for the sinusoidal position encoding table.
    This method follows the implementation in tensor2tensor, but is slightly
    different from the description in "Attention Is All You Need".

    Args:
        n_position (int): 
            The largest position for sequences, that is, the maximum length
            of source or target sequences.
        d_pos_vec (int): 
            The size of positional embedding vector. 
        dtype (str, optional): 
            The output `numpy.array`'s data type. Defaults to "float32".

    Returns:
        numpy.array: 
            The embedding table of sinusoidal position encoding with shape
            `[n_position, d_pos_vec]`.

    Example:
        .. code-block::

            from paddlenlp.transformers import position_encoding_init

            max_length = 256
            emb_dim = 512
            pos_table = position_encoding_init(max_length, emb_dim)
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(
        np.arange(num_timescales) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(
        inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype(dtype)


class WordEmbedding(nn.Layer):
    """
    Word Embedding layer of Transformer. 

    This layer automatically constructs a 2D embedding matrix based on the
    input the size of vocabulary (`vocab_size`) and the size of each embedding
    vector (`emb_dim`). This layer lookups embeddings vector of ids provided
    by input `word`. 

    After the embedding, those weights are multiplied by `sqrt(d_model)` which is
    `sqrt(emb_dim)` in the interface. 

    .. math::

        Out = embedding(word) * sqrt(emb\_dim)

    Args:
        vocab_size (int):
            The size of vocabulary. 
        emb_dim (int):
            Dimensionality of each embedding vector.
        bos_id (int, optional):
            The start token id and also is used as padding id. Defaults to 0.
    """

    def __init__(self, vocab_size, emb_dim, bos_id=0):
        super(WordEmbedding, self).__init__()
        self.emb_dim = emb_dim

        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=bos_id,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(0., emb_dim**-0.5)))

    def forward(self, word):
        r"""
        Computes word embedding.

        Args:
            word (Tensor):
                The input ids which indicates the sequences' words with shape
                `[batch_size, sequence_length]` whose data type can be
                int or int64.

        Returns:
            Tensor:
                The (scaled) embedding tensor of shape
                `(batch_size, sequence_length, emb_dim)` whose data type can be
                float32 or float64.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import WordEmbedding

                word_embedding = WordEmbedding(
                    vocab_size=30000,
                    emb_dim=512,
                    bos_id=0)

                batch_size = 5
                sequence_length = 10
                src_words = paddle.randint(low=3, high=30000, shape=[batch_size, sequence_length])
                src_emb = word_embedding(src_words)
        """
        word_emb = self.emb_dim**0.5 * self.word_embedding(word)
        return word_emb


class PositionalEmbedding(nn.Layer):
    """
    This layer produces sinusoidal positional embeddings of any length.
    While in `forward()` method, this layer lookups embeddings vector of
    ids provided by input `pos`.

    Args:
        emb_dim (int):
            The size of each embedding vector.
        max_length (int):
            The maximum length of sequences.
    """

    def __init__(self, emb_dim, max_length):
        super(PositionalEmbedding, self).__init__()
        self.emb_dim = emb_dim

        self.pos_encoder = nn.Embedding(num_embeddings=max_length,
                                        embedding_dim=self.emb_dim)
        self.pos_encoder.weight.set_value(
            position_encoding_init(max_length,
                                   self.emb_dim,
                                   dtype=paddle.get_default_dtype()))

    def forward(self, pos):
        r"""
        Computes positional embedding.

        Args:
            pos (Tensor):
                The input position ids with shape `[batch_size, sequence_length]` whose
                data type can be int or int64.

        Returns:
            Tensor:
                The positional embedding tensor of shape
                `(batch_size, sequence_length, emb_dim)` whose data type can be
                float32 or float64.
        
        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import PositionalEmbedding

                pos_embedding = PositionalEmbedding(
                    emb_dim=512,
                    max_length=256)

                batch_size = 5
                pos = paddle.tile(paddle.arange(start=0, end=50), repeat_times=[batch_size, 1])
                pos_emb = pos_embedding(pos)
        """
        pos_emb = self.pos_encoder(pos)
        pos_emb.stop_gradient = True
        return pos_emb


class CrossEntropyCriterion(nn.Layer):
    """
    Computes the cross entropy loss for given input with or without label smoothing.

    Args:
        label_smooth_eps (float, optional):
            The weight used to mix up the original ground-truth distribution
            and the fixed distribution. Defaults to None. If given, label smoothing
            will be applied on `label`.
        pad_idx (int, optional):
            The token id used to pad variant sequence. Defaults to 0. 
    """

    def __init__(self, label_smooth_eps=None, pad_idx=0):
        super(CrossEntropyCriterion, self).__init__()
        self.label_smooth_eps = label_smooth_eps
        self.pad_idx = pad_idx

    def forward(self, predict, label):
        r"""
        Computes cross entropy loss with or without label smoothing. 

        Args:
            predict (Tensor):
                The predict results of `TransformerModel` with shape
                `[batch_size, sequence_length, vocab_size]` whose data type can
                be float32 or float64.
            label (Tensor):
                The label for correspoding results with shape
                `[batch_size, sequence_length, 1]`.

        Returns:
            tuple:
                A tuple with items: (`sum_cost`, `avg_cost`, `token_num`).

                With the corresponding fields:

                - `sum_cost` (Tensor):
                    The sum of loss of current batch whose data type can be float32, float64.
                - `avg_cost` (Tensor):
                    The average loss of current batch whose data type can be float32, float64.
                    The relation between `sum_cost` and `avg_cost` can be described as:

                    .. math:

                        avg_cost = sum_cost / token_num

                - `token_num` (Tensor):
                    The number of tokens of current batch. 

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import CrossEntropyCriterion

                criterion = CrossEntropyCriterion(label_smooth_eps=0.1, pad_idx=0)
                batch_size = 1
                seq_len = 2
                vocab_size = 30000
                predict = paddle.rand(shape=[batch_size, seq_len, vocab_size])
                label = paddle.randint(
                    low=3,
                    high=vocab_size,
                    shape=[batch_size, seq_len, 1])

                criterion(predict, label)
        """
        weights = paddle.cast(label != self.pad_idx,
                              dtype=paddle.get_default_dtype())
        if self.label_smooth_eps:
            label = paddle.squeeze(label, axis=[2])
            label = F.label_smooth(label=F.one_hot(
                x=label, num_classes=predict.shape[-1]),
                                   epsilon=self.label_smooth_eps)
            if paddle.get_default_dtype() != "float32":
                label = paddle.cast(label, dtype=paddle.get_default_dtype())

        cost = F.cross_entropy(
            input=predict,
            label=label,
            reduction='none',
            soft_label=True if self.label_smooth_eps else False)
        weighted_cost = cost * weights
        sum_cost = paddle.sum(weighted_cost)
        token_num = paddle.sum(weights)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num
        return sum_cost, avg_cost, token_num


class TransformerDecodeCell(nn.Layer):
    """
    This layer wraps a Transformer decoder combined with embedding
    layer and output layer to produce logits from ids and position.

    Args:
        decoder (callable):
            Can be a `paddle.nn.TransformerDecoder` instance. Or a wrapper that includes an
            embedding layer accepting ids and positions and includes an
            output layer transforming decoder output to logits.
        word_embedding (callable, optional):
            Can be a `WordEmbedding` instance or a callable that accepts ids as
            arguments and return embeddings. It can be None if `decoder`
            includes a embedding layer. Defaults to None.
        pos_embedding (callable, optional):
            Can be a `PositionalEmbedding` instance or a callable that accepts position
            as arguments and return embeddings. It can be None if `decoder`
            includes a positional embedding layer. Defaults to None.
        linear (callable, optional):
            Can be a `paddle.nn.Linear` instance or a callable to transform decoder
            output to logits.
        dropout (float, optional):
            The dropout rate for the results of `word_embedding` and `pos_embedding`.
            Defaults to 0.1.
    """

    def __init__(self,
                 decoder,
                 word_embedding=None,
                 pos_embedding=None,
                 linear=None,
                 dropout=0.1):
        super(TransformerDecodeCell, self).__init__()
        self.decoder = decoder
        self.word_embedding = word_embedding
        self.pos_embedding = pos_embedding
        self.linear = linear
        self.dropout = dropout

    def forward(self, inputs, states, static_cache, trg_src_attn_bias, memory,
                **kwargs):
        r"""
        Produces logits.

        Args:
            inputs (Tensor|tuple|list):
                A tuple/list includes target ids and positions. If `word_embedding` is None,
                then it should be a Tensor which means the input for decoder.
            states (list):
                It is a list and each element of the list is an instance
                of `paddle.nn.MultiheadAttention.Cache` for corresponding decoder
                layer. It can be produced by `paddle.nn.TransformerDecoder.gen_cache`.
            static_cache (list):
                It is a list and each element of the list is an instance of
                `paddle.nn.MultiheadAttention.StaticCache` for corresponding
                decoder layer. It can be produced by `paddle.nn.TransformerDecoder.gen_cache`.
            trg_src_attn_bias (Tensor):
                A tensor used in self attention to prevents attention to some unwanted
                positions, usually the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, target_length, target_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. The data type should be float32 or float64. It can
                be None when nothing wanted or needed to be prevented attention to.
            memory (Tensor):
                The output of Transformer encoder. It is a tensor with shape
                `[batch_size, source_length, d_model]` and its data type can be
                float32 or float64.

        Returns:
            tuple: 
                A tuple with items: `(outputs, new_states)`
                
                With the corresponding fields:

                - `outputs` (Tensor):
                    A float32 or float64 3D tensor representing logits shaped
                    `[batch_size, sequence_length, vocab_size]`.
                - `new_states` (Tensor):
                    This output has the same structure and data type with `states`
                    while the length is one larger since concatanating the
                    intermediate results of current step.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import TransformerDecodeCell
                from paddlenlp.transformers import TransformerBeamSearchDecoder

                def decoder():
                    # do decoder
                    pass

                cell = TransformerDecodeCell(decoder())

                self.decode = TransformerBeamSearchDecoder(
                    cell, start_token=0, end_token=1, beam_size=4,
                    var_dim_in_state=2)
        """

        if states and static_cache:
            states = list(zip(states, static_cache))

        if self.word_embedding:
            if not isinstance(inputs, (list, tuple)):
                inputs = (inputs)

            word_emb = self.word_embedding(inputs[0])
            pos_emb = self.pos_embedding(inputs[1])
            word_emb = word_emb + pos_emb
            inputs = F.dropout(word_emb, p=self.dropout,
                               training=False) if self.dropout else word_emb

            cell_outputs, new_states = self.decoder(inputs, memory, None,
                                                    trg_src_attn_bias, states)
        else:
            cell_outputs, new_states = self.decoder(inputs, memory, None,
                                                    trg_src_attn_bias, states)

        if self.linear:
            cell_outputs = self.linear(cell_outputs)

        new_states = [cache[0] for cache in new_states]

        return cell_outputs, new_states


class TransformerBeamSearchDecoder(nn.decode.BeamSearchDecoder):
    """
    This layer is a subclass of `BeamSearchDecoder` to make
    beam search adapt to Transformer decoder.

    Args:
        cell (`TransformerDecodeCell`):
            An instance of `TransformerDecoderCell`.
        start_token (int):
            The start token id.
        end_token (int):
            The end token id.
        beam_size (int):
            The beam width used in beam search.
        var_dim_in_state (int):
            Indicate which dimension of states is variant.
    """

    def __init__(self, cell, start_token, end_token, beam_size,
                 var_dim_in_state):
        super(TransformerBeamSearchDecoder,
              self).__init__(cell, start_token, end_token, beam_size)
        self.cell = cell
        self.var_dim_in_state = var_dim_in_state

    def _merge_batch_beams_with_var_dim(self, c):
        # Init length of cache is 0, and it increases with decoding carrying on,
        # thus need to reshape elaborately
        var_dim_in_state = self.var_dim_in_state + 1  # count in beam dim
        c = paddle.transpose(
            c,
            list(range(var_dim_in_state, len(c.shape))) +
            list(range(0, var_dim_in_state)))
        c = paddle.reshape(
            c, [0] * (len(c.shape) - var_dim_in_state) +
            [self.batch_size * self.beam_size] +
            [int(size) for size in c.shape[-var_dim_in_state + 2:]])
        c = paddle.transpose(
            c,
            list(range((len(c.shape) + 1 - var_dim_in_state), len(c.shape))) +
            list(range(0, (len(c.shape) + 1 - var_dim_in_state))))
        return c

    def _split_batch_beams_with_var_dim(self, c):
        var_dim_size = paddle.shape(c)[self.var_dim_in_state]
        c = paddle.reshape(
            c, [-1, self.beam_size] +
            [int(size)
             for size in c.shape[1:self.var_dim_in_state]] + [var_dim_size] +
            [int(size) for size in c.shape[self.var_dim_in_state + 1:]])
        return c

    @staticmethod
    def tile_beam_merge_with_batch(t, beam_size):
        r"""
        Tiles the batch dimension of a tensor. Specifically, this function takes
        a tensor t shaped `[batch_size, s0, s1, ...]` composed of minibatch 
        entries `t[0], ..., t[batch_size - 1]` and tiles it to have a shape
        `[batch_size * beam_size, s0, s1, ...]` composed of minibatch entries
        `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
        `beam_size` times.

        Args:
            t (list|tuple):
                A list of tensor with shape `[batch_size, ...]`.
            beam_size (int):
                The beam width used in beam search.

        Returns:
            Tensor:
                A tensor with shape `[batch_size * beam_size, ...]`, whose
                data type is same as `t`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import TransformerBeamSearchDecoder

                t = paddle.rand(shape=[10, 10])
                TransformerBeamSearchDecoder.tile_beam_merge_with_batch(t, beam_size=4)
        """
        return map_structure(
            lambda x: nn.decode.BeamSearchDecoder.tile_beam_merge_with_batch(
                x, beam_size), t)

    def step(self, time, inputs, states, **kwargs):
        # Steps for decoding.
        # Compared to RNN, Transformer has 3D data at every decoding step
        inputs = paddle.reshape(inputs, [-1, 1])  # token
        pos = paddle.ones_like(inputs) * time  # pos

        cell_states = map_structure(self._merge_batch_beams_with_var_dim,
                                    states.cell_states)

        cell_outputs, next_cell_states = self.cell((inputs, pos), cell_states,
                                                   **kwargs)

        # Squeeze to adapt to BeamSearchDecoder which use 2D logits
        cell_outputs = map_structure(
            lambda x: paddle.squeeze(x, [1])
            if len(x.shape) == 3 else x, cell_outputs)
        cell_outputs = map_structure(self._split_batch_beams, cell_outputs)
        next_cell_states = map_structure(self._split_batch_beams_with_var_dim,
                                         next_cell_states)

        beam_search_output, beam_search_state = self._beam_search_step(
            time=time,
            logits=cell_outputs,
            next_cell_states=next_cell_states,
            beam_state=states)

        if kwargs.get("trg_word", None) is not None:
            if paddle.in_dynamic_mode():
                if paddle.shape(kwargs.get("trg_word"))[1] > time:
                    beam_search_output, beam_search_state = self.force_decoding(
                        beam_search_output, beam_search_state,
                        kwargs.get("trg_word"), kwargs.get("trg_length"), time)
            else:

                def condition(trg_word, time):
                    return paddle.shape(trg_word)[1] > time

                def default_fn(beam_search_output, beam_search_state):
                    return beam_search_output, beam_search_state

                from functools import partial
                beam_search_output, beam_search_state = paddle.static.nn.case(
                    [(condition(kwargs.get("trg_word"), time),
                      partial(self.force_decoding,
                              beam_search_output=beam_search_output,
                              beam_search_state=beam_search_state,
                              trg_word=kwargs.get("trg_word"),
                              trg_length=kwargs.get("trg_length"),
                              time=time))],
                    default=partial(default_fn,
                                    beam_search_output=beam_search_output,
                                    beam_search_state=beam_search_state))

        next_inputs, finished = (beam_search_output.predicted_ids,
                                 beam_search_state.finished)

        return (beam_search_output, beam_search_state, next_inputs, finished)

    def force_decoding(self, beam_search_output, beam_search_state, trg_word,
                       trg_length, time):
        batch_size = paddle.shape(beam_search_output.predicted_ids)[0]
        beam_size = paddle.shape(beam_search_output.predicted_ids)[1]

        ids_dtype = beam_search_output.predicted_ids.dtype
        scores_dtype = beam_search_output.scores.dtype
        parent_ids = paddle.zeros(shape=[batch_size, 1], dtype=ids_dtype)
        scores = paddle.ones(shape=[batch_size, beam_size],
                             dtype=scores_dtype) * -10e9
        scores = paddle.scatter(
            scores.flatten(),
            paddle.arange(0,
                          batch_size * beam_size,
                          step=beam_size,
                          dtype=scores_dtype),
            paddle.zeros([batch_size])).reshape([batch_size, beam_size])

        force_position = paddle.unsqueeze(trg_length > time, [1])
        # NOTE: When the date type of the input of paddle.tile is bool
        # and enable static mode, its stop_gradient must be True .
        force_position.stop_gradient = True
        force_position = paddle.tile(force_position, [1, beam_size])
        crt_trg_word = paddle.slice(trg_word,
                                    axes=[1],
                                    starts=[time],
                                    ends=[time + 1])
        crt_trg_word = paddle.tile(crt_trg_word, [1, beam_size])

        predicted_ids = paddle.where(force_position, crt_trg_word,
                                     beam_search_output.predicted_ids)
        scores = paddle.where(force_position, scores, beam_search_output.scores)
        parent_ids = paddle.where(force_position, parent_ids,
                                  beam_search_output.parent_ids)

        cell_states = beam_search_state.cell_states
        log_probs = paddle.where(force_position, scores,
                                 beam_search_state.log_probs)
        finished = beam_search_state.finished
        lengths = beam_search_state.lengths

        return self.OutputWrapper(scores, predicted_ids,
                                  parent_ids), self.StateWrapper(
                                      cell_states, log_probs, finished, lengths)


class TransformerModel(nn.Layer):
    """
    The Transformer model.

    This model is a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        src_vocab_size (int):
            The size of source vocabulary.
        trg_vocab_size (int):
            The size of target vocabulary.
        max_length (int):
            The maximum length of input sequences.
        num_encoder_layers (int):
            The number of sub-layers to be stacked in the encoder.
        num_decoder_layers (int):
            The number of sub-layers to be stacked in the decoder.
        n_head (int):
            The number of head used in multi-head attention.
        d_model (int):
            The dimension for word embeddings, which is also the last dimension of
            the input and output of multi-head attention, position-wise feed-forward
            networks, encoder and decoder.
        d_inner_hid (int):
            Size of the hidden layer in position-wise feed-forward networks.
        dropout (float):
            Dropout rates. Used for pre-process, activation and inside attention.
        weight_sharing (bool):
            Whether to use weight sharing. 
        attn_dropout (float):
            The dropout probability used in MHA to drop some attention target.
            If None, use the value of dropout. Defaults to None.
        act_dropout (float):
            The dropout probability used after FFN activition. If None, use
            the value of dropout. Defaults to None.
        bos_id (int, optional):
            The start token id and also be used as padding id. Defaults to 0.
        eos_id (int, optional):
            The end token id. Defaults to 1.
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 num_encoder_layers,
                 num_decoder_layers,
                 n_head,
                 d_model,
                 d_inner_hid,
                 dropout,
                 weight_sharing,
                 attn_dropout=None,
                 act_dropout=None,
                 bos_id=0,
                 eos_id=1):
        super(TransformerModel, self).__init__()
        self.trg_vocab_size = trg_vocab_size
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dropout = dropout

        self.src_word_embedding = WordEmbedding(vocab_size=src_vocab_size,
                                                emb_dim=d_model,
                                                bos_id=self.bos_id)
        self.src_pos_embedding = PositionalEmbedding(emb_dim=d_model,
                                                     max_length=max_length)
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight sharing."
            )
            self.trg_word_embedding = self.src_word_embedding
            self.trg_pos_embedding = self.src_pos_embedding
        else:
            self.trg_word_embedding = WordEmbedding(vocab_size=trg_vocab_size,
                                                    emb_dim=d_model,
                                                    bos_id=self.bos_id)
            self.trg_pos_embedding = PositionalEmbedding(emb_dim=d_model,
                                                         max_length=max_length)

        self.transformer = paddle.nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_inner_hid,
            dropout=dropout,
            attn_dropout=attn_dropout,
            act_dropout=act_dropout,
            activation="relu",
            normalize_before=True)

        if weight_sharing:
            self.linear = lambda x: paddle.matmul(x=x,
                                                  y=self.trg_word_embedding.
                                                  word_embedding.weight,
                                                  transpose_y=True)
        else:
            self.linear = nn.Linear(in_features=d_model,
                                    out_features=trg_vocab_size,
                                    bias_attr=False)

    def forward(self, src_word, trg_word):
        r"""
        The Transformer forward methods. The input are source/target sequences, and
        returns logits.

        Args:
            src_word (Tensor):
                The ids of source sequences words. It is a tensor with shape
                `[batch_size, source_sequence_length]` and its data type can be
                int or int64.
            trg_word (Tensor):
                The ids of target sequences words. It is a tensor with shape
                `[batch_size, target_sequence_length]` and its data type can be
                int or int64.

        Returns:
            Tensor:
                Output tensor of the final layer of the model whose data
                type can be float32 or float64 with shape
                `[batch_size, sequence_length, vocab_size]`.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import TransformerModel

                transformer = TransformerModel(
                    src_vocab_size=30000,
                    trg_vocab_size=30000,
                    max_length=257,
                    num_encoder_layers=6,
                    num_decoder_layers=6,
                    n_head=8,
                    d_model=512,
                    d_inner_hid=2048,
                    dropout=0.1,
                    weight_sharing=True,
                    bos_id=0,
                    eos_id=1)

                batch_size = 5
                seq_len = 10
                predict = transformer(
                    src_word=paddle.randint(low=3, high=30000, shape=[batch_size, seq_len]),
                    trg_word=paddle.randint(low=3, high=30000, shape=[batch_size, seq_len]))
        """
        src_max_len = paddle.shape(src_word)[-1]
        trg_max_len = paddle.shape(trg_word)[-1]
        src_slf_attn_bias = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()).unsqueeze([1, 2]) * -1e9
        src_slf_attn_bias.stop_gradient = True
        trg_slf_attn_bias = self.transformer.generate_square_subsequent_mask(
            trg_max_len)
        trg_slf_attn_bias.stop_gradient = True
        trg_src_attn_bias = src_slf_attn_bias
        src_pos = paddle.cast(
            src_word != self.bos_id, dtype=src_word.dtype) * paddle.arange(
                start=0, end=src_max_len, dtype=src_word.dtype)
        trg_pos = paddle.cast(
            trg_word != self.bos_id, dtype=src_word.dtype) * paddle.arange(
                start=0, end=trg_max_len, dtype=trg_word.dtype)
        with paddle.static.amp.fp16_guard():
            src_emb = self.src_word_embedding(src_word)
            src_pos_emb = self.src_pos_embedding(src_pos)
            src_emb = src_emb + src_pos_emb
            enc_input = F.dropout(
                src_emb, p=self.dropout,
                training=self.training) if self.dropout else src_emb

            trg_emb = self.trg_word_embedding(trg_word)
            trg_pos_emb = self.trg_pos_embedding(trg_pos)
            trg_emb = trg_emb + trg_pos_emb
            dec_input = F.dropout(
                trg_emb, p=self.dropout,
                training=self.training) if self.dropout else trg_emb

            dec_output = self.transformer(enc_input,
                                          dec_input,
                                          src_mask=src_slf_attn_bias,
                                          tgt_mask=trg_slf_attn_bias,
                                          memory_mask=trg_src_attn_bias)

            predict = self.linear(dec_output)

        return predict
