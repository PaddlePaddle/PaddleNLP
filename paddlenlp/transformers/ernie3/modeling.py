# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn

from .. import PretrainedModel, register_base_model

__all__ = ['Ernie3Model', 'Ernie3PretrainedModel', 'Ernie3ForGeneration']


class Ernie3Embeddings(nn.Layer):
    r"""
    Include embeddings from word, position.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=4096,
                 hidden_dropout_prob=0.0,
                 max_position_embeddings=3072):
        super(Ernie3Embeddings, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones
        position_ids.stop_gradient = True
        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embeddings + position_embeddings
        return embeddings


class Ernie3PretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained ERNIE3models. It provides ERNIE3 related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models. 
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ernie3-10b": {
            "attention_probs_dropout_prob": 0.0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_hidden_layers": 60,
            "num_sharing_layers": 48,
            "branch_attention_probs_dropout_prob": 0.0,
            "branch_hidden_act": "gelu",
            "branch_hidden_dropout_prob": 0.0,
            "branch_hidden_size": 768,
            "branch_num_attention_heads": 12,
            "max_position_embeddings": 3072,
            "initializer_range": 0.02,
            "vocab_size": 30000,
            "bos_token_id": 1,
            "pad_token_id": 0,
            "end_token_id": 29979
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ernie3-10b": "http://0.0.0.0:0/ernie3-10b/ernie3_10b.pdparams"
        }
    }
    base_model_prefix = "ernie3"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            # `normal` does not support float16
            if isinstance(
                    layer.weight,
                    paddle.Tensor) and paddle.get_default_dtype == "float32":
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.ernie3.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class Ernie3Model(Ernie3PretrainedModel):
    r"""
    The bare ERNIE3 Model transformer outputting raw hidden-states.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `ErnieMModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `ErnieMModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, sharing decoder layers and pooler layer. Defaults to `4096`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `60`.
        num_sharing_layers (int, optional):
            Number of hidden layers in the sharing decoder. Defaults to `48`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the sharing decoder.
            Defaults to `64`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `"gelu"`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and sharing decoder.
            Defaults to `0.0`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers of sharing decoder to drop some attention target.
            Defaults to `0.0`.
        branch_hidden_size (int, optional):
            Dimensionality of the embedding layer, branch decoder encoder layers and pooler layer. Defaults to `768`.
        branch_num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the branch decoder.
            Defaults to `12`.
        branch_hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer of branch decoder.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to `"gelu"`.
        branch_hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.0`.
        branch_attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers of branch decoder to drop some attention target.
            Defaults to `0.0`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `3072`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer for initializing all weight matrices.
            Defaults to `0.02`.
            
            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`Ernie3PretrainedModel._init_weights()` for how weights are initialized in `Ernie3Model`.
        bos_token_id(int, optional):
            The index of bos token in the token vocabulary.
            Defaults to `1`.
        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
        end_token_id(int, optional):
            The index of end token in the token vocabulary.
            Defaults to `29979`.

    """

    def __init__(self,
                 vocab_size,
                 hidden_size=4096,
                 num_hidden_layers=60,
                 num_sharing_layers=48,
                 num_attention_heads=64,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.0,
                 attention_probs_dropout_prob=0.0,
                 branch_hidden_size=768,
                 branch_num_attention_heads=12,
                 branch_hidden_act="gelu",
                 branch_hidden_dropout_prob=0.0,
                 branch_attention_probs_dropout_prob=0.0,
                 max_position_embeddings=3072,
                 initializer_range=0.02,
                 bos_token_id=1,
                 pad_token_id=0,
                 end_token_id=29979):
        super(Ernie3Model, self).__init__()
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.end_token_id = end_token_id
        self.initializer_range = initializer_range
        self.embeddings = Ernie3Embeddings(vocab_size, hidden_size,
                                           hidden_dropout_prob,
                                           max_position_embeddings)
        sharing_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=4 * hidden_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            normalize_before=True)
        self.sharing_encoder = nn.TransformerEncoder(sharing_encoder_layer,
                                                     num_sharing_layers)
        self.sharing_layer_norm = nn.LayerNorm(hidden_size)
        self.sharing_to_nlg = nn.Linear(
            hidden_size, branch_hidden_size, bias_attr=False)
        nlg_encoder_layer = nn.TransformerEncoderLayer(
            d_model=branch_hidden_size,
            nhead=branch_num_attention_heads,
            dim_feedforward=4 * branch_hidden_size,
            dropout=branch_hidden_dropout_prob,
            activation=branch_hidden_act,
            attn_dropout=branch_attention_probs_dropout_prob,
            act_dropout=0,
            normalize_before=True)
        self.nlg_encoder = nn.TransformerEncoder(
            nlg_encoder_layer, num_hidden_layers - num_sharing_layers)
        self.nlg_layer_norm = nn.LayerNorm(branch_hidden_size)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                use_cache=False,
                sharing_cache=None,
                nlg_cache=None):
        r"""
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                max_position_embeddings - 1]``.
                Shape as `[batch_size, num_tokens]` and dtype as int64. Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention on to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Defaults to `None`, which means nothing needed to be prevented attention to.
            use_cache (bool, optional):
                Whether or not to use cache. Defaults to `False`. If set to `True`, key value states will be returned and
                can be used to speed up decoding.
            sharing_cache (list, optional):
                It is a list, and each element in the list is a tuple `(incremental_cache, static_cache)` of sharning decoder.
                See `TransformerDecoder.gen_cache <https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/nn/layer/transformer.py#L1060>`__ for more details.
                It is only used for inference and should be None for training.
                Default to `None`.
            nlg_cache (list, optional):
                It is a list, and each element in the list is a tuple `(incremental_cache, static_cache)` of branch decoder.
                See `TransformerDecoder.gen_cache <https://github.com/PaddlePaddle/Paddle/blob/release/2.1/python/paddle/nn/layer/transformer.py#L1060>`__ for more details.
                It is only used for inference and should be None for training.
                Default to `None`.

        Returns:
            Tensor: Returns tensor `encoder_output`, which is the output at the last layer of the model.
            It has a shape of [batch_size, sequence_length, hidden_size].

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import Ernie3Model, Ernie3Tokenizer

                tokenizer = Ernie3Model.from_pretrained('ernie3-10b')
                model = Ernie3Tokenizer.from_pretrained('ernie3-10b')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                output = model(**inputs)

        """
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(paddle.get_default_dtype()) * -1e4,
                axis=[1, 2])
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(
                attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4

        input_length = paddle.shape(input_ids)[-1]
        causal_mask = paddle.triu(
            paddle.full(
                (input_length, input_length),
                -1e4,
                dtype=paddle.get_default_dtype()),
            diagonal=1)
        attention_mask += causal_mask
        attention_mask.stop_gradient = True

        if position_ids is None:
            past_key_values_length = paddle.shape(nlg_cache[0][0])[
                -2] if nlg_cache is not None else 0
            position_ids = paddle.arange(
                past_key_values_length,
                past_key_values_length + input_length,
                dtype="int64")
            position_ids = position_ids.unsqueeze(0).reshape(
                shape=[-1, input_length])
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids)

        if use_cache:
            if sharing_cache is None:
                sharing_cache = self.sharing_encoder.gen_cache(embedding_output)
            sharing_encoder_outputs, sharing_cache = self.sharing_encoder(
                embedding_output, attention_mask, sharing_cache)
            sharing_encoder_outputs = self.sharing_layer_norm(
                sharing_encoder_outputs)
            sharing_encoder_outputs = self.sharing_to_nlg(
                sharing_encoder_outputs)
            if nlg_cache is None:
                nlg_cache = self.sharing_encoder.gen_cache(
                    sharing_encoder_outputs)
            nlg_encoder_outputs, nlg_cache = self.nlg_encoder(
                sharing_encoder_outputs, attention_mask, nlg_cache)
            nlg_encoder_outputs = self.nlg_layer_norm(nlg_encoder_outputs)
            return nlg_encoder_outputs, sharing_cache, nlg_cache
        else:
            sharing_encoder_outputs = self.sharing_encoder(embedding_output,
                                                           attention_mask)
            sharing_encoder_outputs = self.sharing_layer_norm(
                sharing_encoder_outputs)
            sharing_encoder_outputs = self.sharing_to_nlg(
                sharing_encoder_outputs)
            nlg_encoder_outputs = self.nlg_encoder(sharing_encoder_outputs,
                                                   attention_mask)
            nlg_encoder_outputs = self.nlg_layer_norm(nlg_encoder_outputs)
            return nlg_encoder_outputs


class Ernie3TransformHead(nn.Layer):
    def __init__(self, hidden_size, vocab_size, hidden_act):
        super(Ernie3TransformHead, self).__init__()
        self.lm_transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, hidden_act)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lm_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.lm_transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.lm_out(hidden_states)
        return logits


class Ernie3ForGeneration(Ernie3PretrainedModel):
    r"""
        The generate model for Ernie3.

    Args:
        ernie3 (Ernie3Model): 
            An instance of `paddlenlp.transformers.Ernie3Model`.
    """

    def __init__(self, ernie3):
        super(Ernie3ForGeneration, self).__init__()
        self.ernie3 = ernie3
        self.lm_head = Ernie3TransformHead(
            self.ernie3.config['branch_hidden_size'],
            self.ernie3.config['vocab_size'], self.ernie3.config['hidden_act'])
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                use_cache=False,
                sharing_cache=None,
                nlg_cache=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`Ernie3Model`.
            position_ids (Tensor, optional):
                See :class:`Ernie3Model`.
            attention_mask (Tensor, optional):
                See :class:`Ernie3Model`.
            use_cache (bool, optional):
                See :class:`Ernie3Model`.
            sharing_cache (Tensor, optional):
                See :class:`Ernie3Model`.
            nlg_cache (Tensor, optional):
                See :class:`Ernie3Model`.

        Returns:
            Tensor or tuple: Returns tensor `logits` or tuple `(logits, cached_kvs)`. If `use_cache` is True,
            tuple (`logits, sharing_cache, nlg_cache`) will be returned. Otherwise, tensor `logits` will be returned.
            `logits` is the output of the ernie3 model.
            `sharing_cache` is the sharing cache output of ernie3 model if `use_cache` is True.
            `nlg_cache` is the nlg cache output of ernie3 model if `use_cache` is True.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers import Ernie3ForGeneration, Ernie3Tokenizer

                tokenizer = Ernie3Tokenizer.from_pretrained('ernie3-10b')
                model = Ernie3ForGeneration.from_pretrained('ernie3-10b')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                logits = model(**inputs)

        """
        outputs = self.ernie3(input_ids, position_ids, attention_mask,
                              use_cache, sharing_cache, nlg_cache)
        sequence_output = outputs[0] if use_cache else outputs
        logits = self.lm_head(sequence_output)
        if use_cache:
            sharing_cache = outputs[1]
            nlg_cache = outputs[2]
            return logits, sharing_cache, nlg_cache
        else:
            return logits

    def prepare_faster_entry(self, kwargs):
        from paddlenlp.ops import FasterErnie3
        use_fp16_decoding = kwargs.get('use_fp16_decoding', False)
        decode_strategy = kwargs.get('decode_strategy')
        if decode_strategy == "beam_search":
            raise AttributeError(
                "'beam_search' is not supported yet in the faster version of Ernie3"
            )
        # Currently, FasterTransformer only support restricted size_per_head.
        sharing_size_per_head = self.ernie3.config[
            'hidden_size'] // self.ernie3.config['num_attention_heads']
        nlg_size_per_head = self.ernie3.config[
            'branch_hidden_size'] // self.ernie3.config[
                'branch_num_attention_heads']
        if sharing_size_per_head not in [32, 64, 128]:
            raise AttributeError(
                "'sharing_size_per_head = %d' is not supported yet in the faster version of Ernie3"
                % sharing_size_per_head)
        if nlg_size_per_head not in [32, 64, 128]:
            raise AttributeError(
                "'nlg_size_per_head = %d' is not supported yet in the faster version of Ernie3"
                % nlg_size_per_head)
        if kwargs['forced_bos_token_id'] is not None:
            # not support for min_length yet in the faster version
            raise AttributeError(
                "'forced_bos_token_id != None' is not supported yet in the faster version"
            )
        self._faster_entry = FasterErnie3(
            self, use_fp16_decoding=use_fp16_decoding).forward
        return self._faster_entry

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      use_cache=False,
                                      sharing_cache=None,
                                      nlg_cache=None,
                                      **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask[:, -1, -1, :]
            if "int" in paddle.fluid.data_feeder.convert_dtype(
                    attention_mask.dtype):
                attention_mask = (1.0 - attention_mask) * -1e4
        if sharing_cache is not None and nlg_cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "sharing_cache": sharing_cache,
            "nlg_cache": nlg_cache,
        }

    @staticmethod
    def update_model_kwargs_for_generation(outputs,
                                           model_kwargs,
                                           is_encoder_decoder=False):
        # Update the model inputs during generation.
        # Note that If `token_type_ids` and `attention_mask` in `model_kwargs`
        # and they contain pad value, the result vectors updated by this method
        # may be different from expected. In this case, you need to rewrite the
        # method.

        # update cache
        if isinstance(outputs, tuple):
            model_kwargs["sharing_cache"] = outputs[1]
            model_kwargs["nlg_cache"] = outputs[2]

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], axis=-1)

        # update position_ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat(
                [position_ids, position_ids[:, -1].reshape((-1, 1)) + 1],
                axis=-1)

        # update attention_mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            # nn.Pad2D don't support the data type `bool`
            if convert_dtype(attention_mask.dtype) == 'bool':
                attention_mask = paddle.cast(attention_mask, 'int64')
            if len(attention_mask.shape) == 4:
                attention_mask = nn.Pad2D(
                    [0, 0, 0, 1], mode='replicate')(attention_mask)
                attention_mask = nn.Pad2D(
                    [0, 1, 0, 0], value=-1e4)(attention_mask)
                dtype = convert_dtype(attention_mask.dtype)
                if 'int' in dtype:
                    attention_mask[:, :, -1, -1] = 1
                elif 'float' in dtype:
                    attention_mask[:, :, -1, -1] = 0.0
                else:
                    raise ValueError(
                        'The data type of input `attention_mask` must '
                        'be bool, int or float')
            else:
                attention_mask = paddle.concat(
                    [
                        attention_mask, paddle.ones(
                            [attention_mask.shape[0], 1], dtype="int64")
                    ],
                    axis=-1)
            model_kwargs["attention_mask"] = attention_mask

        return model_kwargs

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            try:
                return getattr(getattr(self, self.base_model_prefix), name)
            except AttributeError:
                try:
                    return getattr(self, self.base_model_prefix).config[name]
                except KeyError:
                    raise e
