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
from paddle.fluid.data_feeder import convert_dtype
from .. import PretrainedModel, register_base_model

__all__ = [
    'Ernie3PromptModel', 'Ernie3PromptPretrainedModel',
    'Ernie3PromptForGeneration'
]


class Ernie3PromptEmbeddings(nn.Layer):
    r"""
    Include embeddings from word, position.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.0,
                 max_position_embeddings=2048,
                 use_pos_ids_extra=False):
        super(Ernie3PromptEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.use_pos_ids_extra = use_pos_ids_extra
        if self.use_pos_ids_extra:
            self.position_extra_embeddings = nn.Embedding(
                max_position_embeddings, hidden_size)

    def forward(self, input_ids, position_ids=None, pos_ids_extra=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones
        position_ids.stop_gradient = True
        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embeddings + position_embeddings
        if self.use_pos_ids_extra and pos_ids_extra is not None:
            position_extra_embeddings = self.position_extra_embeddings(
                pos_ids_extra)
            embeddings += position_extra_embeddings
        return embeddings


class Ernie3PromptPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained ERNIE3models. It provides ERNIE3 related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models. 
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ernie3-prompt": {
            "attention_probs_dropout_prob": 0.0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "max_position_embeddings": 2048,
            "initializer_range": 0.02,
            "vocab_size": 55088,
            "vocab_size_output": 46256,
            "bos_token_id": 1,
            "pad_token_id": 0,
            "end_token_id": 29979,
            "gend_token_id": 29983,
            "s_token_id": 29980,
            "use_pos_ids_extra": True,
            "normalize_before": False,
            "weight_sharing": True
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ernie3-prompt":
            "http://10.127.1.139:8901/ernie-prompt/ernie3_prompt.pdparams"
        }
    }
    base_model_prefix = "ernie3_prompt"

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
                        self.ernie3_prompt.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-5


@register_base_model
class Ernie3PromptModel(Ernie3PromptPretrainedModel):
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
                 vocab_size_output,
                 hidden_size=4096,
                 num_hidden_layers=60,
                 num_attention_heads=64,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.0,
                 attention_probs_dropout_prob=0.0,
                 max_position_embeddings=3072,
                 initializer_range=0.02,
                 bos_token_id=1,
                 pad_token_id=0,
                 end_token_id=29979,
                 gend_token_id=29983,
                 s_token_id=29980,
                 use_pos_ids_extra=False,
                 normalize_before=True,
                 weight_sharing=False):
        super(Ernie3PromptModel, self).__init__()
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.end_token_id = end_token_id
        self.gend_token_id = gend_token_id
        self.s_token_id = s_token_id
        self.vocab_size_output = vocab_size_output
        self.initializer_range = initializer_range
        self.normalize_before = normalize_before
        self.embeddings = Ernie3PromptEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, use_pos_ids_extra)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=4 * hidden_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            normalize_before=normalize_before)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                position_ids=None,
                pos_ids_extra=None,
                attention_mask=None,
                use_cache=False,
                cache=None):
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
        attention_mask.stop_gradient = True
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            pos_ids_extra=pos_ids_extra)
        import pdb
        pdb.set_trace()
        if not self.normalize_before:
            embedding_output = self.layer_norm(embedding_output)

        if use_cache:
            if cache is None:
                cache = self.encoder.gen_cache(embedding_output)
            encoder_outputs, cache = self.encoder(embedding_output,
                                                  attention_mask, cache)
            if self.normalize_before:
                encoder_outputs = self.layer_norm(encoder_outputs)
            return encoder_outputs, cache
        else:
            encoder_outputs = self.encoder(embedding_output, attention_mask)
            if self.normalize_before:
                encoder_outputs = self.layer_norm(encoder_outputs)
            return encoder_outputs


class Ernie3PromptTransformHead(nn.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 hidden_act,
                 weight_sharing=False,
                 embeddings_weight=None):
        super(Ernie3PromptTransformHead, self).__init__()
        self.lm_transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, hidden_act)
        self.layer_norm = nn.LayerNorm(hidden_size)
        if not weight_sharing:
            self.lm_out = nn.Linear(hidden_size, vocab_size)
        else:
            weight_attr = paddle.ParamAttr(initializer=nn.initializer.Assign(
                embeddings_weight[:vocab_size].t()))
            self.lm_out = nn.Linear(
                hidden_size, vocab_size, weight_attr=weight_attr)

    def forward(self, hidden_states):
        import pdb
        pdb.set_trace()
        hidden_states = self.lm_transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.lm_out(hidden_states)
        return logits


class Ernie3PromptForGeneration(Ernie3PromptPretrainedModel):
    r"""
        The generate model for Ernie3.

    Args:
        ernie3 (Ernie3Model): 
            An instance of `paddlenlp.transformers.Ernie3Model`.
    """

    def __init__(self, ernie3_prompt):
        super(Ernie3PromptForGeneration, self).__init__()
        self.ernie3_prompt = ernie3_prompt
        self.lm_head = Ernie3PromptTransformHead(
            self.ernie3_prompt.config['hidden_size'],
            self.ernie3_prompt.config['vocab_size_output'],
            self.ernie3_prompt.config['hidden_act'],
            self.ernie3_prompt.config['weight_sharing'],
            self.ernie3_prompt.embeddings.word_embeddings.weight)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                position_ids=None,
                pos_ids_extra=None,
                attention_mask=None,
                use_cache=False,
                cache=None):
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
        outputs = self.ernie3_prompt(input_ids, position_ids, pos_ids_extra,
                                     attention_mask, use_cache, cache)
        sequence_output = outputs[0] if use_cache else outputs
        logits = self.lm_head(sequence_output)
        if use_cache:
            cache = outputs[1]
            return logits, cache
        else:
            return logits

    def prepare_faster_entry(self, kwargs):
        from paddlenlp.ops import FasterErnie3Prompt
        use_fp16_decoding = kwargs.get('use_fp16_decoding', False)
        decoding_lib = kwargs.get('decoding_lib', None)
        # Currently, FasterTransformer only support restricted size_per_head.
        size_per_head = self.ernie3_prompt.config[
            'hidden_size'] // self.ernie3_prompt.config['num_attention_heads']
        if size_per_head not in [32, 64, 128]:
            raise AttributeError(
                "'size_per_head = %d' is not supported yet in the faster version of Ernie3"
                % size_per_head)
        if kwargs['forced_bos_token_id'] is not None:
            # not support for min_length yet in the faster version
            raise AttributeError(
                "'forced_bos_token_id != None' is not supported yet in the faster version"
            )
        self._faster_entry = FasterErnie3Prompt(
            self,
            use_fp16_decoding=use_fp16_decoding,
            decoding_lib=decoding_lib).forward
        return self._faster_entry

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      use_cache=False,
                                      cache=None,
                                      **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        position_ids = kwargs.get("position_ids", None)
        pos_ids_extra = kwargs.get("pos_ids_extra", None)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask[:, -1, -1, :]
            if "int" in paddle.fluid.data_feeder.convert_dtype(
                    attention_mask.dtype):
                attention_mask = (1.0 - attention_mask) * -1e4
        if cache is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
            if pos_ids_extra is not None:
                pos_ids_extra = pos_ids_extra[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "pos_ids_extra": pos_ids_extra,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "cache": cache,
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
            model_kwargs["cache"] = outputs[1]

        # update token_type_ids with last value
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat(
                [position_ids, position_ids[:, -1].unsqueeze(-1)], axis=-1)

        # update position_ids
        if "pos_ids_extra" in model_kwargs:
            pos_ids_extra = model_kwargs["pos_ids_extra"]
            model_kwargs["pos_ids_extra"] = paddle.concat(
                [pos_ids_extra, pos_ids_extra[:, -1].reshape((-1, 1)) + 1],
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
