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
from paddle import nn

from paddlenlp.transformers.ernie.modeling import (
    ErnieEmbeddings,
    ErnieModel,
    ErniePretrainedModel,
)


class ErniePooler(nn.Layer):
    def __init__(self, hidden_size, weight_attr=None):
        super(ErniePooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, weight_attr=weight_attr)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class P_ErnieEmbeddings(ErnieEmbeddings):
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=0,
        weight_attr=None,
        task_type_vocab_size=3,
        task_id=0,
        use_task_id=False,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            pad_token_id,
            weight_attr,
            task_type_vocab_size,
            task_id,
            use_task_id,
        )
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size, padding_idx=pad_token_id, weight_attr=weight_attr
        )
        self.register_buffer("position_ids", paddle.arange(max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = "absolute"


class P_ErnieModel(ErnieModel):

    # _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        task_type_vocab_size=3,
        task_id=0,
        use_task_id=False,
        enable_recompute=False,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            pad_token_id=pad_token_id,
            task_type_vocab_size=task_type_vocab_size,
            task_id=task_id,
            use_task_id=use_task_id,
            enable_recompute=enable_recompute,
        )
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.TruncatedNormal(mean=0.0, std=self.initializer_range)
        )
        self.embeddings = P_ErnieEmbeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            pad_token_id,
            weight_attr,
            task_type_vocab_size,
            task_id,
            use_task_id,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            weight_attr=weight_attr,
            normalize_before=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers, enable_recompute=enable_recompute)
        self.pooler = ErniePooler(hidden_size, weight_attr)
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        prompt_embedding=None,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(self.pooler.dense.weight.dtype) * -1e4, axis=[1, 2]
            )
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
        attention_mask.stop_gradient = True

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        # extend inputs_embeds
        if prompt_embedding is not None:

            bsz, n_prompt_tokens, prompt_dim = prompt_embedding.shape
            prompt_padding = paddle.zeros([bsz, input_shape[1] - n_prompt_tokens - 1, prompt_dim])
            extended_prompt_embedding = paddle.concat([prompt_embedding, prompt_padding], axis=1)
            pre_padding = paddle.zeros([bsz, 1, prompt_dim])
            extended_prompt_embedding = paddle.concat([pre_padding, extended_prompt_embedding], axis=1)  # for <CLS>
            # extended_prompt_embedding = extended_prompt_embedding.repeat(input_shape[0], 1, 1)
            embedding_output = embedding_output + extended_prompt_embedding
        encoder_outputs = self.encoder(
            embedding_output,
            src_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return sequence_output


class ErnieLMHead(nn.Layer):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, embeddings):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.layer_norm = nn.LayerNorm(config["hidden_size"])
        self.activation = getattr(nn.functional, config["hidden_act"])
        # self.decoder_weight = self.create_parameter(
        #     shape=[config['vocab_size'], config['hidden_size']],
        #     dtype=self.dense.weight.dtype,
        #     is_bias=False)
        # self.decoder_bias = self.create_parameter(
        #     shape=[config['vocab_size']], dtype=self.decoder_weight.dtype, is_bias=True)

        self.decoder = nn.Linear(config["hidden_size"], config["vocab_size"])

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x


class P_ErnieForMaskedLM(ErniePretrainedModel):
    # _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    # _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, ernie, n_prompt_tokens):
        super().__init__()
        self.ernie = ernie
        hidden_size = self.ernie.config["hidden_size"]
        num_hidden_layers = self.ernie.config["num_hidden_layers"]
        num_attention_heads = self.ernie.config["num_attention_heads"]
        intermediate_size = self.ernie.config["intermediate_size"]
        hidden_act = self.ernie.config["hidden_act"]
        hidden_dropout_prob = self.ernie.config["hidden_dropout_prob"]
        attention_probs_dropout_prob = self.ernie.config["attention_probs_dropout_prob"]
        max_position_embeddings = self.ernie.config["max_position_embeddings"]
        type_vocab_size = self.ernie.config["type_vocab_size"]
        initializer_range = self.ernie.config["initializer_range"]
        pad_token_id = self.ernie.config["pad_token_id"]
        task_type_vocab_size = self.ernie.config["task_type_vocab_size"]
        task_id = self.ernie.config["task_id"]
        use_task_id = self.ernie.config["use_task_id"]
        enable_recompute = self.ernie.config["enable_recompute"]
        vocab_size = self.ernie.config["vocab_size"]
        self.ernie_model = P_ErnieModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            pad_token_id=pad_token_id,
            task_type_vocab_size=task_type_vocab_size,
            task_id=task_id,
            use_task_id=use_task_id,
            enable_recompute=enable_recompute,
        )
        self.lm_head = ErnieLMHead(self.ernie.config, self.ernie.embeddings)
        self.apply(self.init_weights)

        self.n_prompt_tokens = n_prompt_tokens
        self.prompt_embedding = None

    def set_prompt_embedding(self, prompt_embedding):
        self.prompt_embedding = prompt_embedding

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="roberta-base",
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     mask="<mask>",
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        mask_pos=None,
        prompt_embedding=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if prompt_embedding is None:
            prompt_embedding = self.prompt_embedding
        if prompt_embedding is not None:
            bsz = input_ids.shape[0]
            prompt_dim = prompt_embedding.shape[-1]
            prompt_embedding = prompt_embedding.reshape([-1, self.n_prompt_tokens, prompt_dim])[:bsz]

        outputs = self.ernie_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prompt_embedding=prompt_embedding,
        )

        return {
            "logits": self.lm_head(outputs[paddle.arange(outputs.shape[0]), mask_pos]),
        }
