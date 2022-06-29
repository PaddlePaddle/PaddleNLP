import paddle
import paddle.nn as nn

from paddlenlp.transformers import ErnieModel, ErniePretrainedModel, register_base_model
from paddlenlp.transformers.ernie.modeling import ErnieEmbeddings, ErniePooler
from paddlenlp.transformers.distill_utils import _convert_attention_mask


def transformer_encoder_forward(self, src, src_mask=None, cache=None):
    src_mask = _convert_attention_mask(src_mask, src.dtype)

    output = src
    new_caches = []
    num_layer = len(self.layers)
    for i, mod in enumerate(self.layers):
        if cache is None:
            output = mod(output, src_mask=src_mask)
        else:
            output, new_cache = mod(output, src_mask=src_mask, cache=cache[i])
            new_caches.append(new_cache)
        if i == num_layer - 2:
            real_output = output

    if self.norm is not None:
        output = self.norm(output)
    layer_output_auxiliary = output

    return (layer_output_auxiliary,
            real_output) if cache is None else (layer_output_auxiliary,
                                                real_output, new_caches)


@register_base_model
class ErnieMultiLayerFinetuningModel(ErniePretrainedModel):

    def __init__(self,
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
                 use_task_id=False):
        super(ErnieMultiLayerFinetuningModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.TruncatedNormal(
                mean=0.0, std=self.initializer_range))
        self.embeddings = ErnieEmbeddings(vocab_size, hidden_size,
                                          hidden_dropout_prob,
                                          max_position_embeddings,
                                          type_vocab_size, pad_token_id,
                                          weight_attr, task_type_vocab_size,
                                          task_id, use_task_id)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            weight_attr=weight_attr,
            normalize_before=False)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_hidden_layers + 1)
        self.pooler = ErniePooler(hidden_size, weight_attr)
        self.pooler_auxiliary = ErniePooler(hidden_size, weight_attr)
        self.apply(self.init_weights)
        nn.TransformerEncoder._ori_forward = nn.TransformerEncoder.forward
        nn.TransformerEncoder._forward = transformer_encoder_forward

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                task_type_ids=None):
        if self.training:
            nn.TransformerEncoder.forward = nn.TransformerEncoder._forward
        else:
            nn.TransformerEncoder.forward = nn.TransformerEncoder._ori_forward
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(
                    self.pooler.dense.weight.dtype) * -1e4,
                axis=[1, 2])
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(
                attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e4
        attention_mask.stop_gradient = True
        embedding_output = self.embeddings(input_ids=input_ids,
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids,
                                           task_type_ids=task_type_ids)
        encoder_output = self.encoder(embedding_output, attention_mask)
        if self.training:
            layer_output_auxiliary, real_output = encoder_output

            pooled_output_auxiliary = self.pooler_auxiliary(
                layer_output_auxiliary)
            pooled_output = self.pooler(real_output)
            return layer_output_auxiliary, real_output, pooled_output_auxiliary, pooled_output

        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output


class ErnieMultiLayerFinetuningForSequenceClassification(ErniePretrainedModel):

    def __init__(self, ernie, num_classes=2, dropout=None):
        super(ErnieMultiLayerFinetuningForSequenceClassification,
              self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  ernie.config["hidden_dropout_prob"])
        self.dropout_auxiliary = nn.Dropout(
            dropout if dropout is not None else self.ernie.
            config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    num_classes)
        self.classifier_auxiliary = nn.Linear(self.ernie.config["hidden_size"],
                                              num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        pooled_outputs = self.ernie(input_ids,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    attention_mask=attention_mask)
        if self.training:
            _, _, pooled_output_auxiliary, pooled_output = pooled_outputs

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            pooled_output_auxiliary = self.dropout_auxiliary(
                pooled_output_auxiliary)
            logits_auxiliary = self.classifier_auxiliary(
                pooled_output_auxiliary)

            return logits_auxiliary, logits
        _, pooled_output = pooled_outputs
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class ErnieMultiLayerFinetuningForQuestionAnswering(ErniePretrainedModel):

    def __init__(self, ernie):
        super(ErnieMultiLayerFinetuningForQuestionAnswering, self).__init__()
        self.ernie = ernie  # allow ernie to be config
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], 2)
        self.classifier_auxiliary = nn.Linear(self.ernie.config["hidden_size"],
                                              2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_outputs = self.ernie(input_ids,
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      attention_mask=attention_mask)
        if self.training:
            sequence_output_auxiliary, sequence_output, _, _ = sequence_outputs
            logits = self.classifier(sequence_output)
            logits = paddle.transpose(logits, perm=[2, 0, 1])
            start_logits, end_logits = paddle.unstack(x=logits, axis=0)

            logits_auxiliary = self.classifier_auxiliary(
                sequence_output_auxiliary)
            logits_auxiliary = paddle.transpose(logits_auxiliary,
                                                perm=[2, 0, 1])
            start_logits_auxiliary, end_logits_auxiliary = paddle.unstack(
                x=logits_auxiliary, axis=0)
            return start_logits_auxiliary, end_logits_auxiliary, start_logits, end_logits
        sequence_output, _ = sequence_outputs
        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class ErnieMultiLayerFinetuningForMultipleChoice(ErniePretrainedModel):

    def __init__(self, ernie, num_choices=2, dropout=None):
        super(ErnieMultiLayerFinetuningForMultipleChoice, self).__init__()
        self.num_choices = num_choices
        self.ernie = ernie
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  ernie.config["hidden_dropout_prob"])
        self.dropout_auxiliary = nn.Dropout(
            dropout if dropout is not None else self.ernie.
            config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], 1)
        self.classifier_auxiliary = nn.Linear(self.ernie.config["hidden_size"],
                                              1)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        # input_ids: [bs, num_choice, seq_l]
        input_ids = input_ids.reshape(shape=(
            -1, input_ids.shape[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if position_ids is not None:
            position_ids = position_ids.reshape(shape=(-1,
                                                       position_ids.shape[-1]))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(
                shape=(-1, token_type_ids.shape[-1]))

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(
                shape=(-1, attention_mask.shape[-1]))

        pooled_outputs = self.ernie(input_ids,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    attention_mask=attention_mask)
        if self.training:
            _, _, pooled_output_auxiliary, pooled_output = pooled_outputs
            pooled_output = self.dropout(pooled_output)
            pooled_output_auxiliary = self.dropout_auxiliary(
                pooled_output_auxiliary)
            logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
            logits_auxiliary = self.classifier_auxiliary(
                pooled_output_auxiliary)  # logits: (bs*num_choice,1)
            reshaped_logits = logits.reshape(
                shape=(-1, self.num_choices))  # logits: (bs, num_choice)
            reshaped_logits_auxiliary = logits_auxiliary.reshape(
                shape=(-1, self.num_choices))  # logits: (bs, num_choice)
            return reshaped_logits_auxiliary, reshaped_logits

        _, pooled_output = pooled_outputs
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape(
            shape=(-1, self.num_choices))  # logits: (bs, num_choice)

        return reshaped_logits
