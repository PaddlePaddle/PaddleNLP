# coding=utf-8
import math
import paddle
from paddle import nn
from .. import PretrainedModel, register_base_model

__all__ = [
    'SqueezeBertModel',
    'SqueezeBertForSequenceClassification',
    'SqueezeBertForTokenClassification',
    'SqueezeBertForQuestionAnswering',
]

ACT2FN = {'gelu': nn.GELU()}


def _convert_attention_mask(attention_mask, inputs):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask.unsqueeze(1)
    elif attention_mask.dim() == 2:
        # extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
    extended_attention_mask = paddle.cast(extended_attention_mask, inputs.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def create_config(kwargs):
    class Config:
        def __init__(self, kwargs):
            for k, v in kwargs.items():
                self.__setattr__(k, v)

    return Config(kwargs)


class SqueezeBertEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=None)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", paddle.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64, )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MatMulWrapper(nn.Layer):
    """
    Wrapper for paddle.matmul(). This makes flop-counting easier to implement. Note that if you directly call
    paddle.matmul() in your code, the flop counter will typically ignore the flops of the matmul.
    """

    def __init__(self):
        super().__init__()

    def forward(self, mat1, mat2):
        """
        :param inputs: two paddle tensors :return: matmul of these tensors
        Here are the typical dimensions found in BERT (the B is optional) mat1.shape: [B, <optional extra dims>, M, K]
        mat2.shape: [B, <optional extra dims>, K, N] output shape: [B, <optional extra dims>, M, N]
        """
        return paddle.matmul(mat1, mat2)


class SqueezeBertLayerNorm(nn.LayerNorm):
    def __init__(self, hidden_size, epsilon=1e-12):
        nn.LayerNorm.__init__(self, normalized_shape=hidden_size,
                              epsilon=epsilon)  # instantiates self.{weight, bias, eps}

    def forward(self, x):
        x = x.transpose((0, 2, 1))
        x = nn.LayerNorm.forward(self, x)
        return x.transpose((0, 2, 1))


class ConvDropoutLayerNorm(nn.Layer):
    def __init__(self, cin, cout, groups, dropout_prob):
        super().__init__()

        self.conv1d = nn.Conv1D(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        self.layernorm = SqueezeBertLayerNorm(cout)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        x = self.conv1d(hidden_states)
        x = self.dropout(x)
        x = x + input_tensor
        x = self.layernorm(x)
        return x


class ConvActivation(nn.Layer):
    def __init__(self, cin, cout, groups, act):
        super().__init__()
        self.conv1d = nn.Conv1D(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        self.act = ACT2FN[act]

    def forward(self, x):
        output = self.conv1d(x)
        return self.act(output)


class SqueezeBertSelfAttention(nn.Layer):
    def __init__(self, config, cin, q_groups=1, k_groups=1, v_groups=1):
        """
        config = used for some things; ignored for others (work in progress...) cin = input channels = output channels
        groups = number of groups to use in conv1d layers
        """
        super().__init__()
        if cin % config.num_attention_heads != 0:
            raise ValueError(
                f"cin ({cin}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(cin / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Conv1D(in_channels=cin, out_channels=cin, kernel_size=1, groups=q_groups)
        self.key = nn.Conv1D(in_channels=cin, out_channels=cin, kernel_size=1, groups=k_groups)
        self.value = nn.Conv1D(in_channels=cin, out_channels=cin, kernel_size=1, groups=v_groups)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(axis=-1)

        self.matmul_qk = MatMulWrapper()
        self.matmul_qkv = MatMulWrapper()

    def transpose_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, W, C2] where C1 is the head index, and C2 is one head's contents
        """
        new_x_shape = (x.shape[0], self.num_attention_heads, self.attention_head_size, x.shape[-1])  # [N, C1, C2, W]
        x = x.reshape(new_x_shape)
        return x.transpose((0, 1, 3, 2))  # [N, C1, C2, W] --> [N, C1, W, C2]

    def transpose_key_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, C2, W] where C1 is the head index, and C2 is one head's contents
        """
        new_x_shape = (x.shape[0], self.num_attention_heads, self.attention_head_size, x.shape[-1])  # [N, C1, C2, W]
        x = x.reshape(new_x_shape)
        # no `permute` needed
        return x

    def transpose_output(self, x):
        """
        - input: [N, C1, W, C2]
        - output: [N, C, W]
        """
        x = x.transpose((0, 1, 3, 2))  # [N, C1, C2, W]
        new_x_shape = (x.shape[0], self.all_head_size, x.shape[3])  # [N, C, W]
        x = x.reshape(new_x_shape)
        return x

    def forward(self, hidden_states, attention_mask, output_attentions):
        """
        expects hidden_states in [N, C, W] data layout.
        The attention_mask data layout is [N, W], and it does not need to be transposed.
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_score = self.matmul_qk(query_layer, key_layer)
        attention_score = attention_score / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_score = attention_score + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_score)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = self.matmul_qkv(attention_probs, value_layer)
        context_layer = self.transpose_output(context_layer)

        result = {"context_layer": context_layer}
        if output_attentions:
            result["attention_score"] = attention_score
        return result


class SqueezeBertLayer(nn.Layer):
    def __init__(self, config):
        """
        - hidden_size = input chans = output chans for Q, K, V (they are all the same ... for now) = output chans for
          the module
        - intermediate_size = output chans for intermediate layer
        - groups = number of groups for all layers in the BertLayer. (eventually we could change the interface to
          allow different groups for different layers)
        """
        super().__init__()

        c0 = config.hidden_size
        c1 = config.hidden_size
        c2 = config.intermediate_size
        c3 = config.hidden_size

        self.attention = SqueezeBertSelfAttention(
            config=config, cin=c0, q_groups=config.q_groups, k_groups=config.k_groups, v_groups=config.v_groups
        )
        self.post_attention = ConvDropoutLayerNorm(
            cin=c0, cout=c1, groups=config.post_attention_groups, dropout_prob=config.hidden_dropout_prob
        )
        self.intermediate = ConvActivation(cin=c1, cout=c2, groups=config.intermediate_groups, act=config.hidden_act)
        self.output = ConvDropoutLayerNorm(
            cin=c2, cout=c3, groups=config.output_groups, dropout_prob=config.hidden_dropout_prob
        )

    def forward(self, hidden_states, attention_mask, output_attentions):
        att = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = att["context_layer"]

        post_attention_output = self.post_attention(attention_output, hidden_states)
        intermediate_output = self.intermediate(post_attention_output)
        layer_output = self.output(intermediate_output, post_attention_output)

        output_dict = {"feature_map": layer_output}
        if output_attentions:
            output_dict["attention_score"] = att["attention_score"]

        return output_dict


class SqueezeBertEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()

        assert config.embedding_size == config.hidden_size, (
            "If you want embedding_size != intermediate hidden_size,"
            "please insert a Conv1D layer to adjust the number of channels "
            "before the first SqueezeBertLayer."
        )

        self.layers = nn.LayerList(SqueezeBertLayer(config) for _ in range(config.num_hidden_layers))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):

        if head_mask is None:
            head_mask_is_all_none = True
        elif head_mask.count(None) == len(head_mask):
            head_mask_is_all_none = True
        else:
            head_mask_is_all_none = False
        assert head_mask_is_all_none is True, "head_mask is not yet supported in the SqueezeBert implementation."

        # [batch_size, sequence_length, hidden_size] --> [batch_size, hidden_size, sequence_length]
        hidden_states = hidden_states.transpose((0, 2, 1))

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:

            if output_hidden_states:
                hidden_states = hidden_states.transpose((0, 2, 1))
                all_hidden_states += (hidden_states,)
                hidden_states = hidden_states.transpose((0, 2, 1))

            layer_output = layer.forward(hidden_states, attention_mask, output_attentions)

            hidden_states = layer_output["feature_map"]

            if output_attentions:
                all_attentions += (layer_output["attention_score"],)

        # [batch_size, hidden_size, sequence_length] --> [batch_size, sequence_length, hidden_size]
        hidden_states = hidden_states.transpose((0, 2, 1))

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class SqueezeBertPooler(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SqueezeBertPredictionHeadTransform(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class SqueezeBertLMPredictionHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.transform = SqueezeBertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        self.bias = paddle.create_parameter([config.vocab_size], dtype='float32', is_bias=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SqueezeBertPreTrainingHeads(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.predictions = SqueezeBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class SqueezeBertPreTrainedModel(PretrainedModel):
    base_model_prefix = "squeezebert"
    model__file = "model_json"

    pretrained_init_configuration = {
        "squeezebert-uncased": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "model_type": "squeezebert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 30528,
            "q_groups": 4,
            "k_groups": 4,
            "v_groups": 4,
            "post_attention_groups": 1,
            "intermediate_groups": 4,
            "output_groups": 4,
            "pad_token_id": 0,
            'layer_norm_eps': 1e-12
        },
        "squeezebert-mnli": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "model_type": "squeezebert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 30528,
            "q_groups": 4,
            "k_groups": 4,
            "v_groups": 4,
            "post_attention_groups": 1,
            "intermediate_groups": 4,
            "output_groups": 4,
            "num_labels": 3,
            "pad_token_id": 0,
            'layer_norm_eps': 1e-12
        },
        "squeezebert-mnli-headless": {
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "model_type": "squeezebert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 30528,
            "q_groups": 4,
            "k_groups": 4,
            "v_groups": 4,
            "post_attention_groups": 1,
            "intermediate_groups": 4,
            "output_groups": 4,
            "pad_token_id": 0,
            'layer_norm_eps': 1e-12
        }

    }
    resource_files_names = {"model_state": "model_state.pdparams"}

    pretrained_resource_files_map = {
        "model_state": {
            "squeezebert-uncased":
                "http://paddlenlp.bj.bcebos.com/models/transformers/squeezebert/squeezebert-uncased/model_state.pdparams",
            "squeezebert-mnli":
                "http://paddlenlp.bj.bcebos.com/models/transformers/squeezebert/squeezebert-mnli/model_state.pdparams",
            "squeezebert-mnli-headless":
                "http://paddlenlp.bj.bcebos.com/models/transformers/squeezebert/squeezebert-mnli-headless/model_state.pdparams",
        }
    }

    def init_weights(self):
        """
        Initializes and tie weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)
        # Tie weights if needed
        self.tie_weights()

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        if hasattr(self, "get_output_embeddings") and hasattr(
                self, "get_input_embeddings"):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings,
                                           self.get_input_embeddings())

    def _init_weights(self, layer):
        """Initialize the weights"""
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.squeezebert.cofing["initializer_range"],
                    shape=layer.weight.shape, ))
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.full_like(layer.weight, 1.0))
            layer._epsilon = 1e-12

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone layer weights"""
        if output_embeddings.weight.shape == input_embeddings.weight.shape:
            output_embeddings.weight = input_embeddings.weight
        elif output_embeddings.weight.shape == input_embeddings.weight.t(
        ).shape:
            output_embeddings.weight.set_value(input_embeddings.weight.t())
        else:
            raise ValueError(
                "when tie input/output embeddings, the shape of output embeddings: {}"
                "should be equal to shape of input embeddings: {}"
                "or should be equal to the shape of transpose input embeddings: {}".
                    format(
                    output_embeddings.weight.shape,
                    input_embeddings.weight.shape,
                    input_embeddings.weight.t().shape, ))
        if getattr(output_embeddings, "bias", None) is not None:
            if output_embeddings.weight.shape[
                -1] != output_embeddings.bias.shape[0]:
                raise ValueError(
                    "the weight lase shape: {} of output_embeddings is not equal to the bias shape: {}"
                    "please check output_embeddings uration".format(
                        output_embeddings.weight.shape[-1],
                        output_embeddings.bias.shape[0], ))


@register_base_model
class SqueezeBertModel(SqueezeBertPreTrainedModel):
    def __init__(self, **kwargs):
        super().__init__()
        config = self.config = create_config(kwargs)
        self.embeddings = SqueezeBertEmbeddings(config)
        self.encoder = SqueezeBertEncoder(config)
        self.pooler = SqueezeBertPooler(config)

        # self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
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
            attention_mask = paddle.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        extended_attention_mask = _convert_attention_mask(attention_mask, embedding_output)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return (sequence_output, pooled_output) + encoder_outputs[1:]


class SqueezeBertForPretraining(SqueezeBertPreTrainedModel):

    def __init__(self, squeezebert):
        super().__init__()
        self.squeezebert = squeezebert
        self.initializer_range = self.squeezebert.config['initializer_range']
        self.cls = SqueezeBertPreTrainingHeads(create_config(self.squeezebert.config))
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None):
        with paddle.static.amp.fp16_guard():
            outputs = self.bert(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask)
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_positions)
            return prediction_scores, seq_relationship_score


class SqueezeBertForSequenceClassification(SqueezeBertPreTrainedModel):

    def __init__(self, squeezebert, num_classes=2, dropout=None):
        super().__init__()
        self.num_classes = num_classes
        self.squeezebert = squeezebert

        self.config = self.squeezebert.config
        self.initializer_range = self.config['initializer_range']
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.squeezebert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.squeezebert.config["hidden_size"],
                                    num_classes)
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, pooled_output = self.squeezebert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class SqueezeBertForQuestionAnswering(SqueezeBertPreTrainedModel):

    def __init__(self, squeezebert, dropout=None):
        super().__init__()
        self.squeezebert = squeezebert
        self.config = self.squeezebert.config
        self.initializer_range = self.config['initializer_range']
        self.classifier = nn.Linear(self.squeezebert.config["hidden_size"], 2)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None):
        sequence_output, _ = self.squeezebert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            attention_mask=None)
        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)
        return start_logits, end_logits


class SqueezeBertForTokenClassification(SqueezeBertPreTrainedModel):

    def __init__(self, squeezebert, num_classes=2, dropout=None):
        super().__init__()
        self.num_classes = num_classes
        self.squeezebert = squeezebert
        self.config = self.squeezebert.config
        self.initializer_range = self.config['initializer_range']
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.squeezebert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.squeezebert.config["hidden_size"],
                                    num_classes)
        self.init_weights()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, _ = self.squeezebert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
