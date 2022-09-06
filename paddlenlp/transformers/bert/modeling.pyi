import paddle.nn as nn
import paddle
from ..model_outputs import ModelOutput
from .configuration import BertConfig
from _typeshed import Incomplete
from paddle import Tensor
from paddle.nn import Layer, Embedding, Linear
from paddlenlp.transformers.model_utils import PretrainedModelNew
from typing import Optional, Tuple, Union, overload

class BertEmbeddings(Layer):
    word_embeddings: Embedding
    position_embeddings: Embedding 
    token_type_embeddings: Embedding 
    layer_norm: Layer
    dropout: float
    def __init__(self, config: BertConfig) -> None: ...
    def forward(self, input_ids: Tensor, token_type_ids: Optional[Tensor] = ..., position_ids: Optional[Tensor] = ..., past_key_values_length: Optional[int] = ...): ...

class BertPooler(Layer):
    dense: Linear
    activation: Layer
    pool_act: Layer
    def __init__(self, config: BertConfig) -> None: ...
    def forward(self, hidden_states): ...

class BertPretrainedModel(PretrainedModelNew):
    model_config_file: str
    config_class: Incomplete
    resource_files_names: Incomplete
    base_model_prefix: str
    pretrained_init_configuration: Incomplete
    pretrained_resource_files_map: Incomplete
    def init_weights(self, layer) -> None: ...

class BertModel(BertPretrainedModel):
    pad_token_id: Incomplete
    initializer_range: Incomplete
    embeddings: Incomplete
    fuse: Incomplete
    encoder: Incomplete
    pooler: Incomplete
    
    @overload
    def __init__(self, config: BertConfig) -> None: ...
    @overload
    def __init__(self, *args, **kwargs) -> None: ...

    def get_input_embeddings(self): ...
    def set_input_embeddings(self, value) -> None: ...
    def forward(self, input_ids, token_type_ids: Incomplete | None = ..., position_ids: Incomplete | None = ..., attention_mask: Incomplete | None = ..., past_key_values: Incomplete | None = ..., use_cache: Incomplete | None = ..., output_hidden_states: bool = ..., output_attentions: bool = ..., return_dict: bool = ...): ...

class BertForQuestionAnswering(BertPretrainedModel):
    bert: Incomplete
    dropout: Incomplete
    classifier: Incomplete
    @overload
    def __init__(self, config: BertConfig): ...
    @overload
    def __init__(self, config: BertConfig, dropout: Optional[float] = None): ...
    @overload
    def __init__(self, bert: BertModel, dropout: Optional[float] = None): ...

    def forward(self, input_ids, token_type_ids: Incomplete | None = ..., position_ids: Incomplete | None = ..., attention_mask: Incomplete | None = ..., start_positions: Incomplete | None = ..., end_positions: Incomplete | None = ..., output_hidden_states: bool = ..., output_attentions: bool = ..., return_dict: bool = ...): ...
    def __call__(self, input_ids, token_type_ids: Incomplete | None = ..., position_ids: Incomplete | None = ..., attention_mask: Incomplete | None = ..., start_positions: Incomplete | None = ..., end_positions: Incomplete | None = ..., output_hidden_states: bool = ..., output_attentions: bool = ..., return_dict: bool = ...): ...

class BertForSequenceClassification(BertPretrainedModel):
    bert: Incomplete
    num_labels: Incomplete
    dropout: Incomplete
    classifier: Incomplete
    @overload
    def __init__(self, config: BertConfig): ...
    @overload
    def __init__(self, config: BertConfig, num_labels: Optional[int] = 2, dropout: Optional[float] = None): ...
    @overload
    def __init__(self, bert: BertModel, num_labels: Optional[int] = 2, dropout: Optional[float] = None): ...

    def forward(self, input_ids: Tensor, token_type_ids: Tensor | None = ..., position_ids: Tensor | None = ..., attention_mask: Tensor | None = ..., labels: Tensor | None = ..., output_hidden_states: bool = ..., output_attentions: bool = ..., return_dict: bool = ...): ...
    def __call__(self, input_ids: Tensor, token_type_ids: Tensor | None = ..., position_ids: Tensor | None = ..., attention_mask: Tensor | None = ..., labels: Tensor | None = ..., output_hidden_states: bool = ..., output_attentions: bool = ..., return_dict: bool = ...): ...

class BertForTokenClassification(BertPretrainedModel):
    bert: Incomplete
    num_labels: Incomplete
    dropout: Incomplete
    classifier: Incomplete
    @overload
    def __init__(self, config: BertConfig): ...
    @overload
    def __init__(self, config: BertConfig, num_labels: Optional[int] = 2, dropout: Optional[float] = None): ...
    @overload
    def __init__(self, bert: BertModel, num_labels: Optional[int] = 2, dropout: Optional[float] = None): ...

    def forward(self, input_ids, token_type_ids: Incomplete | None = ..., position_ids: Incomplete | None = ..., attention_mask: Incomplete | None = ..., labels: Incomplete | None = ..., output_hidden_states: bool = ..., output_attentions: bool = ..., return_dict: bool = ...): ...
    def __call__(self, input_ids, token_type_ids: Incomplete | None = ..., position_ids: Incomplete | None = ..., attention_mask: Incomplete | None = ..., labels: Incomplete | None = ..., output_hidden_states: bool = ..., output_attentions: bool = ..., return_dict: bool = ...): ...

class BertLMPredictionHead(Layer):
    transform: Incomplete
    activation: Incomplete
    layer_norm: Incomplete
    decoder_weight: Incomplete
    decoder_bias: Incomplete
    def __init__(self, config: BertConfig, embedding_weights: Incomplete | None = ...) -> None: ...
    def forward(self, hidden_states, masked_positions: Incomplete | None = ...): ...

class BertPretrainingHeads(Layer):
    predictions: Incomplete
    seq_relationship: Incomplete
    def __init__(self, config: BertConfig, embedding_weights: Incomplete | None = ...) -> None: ...
    def forward(self, sequence_output, pooled_output, masked_positions: Incomplete | None = ...): ...

class BertForPreTrainingOutput(ModelOutput):
    loss: Optional[paddle.Tensor]
    prediction_logits: paddle.Tensor
    seq_relationship_logits: paddle.Tensor
    hidden_states: Optional[Tuple[paddle.Tensor]]
    attentions: Optional[Tuple[paddle.Tensor]]
    def __init__(self, loss: Optional[Tensor], prediction_logits: Optional[Tensor], seq_relationship_logits: Optional[Tensor], hidden_states: Optional[Tensor], attentions: Optional[Tensor]) -> None: ...

class BertForPretraining(BertPretrainedModel):
    bert: Incomplete
    cls: Incomplete
    @overload
    def __init__(self, config: BertConfig) -> None: ...
    @overload
    def __init__(self, bert: BertModel) -> None: ...

    def forward(self, input_ids, token_type_ids: Incomplete | None = ..., position_ids: Incomplete | None = ..., attention_mask: Incomplete | None = ..., masked_positions: Incomplete | None = ..., labels: Incomplete | None = ..., next_sentence_label: Incomplete | None = ..., output_hidden_states: bool = ..., output_attentions: bool = ..., return_dict: bool = ...): ...
    def __call__(self, input_ids, token_type_ids: Incomplete | None = ..., position_ids: Incomplete | None = ..., attention_mask: Incomplete | None = ..., masked_positions: Incomplete | None = ..., labels: Incomplete | None = ..., next_sentence_label: Incomplete | None = ..., output_hidden_states: bool = ..., output_attentions: bool = ..., return_dict: bool = ...): ...

class BertPretrainingCriterion(paddle.nn.Layer):
    loss_fn: Incomplete
    vocab_size: Incomplete
    def __init__(self, vocab_size) -> None: ...
    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels, masked_lm_scale): ...
    def __call__(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels, masked_lm_scale): ...

class BertForMultipleChoice(BertPretrainedModel):
    bert: Incomplete
    num_choices: Incomplete
    dropout: Incomplete
    classifier: Incomplete
    @overload
    def __init__(self, config: BertConfig) -> None: ...
    @overload
    def __init__(self, config: BertConfig, num_choices: Optional[int] = 2, dropout: Optional[float] = None) -> None: ...
    @overload
    def __init__(self, bert: BertModel, num_choices: Optional[int] = 2, dropout: Optional[float] = None) -> None: ...

    def forward(self, input_ids, token_type_ids: Incomplete | None = ..., position_ids: Incomplete | None = ..., attention_mask: Incomplete | None = ..., labels: Incomplete | None = ..., output_hidden_states: bool = ..., output_attentions: bool = ..., return_dict: bool = ...): ...

class BertOnlyMLMHead(nn.Layer):
    predictions: Incomplete
    def __init__(self, config: BertConfig, embedding_weights: Incomplete | None = ...) -> None: ...
    def forward(self, sequence_output, masked_positions: Incomplete | None = ...): ...

class BertForMaskedLM(BertPretrainedModel):
    bert: Incomplete
    cls: Incomplete
    @overload
    def __init__(self, config: BertConfig) -> None: ...
    @overload
    def __init__(self, bert: BertModel) -> None: ...

    def forward(self, input_ids, token_type_ids: Incomplete | None = ..., position_ids: Incomplete | None = ..., attention_mask: Incomplete | None = ..., labels: Incomplete | None = ..., output_hidden_states: bool = ..., output_attentions: bool = ..., return_dict: bool = ...): ...
    def __call__(self, input_ids, token_type_ids: Incomplete | None = ..., position_ids: Incomplete | None = ..., attention_mask: Incomplete | None = ..., labels: Incomplete | None = ..., output_hidden_states: bool = ..., output_attentions: bool = ..., return_dict: bool = ...): ...
