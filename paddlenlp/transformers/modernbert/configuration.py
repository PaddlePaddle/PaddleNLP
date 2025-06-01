#
#
#

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = ["ModernBertConfig"]

MODERNBERT_PRETRAINED_INIT_CONFIGURATION = {
    "modernbert-base": {
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "geglu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 8192,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "sliding_window_size": 512,
        "tie_word_embeddings": True,
        "tensor_parallel_degree": 1,
        "rope_theta": 10000.0,
    },
    "modernbert-large": {
        "vocab_size": 30522,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "geglu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 8192,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "sliding_window_size": 512,
        "tie_word_embeddings": True,
        "tensor_parallel_degree": 1,
        "rope_theta": 10000.0,
    },
}

MODERNBERT_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "modernbert-base": "https://bj.bcebos.com/paddlenlp/models/transformers/modernbert/modernbert-base.pdparams",
        "modernbert-large": "https://bj.bcebos.com/paddlenlp/models/transformers/modernbert/modernbert-large.pdparams",
    }
}


class ModernBertConfig(PretrainedConfig):
    model_type = "modernbert"
    attribute_map: dict = {"dropout": "classifier_dropout", "num_classes": "num_labels"}
    pretrained_init_configuration = MODERNBERT_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "geglu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 8192,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        sliding_window_size: int = 512,
        tie_word_embeddings: bool = True,
        tensor_parallel_degree: int = 1,
        rope_theta: float = 10000.0,
        use_cache: bool = False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.sliding_window_size = sliding_window_size
        self.tie_word_embeddings = tie_word_embeddings
        self.tensor_parallel_degree = tensor_parallel_degree
        self.rope_theta = rope_theta
        self.use_cache = use_cache
