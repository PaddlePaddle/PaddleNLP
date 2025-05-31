#
#
#
""" Gemma2 model configuration"""

from paddlenlp.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "GEMMA2_PRETRAINED_INIT_CONFIGURATION", 
    "Gemma2Config",
    "GEMMA2_PRETRAINED_RESOURCE_FILES_MAP",
]

GEMMA2_PRETRAINED_INIT_CONFIGURATION = {
    "google/gemma-2-2b": {
        "architectures": ["Gemma2ForCausalLM"],
        "hidden_size": 2304,
        "initializer_range": 0.02,
        "intermediate_size": 9216,
        "max_position_embeddings": 8192,
        "model_type": "gemma2",
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "num_hidden_layers": 26,
        "rms_norm_eps": 1e-06,
        "vocab_size": 256000,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "use_cache": True,
        "layer_types": ["eager"] * 26,
        "sliding_window": 4096,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "query_pre_attn_scalar": 256,
    },
    "google/gemma-2-9b": {
        "architectures": ["Gemma2ForCausalLM"],
        "hidden_size": 3584,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "model_type": "gemma2",
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_hidden_layers": 40,
        "rms_norm_eps": 1e-06,
        "vocab_size": 256000,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "use_cache": True,
        "layer_types": ["eager"] * 40,
        "sliding_window": 4096,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "query_pre_attn_scalar": 256,
    },
    "google/gemma-2-27b": {
        "architectures": ["Gemma2ForCausalLM"],
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 20480,
        "max_position_embeddings": 8192,
        "model_type": "gemma2",
        "num_attention_heads": 32,
        "num_key_value_heads": 16,
        "num_hidden_layers": 48,
        "rms_norm_eps": 1e-06,
        "vocab_size": 256000,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "use_cache": True,
        "layer_types": ["eager"] * 48,
        "sliding_window": 4096,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "query_pre_attn_scalar": 256,
    },
}

GEMMA2_PRETRAINED_RESOURCE_FILES_MAP = {
    "model_state": {
        "google/gemma-2-2b": "https://bj.bcebos.com/paddlenlp/models/transformers/gemma2/gemma-2-2b/model_state.pdparams",
        "google/gemma-2-9b": "https://bj.bcebos.com/paddlenlp/models/transformers/gemma2/gemma-2-9b/model_state.pdparams",
        "google/gemma-2-27b": "https://bj.bcebos.com/paddlenlp/models/transformers/gemma2/gemma-2-27b/model_state.pdparams",
    }
}

class Gemma2Config(PretrainedConfig):
    """
    Configuration class for Gemma2 model. Extends the PretrainedConfig class and contains all the parameters
    required to initialize the Gemma2 model architecture.
    
    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Gemma2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Gemma2Model`].
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 26):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used.
        head_dim (`int`, *optional*, defaults to 256):
            Dimension of the attention heads.
        hidden_activation (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        seq_length (`int`, *optional*, defaults to 8192):
            The maximum sequence length for training.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        layer_types (`List[str]`, *optional*):
            The type of each layer in the decoder. If None, defaults to ["eager"] * num_hidden_layers.
        sliding_window (`int`, *optional*, defaults to 4096):
            The size of the sliding window attention.
        attn_logit_softcapping (`float`, *optional*, defaults to 50.0):
            The soft capping value for attention logits.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0):
            The soft capping value for final logits.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256):
            The scalar to apply to the query before attention.
    """
    model_type = "gemma2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        seq_length=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        layer_types=None,
        sliding_window=4096,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=30.0,
        query_pre_attn_scalar=256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_activation = hidden_activation
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        
        if layer_types is None:
            layer_types = ["eager"] * num_hidden_layers
        self.layer_types = layer_types
        self.sliding_window = sliding_window
        self.attn_logit_softcapping = attn_logit_softcapping
        self.final_logit_softcapping = final_logit_softcapping
        self.query_pre_attn_scalar = query_pre_attn_scalar
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
