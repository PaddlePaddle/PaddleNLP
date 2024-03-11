import importlib

class LongSequenceStrategies():
    @classmethod
    def build_long_sequence_strategy(cls,strategy_type=None,stratety_name=None,**init_args):
        '''
        
        **init_args:   head_dim, 
                       max_position_embeddings, 
                       rope_scaling_type, 
                       rope_scaling_factor,
                       ...
        
        strategy_type: "None" ---------------走原始的build-in模块
                       "EmbeddingStrategies"、
                       "AttentionStrategies"、
                       "EmbeddingAttentionMixStrategies"
                       ...
                       
        stratety_name: "RotaryEmbedding"、
                       "LinearScalingRotaryEmbedding"、
                       "NTKScalingRotaryEmbedding"、
                       "DynamicNTKScalingRotaryEmbedding"
                       "AttentionWithLinearBias" 
                       ...
                                       
        '''
  
        ''' 
        paddlenlp.transformers.LongSequenceStrategies.{strategy_type<->import_class)}.{stratety_name<->strategy_class)}
        paddlenlp.transformers.LongSequenceStrategies.{EmbeddingStrategies}.{RoPE,...}
        paddlenlp.transformers.LongSequenceStrategies.{AttentionStrategies}.{ALiBi,...}
        '''   
        try:
            import_class = importlib.import_module(f"paddlenlp.transformers.LongSequenceStrategies.{strategy_type}")
        except ValueError:
            raise ValueError(
                f"Wrong strategy type {strategy_type}."
            )  
        try:
            strategy_class = getattr(import_class, stratety_name)
            strategy_instance = strategy_class(**init_args)
            return strategy_instance
        except AttributeError:
            all_strategy_classes = import_class.__all__
            raise AttributeError(
                f"module '{import_class.__name__}' only supports the following classes: "
                + ", ".join(m for m in all_strategy_classes)
            )     