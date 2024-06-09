from .constants import *
from .llama.modelings_intervenable_llama import *


#########################################################################
"""
Below are functions that you need to modify if you add
a new model arch type in this library.

We put them in front so it is easier to keep track of
things that need to be changed.
"""

# import paddlenlp.transformers.models as hf_models
import paddlenlp


global type_to_module_mapping
global type_to_dimension_mapping
global output_to_subcomponent_fn_mapping
global scatter_intervention_output_fn_mapping



type_to_module_mapping = {
    paddlenlp.transformers.llama.modeling.LlamaModel: llama_type_to_module_mapping,
    paddlenlp.transformers.llama.modeling.LlamaForCausalLM: llama_lm_type_to_module_mapping,
    # paddlenlp.transformers.llama.modeling.LlamaForSequenceClassification: llama_classifier_type_to_module_mapping,
    # 定义模型文件后添加新的模型类型
}


type_to_dimension_mapping = {
    # hf_models.llama.modeling_llama.LlamaModel: llama_type_to_dimension_mapping,
    # hf_models.llama.modeling_llama.LlamaForCausalLM: llama_lm_type_to_dimension_mapping,
    # hf_models.llama.modeling_llama.LlamaForSequenceClassification: llama_classifier_type_to_dimension_mapping,
    # new model type goes here after defining the model files
    paddlenlp.transformers.llama.modeling.LlamaForCausalLM: llama_lm_type_to_dimension_mapping,
}
#########################################################################
