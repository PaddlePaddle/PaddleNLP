# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


from .configuration_utils import PretrainedConfig
from .model_utils import PretrainedModel, register_base_model
from .tokenizer_utils import (
    PretrainedTokenizer,
    BPETokenizer,
    tokenize_chinese_chars,
    is_chinese_char,
    AddedToken,
    normalize_chars,
    tokenize_special_chars,
    convert_to_unicode,
)
from .processing_utils import ProcessorMixin
from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .image_processing_utils import ImageProcessingMixin
from .attention_utils import create_bigbird_rand_mask_idx_list
from .sequence_parallel_utils import (
    GatherOp,
    ScatterOp,
    AllGatherOp,
    ReduceScatterOp,
    ColumnSequenceParallelLinear,
    RowSequenceParallelLinear,
    mark_as_sequence_parallel_parameter,
    register_sequence_parallel_allreduce_hooks,
)
from .export import export_model

# isort: split
from .bert.modeling import *
from .bert.tokenizer import *
from .bert.configuration import *

# isort: split
from .gpt.modeling import *
from .gpt.tokenizer import *
from .gpt.configuration import *
from .roberta.modeling import *
from .roberta.tokenizer import *
from .roberta.configuration import *
from .electra.modeling import *
from .electra.tokenizer import *
from .electra.configuration import *
from .albert.configuration import *
from .albert.modeling import *
from .albert.tokenizer import *
from .bit.modeling import *
from .bit.configuration import *
from .bit.image_processing import *
from .bart.modeling import *
from .bart.tokenizer import *
from .bart.configuration import *
from .bert_japanese.tokenizer import *
from .bigbird.modeling import *
from .bigbird.configuration import *
from .bigbird.tokenizer import *
from .blenderbot.modeling import *
from .blenderbot.tokenizer import *
from .blenderbot.configuration import *
from .blenderbot_small.modeling import *
from .blenderbot_small.tokenizer import *
from .blenderbot_small.configuration import *
from .blip.modeling import *
from .blip.modeling_text import *
from .blip.configuration import *
from .blip.processing import *
from .blip.image_processing import *
from .chinesebert.configuration import *
from .chinesebert.modeling import *
from .chinesebert.tokenizer import *
from .convbert.configuration import *
from .convbert.modeling import *
from .convbert.tokenizer import *
from .ctrl.modeling import *
from .ctrl.tokenizer import *
from .ctrl.configuration import *
from .dpt.modeling import *
from .dpt.configuration import *
from .dpt.image_processing import *
from .distilbert.configuration import *
from .distilbert.modeling import *
from .distilbert.tokenizer import *
from .ernie.configuration import *
from .ernie.modeling import *
from .ernie.tokenizer import *
from .ernie_ctm.modeling import *
from .ernie_ctm.tokenizer import *
from .ernie_ctm.configuration import *
from .ernie_doc.modeling import *
from .ernie_doc.tokenizer import *
from .ernie_doc.configuration import *
from .ernie_gen.modeling import ErnieForGeneration
from .ernie_gram.modeling import *
from .ernie_gram.tokenizer import *
from .ernie_gram.configuration import *
from .ernie_layout.modeling import *
from .ernie_layout.tokenizer import *
from .ernie_layout.configuration import *
from .ernie_m.configuration import *
from .ernie_m.modeling import *
from .ernie_m.tokenizer import *
from .fnet.modeling import *
from .fnet.tokenizer import *
from .fnet.configuration import *
from .funnel.modeling import *
from .funnel.tokenizer import *
from .funnel.configuration import *
from .llama.configuration import *
from .llama.modeling import *
from .llama.tokenizer import *
from .layoutlm.configuration import *
from .layoutlm.modeling import *
from .layoutlm.tokenizer import *
from .layoutlmv2.modeling import *
from .layoutlmv2.tokenizer import *
from .layoutlmv2.configuration import *
from .layoutxlm.modeling import *
from .layoutxlm.tokenizer import *
from .layoutxlm.configuration import *
from .luke.modeling import *
from .luke.tokenizer import *
from .luke.configuration import *
from .mbart.modeling import *
from .mbart.tokenizer import *
from .mbart.configuration import *
from .megatronbert.modeling import *
from .megatronbert.tokenizer import *
from .megatronbert.configuration import *
from .prophetnet.modeling import *
from .prophetnet.tokenizer import *
from .prophetnet.configuration import *
from .mobilebert.configuration import *
from .mobilebert.modeling import *
from .mobilebert.tokenizer import *
from .mpnet.configuration import *
from .mpnet.modeling import *
from .mpnet.tokenizer import *
from .mt5.configuration import *
from .mt5.modeling import *
from .nezha.configuration import *
from .nezha.modeling import *
from .nezha.tokenizer import *
from .ppminilm.modeling import *
from .ppminilm.tokenizer import *
from .reformer.modeling import *
from .reformer.tokenizer import *
from .reformer.configuration import *
from .rembert.modeling import *
from .rembert.tokenizer import *
from .rembert.configuration import *
from .roformer.modeling import *
from .roformer.configuration import *
from .roformer.tokenizer import *
from .semantic_search.modeling import *
from .skep.configuration import *
from .skep.modeling import *
from .skep.tokenizer import *
from .squeezebert.modeling import *
from .squeezebert.tokenizer import *
from .squeezebert.configuration import *
from .t5.modeling import *
from .t5.tokenizer import *
from .t5.configuration import *
from .tinybert.configuration import *
from .tinybert.modeling import *
from .tinybert.tokenizer import *
from .transformer.modeling import *
from .unified_transformer.modeling import *
from .unified_transformer.tokenizer import *
from .unified_transformer.configuration import *
from .ernie_code.tokenizer import *
from .ernie_code.modeling import *
from .ernie_code.configuration import *
from .ernie_vil.configuration import *
from .ernie_vil.modeling import *
from .ernie_vil.feature_extraction import *
from .ernie_vil.tokenizer import *
from .ernie_vil.processing import *
from .ernie_vil.image_processing import *
from .unimo.modeling import *
from .unimo.tokenizer import *
from .unimo.configuration import *
from .xlnet.modeling import *
from .xlnet.tokenizer import *
from .xlnet.configuration import *
from .xlm.modeling import *
from .xlm.tokenizer import *
from .xlm.configuration import *
from .gau_alpha.modeling import *
from .gau_alpha.tokenizer import *
from .gau_alpha.configuration import *
from .roformerv2.modeling import *
from .roformerv2.tokenizer import *
from .roformerv2.configuration import *
from .optimization import *
from .opt.configuration import *
from .opt.modeling import *
from .auto.modeling import *
from .auto.tokenizer import *
from .auto.processing import *
from .auto.configuration import *
from .codegen.modeling import *
from .codegen.tokenizer import *
from .codegen.configuration import *
from .artist.modeling import *
from .artist.tokenizer import *
from .artist.configuration import *
from .dallebart.modeling import *
from .dallebart.tokenizer import *
from .dallebart.configuration import *
from .clip.modeling import *
from .clip.configuration import *
from .clip.feature_extraction import *
from .clip.tokenizer import *
from .clip.processing import *
from .clip.image_processing import *
from .chineseclip.modeling import *
from .chineseclip.configuration import *
from .chineseclip.feature_extraction import *
from .chineseclip.processing import *
from .chineseclip.image_processing import *
from .chineseclip.tokenizer import *
from .gptj.modeling import *
from .gptj.tokenizer import *
from .gptj.configuration import *
from .pegasus.modeling import *
from .pegasus.tokenizer import *
from .pegasus.configuration import *
from .glm.configuration import *
from .glm.modeling import *
from .glm.tokenizer import *
from .nystromformer.configuration import *
from .nystromformer.modeling import *
from .nystromformer.tokenizer import *
from .bloom.configuration import *
from .bloom.modeling import *
from .bloom.tokenizer import *
from .clipseg.configuration import *
from .clipseg.modeling import *
from .clipseg.processing import *
from .clipseg.image_processing import *
from .blip_2.modeling import *
from .blip_2.configuration import *
from .blip_2.processing import *
from .chatglm.configuration import *
from .chatglm.modeling import *
from .chatglm.tokenizer import *
from .chatglm_v2.configuration import *
from .chatglm_v2.modeling import *
from .chatglm_v2.tokenizer import *
from .speecht5.configuration import *
from .speecht5.modeling import *
from .speecht5.tokenizer import *
from .speecht5.processing import *
from .speecht5.feature_extraction import *
from .minigpt4.modeling import *
from .minigpt4.configuration import *
from .minigpt4.processing import *
from .minigpt4.image_processing import *
from .clap.configuration import *
from .clap.feature_extraction import *
from .clap.modeling import *
from .clap.processing import *
from .visualglm.modeling import *
from .visualglm.configuration import *
from .visualglm.processing import *
from .visualglm.image_processing import *
from .rw.modeling import *
from .rw.configuration import *
from .rw.tokenizer import *
from .qwen.modeling import *
from .qwen.configuration import *
from .qwen.tokenizer import *

# For faster tokenizer
from ..utils.import_utils import is_fast_tokenizer_available

if is_fast_tokenizer_available():
    from .tokenizer_utils_fast import PretrainedFastTokenizer
    from .bert.fast_tokenizer import *
    from .ernie.fast_tokenizer import *
    from .tinybert.fast_tokenizer import *
    from .ernie_m.fast_tokenizer import *
    from .nystromformer.fast_tokenizer import *
