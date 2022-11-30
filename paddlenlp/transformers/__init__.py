# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from .attention_utils import create_bigbird_rand_mask_idx_list  # noqa: F401
from .export import export_model  # noqa: F401
from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin  # noqa: F401
from .model_utils import PretrainedModel, register_base_model  # noqa: F401
from .processing_utils import ProcessorMixin  # noqa: F401
from .tokenizer_utils import AddedToken  # noqa: F401
from .tokenizer_utils import BPETokenizer  # noqa: F401
from .tokenizer_utils import PretrainedTokenizer  # noqa: F401
from .tokenizer_utils import convert_to_unicode  # noqa: F401
from .tokenizer_utils import is_chinese_char  # noqa: F401
from .tokenizer_utils import normalize_chars  # noqa: F401
from .tokenizer_utils import tokenize_chinese_chars  # noqa: F401
from .tokenizer_utils import tokenize_special_chars  # noqa: F401

# isort: split

from .bert.configuration import *  # noqa: F401
from .bert.converter import *  # noqa: F401
from .bert.modeling import *  # noqa: F401
from .bert.tokenizer import *  # noqa: F401

# isort: split

from .albert.modeling import *  # noqa: F401
from .albert.tokenizer import *  # noqa: F401
from .artist.modeling import *  # noqa: F401
from .artist.tokenizer import *  # noqa: F401
from .auto.modeling import *  # noqa: F401
from .auto.tokenizer import *  # noqa: F401
from .bart.modeling import *  # noqa: F401
from .bart.tokenizer import *  # noqa: F401
from .bert_japanese.tokenizer import *  # noqa: F401
from .bigbird.modeling import *  # noqa: F401
from .bigbird.tokenizer import *  # noqa: F401
from .blenderbot.modeling import *  # noqa: F401
from .blenderbot.tokenizer import *  # noqa: F401
from .blenderbot_small.modeling import *  # noqa: F401
from .blenderbot_small.tokenizer import *  # noqa: F401
from .chinesebert.modeling import *  # noqa: F401
from .chinesebert.tokenizer import *  # noqa: F401
from .clip.converter import *  # noqa: F401
from .clip.feature_extraction import *  # noqa: F401
from .clip.modeling import *  # noqa: F401
from .clip.procesing import *  # noqa: F401
from .clip.tokenizer import *  # noqa: F401
from .codegen.modeling import *  # noqa: F401
from .codegen.tokenizer import *  # noqa: F401
from .convbert.modeling import *  # noqa: F401
from .convbert.tokenizer import *  # noqa: F401
from .ctrl.modeling import *  # noqa: F401
from .ctrl.tokenizer import *  # noqa: F401
from .dallebart.modeling import *  # noqa: F401
from .dallebart.tokenizer import *  # noqa: F401
from .distilbert.modeling import *  # noqa: F401
from .distilbert.tokenizer import *  # noqa: F401
from .electra.converter import *  # noqa: F401
from .electra.modeling import *  # noqa: F401
from .electra.tokenizer import *  # noqa: F401
from .ernie.modeling import *  # noqa: F401
from .ernie.tokenizer import *  # noqa: F401
from .ernie_ctm.modeling import *  # noqa: F401
from .ernie_ctm.tokenizer import *  # noqa: F401
from .ernie_doc.modeling import *  # noqa: F401
from .ernie_doc.tokenizer import *  # noqa: F401
from .ernie_gen.modeling import ErnieForGeneration  # noqa: F401
from .ernie_gram.modeling import *  # noqa: F401
from .ernie_gram.tokenizer import *  # noqa: F401
from .ernie_layout.modeling import *  # noqa: F401
from .ernie_layout.tokenizer import *  # noqa: F401
from .ernie_m.modeling import *  # noqa: F401
from .ernie_m.tokenizer import *  # noqa: F401
from .ernie_vil.feature_extraction import *  # noqa: F401
from .ernie_vil.modeling import *  # noqa: F401
from .ernie_vil.procesing import *  # noqa: F401
from .ernie_vil.tokenizer import *  # noqa: F401
from .fnet.modeling import *  # noqa: F401
from .fnet.tokenizer import *  # noqa: F401
from .funnel.modeling import *  # noqa: F401
from .funnel.tokenizer import *  # noqa: F401
from .gau_alpha.modeling import *  # noqa: F401
from .gau_alpha.tokenizer import *  # noqa: F401
from .gpt.modeling import *  # noqa: F401
from .gpt.tokenizer import *  # noqa: F401
from .gptj.modeling import *  # noqa: F401
from .gptj.tokenizer import *  # noqa: F401
from .layoutlm.modeling import *  # noqa: F401
from .layoutlm.tokenizer import *  # noqa: F401
from .layoutlmv2.modeling import *  # noqa: F401
from .layoutlmv2.tokenizer import *  # noqa: F401
from .layoutxlm.modeling import *  # noqa: F401
from .layoutxlm.tokenizer import *  # noqa: F401
from .luke.modeling import *  # noqa: F401
from .luke.tokenizer import *  # noqa: F401
from .mbart.modeling import *  # noqa: F401
from .mbart.tokenizer import *  # noqa: F401
from .megatronbert.modeling import *  # noqa: F401
from .megatronbert.tokenizer import *  # noqa: F401
from .mobilebert.modeling import *  # noqa: F401
from .mobilebert.tokenizer import *  # noqa: F401
from .mpnet.modeling import *  # noqa: F401
from .mpnet.tokenizer import *  # noqa: F401
from .nezha.modeling import *  # noqa: F401
from .nezha.tokenizer import *  # noqa: F401
from .opt.modeling import *  # noqa: F401
from .optimization import *  # noqa: F401
from .pegasus.modeling import *  # noqa: F401
from .pegasus.tokenizer import *  # noqa: F401
from .ppminilm.modeling import *  # noqa: F401
from .ppminilm.tokenizer import *  # noqa: F401
from .prophetnet.modeling import *  # noqa: F401
from .prophetnet.tokenizer import *  # noqa: F401
from .reformer.modeling import *  # noqa: F401
from .reformer.tokenizer import *  # noqa: F401
from .rembert.modeling import *  # noqa: F401
from .rembert.tokenizer import *  # noqa: F401
from .roberta.configuration import *  # noqa: F401
from .roberta.converter import *  # noqa: F401
from .roberta.modeling import *  # noqa: F401
from .roberta.tokenizer import *  # noqa: F401
from .roformer.modeling import *  # noqa: F401
from .roformer.tokenizer import *  # noqa: F401
from .roformerv2.modeling import *  # noqa: F401
from .roformerv2.tokenizer import *  # noqa: F401
from .semantic_search.modeling import *  # noqa: F401
from .skep.modeling import *  # noqa: F401
from .skep.tokenizer import *  # noqa: F401
from .squeezebert.modeling import *  # noqa: F401
from .squeezebert.tokenizer import *  # noqa: F401
from .t5.modeling import *  # noqa: F401
from .t5.tokenizer import *  # noqa: F401
from .tinybert.modeling import *  # noqa: F401
from .tinybert.tokenizer import *  # noqa: F401
from .transformer.modeling import *  # noqa: F401
from .unified_transformer.modeling import *  # noqa: F401
from .unified_transformer.tokenizer import *  # noqa: F401
from .unimo.modeling import *  # noqa: F401
from .unimo.tokenizer import *  # noqa: F401
from .xlm.modeling import *  # noqa: F401
from .xlm.tokenizer import *  # noqa: F401
from .xlnet.converter import *  # noqa: F401
from .xlnet.modeling import *  # noqa: F401
from .xlnet.tokenizer import *  # noqa: F401

# isort: split

# For faster tokenizer
from ..utils.import_utils import is_fast_tokenizer_available  # noqa: F401

if is_fast_tokenizer_available():
    from .bert.fast_tokenizer import *  # noqa: F401
    from .ernie.fast_tokenizer import *  # noqa: F401
    from .ernie_m.fast_tokenizer import *  # noqa: F401
    from .tinybert.fast_tokenizer import *  # noqa: F401
