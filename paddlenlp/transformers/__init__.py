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

from .model_utils import PretrainedModel, register_base_model
from .tokenizer_utils import PretrainedTokenizer, BPETokenizer, tokenize_chinese_chars, is_chinese_char, AddedToken, normalize_chars, tokenize_special_chars, convert_to_unicode
from .attention_utils import create_bigbird_rand_mask_idx_list
from .export import export_model

from .bert.modeling import *
from .bert.tokenizer import *
from .gpt.modeling import *
from .gpt.tokenizer import *
from .roberta.modeling import *
from .roberta.tokenizer import *
from .electra.modeling import *
from .electra.tokenizer import *
from .albert.modeling import *
from .albert.tokenizer import *
from .bart.modeling import *
from .bart.tokenizer import *
from .bert_japanese.tokenizer import *
from .bigbird.modeling import *
from .bigbird.tokenizer import *
from .blenderbot.modeling import *
from .blenderbot.tokenizer import *
from .blenderbot_small.modeling import *
from .blenderbot_small.tokenizer import *
from .chinesebert.modeling import *
from .chinesebert.tokenizer import *
from .convbert.modeling import *
from .convbert.tokenizer import *
from .ctrl.modeling import *
from .ctrl.tokenizer import *
from .distilbert.modeling import *
from .distilbert.tokenizer import *
from .ernie.modeling import *
from .ernie.tokenizer import *
from .ernie_ctm.modeling import *
from .ernie_ctm.tokenizer import *
from .ernie_doc.modeling import *
from .ernie_doc.tokenizer import *
from .ernie_gen.modeling import ErnieForGeneration
from .ernie_gram.modeling import *
from .ernie_gram.tokenizer import *
from .ernie_m.modeling import *
from .ernie_m.tokenizer import *
from .fnet.modeling import *
from .fnet.tokenizer import *
from .funnel.modeling import *
from .funnel.tokenizer import *
from .layoutlm.modeling import *
from .layoutlm.tokenizer import *
from .layoutlmv2.modeling import *
from .layoutlmv2.tokenizer import *
from .layoutxlm.modeling import *
from .layoutxlm.tokenizer import *
from .luke.modeling import *
from .luke.tokenizer import *
from .mbart.modeling import *
from .mbart.tokenizer import *
from .megatronbert.modeling import *
from .megatronbert.tokenizer import *
from .prophetnet.modeling import *
from .prophetnet.tokenizer import *
from .mobilebert.modeling import *
from .mobilebert.tokenizer import *
from .mpnet.modeling import *
from .mpnet.tokenizer import *
from .nezha.modeling import *
from .nezha.tokenizer import *
from .ppminilm.modeling import *
from .ppminilm.tokenizer import *
from .prophetnet.modeling import *
from .prophetnet.tokenizer import *
from .reformer.modeling import *
from .reformer.tokenizer import *
from .rembert.modeling import *
from .rembert.tokenizer import *
from .roformer.modeling import *
from .roformer.tokenizer import *
from .semantic_search.modeling import *
from .skep.modeling import *
from .skep.tokenizer import *
from .squeezebert.modeling import *
from .squeezebert.tokenizer import *
from .t5.modeling import *
from .t5.tokenizer import *
from .tinybert.modeling import *
from .tinybert.tokenizer import *
from .transformer.modeling import *
from .unified_transformer.modeling import *
from .unified_transformer.tokenizer import *
from .unimo.modeling import *
from .unimo.tokenizer import *
from .xlnet.modeling import *
from .xlnet.tokenizer import *
from .xlm.modeling import *
from .xlm.tokenizer import *
from .gau_alpha.modeling import *
from .gau_alpha.tokenizer import *
from .roformerv2.modeling import *
from .roformerv2.tokenizer import *
from .optimization import *
from .auto.modeling import *
from .auto.tokenizer import *
from .codegen.modeling import *
from .codegen.tokenizer import *

# For faster tokenizer
from ..utils.import_utils import is_faster_tokenizer_available
if is_faster_tokenizer_available():
    from .bert.faster_tokenizer import *
    from .ernie.faster_tokenizer import *
    from .tinybert.faster_tokenizer import *
