# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import threading

import paddle

from ..utils.tools import get_env_device
from .code_generation import CodeGenerationTask
from .dependency_parsing import DDParserTask
from .dialogue import DialogueTask
from .document_intelligence import DocPromptTask
from .fill_mask import FillMaskTask
from .information_extraction import GPTask, UIETask
from .knowledge_mining import NPTagTask, WordTagTask
from .lexical_analysis import LacTask
from .multimodal_feature_extraction import MultimodalFeatureExtractionTask
from .named_entity_recognition import NERLACTask, NERWordTagTask
from .poetry_generation import PoetryGenerationTask
from .pos_tagging import POSTaggingTask
from .question_answering import QuestionAnsweringTask
from .question_generation import QuestionGenerationTask
from .sentiment_analysis import SentaTask, SkepTask, UIESentaTask
from .text2text_generation import ChatGLMTask
from .text_classification import TextClassificationTask
from .text_correction import CSCTask
from .text_feature_extraction import (
    SentenceFeatureExtractionTask,
    TextFeatureExtractionTask,
)
from .text_similarity import TextSimilarityTask
from .text_summarization import TextSummarizationTask
from .word_segmentation import SegJiebaTask, SegLACTask, SegWordTagTask
from .zero_shot_text_classification import ZeroShotTextClassificationTask

TASKS = {
    "dependency_parsing": {
        "models": {
            "ddparser": {
                "task_class": DDParserTask,
                "task_flag": "dependency_parsing-biaffine",
            },
            "ddparser-ernie-1.0": {
                "task_class": DDParserTask,
                "task_flag": "dependency_parsing-ernie-1.0",
            },
            "ddparser-ernie-gram-zh": {
                "task_class": DDParserTask,
                "task_flag": "dependency_parsing-ernie-gram-zh",
            },
        },
        "default": {
            "model": "ddparser",
        },
    },
    "dialogue": {
        "models": {
            "plato-mini": {"task_class": DialogueTask, "task_flag": "dialogue-plato-mini"},
            "__internal_testing__/tiny-random-plato": {
                "task_class": DialogueTask,
                "task_flag": "dialogue-tiny-random-plato",
            },
        },
        "default": {
            "model": "plato-mini",
        },
    },
    "fill_mask": {
        "models": {
            "fill_mask": {"task_class": FillMaskTask, "task_flag": "fill_mask-fill_mask"},
        },
        "default": {
            "model": "fill_mask",
        },
    },
    "knowledge_mining": {
        "models": {
            "wordtag": {
                "task_class": WordTagTask,
                "task_flag": "knowledge_mining-wordtag",
                "task_priority_path": "wordtag",
            },
            "nptag": {
                "task_class": NPTagTask,
                "task_flag": "knowledge_mining-nptag",
            },
        },
        "default": {
            "model": "wordtag",
        },
    },
    "lexical_analysis": {
        "models": {
            "lac": {
                "task_class": LacTask,
                "hidden_size": 128,
                "emb_dim": 128,
                "task_flag": "lexical_analysis-gru_crf",
                "task_priority_path": "lac",
            }
        },
        "default": {"model": "lac"},
    },
    "ner": {
        "modes": {
            "accurate": {
                "task_class": NERWordTagTask,
                "task_flag": "ner-wordtag",
                "task_priority_path": "wordtag",
                "linking": False,
            },
            "fast": {
                "task_class": NERLACTask,
                "hidden_size": 128,
                "emb_dim": 128,
                "task_flag": "ner-lac",
                "task_priority_path": "lac",
            },
        },
        "default": {"mode": "accurate"},
    },
    "poetry_generation": {
        "models": {
            "gpt-cpm-large-cn": {
                "task_class": PoetryGenerationTask,
                "task_flag": "poetry_generation-gpt-cpm-large-cn",
                "task_priority_path": "gpt-cpm-large-cn",
            },
        },
        "default": {
            "model": "gpt-cpm-large-cn",
        },
    },
    "pos_tagging": {
        "models": {
            "lac": {
                "task_class": POSTaggingTask,
                "hidden_size": 128,
                "emb_dim": 128,
                "task_flag": "pos_tagging-gru_crf",
                "task_priority_path": "lac",
            }
        },
        "default": {"model": "lac"},
    },
    "question_answering": {
        "models": {
            "gpt-cpm-large-cn": {
                "task_class": QuestionAnsweringTask,
                "task_flag": "question_answering-gpt-cpm-large-cn",
                "task_priority_path": "gpt-cpm-large-cn",
            },
        },
        "default": {
            "model": "gpt-cpm-large-cn",
        },
    },
    "sentiment_analysis": {
        "models": {
            "bilstm": {
                "task_class": SentaTask,
                "task_flag": "sentiment_analysis-bilstm",
            },
            "skep_ernie_1.0_large_ch": {
                "task_class": SkepTask,
                "task_flag": "sentiment_analysis-skep_ernie_1.0_large_ch",
            },
            "uie-senta-base": {
                "task_class": UIESentaTask,
                "task_flag": "sentiment_analysis-uie-senta-base",
            },
            "uie-senta-medium": {
                "task_class": UIESentaTask,
                "task_flag": "sentiment_analysis-uie-senta-medium",
            },
            "uie-senta-mini": {
                "task_class": UIESentaTask,
                "task_flag": "sentiment_analysis-uie-senta-mini",
            },
            "uie-senta-micro": {
                "task_class": UIESentaTask,
                "task_flag": "sentiment_analysis-uie-senta-micro",
            },
            "uie-senta-nano": {
                "task_class": UIESentaTask,
                "task_flag": "sentiment_analysis-uie-senta-nano",
            },
            "__internal_testing__/tiny-random-skep": {
                "task_class": SkepTask,
                "task_flag": "sentiment_analysis-tiny-random-skep",
            },
        },
        "default": {"model": "bilstm"},
    },
    "text_correction": {
        "models": {
            "ernie-csc": {"task_class": CSCTask, "task_flag": "text_correction-ernie-csc"},
        },
        "default": {"model": "ernie-csc"},
    },
    "text_similarity": {
        "models": {
            "simbert-base-chinese": {
                "task_class": TextSimilarityTask,
                "task_flag": "text_similarity-simbert-base-chinese",
            },
            "rocketqa-zh-dureader-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag": "text_similarity-rocketqa-zh-dureader-cross-encoder",
            },
            "rocketqa-base-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag": "text_similarity-rocketqa-base-cross-encoder",
            },
            "rocketqa-medium-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag": "text_similarity-rocketqa-medium-cross-encoder",
            },
            "rocketqa-mini-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag": "text_similarity-rocketqa-mini-cross-encoder",
            },
            "rocketqa-micro-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag": "text_similarity-rocketqa-micro-cross-encoder",
            },
            "rocketqa-nano-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag": "text_similarity-rocketqa-nano-cross-encoder",
            },
            "rocketqav2-en-marco-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag": "text_similarity-rocketqav2-en-marco-cross-encoder",
            },
            "ernie-search-large-cross-encoder-marco-en": {
                "task_class": TextSimilarityTask,
                "task_flag": "text_similarity-ernie-search-large-cross-encoder-marco-en",
            },
            "__internal_testing__/tiny-random-bert": {
                "task_class": TextSimilarityTask,
                "task_flag": "text_similarity-tiny-random-bert",
            },
        },
        "default": {"model": "simbert-base-chinese"},
    },
    "text_summarization": {
        "models": {
            "unimo-text-1.0-summary": {
                "task_class": TextSummarizationTask,
                "task_flag": "text_summarization-unimo-text-1.0-summary",
                "task_priority_path": "unimo-text-1.0-summary",
            },
            "IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese": {
                "task_class": TextSummarizationTask,
                "task_flag": "text_summarization-IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese",
                "task_priority_path": "IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese",
            },
            "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese": {
                "task_class": TextSummarizationTask,
                "task_flag": "text_summarization-IDEA-CCNL/Randeng-Pegasus523M-Summary-Chinese",
                "task_priority_path": "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese",
            },
            "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1": {
                "task_class": TextSummarizationTask,
                "task_flag": "text_summarization-IDEA-CCNL/Randeng-Pegasus523M-Summary-Chinese-V1",
                "task_priority_path": "IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1",
            },
            "PaddlePaddle/Randeng-Pegasus-238M-Summary-Chinese-SSTIA": {
                "task_class": TextSummarizationTask,
                "task_flag": "text_summarization-PaddlePaddle/Randeng-Pegasus-238M-Summary-Chinese-SSTIA",
                "task_priority_path": "PaddlePaddle/Randeng-Pegasus-238M-Summary-Chinese-SSTIA",
            },
            "PaddlePaddle/Randeng-Pegasus-523M-Summary-Chinese-SSTIA": {
                "task_class": TextSummarizationTask,
                "task_flag": "text_summarization-PaddlePaddle/Randeng-Pegasus-523M-Summary-Chinese-SSTIA",
                "task_priority_path": "PaddlePaddle/Randeng-Pegasus-523M-Summary-Chinese-SSTIA",
            },
        },
        "default": {"model": "PaddlePaddle/Randeng-Pegasus-523M-Summary-Chinese-SSTIA"},
    },
    "word_segmentation": {
        "modes": {
            "fast": {
                "task_class": SegJiebaTask,
                "task_flag": "word_segmentation-jieba",
            },
            "base": {
                "task_class": SegLACTask,
                "hidden_size": 128,
                "emb_dim": 128,
                "task_flag": "word_segmentation-gru_crf",
                "task_priority_path": "lac",
            },
            "accurate": {
                "task_class": SegWordTagTask,
                "task_flag": "word_segmentation-wordtag",
                "task_priority_path": "wordtag",
                "linking": False,
            },
        },
        "default": {"mode": "base"},
    },
    "information_extraction": {
        "models": {
            "uie-base": {"task_class": UIETask, "hidden_size": 768, "task_flag": "information_extraction-uie-base"},
            "uie-medium": {
                "task_class": UIETask,
                "hidden_size": 768,
                "task_flag": "information_extraction-uie-medium",
            },
            "uie-mini": {"task_class": UIETask, "hidden_size": 384, "task_flag": "information_extraction-uie-mini"},
            "uie-micro": {"task_class": UIETask, "hidden_size": 384, "task_flag": "information_extraction-uie-micro"},
            "uie-nano": {"task_class": UIETask, "hidden_size": 312, "task_flag": "information_extraction-uie-nano"},
            "uie-tiny": {"task_class": UIETask, "hidden_size": 768, "task_flag": "information_extraction-uie-tiny"},
            "uie-medical-base": {
                "task_class": UIETask,
                "hidden_size": 768,
                "task_flag": "information_extraction-uie-medical-base",
            },
            "uie-base-en": {
                "task_class": UIETask,
                "hidden_size": 768,
                "task_flag": "information_extraction-uie-base-en",
            },
            "uie-m-base": {
                "task_class": UIETask,
                "hidden_size": 768,
                "task_flag": "information_extraction-uie-m-base",
            },
            "uie-m-large": {
                "task_class": UIETask,
                "hidden_size": 1024,
                "task_flag": "information_extraction-uie-m-large",
            },
            "uie-x-base": {
                "task_class": UIETask,
                "hidden_size": 768,
                "task_flag": "information_extraction-uie-x-base",
            },
            "uie-data-distill-gp": {"task_class": GPTask, "task_flag": "information_extraction-uie-data-distill-gp"},
            "__internal_testing__/tiny-random-uie": {
                "task_class": UIETask,
                "hidden_size": 8,
                "task_flag": "information_extraction-tiny-random-uie",
            },
            "__internal_testing__/tiny-random-uie-m": {
                "task_class": UIETask,
                "hidden_size": 8,
                "task_flag": "information_extraction-tiny-random-uie-m",
            },
            "__internal_testing__/tiny-random-uie-x": {
                "task_class": UIETask,
                "hidden_size": 8,
                "task_flag": "information_extraction-tiny-random-uie-x",
            },
        },
        "default": {"model": "uie-base"},
    },
    "code_generation": {
        "models": {
            "Salesforce/codegen-350M-mono": {
                "task_class": CodeGenerationTask,
                "task_flag": "code_generation-Salesforce/codegen-350M-mono",
                "task_priority_path": "Salesforce/codegen-350M-mono",
            },
            "Salesforce/codegen-2B-mono": {
                "task_class": CodeGenerationTask,
                "task_flag": "code_generation-Salesforce/codegen-2B-mono",
                "task_priority_path": "Salesforce/codegen-2B-mono",
            },
            "Salesforce/codegen-6B-mono": {
                "task_class": CodeGenerationTask,
                "task_flag": "code_generation-Salesforce/codegen-6B-mono",
                "task_priority_path": "Salesforce/codegen-6B-mono",
            },
            "Salesforce/codegen-350M-nl": {
                "task_class": CodeGenerationTask,
                "task_flag": "code_generation-Salesforce/codegen-350M-nl",
                "task_priority_path": "Salesforce/codegen-350M-nl",
            },
            "Salesforce/codegen-2B-nl": {
                "task_class": CodeGenerationTask,
                "task_flag": "code_generation-Salesforce/codegen-2B-nl",
                "task_priority_path": "Salesforce/codegen-2B-nl",
            },
            "Salesforce/codegen-6B-nl": {
                "task_class": CodeGenerationTask,
                "task_flag": "code_generation-Salesforce/codegen-6B-nl",
                "task_priority_path": "Salesforce/codegen-6B-nl",
            },
            "Salesforce/codegen-350M-multi": {
                "task_class": CodeGenerationTask,
                "task_flag": "code_generation-Salesforce/codegen-350M-multi",
                "task_priority_path": "Salesforce/codegen-350M-multi",
            },
            "Salesforce/codegen-2B-multi": {
                "task_class": CodeGenerationTask,
                "task_flag": "code_generation-Salesforce/codegen-2B-multi",
                "task_priority_path": "Salesforce/codegen-2B-multi",
            },
            "Salesforce/codegen-6B-multi": {
                "task_class": CodeGenerationTask,
                "task_flag": "code_generation-Salesforce/codegen-6B-multi",
                "task_priority_path": "Salesforce/codegen-6B-multi",
            },
        },
        "default": {
            "model": "Salesforce/codegen-350M-mono",
        },
    },
    "text_classification": {
        "modes": {
            "finetune": {
                "task_class": TextClassificationTask,
                "task_flag": "text_classification-finetune",
            },
            "prompt": {
                "task_class": TextClassificationTask,
                "task_flag": "text_classification-prompt",
            },
        },
        "default": {"mode": "finetune"},
    },
    "document_intelligence": {
        "models": {
            "docprompt": {
                "task_class": DocPromptTask,
                "task_flag": "document_intelligence-docprompt",
            },
        },
        "default": {"model": "docprompt"},
    },
    "question_generation": {
        "models": {
            "unimo-text-1.0": {
                "task_class": QuestionGenerationTask,
                "task_flag": "question_generation-unimo-text-1.0",
            },
            "unimo-text-1.0-dureader_qg": {
                "task_class": QuestionGenerationTask,
                "task_flag": "question_generation-unimo-text-1.0-dureader_qg",
            },
            "unimo-text-1.0-question-generation": {
                "task_class": QuestionGenerationTask,
                "task_flag": "question_generation-unimo-text-1.0-question-generation",
            },
            "unimo-text-1.0-question-generation-dureader_qg": {
                "task_class": QuestionGenerationTask,
                "task_flag": "question_generation-unimo-text-1.0-question-generation-dureader_qg",
            },
        },
        "default": {"model": "unimo-text-1.0-dureader_qg"},
    },
    "text2text_generation": {
        "models": {
            "THUDM/chatglm-6b": {
                "task_class": ChatGLMTask,
                "task_flag": "text_generation-THUDM/chatglm-6b",
            },
            "THUDM/chatglm2-6b": {
                "task_class": ChatGLMTask,
                "task_flag": "text_generation-THUDM/chatglm2-6b",
            },
            "__internal_testing__/tiny-random-chatglm": {
                "task_class": ChatGLMTask,
                "task_flag": "text_generation-tiny-random-chatglm",
            },
            "THUDM/chatglm-6b-v1.1": {
                "task_class": ChatGLMTask,
                "task_flag": "text_generation-THUDM/chatglm-6b-v1.1",
            },
        },
        "default": {"model": "THUDM/chatglm-6b-v1.1"},
    },
    "zero_shot_text_classification": {
        "models": {
            "utc-large": {
                "task_class": ZeroShotTextClassificationTask,
                "task_flag": "zero_shot_text_classification-utc-large",
            },
            "utc-xbase": {
                "task_class": ZeroShotTextClassificationTask,
                "task_flag": "zero_shot_text_classification-utc-xbase",
            },
            "utc-base": {
                "task_class": ZeroShotTextClassificationTask,
                "task_flag": "zero_shot_text_classification-utc-base",
            },
            "utc-medium": {
                "task_class": ZeroShotTextClassificationTask,
                "task_flag": "zero_shot_text_classification-utc-medium",
            },
            "utc-micro": {
                "task_class": ZeroShotTextClassificationTask,
                "task_flag": "zero_shot_text_classification-utc-micro",
            },
            "utc-mini": {
                "task_class": ZeroShotTextClassificationTask,
                "task_flag": "zero_shot_text_classification-utc-mini",
            },
            "utc-nano": {
                "task_class": ZeroShotTextClassificationTask,
                "task_flag": "zero_shot_text_classification-utc-nano",
            },
            "utc-pico": {
                "task_class": ZeroShotTextClassificationTask,
                "task_flag": "zero_shot_text_classification-utc-pico",
            },
            "__internal_testing__/tiny-random-utc": {
                "task_class": ZeroShotTextClassificationTask,
                "task_flag": "zero_shot_text_classification-tiny-random-utc",
            },
        },
        "default": {"model": "utc-base"},
    },
    "feature_extraction": {
        "models": {
            "rocketqa-zh-dureader-query-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqa-zh-dureader-query-encoder",
                "task_priority_path": "rocketqa-zh-dureader-query-encoder",
            },
            "rocketqa-zh-dureader-para-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqa-zh-dureader-para-encoder",
                "task_priority_path": "rocketqa-rocketqa-zh-dureader-para-encoder",
            },
            "rocketqa-zh-base-query-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqa-zh-base-query-encoder",
                "task_priority_path": "rocketqa-zh-base-query-encoder",
            },
            "rocketqa-zh-base-para-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqa-zh-base-para-encoder",
                "task_priority_path": "rocketqa-zh-base-para-encoder",
            },
            "rocketqa-zh-medium-query-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqa-zh-medium-query-encoder",
                "task_priority_path": "rocketqa-zh-medium-query-encoder",
            },
            "rocketqa-zh-medium-para-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqa-zh-medium-para-encoder",
                "task_priority_path": "rocketqa-zh-medium-para-encoder",
            },
            "rocketqa-zh-mini-query-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqa-zh-mini-query-encoder",
                "task_priority_path": "rocketqa-zh-mini-query-encoder",
            },
            "rocketqa-zh-mini-para-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqa-rocketqa-zh-mini-para-encoder",
                "task_priority_path": "rocketqa-zh-mini-para-encoder",
            },
            "rocketqa-zh-micro-query-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqa-zh-micro-query-encoder",
                "task_priority_path": "rocketqa-zh-micro-query-encoder",
            },
            "rocketqa-zh-micro-para-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqa-zh-micro-para-encoder",
                "task_priority_path": "rocketqa-zh-micro-para-encoder",
            },
            "rocketqa-zh-nano-query-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqa-zh-nano-query-encoder",
                "task_priority_path": "rocketqa-zh-nano-query-encoder",
            },
            "rocketqa-zh-nano-para-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqa-zh-nano-para-encoder",
                "task_priority_path": "rocketqa-zh-nano-para-encoder",
            },
            "rocketqav2-en-marco-query-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqav2-en-marco-query-encoder",
                "task_priority_path": "rocketqav2-en-marco-query-encoder",
            },
            "rocketqav2-en-marco-para-encoder": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-rocketqav2-en-marco-para-encoder",
                "task_priority_path": "rocketqav2-en-marco-para-encoder",
            },
            "ernie-search-base-dual-encoder-marco-en": {
                "task_class": TextFeatureExtractionTask,
                "task_flag": "feature_extraction-ernie-search-base-dual-encoder-marco-en",
                "task_priority_path": "ernie-search-base-dual-encoder-marco-en",
            },
            "PaddlePaddle/ernie_vil-2.0-base-zh": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-PaddlePaddle/ernie_vil-2.0-base-zh",
                "task_priority_path": "PaddlePaddle/ernie_vil-2.0-base-zh",
            },
            "OFA-Sys/chinese-clip-vit-base-patch16": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-OFA-Sys/chinese-clip-vit-base-patch16",
                "task_priority_path": "OFA-Sys/chinese-clip-vit-base-patch16",
            },
            "OFA-Sys/chinese-clip-vit-huge-patch14": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-OFA-Sys/chinese-clip-vit-huge-patch14",
                "task_priority_path": "OFA-Sys/chinese-clip-vit-huge-patch14",
            },
            "OFA-Sys/chinese-clip-vit-large-patch14": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-OFA-Sys/chinese-clip-vit-large-patch14",
                "task_priority_path": "OFA-Sys/chinese-clip-vit-large-patch14",
            },
            "OFA-Sys/chinese-clip-vit-large-patch14-336px": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-OFA-Sys/chinese-clip-vit-large-patch14-336px",
                "task_priority_path": "OFA-Sys/chinese-clip-vit-large-patch14-336px",
            },
            "openai/clip-vit-base-patch32": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-openai/clip-vit-base-patch32",
                "task_priority_path": "openai/clip-vit-base-patch32",
            },
            "openai/clip-vit-base-patch16": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-openai/clip-vit-base-patch16",
                "task_priority_path": "openai/clip-vit-base-patch16",
            },
            "openai/clip-vit-large-patch14": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-openai/clip-vit-large-patch14",
                "task_priority_path": "openai/clip-vit-large-patch14",
            },
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                "task_priority_path": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            },
            "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                "task_priority_path": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            },
            "openai/clip-rn50": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-openai/clip-rn50",
                "task_priority_path": "openai/clip-rn50",
            },
            "openai/clip-rn101": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-openai/clip-rn101",
                "task_priority_path": "openai/clip-rn101",
            },
            "openai/clip-rn50x4": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-openai/clip-rn50x4",
                "task_priority_path": "openai/clip-rn50x4",
            },
            "__internal_testing__/tiny-random-ernievil2": {
                "task_class": MultimodalFeatureExtractionTask,
                "task_flag": "feature_extraction-tiny-random-ernievil2",
                "task_priority_path": "__internal_testing__/tiny-random-ernievil2",
            },
            "moka-ai/m3e-base": {
                "task_class": SentenceFeatureExtractionTask,
                "task_flag": "feature_extraction-moka-ai/m3e-base",
                "task_priority_path": "moka-ai/m3e-base",
            },
            "__internal_testing__/tiny-random-m3e": {
                "task_class": SentenceFeatureExtractionTask,
                "task_flag": "__internal_testing__/tiny-random-m3e",
                "task_priority_path": "__internal_testing__/tiny-random-m3e",
            },
        },
        "default": {"model": "PaddlePaddle/ernie_vil-2.0-base-zh"},
    },
}

support_schema_list = [
    "uie-base",
    "uie-medium",
    "uie-mini",
    "uie-micro",
    "uie-nano",
    "uie-tiny",
    "uie-medical-base",
    "uie-base-en",
    "wordtag",
    "uie-m-large",
    "uie-m-base",
    "uie-x-base",
    "uie-senta-base",
    "uie-senta-medium",
    "uie-senta-mini",
    "uie-senta-micro",
    "uie-senta-nano",
    "utc-large",
    "utc-xbase",
    "utc-base",
    "utc-medium",
    "utc-micro",
    "utc-mini",
    "utc-nano",
    "utc-pico",
    "utc-tiny",
    "__internal_testing__/tiny-random-uie",
    "__internal_testing__/tiny-random-uie-m",
    "__internal_testing__/tiny-random-uie-x",
]

support_argument_list = [
    "dalle-mini",
    "dalle-mega",
    "dalle-mega-v16",
    "pai-painter-painting-base-zh",
    "pai-painter-scenery-base-zh",
    "pai-painter-commercial-base-zh",
    "CompVis/stable-diffusion-v1-4",
    "openai/disco-diffusion-clip-vit-base-patch32",
    "openai/disco-diffusion-clip-rn50",
    "openai/disco-diffusion-clip-rn101",
    "PaddlePaddle/disco_diffusion_ernie_vil-2.0-base-zh",
    "uie-base",
    "uie-medium",
    "uie-mini",
    "uie-micro",
    "uie-nano",
    "uie-tiny",
    "uie-medical-base",
    "uie-base-en",
    "uie-m-large",
    "uie-m-base",
    "uie-x-base",
    "__internal_testing__/tiny-random-uie-m",
    "__internal_testing__/tiny-random-uie-x",
    "THUDM/chatglm-6b",
    "THUDM/chatglm2-6b",
    "THUDM/chatglm-6b-v1.1",
]


class Taskflow(object):
    """
    The Taskflow is the end2end interface that could convert the raw text to model result, and decode the model result to task result. The main functions as follows:
        1) Convert the raw text to task result.
        2) Convert the model to the inference model.
        3) Offer the usage and help message.
    Args:
        task (str): The task name for the Taskflow, and get the task class from the name.
        model (str, optional): The model name in the task, if set None, will use the default model.
        mode (str, optional): Select the mode of the task, only used in the tasks of word_segmentation and ner.
            If set None, will use the default mode.
        device_id (int, optional): The device id for the gpu, xpu and other devices, the defalut value is 0.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.

    """

    def __init__(self, task, model=None, mode=None, device_id=0, from_hf_hub=False, **kwargs):
        assert task in TASKS, f"The task name:{task} is not in Taskflow list, please check your task name."
        self.task = task
        # Set the device for the task
        device = get_env_device()
        if device == "cpu" or device_id == -1:
            paddle.set_device("cpu")
        else:
            paddle.set_device(device + ":" + str(device_id))

        if self.task in ["word_segmentation", "ner", "text_classification"]:
            tag = "modes"
            ind_tag = "mode"
            self.model = mode
        else:
            tag = "models"
            ind_tag = "model"
            self.model = model

        if self.model is not None:
            assert self.model in set(TASKS[task][tag].keys()), f"The {tag} name: {model} is not in task:[{task}]"
        else:
            self.model = TASKS[task]["default"][ind_tag]

        if "task_priority_path" in TASKS[self.task][tag][self.model]:
            self.priority_path = TASKS[self.task][tag][self.model]["task_priority_path"]
        else:
            self.priority_path = None

        # Update the task config to kwargs
        config_kwargs = TASKS[self.task][tag][self.model]
        kwargs["device_id"] = device_id
        kwargs.update(config_kwargs)
        self.kwargs = kwargs
        task_class = TASKS[self.task][tag][self.model]["task_class"]
        self.task_instance = task_class(
            model=self.model, task=self.task, priority_path=self.priority_path, from_hf_hub=from_hf_hub, **self.kwargs
        )
        task_list = TASKS.keys()
        Taskflow.task_list = task_list

        # Add the lock for the concurrency requests
        self._lock = threading.Lock()

    def __call__(self, *inputs, **kwargs):
        """
        The main work function in the taskflow.
        """
        results = self.task_instance(inputs, **kwargs)
        return results

    def help(self):
        """
        Return the task usage message.
        """
        return self.task_instance.help()

    def task_path(self):
        """
        Return the path of current task
        """
        return self.task_instance._task_path

    @staticmethod
    def tasks():
        """
        Return the available task list.
        """
        task_list = list(TASKS.keys())
        return task_list

    def from_segments(self, *inputs):
        results = self.task_instance.from_segments(inputs)
        return results

    def interactive_mode(self, max_turn):
        with self.task_instance.interactive_mode(max_turn):
            while True:
                human = input("[Human]:").strip()
                if human.lower() == "exit":
                    exit()
                robot = self.task_instance(human)[0]
                print("[Bot]:%s" % robot)

    def set_schema(self, schema):
        assert (
            self.task_instance.model in support_schema_list
        ), "This method can only be used by the task based on the model of uie or wordtag."
        self.task_instance.set_schema(schema)

    def set_argument(self, argument):
        assert self.task_instance.model in support_argument_list, (
            "This method can only be used by the task of text-to-image generation, information extraction "
            "or zero-text-classification."
        )
        self.task_instance.set_argument(argument)
