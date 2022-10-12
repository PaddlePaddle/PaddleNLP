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

import contextlib
from collections import deque
import warnings
import paddle
from ..utils.tools import get_env_device
from ..transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer
from .knowledge_mining import WordTagTask, NPTagTask
from .named_entity_recognition import NERWordTagTask
from .named_entity_recognition import NERLACTask
from .sentiment_analysis import SentaTask, SkepTask
from .lexical_analysis import LacTask
from .word_segmentation import SegJiebaTask
from .word_segmentation import SegLACTask
from .word_segmentation import SegWordTagTask
from .pos_tagging import POSTaggingTask
from .text_generation import TextGenerationTask
from .poetry_generation import PoetryGenerationTask
from .question_answering import QuestionAnsweringTask
from .dependency_parsing import DDParserTask
from .text_correction import CSCTask
from .text_similarity import TextSimilarityTask
from .dialogue import DialogueTask
from .information_extraction import UIETask, GPTask
from .code_generation import CodeGenerationTask
from .text_to_image import TextToImageGenerationTask, TextToImageDiscoDiffusionTask, TextToImageStableDiffusionTask
from .text_summarization import TextSummarizationTask
from .document_intelligence import DocPromptTask

warnings.simplefilter(action='ignore', category=Warning, lineno=0, append=False)

TASKS = {
    'dependency_parsing': {
        "models": {
            "ddparser": {
                "task_class": DDParserTask,
                "task_flag": 'dependency_parsing-biaffine',
            },
            "ddparser-ernie-1.0": {
                "task_class": DDParserTask,
                "task_flag": 'dependency_parsing-ernie-1.0',
            },
            "ddparser-ernie-gram-zh": {
                "task_class": DDParserTask,
                "task_flag": 'dependency_parsing-ernie-gram-zh',
            },
        },
        "default": {
            "model": "ddparser",
        }
    },
    'dialogue': {
        "models": {
            "plato-mini": {
                "task_class": DialogueTask,
                "task_flag": "dialogue-plato-mini"
            },
        },
        "default": {
            "model": "plato-mini",
        }
    },
    "knowledge_mining": {
        "models": {
            "wordtag": {
                "task_class": WordTagTask,
                "task_flag": 'knowledge_mining-wordtag',
                "task_priority_path": "wordtag",
            },
            "nptag": {
                "task_class": NPTagTask,
                "task_flag": 'knowledge_mining-nptag',
            },
        },
        "default": {
            "model": "wordtag",
        }
    },
    "lexical_analysis": {
        "models": {
            "lac": {
                "task_class": LacTask,
                "hidden_size": 128,
                "emb_dim": 128,
                "task_flag": 'lexical_analysis-gru_crf',
                "task_priority_path": "lac",
            }
        },
        "default": {
            "model": "lac"
        }
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
            }
        },
        "default": {
            "mode": "accurate"
        }
    },
    "poetry_generation": {
        "models": {
            "gpt-cpm-large-cn": {
                "task_class": PoetryGenerationTask,
                "task_flag": 'poetry_generation-gpt-cpm-large-cn',
                "task_priority_path": "gpt-cpm-large-cn",
            },
        },
        "default": {
            "model": "gpt-cpm-large-cn",
        }
    },
    "pos_tagging": {
        "models": {
            "lac": {
                "task_class": POSTaggingTask,
                "hidden_size": 128,
                "emb_dim": 128,
                "task_flag": 'pos_tagging-gru_crf',
                "task_priority_path": "lac",
            }
        },
        "default": {
            "model": "lac"
        }
    },
    "question_answering": {
        "models": {
            "gpt-cpm-large-cn": {
                "task_class": QuestionAnsweringTask,
                "task_flag": 'question_answering-gpt-cpm-large-cn',
                "task_priority_path": "gpt-cpm-large-cn",
            },
        },
        "default": {
            "model": "gpt-cpm-large-cn",
        }
    },
    'sentiment_analysis': {
        "models": {
            "bilstm": {
                "task_class": SentaTask,
                "task_flag": 'sentiment_analysis-bilstm',
            },
            "skep_ernie_1.0_large_ch": {
                "task_class": SkepTask,
                "task_flag": 'sentiment_analysis-skep_ernie_1.0_large_ch',
            }
        },
        "default": {
            "model": "bilstm"
        }
    },
    'text_correction': {
        "models": {
            "ernie-csc": {
                "task_class": CSCTask,
                "task_flag": "text_correction-ernie-csc"
            },
        },
        "default": {
            "model": "ernie-csc"
        }
    },
    'text_similarity': {
        "models": {
            "simbert-base-chinese": {
                "task_class": TextSimilarityTask,
                "task_flag": "text_similarity-simbert-base-chinese"
            },
            "rocketqa-zh-dureader-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag":
                'text_similarity-rocketqa-zh-dureader-cross-encoder',
            },
            "rocketqa-base-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag": 'text_similarity-rocketqa-base-cross-encoder',
            },
            "rocketqa-medium-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag": 'text_similarity-rocketqa-medium-cross-encoder',
            },
            "rocketqa-mini-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag": 'text_similarity-rocketqa-mini-cross-encoder',
            },
            "rocketqa-micro-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag": 'text_similarity-rocketqa-micro-cross-encoder',
            },
            "rocketqa-nano-cross-encoder": {
                "task_class": TextSimilarityTask,
                "task_flag": 'text_similarity-rocketqa-nano-cross-encoder',
            },
        },
        "default": {
            "model": "rocketqa-zh-dureader-cross-encoder"
        }
    },
    'text_summarization': {
        "models": {
            "unimo-text-1.0-summary": {
                "task_class": TextSummarizationTask,
                "task_flag": "text_summarization-unimo-text-1.0-summary",
                "task_priority_path": "unimo-text-1.0-summary",
            },
        },
        "default": {
            "model": "unimo-text-1.0-summary"
        }
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
        "default": {
            "mode": "base"
        }
    },
    'information_extraction': {
        "models": {
            "uie-base": {
                "task_class": UIETask,
                "hidden_size": 768,
                "task_flag": "information_extraction-uie-base"
            },
            "uie-medium": {
                "task_class": UIETask,
                "hidden_size": 768,
                "task_flag": "information_extraction-uie-medium"
            },
            "uie-mini": {
                "task_class": UIETask,
                "hidden_size": 384,
                "task_flag": "information_extraction-uie-mini"
            },
            "uie-micro": {
                "task_class": UIETask,
                "hidden_size": 384,
                "task_flag": "information_extraction-uie-micro"
            },
            "uie-nano": {
                "task_class": UIETask,
                "hidden_size": 312,
                "task_flag": "information_extraction-uie-nano"
            },
            "uie-tiny": {
                "task_class": UIETask,
                "hidden_size": 768,
                "task_flag": "information_extraction-uie-tiny"
            },
            "uie-medical-base": {
                "task_class": UIETask,
                "hidden_size": 768,
                "task_flag": "information_extraction-uie-medical-base"
            },
            "uie-base-en": {
                "task_class": UIETask,
                "hidden_size": 768,
                "task_flag": "information_extraction-uie-base-en"
            },
            "uie-m-base": {
                "task_class": UIETask,
                "hidden_size": 768,
                "task_flag": "information_extraction-uie-m-base"
            },
            "uie-m-large": {
                "task_class": UIETask,
                "hidden_size": 1024,
                "task_flag": "information_extraction-uie-m-large"
            },
            "uie-data-distill-gp": {
                "task_class": GPTask,
                "task_flag": "information_extraction-uie-data-distill-gp"
            },
        },
        "default": {
            "model": "uie-base"
        }
    },
    "code_generation": {
        "models": {
            "Salesforce/codegen-350M-mono": {
                "task_class": CodeGenerationTask,
                "task_flag": 'code_generation-Salesforce/codegen-350M-mono',
                "task_priority_path": "Salesforce/codegen-350M-mono",
            },
            "Salesforce/codegen-2B-mono": {
                "task_class": CodeGenerationTask,
                "task_flag": 'code_generation-Salesforce/codegen-2B-mono',
                "task_priority_path": "Salesforce/codegen-2B-mono",
            },
            "Salesforce/codegen-6B-mono": {
                "task_class": CodeGenerationTask,
                "task_flag": 'code_generation-Salesforce/codegen-6B-mono',
                "task_priority_path": "Salesforce/codegen-6B-mono",
            },
            "Salesforce/codegen-350M-nl": {
                "task_class": CodeGenerationTask,
                "task_flag": 'code_generation-Salesforce/codegen-350M-nl',
                "task_priority_path": "Salesforce/codegen-350M-nl",
            },
            "Salesforce/codegen-2B-nl": {
                "task_class": CodeGenerationTask,
                "task_flag": 'code_generation-Salesforce/codegen-2B-nl',
                "task_priority_path": "Salesforce/codegen-2B-nl",
            },
            "Salesforce/codegen-6B-nl": {
                "task_class": CodeGenerationTask,
                "task_flag": 'code_generation-Salesforce/codegen-6B-nl',
                "task_priority_path": "Salesforce/codegen-6B-nl",
            },
            "Salesforce/codegen-350M-multi": {
                "task_class": CodeGenerationTask,
                "task_flag": 'code_generation-Salesforce/codegen-350M-multi',
                "task_priority_path": "Salesforce/codegen-350M-multi",
            },
            "Salesforce/codegen-2B-multi": {
                "task_class": CodeGenerationTask,
                "task_flag": 'code_generation-Salesforce/codegen-2B-multi',
                "task_priority_path": "Salesforce/codegen-2B-multi",
            },
            "Salesforce/codegen-6B-multi": {
                "task_class": CodeGenerationTask,
                "task_flag": 'code_generation-Salesforce/codegen-6B-multi',
                "task_priority_path": "Salesforce/codegen-6B-multi",
            },
        },
        "default": {
            "model": "Salesforce/codegen-350M-mono",
        },
    },
    "text_to_image": {
        "models": {
            "dalle-mini": {
                "task_class": TextToImageGenerationTask,
                "task_flag": "text_to_image-dalle-mini",
                "task_priority_path": "dalle-mini",
            },
            "dalle-mega-v16": {
                "task_class": TextToImageGenerationTask,
                "task_flag": "text_to_image-dalle-mega-v16",
                "task_priority_path": "dalle-mega-v16",
            },
            "dalle-mega": {
                "task_class": TextToImageGenerationTask,
                "task_flag": "text_to_image-dalle-mega",
                "task_priority_path": "dalle-mega",
            },
            "pai-painter-painting-base-zh": {
                "task_class": TextToImageGenerationTask,
                "task_flag": "text_to_image-pai-painter-painting-base-zh",
                "task_priority_path": "pai-painter-painting-base-zh",
            },
            "pai-painter-scenery-base-zh": {
                "task_class": TextToImageGenerationTask,
                "task_flag": "text_to_image-pai-painter-scenery-base-zh",
                "task_priority_path": "pai-painter-scenery-base-zh",
            },
            "pai-painter-commercial-base-zh": {
                "task_class": TextToImageGenerationTask,
                "task_flag": "text_to_image-pai-painter-commercial-base-zh",
                "task_priority_path": "pai-painter-commercial-base-zh",
            },
            "openai/disco-diffusion-clip-vit-base-patch32": {
                "task_class":
                TextToImageDiscoDiffusionTask,
                "task_flag":
                "text_to_image-openai/disco-diffusion-clip-vit-base-patch32",
                "task_priority_path":
                "openai/disco-diffusion-clip-vit-base-patch32",
            },
            "openai/disco-diffusion-clip-rn50": {
                "task_class": TextToImageDiscoDiffusionTask,
                "task_flag": "text_to_image-openai/disco-diffusion-clip-rn50",
                "task_priority_path": "openai/disco-diffusion-clip-rn50",
            },
            "openai/disco-diffusion-clip-rn101": {
                "task_class": TextToImageDiscoDiffusionTask,
                "task_flag": "text_to_image-openai/disco-diffusion-clip-rn101",
                "task_priority_path": "openai/disco-diffusion-clip-rn101",
            },
            "disco_diffusion_ernie_vil-2.0-base-zh": {
                "task_class": TextToImageDiscoDiffusionTask,
                "task_flag":
                "text_to_image-disco_diffusion_ernie_vil-2.0-base-zh",
                "task_priority_path": "disco_diffusion_ernie_vil-2.0-base-zh",
            },
            "CompVis/stable-diffusion-v1-4": {
                "task_class": TextToImageStableDiffusionTask,
                "task_flag": "text_to_image-CompVis/stable-diffusion-v1-4",
                "task_priority_path": "CompVis/stable-diffusion-v1-4",
            },
        },
        "default": {
            "model": "pai-painter-painting-base-zh",
        }
    },
    "document_intelligence": {
        "models": {
            "docprompt": {
                "task_class": DocPromptTask,
                "task_flag": "document_intelligence-docprompt",
            },
        },
        "default": {
            "model": "docprompt"
        }
    },
}

support_schema_list = [
    "uie-base", "uie-medium", "uie-mini", "uie-micro", "uie-nano", "uie-tiny",
    "uie-medical-base", "uie-base-en", "wordtag", "uie-m-large", "uie-m-base"
]

support_argument_list = [
    "dalle-mini", "dalle-mega", "dalle-mega-v16",
    "pai-painter-painting-base-zh", "pai-painter-scenery-base-zh",
    "pai-painter-commercial-base-zh", "CompVis/stable-diffusion-v1-4",
    "openai/disco-diffusion-clip-vit-base-patch32",
    "openai/disco-diffusion-clip-rn50", "openai/disco-diffusion-clip-rn101",
    "disco_diffusion_ernie_vil-2.0-base-zh"
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

    def __init__(self, task, model=None, mode=None, device_id=0, **kwargs):
        assert task in TASKS, "The task name:{} is not in Taskflow list, please check your task name.".format(
            task)
        self.task = task

        if self.task in ["word_segmentation", "ner"]:
            tag = "modes"
            ind_tag = "mode"
            self.model = mode
        else:
            tag = "models"
            ind_tag = "model"
            self.model = model

        if self.model is not None:
            assert self.model in set(TASKS[task][tag].keys(
            )), "The {} name: {} is not in task:[{}]".format(tag, model, task)
        else:
            self.model = TASKS[task]['default'][ind_tag]

        if "task_priority_path" in TASKS[self.task][tag][self.model]:
            self.priority_path = TASKS[self.task][tag][
                self.model]["task_priority_path"]
        else:
            self.priority_path = None

        # Set the device for the task
        device = get_env_device()
        if device == 'cpu' or device_id == -1:
            paddle.set_device('cpu')
        else:
            paddle.set_device(device + ":" + str(device_id))

        # Update the task config to kwargs
        config_kwargs = TASKS[self.task][tag][self.model]
        kwargs['device_id'] = device_id
        kwargs.update(config_kwargs)
        self.kwargs = kwargs
        task_class = TASKS[self.task][tag][self.model]['task_class']
        self.task_instance = task_class(model=self.model,
                                        task=self.task,
                                        priority_path=self.priority_path,
                                        **self.kwargs)
        task_list = TASKS.keys()
        Taskflow.task_list = task_list

    def __call__(self, *inputs):
        """
        The main work function in the taskflow.
        """
        results = self.task_instance(inputs)
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
        assert self.task_instance.model in support_schema_list, 'This method can only be used by the task with the model of uie or wordtag.'
        self.task_instance.set_schema(schema)

    def set_argument(self, argument):
        assert self.task_instance.model in support_argument_list, 'This method can only be used by the task with the model of text_to_image generation.'
        self.task_instance.set_argument(argument)
