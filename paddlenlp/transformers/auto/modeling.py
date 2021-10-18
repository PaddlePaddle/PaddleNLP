# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
"""Modeling classes for ALBERT model."""

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Layer

from .. import PretrainedModel, register_base_model

__all__ = [
    "AutoModel",
    "AutoModelForPretraining",
    "AutoModelForCausalLM",
    "AutoModelForMaskedLM",
    "AutoModelForSequenceClassification",
    "AutoModelForTokenClassification",
    "AutoModelForQuestionAnswering",
    "AutoModelForMultipleChoice",
    "AutoModelForNextSentencePrediction",
]
"""
__all__ = [
    "AlbertPretrainedModel",
    "AlbertModel",
    "AlbertForPretraining",
    "AlbertForMaskedLM",
    "AlbertForSequenceClassification",
    "AlbertForTokenClassification",
    "AlbertForMultipleChoice",
]

__all__ = [
    'BartModel', 'BartPretrainedModel', 'BartEncoder', 'BartDecoder',
    'BartClassificationHead', 'BartForSequenceClassification',
    'BartForQuestionAnswering', 'BartForConditionalGeneration'
]

__all__ = [
    'BertModel',
    "BertPretrainedModel",
    'BertForPretraining',
    'BertPretrainingCriterion',
    'BertPretrainingHeads',
    'BertForSequenceClassification',
    'BertForTokenClassification',
    'BertForQuestionAnswering',
]

__all__ = [
    'BigBirdModel',
    'BigBirdPretrainedModel',
    'BigBirdForPretraining',
    'BigBirdPretrainingCriterion',
    'BigBirdForSequenceClassification',
    'BigBirdPretrainingHeads',
]

__all__ = [
    "ConvBertModel", "ConvBertPretrainedModel", "ConvBertForTotalPretraining",
    "ConvBertDiscriminator", "ConvBertGenerator", "ConvBertClassificationHead",
    "ConvBertForSequenceClassification", "ConvBertForTokenClassification",
    "ConvBertPretrainingCriterion", "ConvBertForQuestionAnswering",
    "ConvBertForMultipleChoice"
]

__all__ = [
    'DistilBertModel',
    'DistilBertPretrainedModel',
    'DistilBertForSequenceClassification',
    'DistilBertForTokenClassification',
    'DistilBertForQuestionAnswering',
    'DistilBertForMaskedLM',
]

__all__ = [
    'ElectraModel', 'ElectraPretrainedModel', 'ElectraForTotalPretraining',
    'ElectraDiscriminator', 'ElectraGenerator', 'ElectraClassificationHead',
    'ElectraForSequenceClassification', 'ElectraForTokenClassification',
    'ElectraPretrainingCriterion'
]

__all__ = [
    'ErnieModel', 'ErniePretrainedModel', 'ErnieForSequenceClassification',
    'ErnieForTokenClassification', 'ErnieForQuestionAnswering',
    'ErnieForPretraining', 'ErniePretrainingCriterion'
]

__all__ = [
    'ErnieCtmPretrainedModel', 'ErnieCtmModel', 'ErnieCtmWordtagModel',
    'ErnieCtmForTokenClassification'
]

__all__ = [
    'ErnieDocModel',
    'ErnieDocPretrainedModel',
    'ErnieDocForSequenceClassification',
    'ErnieDocForTokenClassification',
    'ErnieDocForQuestionAnswering',
]

__all__ = ["ErnieGenPretrainedModel", "ErnieForGeneration"]

__all__ = [
    'ErnieGramModel',
    'ErnieGramForSequenceClassification',
    'ErnieGramForTokenClassification',
    'ErnieGramForQuestionAnswering',
]

__all__ = [
    'GPTModel',
    "GPTPretrainedModel",
    'GPTForPretraining',
    'GPTPretrainingCriterion',
    'GPTForGreedyGeneration',
    'GPTLMHeadModel',
]

__all__ = [
    "MPNetModel",
    "MPNetPretrainedModel",
    "MPNetForMaskedLM",
    "MPNetForSequenceClassification",
    "MPNetForMultipleChoice",
    "MPNetForTokenClassification",
    "MPNetForQuestionAnswering",
]

__all__ = [
    'NeZhaModel', "NeZhaPretrainedModel", 'NeZhaForPretraining',
    'NeZhaForSequenceClassification', 'NeZhaPretrainingHeads',
    'NeZhaForTokenClassification', 'NeZhaForQuestionAnswering',
    'NeZhaForMultipleChoice'
]

__all__ = [
    'RobertaModel',
    'RobertaPretrainedModel',
    'RobertaForSequenceClassification',
    'RobertaForTokenClassification',
    'RobertaForQuestionAnswering',
]

__all__ = [
    "RoFormerModel",
    "RoFormerPretrainedModel",
    "RoFormerForPretraining",
    "RoFormerPretrainingCriterion",
    "RoFormerPretrainingHeads",
    "RoFormerForSequenceClassification",
    "RoFormerForTokenClassification",
    "RoFormerForQuestionAnswering",
]

__all__ = [
    'SkepModel', 'SkepPretrainedModel', 'SkepForSequenceClassification',
    'SkepForTokenClassification', 'SkepCrfForTokenClassification'
]

__all__ = [
    'TinyBertModel',
    'TinyBertPretrainedModel',
    'TinyBertForPretraining',
    'TinyBertForSequenceClassification',
]

__all__ = [
    "UNIMOPretrainedModel",
    'UNIMOModel',
    'UNIMOLMHeadModel',
]

__all__ = [
    "XLNetPretrainedModel",
    "XLNetModel",
    "XLNetForSequenceClassification",
    "XLNetForTokenClassification",
]

"""
MODEL_MAPPING_NAMES = OrderedDict([
    # Base model mapping
    ("albert", "AlbertModel"),
    ("bart", "BartModel"),
    ("bert", "BertModel"),
    ("bigbird", "BigBirdModel"),
    ("convbert", "ConvBertModel"),
    ("distilbert", "DistilBertModel"),
    ("electra", "ElectraModel"),
    ("ernie", "ErnieModel"),
    ("ernie-ctm", "ErnieCtmModel"),
    ("ernie-doc", "ErnieDocModel"),
    ("ernie-gen", "ErnieForGeneration"),
    ("ernie-gram", "ErnieGramModel"),
    ("gpt", "GPTModel"),
    ("mpnet", "MPNetModel"),
    ("nezha", "NeZhaModel"),
    ("roberta", "RobertaModel"),
    ("roformer", "RoFormerModel"),
    ("skep", "SkepModel"),
    ("tinybert", "TinyBertModel"),
    ("unimo", "UNIMOModel"),
    ("xlnet", "XLNetModel"),
])

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict([
    # Model for pre-training mapping
    ("albert", "AlbertForPreTraining"),
    ("bart", "BartForConditionalGeneration"),
    ("bert", "BertForPreTraining"),
    ("bigbird", "BigBirdForPreTraining"),
    ("convbert", "ConvBertForTotalPretraining"),
    ("electra", "ElectraForTotalPreTraining"),
    ("ernie", "ErnieForPreTraining"),
    ("gpt", "GPTForPretraining"),
    ("nezha", "NeZhaForPretraining"),
    ("roformer", "RoformerForPretraining"),
    ("tinybert", "TinyBertForPretraining"),
])

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict([
    # Model with LM heads mapping
    ("albert", "AlbertForMaskedLM"),
    ("bart", "BartForConditionalGeneration"),
    ("bert", "BertPretrainingHeads"),
    ("bigbird", "BigBirdPretrainingHeads"),
    ("convbert", "ConvBertClassificationHead"),
    ("distilbert", "DistilBertForMaskedLM"),
    ("electra", "ElectraClassificationHead"),
    ("gpt", "GPTLMHeadModel"),
    ("mpnet", "MPNetForMaskedLM"),
    ("nezha", "NeZhaPretrainingHeads"),
    ("roformer", "RoFormerPretrainingHeads"),
    ("unimo", "UNIMOLMHeadModel"),
])

MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict([
    # Model for Masked LM mapping
    ("albert", "AlbertForMaskedLM"),
    ("bart", "BartForConditionalGeneration"),
    ("distilbert", "DistilBertForMaskedLM"),
    ("electra", "ElectraForMaskedLM"),
    ("mpnet", "MPNetForMaskedLM"),
    ("roberta", "RobertaForMaskedLM"),
])

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict([
    # Model for Sequence Classification mapping
    ("albert", "AlbertForSequenceClassification"),
    ("bart", "BartForSequenceClassification"),
    ("bert", "BertForSequenceClassification"),
    ("bigbird", "BigBirdForSequenceClassification"),
    ("convbert", "ConvBertForSequenceClassification"),
    ("distilbert", "DistilBertForSequenceClassification"),
    ("electra", "ElectraForSequenceClassification"),
    ("ernie", "ErnieForSequenceClassification"),
    ("ernie-doc", "ErnieDocForSequenceClassification"),
    ("ernie-gram", "ErnieGramForSequenceClassification"),
    ("gpt", "GPTForSequenceClassification"),
    ("mpnet", "MPNetForSequenceClassification"),
    ("nezha", "NeZhaForSequenceClassification"),
    ("roberta", "RobertaForSequenceClassification"),
    ("roformer", "RoFormerForSequenceClassification"),
    ("skep", "SkepForSequenceClassification"),
    ("xlnet", "XLNetForSequenceClassification"),
])

MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict([
    # Model for Question Answering mapping
    ("bart", "BartForQuestionAnswering"),
    ("bert", "BertForQuestionAnswering"),
    ("convbert", "ConvBertForQuestionAnswering"),
    ("distilbert", "DistilBertForQuestionAnswering"),
    ("ernie", "ErnieForQuestionAnswering"),
    ("ernie-doc", "ErnieDocForQuestionAnswering"),
    ("ernie-gram", "ErnieGramForQuestionAnswering"),
    ("mpnet", "MPNetForQuestionAnswering"),
    ("nezha", "NeZhaForQuestionAnswering"),
    ("roberta", "RobertaForQuestionAnswering"),
    ("roformer", "RoFormerForQuestionAnswering"),
])

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict([
    # Model for Token Classification mapping
    ("albert", "AlbertForTokenClassification"),
    ("bert", "BertForTokenClassification"),
    ("bigbird", "BigBirdForTokenClassification"),
    ("convbert", "ConvBertForTokenClassification"),
    ("distilbert", "DistilBertForTokenClassification"),
    ("electra", "ElectraForTokenClassification"),
    ("ernie", "ErnieForTokenClassification"),
    ("ernie-ctm", "ErnieCtmForTokenClassification"),
    ("ernie-doc", "ErnieDocForTokenClassification"),
    ("ernie-gram", "ErnieGramForTokenClassification"),
    ("mpnet", "MPNetForTokenClassification"),
    ("nezha", "NeZhaForTokenClassification"),
    ("roberta", "RobertaForTokenClassification"),
    ("roformer", "RoformerForTokenClassification"),
    ("skep", "SkepForTokenClassification"),
    ("xlnet", "XlnetForTokenClassification"),
])

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict([
    # Model for Multiple Choice mapping
    ("albert", "AlbertForMultipleChoice"),
    ("convbert", "ConvbertForMultipleChoice"),
    ("mpnet", "MPNetForMultipleChoice"),
    ("nezha", "NeZhaForMultipleChoice"),
])

MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)

MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)

MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES,
                                              MODEL_WITH_LM_HEAD_MAPPING_NAMES)

MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)

MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES)

MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES)

MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES)


class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


AutoModel = auto_class_update(AutoModel)


class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


AutoModelForPreTraining = auto_class_update(
    AutoModelForPreTraining, head_doc="pretraining")


# Private on purpose, the public class will add the deprecation warnings.
class _AutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = MODEL_WITH_LM_HEAD_MAPPING


_AutoModelWithLMHead = auto_class_update(
    _AutoModelWithLMHead, head_doc="language modeling")


class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING


AutoModelForCausalLM = auto_class_update(
    AutoModelForCausalLM, head_doc="causal language modeling")


class AutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING


AutoModelForMaskedLM = auto_class_update(
    AutoModelForMaskedLM, head_doc="masked language modeling")


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


AutoModelForSeq2SeqLM = auto_class_update(
    AutoModelForSeq2SeqLM,
    head_doc="sequence-to-sequence language modeling",
    checkpoint_for_example="t5-base")


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


AutoModelForSequenceClassification = auto_class_update(
    AutoModelForSequenceClassification, head_doc="sequence classification")


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING


AutoModelForQuestionAnswering = auto_class_update(
    AutoModelForQuestionAnswering, head_doc="question answering")


class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING


AutoModelForTableQuestionAnswering = auto_class_update(
    AutoModelForTableQuestionAnswering,
    head_doc="table question answering",
    checkpoint_for_example="google/tapas-base-finetuned-wtq", )


class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


AutoModelForTokenClassification = auto_class_update(
    AutoModelForTokenClassification, head_doc="token classification")


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING


AutoModelForMultipleChoice = auto_class_update(
    AutoModelForMultipleChoice, head_doc="multiple choice")


class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


AutoModelForNextSentencePrediction = auto_class_update(
    AutoModelForNextSentencePrediction, head_doc="next sentence prediction")


class AutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


AutoModelForImageClassification = auto_class_update(
    AutoModelForImageClassification, head_doc="image classification")


class AutoModelForImageSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING


AutoModelForImageSegmentation = auto_class_update(
    AutoModelForImageSegmentation, head_doc="image segmentation")


class AutoModelForObjectDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING


AutoModelForObjectDetection = auto_class_update(
    AutoModelForObjectDetection, head_doc="object detection")


class AutoModelForAudioClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING


AutoModelForAudioClassification = auto_class_update(
    AutoModelForAudioClassification, head_doc="audio classification")


class AutoModelForCTC(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CTC_MAPPING


AutoModelForCTC = auto_class_update(
    AutoModelForCTC, head_doc="connectionist temporal classification")


class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


AutoModelForSpeechSeq2Seq = auto_class_update(
    AutoModelForSpeechSeq2Seq,
    head_doc="sequence-to-sequence speech-to-text modeing")


class AutoModelWithLMHead(_AutoModelWithLMHead):
    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning, )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning, )
        return super().from_pretrained(pretrained_model_name_or_path,
                                       *model_args, **kwargs)
