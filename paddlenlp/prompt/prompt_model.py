# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
from typing import Any, Dict, List, Optional

import paddle
import paddle.nn.functional as F
from paddle.static import InputSpec

from ..transformers.generation_utils import GenerationMixin
from ..transformers.model_outputs import (
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
)
from .prompt_utils import signature
from .template import PrefixTemplate, Template
from .verbalizer import Verbalizer


class PromptModelForSequenceClassification(paddle.nn.Layer):
    """
    PromptModel for classification tasks.
    """

    def __init__(
        self,
        model: paddle.nn.Layer,
        template: Template,
        verbalizer: Optional[Verbalizer] = None,
        freeze_plm: bool = False,
        freeze_dropout: bool = False,
    ):
        super(PromptModelForSequenceClassification, self).__init__()
        self.plm = model
        self.template = template
        self.verbalizer = verbalizer
        self.freeze_plm = freeze_plm
        self.freeze_dropout = freeze_dropout
        if self.freeze_plm:
            for param in self.plm.parameters():
                param.stop_gradient = True
            if self.freeze_dropout:
                self.plm.eval()
        self.forward_keys = signature(self.plm.forward)
        self._mask_token_id = self.template.tokenizer.mask_token_id
        self._pad_token_id = self.template.tokenizer.pad_token_id
        if isinstance(self.template, PrefixTemplate):
            self.plm = self.template.process_model(self.plm)
            self.forward_keys.append("past_key_values")

    def forward(
        self,
        input_ids: paddle.Tensor,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        masked_positions: Optional[paddle.Tensor] = None,
        soft_token_ids: Optional[paddle.Tensor] = None,
        encoder_ids: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Dict[str, Any]
    ):
        return_dict = return_dict if return_dict is not None else False
        return_hidden_states = kwargs.get("return_hidden_states", False)
        input_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "masked_positions": masked_positions,
            "soft_token_ids": soft_token_ids,
            "attention_mask": attention_mask,
            "encoder_ids": encoder_ids,
            **kwargs,
        }
        input_dict = self.template.process_batch(input_dict)
        input_dict = {**input_dict, **kwargs}
        model_inputs = {k: input_dict[k] for k in input_dict if k in self.forward_keys}
        if "masked_positions" in model_inputs:
            model_inputs.pop("masked_positions")
        model_outputs = self.plm(**model_inputs, return_dict=True)
        if isinstance(model_outputs, MaskedLMOutput):
            if self.verbalizer is not None:
                logits = self.verbalizer.process_outputs(model_outputs.logits, input_dict["masked_positions"])
                num_labels = len(self.verbalizer.label_words)
            else:
                raise Exception("Verbalizer is required when model uses the MaskedLM head")
        elif isinstance(model_outputs, SequenceClassifierOutput):
            logits = model_outputs.logits
            num_labels = self.plm.num_labels if self.plm.num_labels is not None else self.plm.num_labels
        elif isinstance(model_outputs, MultipleChoiceModelOutput):
            logits = model_outputs.logits
            num_labels = -1
        else:
            raise Exception(f"Model type not support yet: {type(model_outputs)}")

        loss = None
        if labels is not None:
            if num_labels == 1:
                loss_fct = paddle.nn.MSELoss()
                loss = loss_fct(logits, labels)
            elif num_labels > 0 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape((-1, num_labels)), labels.reshape((-1,)))
            else:
                loss_fct = paddle.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,)
            if return_hidden_states:
                output = output + (model_outputs.logits,)
            if loss is not None:
                return (loss,) + output
            if isinstance(output, (list, tuple)) and len(output) == 1:
                output = output[0]
            return output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.logits,
        )

    def prompt_parameters(self):
        """
        Get the parameters of template and verbalizer.
        """
        params = [p for p in self.template.parameters()]
        if self.verbalizer is not None:
            params += [p for p in self.verbalizer.parameters()]
        return params

    def get_input_spec(self):
        template_keywords = self.template.extract_template_keywords(self.template.prompt)
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            InputSpec(shape=[None, None, None, None], dtype="float32", name="attention_mask"),
        ]
        if "mask" in template_keywords:
            input_spec.append(InputSpec(shape=[None], dtype="int64", name="masked_positions"))
        if "soft" in template_keywords:
            # Add placeholder for argument `masked_positions` if not exists.
            if "mask" not in template_keywords:
                input_spec.append(None)
            input_spec.append(InputSpec(shape=[None, None], dtype="int64", name="soft_token_ids"))
            if "encoder" in template_keywords:
                input_spec.append(InputSpec(shape=[None, None], dtype="int64", name="encoder_ids"))
        return input_spec


class PromptModelForGeneration(paddle.nn.Layer, GenerationMixin):
    """
    PromptModel for classification tasks.
    """

    def __init__(
        self,
        model: paddle.nn.Layer,
        template: Template,
        freeze_plm: bool = False,
        freeze_dropout: bool = False,
    ):
        super(PromptModelForGeneration, self).__init__()
        self.plm = model
        self.template = template
        self.freeze_plm = freeze_plm
        self.freeze_dropout = freeze_dropout
        if self.freeze_plm:
            for param in self.plm.parameters():
                param.stop_gradient = True
            if self.freeze_dropout:
                self.plm.eval()
        self.forward_keys = signature(self.plm.forward)
        self._mask_token_id = self.template.tokenizer.mask_token_id
        self._pad_token_id = self.template.tokenizer.pad_token_id
        if not isinstance(self.template, PrefixTemplate):
            raise TypeError(f"{self.__class__.__name__} is not compatible with {self.template.__class__.__name__} ")
        self.plm = self.template.process_model(self.plm)
        self.forward_keys.append("past_key_values")

    def forward(
        self,
        input_ids: paddle.Tensor,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        soft_token_ids: Optional[paddle.Tensor] = None,
        encoder_ids: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Dict[str, Any]
    ):
        return_dict = return_dict if return_dict is not None else False
        if soft_token_ids is None:
            outputs = self.plm(input_ids)
            return outputs

        return_hidden_states = kwargs.get("return_hidden_states", False)
        input_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "soft_token_ids": soft_token_ids,
            "encoder_ids": encoder_ids,
            "labels": labels,
            **kwargs,
        }
        input_dict = self.template.process_batch(input_dict)
        input_dict = {**input_dict, **kwargs}
        model_inputs = {k: input_dict[k] for k in input_dict if k in self.forward_keys}
        if "cache" in self.forward_keys:
            model_inputs["cache"] = []
            for i in range(len(model_inputs["past_key_values"])):
                from paddlenlp.transformers.gpt.modeling import MultiHeadAttention

                model_inputs["cache"].append(
                    MultiHeadAttention.Cache(
                        k=model_inputs["past_key_values"][i][0], v=model_inputs["past_key_values"][i][1]
                    )
                )
            model_inputs.pop("past_key_values")
        model_inputs.pop("labels")
        model_outputs = self.plm(**model_inputs, return_dict=True, use_cache=True)
        logits = model_outputs.logits

        loss = None
        if labels is not None:
            shift_labels = labels[..., 1:]
            shift_logits = logits[..., : shift_labels.shape[1], :]
            loss_fct = paddle.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
            loss = loss_fct(shift_logits.reshape((-1, shift_logits.shape[-1])), shift_labels.reshape((-1,))).reshape(
                [-1]
            )

        if not return_dict:
            output = (logits,)
            if return_hidden_states:
                output = output + (model_outputs.logits,)
            if loss is not None:
                return (loss,) + output
            if isinstance(output, (list, tuple)) and len(output) == 1:
                output = output[0]
            return output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.logits,
        )

    def greedy_search(self, input_ids, logits_processors, max_length, pad_token_id, eos_token_id, **model_kwargs):
        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())
        while cur_len < max_length:
            # prepare model inputs & get model output
            if "use_cache" in model_kwargs:
                del model_kwargs["use_cache"]
            if "attention_mask" in model_kwargs:
                del model_kwargs["attention_mask"]
            if "labels" in model_kwargs:
                del model_kwargs["labels"]
            outputs = self(input_ids, **model_kwargs)
            outputs = outputs[1] if isinstance(outputs, tuple) else outputs

            # To hundle the logits is a ModelOutput
            logits = outputs.logits if isinstance(outputs, ModelOutput) else outputs

            # [batch_size, vocab_size]
            next_token_logits = logits[:, -1, :]

            # pre-process distribution
            next_token_logits = self.adjust_logits_during_generation(next_token_logits)
            next_tokens_scores = logits_processors(input_ids, next_token_logits)
            # greedy
            probs = F.softmax(next_tokens_scores)
            probs = paddle.log(probs)
            next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_scores = paddle.index_sample(probs.astype("float32"), next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(unfinished_flag, next_tokens != eos_token_id)

            # Stop when there is a </s> in all sentences
            if not paddle.any(unfinished_flag):
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
            model_kwargs["soft_token_ids"] = paddle.concat(
                [model_kwargs["soft_token_ids"], paddle.to_tensor([[0]])], axis=1
            )

        return input_ids[:, origin_len:], scores


class LogitsProcessorList(List):
    def __call__(self, input_ids, logits, **kwargs):
        for processor in self:
            processor_args = inspect.signature(processor.__call__).parameters
            if len(processor_args) > 2:
                assert all(
                    arg in kwargs for arg in list(processor_args.keys())[2:]
                ), f"The parameters don't match for {processor.__class__}"
                logits = processor(input_ids, logits, **kwargs)
            else:
                logits = processor(input_ids, logits)
        return logits
