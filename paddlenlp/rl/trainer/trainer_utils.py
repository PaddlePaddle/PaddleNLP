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


import inspect
from contextlib import contextmanager

import paddle
from paddle.distributed import fleet

from ...generation.utils import GenerationMixin
from ...trainer.trainer import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    logger,
)
from ...transformers.configuration_utils import PretrainedConfig
from ...transformers.model_outputs import ModelOutput
from ...transformers.tokenizer_utils import PretrainedTokenizer
from ...transformers.tokenizer_utils_base import BatchEncoding, PaddingStrategy
from ..models.ppo_model_utils import make_attention_mask, make_position_ids


class MuteDefaultFlowCallback(TrainerCallback):
    """
    Add this callback can cencel logging/evaluation/saving by DefaultFlowCallback.
    Use this when having multi trainer.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called at the end of a step, can be used to update the control flow.

        Args:
            args (TrainingArguments): The training arguments object.
            state (TrainerState): The trainer state object.
            control (TrainerControl): The trainer control object, containing control information during training,
                such as whether to save the model, whether to evaluate, and whether to log.
            kwargs (dict, optional): Other keyword arguments, default is None, not used.

        Returns:
            TrainerControl: Returns a TrainerControl object containing control information during training,
                such as whether to save the model, whether to evaluate, and whether to log.

        Raises:
            None
        """
        control.should_save = False
        control.should_evaluate = False
        control.should_log = False
        return control


@contextmanager
def guard_set_args(args, arg_name_values):
    """
    Temporarily set given argument names and values within a context, restoring them after the context ends.

    Args:
        args (object): The object whose attributes need to be modified, typically an instance of a command-line parser.
        arg_name_values (dict[str, Any]): A dictionary containing argument names and their new values. These arguments will be modified within the context.
            key (str): The name of the argument.
            value (Any): The new value for the argument.

    Yields:
        None: No return value, used for context management.

    Returns:
        None: No return value, used for context management.

    Raises:
        None: Does not raise any exceptions.
    """
    for k, v in arg_name_values.items():
        old_value = getattr(args, k, None)
        setattr(args, k, v)
        arg_name_values[k] = old_value
    yield
    for k, v in arg_name_values.items():
        old_value = getattr(args, k)
        setattr(args, k, v)
        arg_name_values[k] = old_value


class PipeEvalModel(GenerationMixin):
    """
    Wrapper for PipelineParallel to do evaluate and generate. Currently only
    support .
    """

    def __init__(self, trainer: Trainer):
        """
        Args:
        trainer (Trainer): Trainer object.
            The trainer should have a attribute named `_inner_eval_model` which is the model used for evaluation.
            If it does not exist, then the model in `trainer.model_wrapped` will be used.
        """
        eval_model = getattr(trainer, "_inner_eval_model", None)
        self.model: fleet.model.PipelineParallel = trainer.model_wrapped if eval_model is None else eval_model
        self.config: PretrainedConfig = trainer.model.config
        self._is_gen = False
        self.update_model_kwargs_for_generation = (
            self.model._layers._non_pipe_model_class.update_model_kwargs_for_generation
        )

    @property
    def pp_group(self):
        """
        Get the property group of the current model. The return value is of type str.
        If no property group is set for the model, return None.

        Returns:
            str, optional: The property group of the current model, default is None.
        """
        return self.model.pp_group

    def eval(self):
        """
        Put the model in evaluation mode, disabling gradient computation and dropout.
        Returns:
            None
        """
        self.model.eval()

    def train(self):
        """
        Set the model to training mode.
        This function must be called before any forward pass functions are invoked.

        Returns:
            None: No return value.
        """
        self.model.train()

    def __getattr__(self, name):
        """
        If the attribute is not found in the current class, try to get it from the model.
        If the attribute is not found in the model either, an AttributeError exception will be raised.

        Args:
            name (str): The name of the attribute to query.

        Returns:
            Any: The value of the attribute. If the attribute is not found in the current class or the model, an AttributeError exception will be raised.

        Raises:
            AttributeError: If the attribute is not found in the current class or the model.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _broadcast_outputs(self, outputs):
        """
        Broadcast the outputs to all processes. If it is not the last stage, return a tuple; otherwise, return ModelOutput or paddle.Tensor.
        If it is not the last stage, create a new empty tensor with the same shape and type as the input tensor for each input tensor and broadcast these tensors.

        Args:
            outputs (Union[paddle.Tensor, Tuple[paddle.Tensor], ModelOutput]): The output of the model, which can be a single tensor, a tuple of tensors, or ModelOutput.

        Returns:
            Union[paddle.Tensor, Tuple[paddle.Tensor], ModelOutput]: If it is not the last stage, return a tuple; otherwise, return ModelOutput or paddle.Tensor.
        """
        # outputs is PipelineParallel.eval_batch which is a list of batches.
        out = []
        outputs = (outputs,) if isinstance(outputs, paddle.Tensor) else outputs
        for tensors in outputs:
            if not self.model.is_pipeline_last_stage():
                tensor = tensors if isinstance(tensors, paddle.Tensor) else tensors[0]
                head_out_meta = (
                    (self.model._layers.head_out_meta,)
                    if isinstance(
                        self.model._layers.head_out_meta,
                        paddle.static.InputSpec,
                    )
                    else self.model._layers.head_out_meta
                )
                tensors = tuple(
                    paddle.empty(
                        shape=[
                            (tensor.shape[i] if (meta.shape[i] is None or meta.shape[i] < 0) else meta.shape[i])
                            for i in range(len(meta.shape))
                        ],
                        dtype=(tensor.dtype if meta.dtype is None else meta.dtype),
                    )
                    for meta in head_out_meta
                )
            else:
                # Currently use tuple instead of ModelOutput and require the caller to use the return result as a tuple.
                tensors = (
                    (tensors,)
                    if isinstance(tensors, paddle.Tensor)
                    else (tensors.to_tuple() if isinstance(tensors, ModelOutput) else tensors)
                )

            # use map_structure seems hung
            for tensor in tensors:
                paddle.distributed.broadcast(
                    tensor,
                    src=self.model.pp_group.ranks[-1],
                    group=self.model.pp_group,
                )
            out.append(tensors[0] if len(tensors) == 1 else tensors)
        return out[0] if len(out) == 1 else out

    def __call__(self, *args, **kwargs):
        """
        Call the method to generate output from given input.

        Args:
            *args (tuple, optional): Input arguments to the method. Defaults to ().
            **kwargs (dict, optional): Keyword arguments to the method. Defaults to {}.

        Returns:
            Union[List[Any], Tuple[Any]]: Output generated from the input. If the method is
                called multiple times, each call returns one output. The type of the output
                depends on the implementation of the method.
        """
        model = self.model
        assert self.model.training is False
        if self._is_gen:
            # inputs by `prepare_inputs_for_generation` is a dict with following keys:
            # "input_ids", "position_ids", "past_key_values", "use_cache", "attention_mask"
            # NOTE: 1. cache/past_key_values should be passed across decoding steps
            # by using as model attr rather than input args to reduce comm overhead.
            # Also, pipe model defined for training not support this cache input.
            # 2. ignore use_cache since _check_data_vaild requires tensor if not None.
            # 3. attention_mask can reuse _prepare_decoder_attention_mask in LlamaEmbeddingPipe.
            # 4. position_ids pass through _prepare_pipeline_inputs_func and PipeLayer.
            inputs, labels = model._prepare_pipeline_inputs_func(*args, **kwargs)
            # currently, set accumulate_steps to 1 to avoid multi-batch eval/gen
            with guard_set_args(model, {"_compute_loss": False, "accumulate_steps": 1}):
                outputs = model.eval_batch([inputs, labels], compute_loss=False)
            # TODO(guosheng): Broadcasted logits are used to get next_scores, remove
            # it to reduce comm overhead. Also note that we still need broadcast
            # next_tokens though logits are broadcasted since pp ranks' seeds differs.
            # Currently, just slice the last token to reduce comm overhead.
            outputs = [
                (
                    micro_batch_output[:, -1, :].unsqueeze(1).contiguous()
                    if isinstance(micro_batch_output, paddle.Tensor)
                    else micro_batch_output[0][:, -1, :].unsqueeze(1).contiguous()
                )
                for micro_batch_output in outputs
            ]
            outputs = self._broadcast_outputs(outputs)
        else:
            # use _prepare_pipeline_inputs_func to convert pipeline inputs
            inputs, labels = model._prepare_pipeline_inputs_func(*args, **kwargs)
            # NOTE(guosheng): bug seems exist. pp.eval_batch(compute_loss=False)
            # will set pp._compute_loss to False and would not set it back. Thus
            # hack here to set it back.
            with guard_set_args(model, {"_compute_loss": False, "accumulate_steps": 1}):
                outputs = model.eval_batch([inputs, labels], compute_loss=False)
            outputs = self._broadcast_outputs(outputs)
        return outputs

    def generate(self, *args, **kwargs):
        """
        Override the parent class method to use caching during text generation.
        First, set self._is_gen to True and modify DecoderLayerPipe to use caching.
        Next, call super().generate(*args, **kwargs) to perform text generation.
        Finally, clear the cache in all layers (including sublayers) and set self._has_cache to False.

        Args:
            args (Tuple[Any], optional): A variable argument list, default is an empty tuple.
            kwargs (Dict[str, Any], optional): A dictionary of keyword arguments, default is an empty dictionary.

        Returns:
            Tuple[Any]: Returns a tuple containing the generated text and the corresponding probability distribution.

        Raises:
            None
        """
        self._is_gen = True
        # patch DecoderLayerPipe to use cache, DecoderLayerPipe is subclass of
        # DecoderLayer, and would call super().forward
        ori_decoder_layer_forward = self.model._layers._non_pipe_decoder_layer_class.forward

        def decoder_layer_forward(layer_self, *args, **kwargs):
            kwargs.update(
                {
                    "use_cache": True,
                    "cache": getattr(layer_self, "_cache", None),
                }
            )
            outputs = ori_decoder_layer_forward(layer_self, *args, **kwargs)
            output = outputs[0]
            layer_self._cache = outputs[1]
            self._has_cache = True
            return output

        with guard_set_args(
            self.model._layers._non_pipe_decoder_layer_class,
            {"forward": decoder_layer_forward},
        ):
            outputs = super().generate(*args, **kwargs)
        self._is_gen = False
        # clear cache of decoder layers, sublayers is incursive thus suitable
        # to both 1F1B and interleave
        for layer in self.model._layers.sublayers():
            if isinstance(layer, self.model._layers._non_pipe_decoder_layer_class):
                layer._cache = None
        self._has_cache = False
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
            Prepare the input for generation. This method is used by
        :meth:`~transformers.Pipeline.__call__` to generate text from prompts.

        Args:
            *args (tuple, optional): Arguments passed to :meth:`~transformers.Pipeline.__call__`.
            **kwargs (dict, optional): Keyword arguments passed to :meth:`~transformers.Pipeline.__call__`.

        Returns:
            dict: A dictionary containing the prepared inputs for generation. The keys are:

                - "prompt" (:obj:`str`, `optional`, defaults to :obj:`None`):
                  Text to be decoded. If not provided, the pipeline will try to use the cached prompts.
                - "cache" (:obj:`bool`, `optional`, defaults to :obj:`False`):
                  Whether to use the cached past key values. If not provided, it will be set to :obj:`True` when
                  the pipeline has cache.
                - Other keyword arguments are passed to :meth:`~transformers.Pipeline.__call__`.

        Raises:
            ValueError: If both ``prompt`` and ``cache`` are not provided.
        """
        arg_bind = inspect.signature(self.model._layers._non_pipe_model_class.prepare_inputs_for_generation).bind(
            *((self,) + args), **kwargs
        )
        arg_bind.apply_defaults()
        arg_dict = arg_bind.arguments
        last_arg_name, last_arg_value = arg_dict.popitem()
        if arg_bind.signature.parameters[last_arg_name].kind == inspect.Parameter.VAR_KEYWORD:
            arg_dict.update(last_arg_value)
        else:
            arg_dict[last_arg_name] = last_arg_value
        arg_dict.pop("self")
        cache = arg_dict.get("cache", None)
        # prepare_inputs_for_generation use cache to discrimate prefill
        # or decode and slice inputs accordingly.
        if getattr(self, "_has_cache", False):
            arg_dict.update({"cache": True})
        model_inputs = self.model._layers._non_pipe_model_class.prepare_inputs_for_generation(self, **arg_dict)
        model_inputs.update({"cache": cache})
        return model_inputs


def is_same_tokenizer(
    tokenizer: PretrainedTokenizer,
    other_tokenizer: PretrainedTokenizer,
) -> bool:
    """Check if two tokenizers are the same."""
    return tokenizer is other_tokenizer or (
        tokenizer.__class__ == other_tokenizer.__class__ and tokenizer.get_vocab() == other_tokenizer.get_vocab()
    )


def retokenize(src_tokenizer, dest_tokenizer, token_ids, skip_special_tokens):
    """Retokenize a sequence of token ids from one tokenizer to another."""
    tokens = src_tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
    part_tokens = []
    result_ids = []
    for token in tokens:
        if token in src_tokenizer.all_special_tokens:
            if part_tokens:
                decoded_text = src_tokenizer.decode(
                    src_tokenizer.convert_tokens_to_ids(part_tokens),
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=False,
                )
                tmp_tokens = dest_tokenizer.tokenize(decoded_text)
                result_ids.extend(dest_tokenizer.convert_tokens_to_ids(tmp_tokens))
                part_tokens = []  # 清空
            # 转换当前特殊 token
            special_token = dest_tokenizer.convert_tokens_to_ids(token)
            result_ids.append(special_token)
        else:
            part_tokens.append(token)
    # 如果有，处理最后一段(一般不应该走到, 应该以special token结尾)
    if part_tokens:
        decoded_text = src_tokenizer.decode(
            src_tokenizer.convert_tokens_to_ids(part_tokens),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        tmp_tokens = dest_tokenizer.tokenize(decoded_text)
        result_ids.extend(dest_tokenizer.convert_tokens_to_ids(tmp_tokens))
    return result_ids


def batch_retokenize(
    input_ids: paddle.Tensor,
    src_tokenizer: PretrainedTokenizer,
    dest_tokenizer: PretrainedTokenizer,
    *,
    padding: bool | str | PaddingStrategy = PaddingStrategy.LONGEST,
    skip_special_tokens: bool = False,
) -> BatchEncoding:
    """Re-tokenize a batch of input ids from one tokenizer to another."""
    all_ids = []
    for token_ids in input_ids:
        tmp_ids = retokenize(src_tokenizer, dest_tokenizer, token_ids, skip_special_tokens)
        all_ids.append(tmp_ids)
    output = {}

    output["input_ids"] = dest_tokenizer.pad(
        {"input_ids": all_ids},
        padding=padding,
        return_attention_mask=False,
        return_tensors="pd",
    )["input_ids"]
    output["attention_mask"] = make_attention_mask(
        output["input_ids"],
        pad_id=dest_tokenizer.pad_token_id,
        eos_id=dest_tokenizer.eos_token_id,
        unk_id=dest_tokenizer.unk_token_id,
        causal_mask=True,
    ).cast(paddle.bfloat16)
    output["position_ids"] = make_position_ids(output["attention_mask"])
    return output


def process_row(row, remove_value=0, remove_side="both", eos_token_id=None):
    """
    Remove leading/trailing specific values from a tensor.

    Args:
        row (paddle.Tensor): The 1D tensor to be processed.
        remove_value (int, optional): The value to be removed, default is 0.
        remove_side (str, optional): The side to remove values from, can be "left" (remove leading only), "right" (remove trailing only),
            or "both" (remove both leading and trailing), default is "both".

    Returns:
        paddle.Tensor: The processed 1D tensor.
    """
    if eos_token_id is not None and remove_value == eos_token_id:
        # 特殊处理：保留最后一个 eos_token_id 的 index
        is_not_remove_value = row != remove_value
        last_eos_idx = paddle.nonzero(row == eos_token_id).flatten()
        if last_eos_idx.shape[0] > 0:
            last_eos_idx = last_eos_idx[-1]
            is_not_remove_value[last_eos_idx] = True  # 保留 eos
    else:
        is_not_remove_value = row != remove_value

    non_zero_indices = paddle.nonzero(is_not_remove_value).flatten()
    if non_zero_indices.shape[0] == 0:
        # If the row is all zeros, log a warning and return the original row.
        logger.warning("Row is all zeros, no trimming will be performed.")
        return row
    start_index = non_zero_indices[0]
    end_index = non_zero_indices[-1]
    # Slice the middle non-zero part.
    if remove_side == "left":
        trimmed_row = row[start_index:]
    elif remove_side == "right":
        trimmed_row = row[: end_index + 1]
    elif remove_side == "both":
        trimmed_row = row[start_index : end_index + 1]
    else:
        # If an unknown remove_side is provided, log a warning and use "both".
        logger.warning("Unknown remove_side, using 'both' remove_side.")
        trimmed_row = row[start_index : end_index + 1]

    return trimmed_row
