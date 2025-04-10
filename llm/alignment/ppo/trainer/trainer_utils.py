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

from paddlenlp.generation.utils import GenerationMixin
from paddlenlp.trainer.trainer import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    logger,
)
from paddlenlp.transformers import BatchEncoding, PretrainedTokenizer
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_outputs import ModelOutput
from paddlenlp.transformers.tokenizer_utils_base import PaddingStrategy

from models.ppo_model_utils import make_attention_mask, make_position_ids  # isort:skip


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
        在一个步骤结束时调用，可以用来更新控制流程。

        Args:
            args (TrainingArguments): 训练参数对象。
            state (TrainerState): 训练器状态对象。
            control (TrainerControl): 训练控制对象，包含了训练过程中的控制信息，如是否保存模型、是否进行评估和是否记录日志等。
            kwargs (dict, optional): 其他关键字参数，默认为None，没有使用。

        Returns:
            TrainerControl: 返回一个TrainerControl对象，包含了训练过程中的控制信息，如是否保存模型、是否进行评估和是否记录日志等。

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
    在一个上下文中，设置给定的参数名称和值，并在上下文结束后将其还原。

    Args:
        args (object): 需要修改参数的对象，通常是命令行解析器的实例。
        arg_name_values (dict[str, Any]): 包含参数名称和新值的字典，该函数会在上下文中修改这些参数。
            key (str): 参数名称。
            value (Any): 参数的新值。

    Yields:
        None: 无返回值，只是用于上下文管理。

    Returns:
        None: 无返回值，只是用于上下文管理。

    Raises:
        None: 不会引发任何异常。
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
        获取当前模型的属性分组，返回值为str类型。
        如果模型没有设置属性分组，则返回None。

        Returns:
            str, optional: 当前模型的属性分组，默认为None。
        """
        return self.model.pp_group

    def eval(self):
        """
        将模型置于评估模式，禁用梯度计算和 dropout。
        返回：None
        """
        self.model.eval()

    def train(self):
        """
        将模型设置为训练模式。
        在调用任何前向传播函数之前，必须先调用此函数。

        Returns:
            None, 无返回值。
        """
        self.model.train()

    def __getattr__(self, name):
        """
        如果在当前类中没有找到对应的属性，则尝试从模型中获取。
        如果在模型中也没有找到对应的属性，则会引发AttributeError异常。

        Args:
            name (str): 要查询的属性名称。

        Returns:
            Any: 返回属性值，如果在当前类和模型中都没有找到该属性，则会引发AttributeError异常。

        Raises:
            AttributeError: 如果在当前类和模型中都没有找到对应的属性。
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _broadcast_outputs(self, outputs):
        """
        将输出广播到所有进程中，如果不是最后一个阶段则返回元组，否则返回ModelOutput或者paddle.Tensor。
        如果不是最后一个阶段，会对输入的每个张量创建一个与其形状、类型相同但内容为空的新张量，并广播这些张量。

        Args:
            outputs (Union[paddle.Tensor, Tuple[paddle.Tensor], ModelOutput]): 模型的输出，可以是单个张量或张量元组，也可以是ModelOutput。

        Returns:
            Union[paddle.Tensor, Tuple[paddle.Tensor], ModelOutput]: 如果不是最后一个阶段，返回元组；否则返回ModelOutput或者paddle.Tensor。
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
                # Currently use tuple instead of ModelOutput and require the
                # caller use the return result as tuple.
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
            重写父类的方法，在生成文本时使用缓存。
        首先将self._is_gen设置为True，然后修改DecoderLayerPipe以使用缓存。
        接下来，调用super().generate(*args, **kwargs)进行文本生成。
        最后，清除所有层中的缓存（包括子层），并将self._has_cache设置为False。

        Args:
            args (Tuple[Any], optional): 可变参数列表，默认为空元组。
            kwargs (Dict[str, Any], optional): 关键字参数字典，默认为空字典。

        Returns:
            Tuple[Any]: 返回一个元组，其中包含了生成的文本和相应的概率分布。

        Raises:
            无。
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


def process_row(row, remove_value=0, remove_side="both"):
    """
    从张量中去除前导/尾随的特定值。

    Args:
        row (paddle.Tensor): 待处理的张量，一维。
        remove_value (int, optional): 要去除的值，默认为0。
        remove_side (str, optional): 去除的位置，可选"left"（只去除前导）、"right"（只去除尾随）、"both"（去除前导和尾随），默认为"both"。

    Returns:
        paddle.Tensor: 处理后的张量，一维。

    """
    non_zero_indices = paddle.nonzero(row != remove_value).flatten()
    if non_zero_indices.shape[0] == 0:
        # 行全为0，警告，不处理
        logger.warning("Row is all zeros, no trimming will be performed.")
        return row
    start_index = non_zero_indices[0]
    end_index = non_zero_indices[-1]
    # 切取中间的非零部分
    if remove_side == "left":
        trimmed_row = row[start_index:]
    elif remove_side == "right":
        trimmed_row = row[: end_index + 1]
    elif remove_side == "both":
        trimmed_row = row[start_index : end_index + 1]
    else:
        logger.warning("unknown remove_side, using both remove_side.")
        trimmed_row = row[start_index : end_index + 1]

    return trimmed_row
