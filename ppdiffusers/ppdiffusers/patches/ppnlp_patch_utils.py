# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import builtins
import contextlib
import copy
import functools
import json
import math
import weakref
from collections import OrderedDict
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils import (
    DIFFUSERS_CACHE,
    FROM_DIFFUSERS,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    LOW_CPU_MEM_USAGE_DEFAULT,
    PPDIFFUSERS_CACHE,
    TO_DIFFUSERS,
    _add_variant,
    _get_model_file,
    get_logger,
    is_paddle_available,
    is_paddlenlp_available,
    is_ppxformers_available,
    is_safetensors_available,
    is_torch_available,
    is_torch_file,
    smart_load,
)

logger = get_logger(__name__)

__all__ = []

from contextlib import ExitStack


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


def copy_func(f):
    "Copy a non-builtin function (NB `copy.copy` does not work for this)"
    if not isinstance(f, FunctionType):
        return copy.copy(f)
    fn = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    fn.__kwdefaults__ = f.__kwdefaults__
    fn.__dict__.update(f.__dict__)
    fn.__annotations__.update(f.__annotations__)
    fn.__qualname__ = f.__qualname__
    return fn


class _clsmethod:
    def __init__(self, f):
        self.f = f

    def __get__(self, _, f_cls):
        return MethodType(self.f, f_cls)


# copied from https://github.com/fastai/fastcore/blob/c9b4c088d3706569c076e7c197c724730be190ab/fastcore/basics.py#L938-L954
def patch_to(cls, as_prop=False, cls_method=False):
    "Decorator: add `f` to `cls`"
    if not isinstance(cls, (tuple, list)):
        cls = (cls,)

    def _inner(f):
        for c_ in cls:
            nf = copy_func(f)
            nm = f.__name__
            # `functools.update_wrapper` when passing patched function to `Pipeline`, so we do it manually
            for o in functools.WRAPPER_ASSIGNMENTS:
                setattr(nf, o, getattr(f, o))
            nf.__qualname__ = f"{c_.__name__}.{nm}"
            if cls_method:
                # fix https://github.com/fastai/fastcore/issues/510
                setattr(c_, nm, _clsmethod(nf))
            else:
                setattr(c_, nm, property(nf) if as_prop else nf)
        # Avoid clobbering existing functions
        return globals().get(nm, builtins.__dict__.get(nm, None))

    return _inner


if is_paddle_available():
    import paddle
    import paddle.nn as nn

    def is_floating_point(x):
        if not isinstance(x, (paddle.Tensor, paddle.static.Variable)):
            raise TypeError("Expected Tensor, but received type of x: {}".format(type(x)))
        dtype = x.dtype
        is_fp_dtype = (
            dtype == paddle.float32 or dtype == paddle.float64 or dtype == paddle.float16 or dtype == paddle.bfloat16
        )
        return is_fp_dtype

    if not hasattr(paddle, "is_floating_point"):
        paddle.is_floating_point = is_floating_point

    # paddle.long = paddle.int64
    # paddle.int = paddle.int32
    # paddle.double = paddle.float64
    # paddle.half = paddle.float16
    # paddle.Tensor.half = lambda x: paddle.cast(x, paddle.float16)
    # paddle.Tensor.float = lambda x: paddle.cast(x, paddle.float32)
    # paddle.Tensor.double = lambda x: paddle.cast(x, paddle.float64)
    # paddle.Tensor.int = lambda x: paddle.cast(x, paddle.int32)
    # paddle.Tensor.long = lambda x: paddle.cast(x, paddle.int64)
    # paddle.Tensor.bool = lambda x: paddle.cast(x, paddle.bool)
    # paddle.Tensor.clamp = paddle.clip
    # paddle.clamp = paddle.clip

    def view_pt(x, *shape: builtins.int, name=None):
        return paddle.reshape(x, shape=shape, name=name)

    paddle.view = view_pt
    paddle.Tensor.view = view_pt

    if not hasattr(paddle.Tensor, "data_ptr"):
        paddle.Tensor.data_ptr = lambda x: x.value().get_tensor()._ptr()

    def permute_pt(x, *perm: builtins.int, name=None):
        return paddle.transpose(x, perm=perm, name=name)

    paddle.permute = permute_pt
    paddle.Tensor.permute = permute_pt
    paddle.Tensor.softmax = nn.functional.softmax

    # patch repeat_interleave
    raw_repeat_interleave = paddle.repeat_interleave

    @paddle.jit.not_to_static
    def repeat_interleave(x, repeats, axis=None, name=None):
        fp16 = False
        if x.dtype == paddle.float16:
            x = x.cast(paddle.float32)
            fp16 = True

        out = raw_repeat_interleave(x, repeats=repeats, axis=axis, name=name)

        if fp16:
            out = out.cast(paddle.float16)
        return out

    paddle.repeat_interleave = repeat_interleave
    paddle.Tensor.repeat_interleave = repeat_interleave

    # patch max
    raw_max = paddle.max

    @paddle.jit.not_to_static
    def max(x, axis=None, keepdim=False, name=None):
        fp16 = False
        if x.dtype == paddle.float16:
            x = x.cast(paddle.float32)
            fp16 = True

        out = raw_max(x, axis=axis, keepdim=keepdim, name=name)

        if fp16:
            out = out.cast(paddle.float16)
        return out

    paddle.max = max
    paddle.Tensor.max = max

    # patch gather_nd support bfloat16
    raw_gather_nd = paddle.gather_nd

    @paddle.jit.not_to_static
    def gather_nd(x, index, name=None):
        bfp16 = False
        if x.dtype == paddle.bfloat16:
            x = x.cast(paddle.float16)
            bfp16 = True

        out = raw_gather_nd(x, index=index, name=name)

        if bfp16:
            out = out.cast(paddle.bfloat16)
        return out

    paddle.gather_nd = gather_nd
    paddle.Tensor.gather_nd = gather_nd
    paddle.Tensor.contiguous = lambda x: x

    # must return self!
    def eval(self):
        # Layer-level setting
        self.training = False
        for layer in self.sublayers():
            layer.training = False
        return self

    nn.Layer.eval = eval

    def Parameter(data: paddle.Tensor, requires_grad=True):
        tensor = paddle.create_parameter(data.shape, dtype=data.dtype, default_initializer=nn.initializer.Assign(data))
        if not requires_grad:
            tensor.stop_gradient = True
        return tensor

    nn.Parameter = Parameter

    @contextlib.contextmanager
    def device_scope(device="cpu"):
        new_device = device.replace("cuda", "gpu")
        old_device = paddle.get_device()
        try:
            paddle.set_device(new_device)
            yield
        finally:
            paddle.set_device(old_device)

    paddle.device_scope = device_scope

    def get_sublayer(self, target: str):
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: nn.Layer = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod.__class__.__name__ + " has no " "attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, nn.Layer):
                raise AttributeError("`" + item + "` is not " "an nn.Layer")
        return mod

    nn.Layer.get_sublayer = get_sublayer

    class _WrappedHook:
        def __init__(self, hook: Callable, module: Optional["nn.Layer"] = None):
            self.hook: Callable = hook
            functools.update_wrapper(self, hook)

            self.with_module: bool = False

            if module is not None:
                self.module: weakref.ReferenceType["nn.Layer"] = weakref.ref(module)
                self.with_module = True

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if self.with_module:
                module = self.module()
                if module is None:
                    raise RuntimeError("You are trying to call the hook of a dead Module!")
                return self.hook(module, *args, **kwargs)
            return self.hook(*args, **kwargs)

        def __getstate__(self) -> Dict:
            result = {"hook": self.hook, "with_module": self.with_module}
            if self.with_module:
                result["module"] = self.module()

            return result

        def __setstate__(self, state: Dict):
            self.hook = state["hook"]
            self.with_module = state["with_module"]

            if self.with_module:
                if state["module"] is None:
                    raise RuntimeError("You are trying to revive the hook of a dead Module!")
                self.module = weakref.ref(state["module"])

    try:
        from paddle.nn.layer.layers import HookRemoveHelper
    except ImportError:
        from paddle.fluid.dygraph.layers import HookRemoveHelper

    def register_load_state_dict_pre_hook(self, hook, with_module=False):
        if not hasattr(self, "load_state_dict_pre_hooks"):
            self.load_state_dict_pre_hooks = OrderedDict()
        handle = HookRemoveHelper(self.load_state_dict_pre_hooks)
        self.load_state_dict_pre_hooks[handle._hook_id] = _WrappedHook(hook, self if with_module else None)
        return handle

    nn.Layer.register_load_state_dict_pre_hook = register_load_state_dict_pre_hook

    raw_set_state_dict = nn.Layer.set_state_dict

    def set_state_dict(self, state_dict, use_structured_name: bool = True):
        if hasattr(self, "load_state_dict_pre_hooks"):
            for hook in self.load_state_dict_pre_hooks.values():
                hook(state_dict)
        # POP is_torch_weight
        state_dict.pop("is_torch_weight", None)
        return raw_set_state_dict(self, state_dict, use_structured_name=use_structured_name)

    nn.Layer.set_state_dict = set_state_dict
    nn.Layer.load_dict = nn.Layer.set_state_dict
    nn.Layer.set_dict = nn.Layer.set_state_dict

if is_paddle_available() and is_paddlenlp_available():
    import paddle

    import paddlenlp.transformers
    from paddlenlp import __version__
    from paddlenlp.transformers import PretrainedConfig, PretrainedModel

    try:
        from paddlenlp.transformers.model_utils import no_init_weights
    except ImportError:
        from ..utils.paddle_utils import no_init_weights

    if is_ppxformers_available():
        from paddle.incubate.nn.memory_efficient_attention import (
            memory_efficient_attention,
        )
        from paddle.nn.functional.flash_attention import flash_attention

        def scaled_dot_product_attention_(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            training=True,
            attention_op="cutlass",
        ):
            if attn_mask is not None or attention_op == "math":
                if scale is None:
                    scale = 1 / math.sqrt(query.shape[-1])
                qt = paddle.transpose(query, [0, 2, 1, 3])
                kt = paddle.transpose(key, [0, 2, 1, 3])
                vt = paddle.transpose(value, [0, 2, 1, 3])
                s = paddle.matmul(qt * scale, kt, transpose_y=True)
                if is_causal:
                    p = paddle.incubate.softmax_mask_fuse_upper_triangle(s)
                else:
                    if attn_mask is not None:
                        attn_mask = paddle.transpose(attn_mask, [0, 2, 1, 3])
                        if attn_mask.cast("float32").min() == 0 and attn_mask.cast("float32").max() == 1:
                            attn_mask = (attn_mask.cast(s.dtype) - 1) * 10000.0
                        s = s + attn_mask
                    p = paddle.nn.functional.softmax(s)
                if dropout_p > 0.0:
                    p = paddle.nn.functional.dropout(p, dropout_p, training=training, mode="upscale_in_train")
                o = paddle.matmul(p, vt)
                return paddle.transpose(o, [0, 2, 1, 3])
            elif attention_op is None or attention_op == "cutlass" or training:
                if scale is None:
                    scale = 1 / math.sqrt(query.shape[-1])
                # support fp32, fp16, bfp16
                output = memory_efficient_attention(
                    query,
                    key,
                    value,
                    None,
                    p=dropout_p,
                    scale=scale,
                    training=training,
                )
            elif attention_op == "flash":
                raw_dtype = query.dtype
                if raw_dtype == paddle.float32:
                    query, key, value = (
                        query.cast(paddle.float16),
                        key.cast(paddle.float16),
                        value.cast(paddle.float16),
                    )
                output = flash_attention(query, key, value, dropout=dropout_p, causal=is_causal, return_softmax=False)[
                    0
                ]
                if raw_dtype == paddle.float32:
                    output = output.cast(raw_dtype)
            else:
                raise ValueError("ppxformers's attention_op shoulde be in ['cutlass', 'flash', 'math']")
            return output

        paddle.nn.functional.scaled_dot_product_attention_ = scaled_dot_product_attention_

    @patch_to(nn.Layer, as_prop=True)
    def dtype(parameter: nn.Layer) -> paddle.dtype:
        try:
            return next(parameter.named_parameters())[1].dtype
        except StopIteration:
            try:
                return next(parameter.named_buffers())[1].dtype
            except StopIteration:
                return parameter._dtype

    @patch_to(PretrainedModel, as_prop=True)
    def device(self):
        try:
            return next(self.named_parameters())[1].place
        except StopIteration:
            try:
                return next(self.named_buffers())[1].place
            except StopIteration:
                return paddle.get_device()

    try:
        from paddlenlp.transformers import XLMRobertaTokenizer
    except ImportError:
        # patch xlm-roberta tokenizer
        """Tokenization classes for XLM-RoBERTa model."""
        import os
        from shutil import copyfile

        import sentencepiece as spm

        from paddlenlp.transformers.tokenizer_utils import (
            AddedToken,
            PretrainedTokenizer,
        )

        SPIECE_UNDERLINE = "▁"

        class XLMRobertaTokenizer(PretrainedTokenizer):

            resource_files_names = {"vocab_file": "sentencepiece.bpe.model"}
            pretrained_resource_files_map = {}
            pretrained_init_configuration = {}
            max_model_input_sizes = {
                "xlm-roberta-base": 512,
                "xlm-roberta-large": 512,
                "xlm-roberta-large-finetuned-conll02-dutch": 512,
                "xlm-roberta-large-finetuned-conll02-spanish": 512,
                "xlm-roberta-large-finetuned-conll03-english": 512,
                "xlm-roberta-large-finetuned-conll03-german": 512,
            }
            model_input_names = ["input_ids", "attention_mask"]

            def __init__(
                self,
                vocab_file,
                bos_token="<s>",
                eos_token="</s>",
                sep_token="</s>",
                cls_token="<s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
                sp_model_kwargs: Optional[Dict[str, Any]] = None,
                **kwargs
            ) -> None:
                # Mask token behave like a normal word, i.e. include the space before it
                mask_token = (
                    AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
                )

                self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

                super().__init__(
                    bos_token=bos_token,
                    eos_token=eos_token,
                    unk_token=unk_token,
                    sep_token=sep_token,
                    cls_token=cls_token,
                    pad_token=pad_token,
                    mask_token=mask_token,
                    sp_model_kwargs=self.sp_model_kwargs,
                    **kwargs,
                )

                self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
                self.sp_model.Load(str(vocab_file))
                self.vocab_file = vocab_file

                # Original fairseq vocab and spm vocab must be "aligned":
                # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
                # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
                # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
                # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'

                # Mimic fairseq token-to-id alignment for the first 4 token
                self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

                # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
                self.fairseq_offset = 1

                self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
                self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

            def __getstate__(self):
                state = self.__dict__.copy()
                state["sp_model"] = None
                state["sp_model_proto"] = self.sp_model.serialized_model_proto()
                return state

            def __setstate__(self, d):
                self.__dict__ = d

                # for backward compatibility
                if not hasattr(self, "sp_model_kwargs"):
                    self.sp_model_kwargs = {}

                self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
                self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

            def build_inputs_with_special_tokens(
                self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
            ) -> List[int]:
                """
                Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
                adding special tokens. An XLM-RoBERTa sequence has the following format:
                - single sequence: `<s> X </s>`
                - pair of sequences: `<s> A </s></s> B </s>`
                Args:
                    token_ids_0 (`List[int]`):
                        List of IDs to which the special tokens will be added.
                    token_ids_1 (`List[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.
                Returns:
                    `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
                """

                if token_ids_1 is None:
                    return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
                cls = [self.cls_token_id]
                sep = [self.sep_token_id]
                return cls + token_ids_0 + sep + sep + token_ids_1 + sep

            def get_special_tokens_mask(
                self,
                token_ids_0: List[int],
                token_ids_1: Optional[List[int]] = None,
                already_has_special_tokens: bool = False,
            ) -> List[int]:
                """
                Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
                special tokens using the tokenizer `prepare_for_model` method.
                Args:
                    token_ids_0 (`List[int]`):
                        List of IDs.
                    token_ids_1 (`List[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.
                    already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                        Whether or not the token list is already formatted with special tokens for the model.
                Returns:
                    `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
                """

                if already_has_special_tokens:
                    return super().get_special_tokens_mask(
                        token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
                    )

                if token_ids_1 is None:
                    return [1] + ([0] * len(token_ids_0)) + [1]
                return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

            def create_token_type_ids_from_sequences(
                self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
            ) -> List[int]:
                """
                Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
                not make use of token type ids, therefore a list of zeros is returned.
                Args:
                    token_ids_0 (`List[int]`):
                        List of IDs.
                    token_ids_1 (`List[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.
                Returns:
                    `List[int]`: List of zeros.
                """

                sep = [self.sep_token_id]
                cls = [self.cls_token_id]

                if token_ids_1 is None:
                    return len(cls + token_ids_0 + sep) * [0]
                return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

            @property
            def vocab_size(self):
                return len(self.sp_model) + self.fairseq_offset + 1  # Add the <mask> token

            def get_vocab(self):
                vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
                vocab.update(self.added_tokens_encoder)
                return vocab

            def _tokenize(self, text: str) -> List[str]:
                return self.sp_model.encode(text, out_type=str)

            def _convert_token_to_id(self, token):
                """Converts a token (str) in an id using the vocab."""
                if token in self.fairseq_tokens_to_ids:
                    return self.fairseq_tokens_to_ids[token]
                spm_id = self.sp_model.PieceToId(token)

                # Need to return unknown token if the SP model returned 0
                return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

            def _convert_id_to_token(self, index):
                """Converts an index (integer) in a token (str) using the vocab."""
                if index in self.fairseq_ids_to_tokens:
                    return self.fairseq_ids_to_tokens[index]
                return self.sp_model.IdToPiece(index - self.fairseq_offset)

            def convert_tokens_to_string(self, tokens):
                """Converts a sequence of tokens (strings for sub-words) in a single string."""
                out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
                return out_string

            def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
                if not os.path.isdir(save_directory):
                    logger.error(f"Vocabulary path ({save_directory}) should be a directory")
                    return
                out_vocab_file = os.path.join(
                    save_directory,
                    (filename_prefix + "-" if filename_prefix else "") + self.resource_files_names["vocab_file"],
                )

                if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(
                    self.vocab_file
                ):
                    copyfile(self.vocab_file, out_vocab_file)
                elif not os.path.isfile(self.vocab_file):
                    with open(out_vocab_file, "wb") as fi:
                        content_spiece_model = self.sp_model.serialized_model_proto()
                        fi.write(content_spiece_model)

                return (out_vocab_file,)

        paddlenlp.transformers.XLMRobertaTokenizer = XLMRobertaTokenizer

    # patch BertModel forward
    from paddlenlp.transformers import BertModel

    BertModel.raw_forward = BertModel.forward

    def forward_new(
        self,
        input_ids: paddle.Tensor,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)
        return self.raw_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

    BertModel.forward = forward_new

    TRANSFORMERS_SAFE_WEIGHTS_NAME = "model.safetensors"
    TRANSFORMERS_WEIGHTS_NAME = "pytorch_model.bin"

    # patch from_pretrained and save_pretrained
    def from_pretrained_v3(cls, pretrained_model_name_or_path, *args, from_hf_hub: bool = False, **kwargs):
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_diffusers = kwargs.pop("from_diffusers", None)
        if from_diffusers is None:
            from_diffusers = FROM_DIFFUSERS
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        paddle_dtype = kwargs.pop("paddle_dtype", None)
        # do not use paddlenlp dtype
        _dtype = kwargs.pop("dtype", None)
        if _dtype is not None and paddle_dtype is None:
            paddle_dtype = _dtype
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", LOW_CPU_MEM_USAGE_DEFAULT)

        user_agent = {
            "ppdiffusers": __version__,
            "file_type": "model",
            "framework": "paddle",
        }

        config = None

        model_kwargs = kwargs
        # 1. get the PretrainedConfig to init model
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path

            # TODO fix config  from_pretrained
            # must from hf hub
            if from_hf_hub:
                if subfolder is not None:
                    kwargs["subfolder"] = subfolder
            else:
                if subfolder is not None:
                    config_path = (
                        os.path.join(config_path, subfolder)
                        if os.path.isdir(config_path)
                        else "/".join([config_path, subfolder])
                    )

            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                from_hf_hub=from_hf_hub,
                **kwargs,
            )
        assert config is not None

        # we will remove in the future.
        if not from_hf_hub and not os.path.exists(os.path.join(cache_dir, config_path, "config.json")):
            config.save_pretrained(os.path.join(cache_dir, config_path))

        if paddle_dtype is None:
            paddle_dtype = config.get("dtype", paddle.get_default_dtype())
        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # Load model
        model_file = None
        if from_diffusers:
            if is_safetensors_available():
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(TRANSFORMERS_SAFE_WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        from_hf_hub=from_hf_hub,
                    )
                except Exception:  # noqa: E722
                    model_file = None
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(TRANSFORMERS_WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    from_hf_hub=from_hf_hub,
                )
        else:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=_add_variant("model_state.pdparams", variant),
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                from_hf_hub=from_hf_hub,
            )
        assert model_file is not None

        # try load model_file with paddle / torch / safetensor
        state_dict = smart_load(model_file)
        init_contexts = []

        dtype = set(v.dtype for v in state_dict.values() if paddle.is_tensor(v) and paddle.is_floating_point(v))
        if len(dtype) > 1 and paddle.float32 not in dtype:
            raise ValueError(
                f"The weights of the model file {model_file} have a mixture of incompatible dtypes {dtype}. Please"
                f" make sure that {model_file} weights have only one dtype."
            )
        elif len(dtype) > 1 and paddle.float32 in dtype:
            dtype = paddle.float32
        elif len(dtype) == 0:
            dtype = paddle.float32
        else:
            dtype = dtype.pop()

        init_contexts.append(paddle.dtype_guard(dtype))

        if low_cpu_mem_usage:
            # Instantiate model.
            init_contexts.append(no_init_weights(_enable=True))
            if hasattr(paddle, "LazyGuard"):
                init_contexts.append(paddle.LazyGuard())

        with ContextManagers(init_contexts):
            model = cls(config, **model_kwargs)

        # convert weights
        if (from_diffusers or is_torch_file(model_file)) and hasattr(cls, "smart_convert"):
            state_dict = cls.smart_convert(state_dict, model)

        loaded_state_dict_keys = list(state_dict.keys())

        model, missing_keys, unexpected_keys, mismatched_keys = cls._load_pretrained_model_old(
            model=model,
            state_dict=state_dict,
            loaded_keys=loaded_state_dict_keys,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            dtype=None,
        )
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
            "error_msgs": "",
        }

        # if paddle_dtype is not None and not isinstance(paddle_dtype, paddle.dtype):
        #     raise ValueError(
        #         f"{paddle_dtype} needs to be of type `paddle.dtype`, e.g. `paddle.float16`, but is {type(paddle_dtype)}."
        #     )
        if paddle_dtype is not None:
            model = model.to(dtype=paddle_dtype)

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        if output_loading_info:
            return model, loading_info

        return model

    import re

    import numpy as np

    @classmethod
    def _load_pretrained_model_old(
        cls,
        model: PretrainedModel,
        state_dict: Dict[str, paddle.Tensor],
        loaded_keys: List[str],
        ignore_mismatched_sizes=False,
        dtype=None,
    ) -> Tuple[List[str]]:
        model_state_dict = model.state_dict()

        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            expected_keys = [".".join(s.split(".")[1:]) if s.startswith(prefix) else s for s in expected_keys]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{checkpoint_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                        model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        # Whole checkpoint
        mismatched_keys = _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        )

        start_prefix = prefix + "."

        # `add_prefix_to_model` and `remove_prefix_from_model` are for different situation,
        # you can check the following matrix, which means:
        # the value of cell: (add_prefix_to_model, remove_prefix_from_model)
        # the load/Init-Base is the state-dict which don't contain `prefix`.
        # the load/Init-DownStream is the state-dict which contain the `prefix`
        #
        # |                 | load-Base | load-DownStream |
        # |-----------------|-----------|-----------------|
        # | Init-Base       | F,F       | T,F             |
        # | Init-DonwStream | F,T       | F,F             |
        #
        # the above value matrix will help you understand the following code.
        if add_prefix_to_model:
            for key in list(state_dict.keys()):
                if key.startswith(start_prefix):
                    state_dict[key.replace(start_prefix, "")] = state_dict.pop(key)

        if remove_prefix_from_model:
            for key in list(state_dict.keys()):
                state_dict[start_prefix + key] = state_dict.pop(key)

        # convert the dtype of state dict
        if dtype is not None:
            if isinstance(dtype, paddle.dtype):
                dtype = str(dtype)[7:]

            if dtype not in ["float32", "float16", "bfloat16"]:
                raise ValueError(
                    f"the value of `dtype` should be one of [`float32`, `float16`, `bfloat16`], but received {dtype}"
                )
            for key in state_dict.keys():
                target_dtype = dtype
                if isinstance(state_dict[key], np.ndarray):
                    if not issubclass(state_dict[key].dtype.type, np.floating):
                        continue

                    # TODO(wj-Mcat): add `keep_in_fp32` feature to enable hybrid fp32 state-dict
                    # this is the temp hard code for fused-mt transformer
                    if model.keep_in_fp32_modules(key, model.config, dtype):
                        target_dtype = "float32"
                    # state_dict[key] = convert_ndarray_dtype(state_dict[key], target_dtype)

                elif isinstance(state_dict[key], paddle.Tensor):
                    if not state_dict[key].is_floating_point():
                        continue

                    # TODO(wj-Mcat): add `keep_in_fp32` feature to enable hybrid fp32 state-dict
                    # this is the temp hard code for fused-mt transformer
                    if model.keep_in_fp32_modules(key, model.config, dtype):
                        target_dtype = "float32"
                    state_dict[key] = paddle.cast(state_dict[key], dtype=target_dtype)
                else:
                    raise ValueError(f"the dtype<{state_dict[key].dtype}> of current state-dict[{key}] is not valid")
        else:
            dtype_prefix_len = len("paddle.")
            for k, v in model_to_load.state_dict().items():
                if not isinstance(v, np.ndarray):
                    dtype = str(v.dtype)[dtype_prefix_len:]
                if k in state_dict:
                    if paddle.in_dynamic_mode():
                        if isinstance(state_dict[k], np.ndarray):
                            state_dict[k] = state_dict[k].astype(dtype)
                        else:
                            state_dict[k] = paddle.cast(state_dict[k], dtype)
                    else:
                        # there are some latent error when case dtype in static-mode, so let's:
                        # 1. convert fluid.*.Tensor -> numpy.ndarray
                        # 2. cast the dtype with numpy tools
                        # 3. paddle works well with ndarray state-dict
                        state_dict[k] = np.array(state_dict[k])
                        state_dict[k] = state_dict[k].astype(dtype)

        # For model parallel if FastGeneration
        # To avoid recursive import temporarily.
        import paddlenlp.ops.fast_transformer.transformer.decoding as ft_decoding

        state_to_load = ft_decoding.get_ft_para_conf().fit_partial_model(model_to_load, state_dict)
        if paddle.in_dynamic_mode():
            model_to_load.set_state_dict(state_to_load)

        return model_to_load, missing_keys, unexpected_keys, mismatched_keys

    PretrainedModel._load_pretrained_model_old = _load_pretrained_model_old

    # PretrainedModel.from_pretrained is classmethod
    raw_from_pretrained = PretrainedModel.from_pretrained.__func__
    raw_save_pretrained = PretrainedModel.save_pretrained

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        from_hf_hub=False,
        subfolder=None,
        paddle_dtype=None,
        from_diffusers=None,
        variant=None,
        **kwargs
    ):
        try:
            if cls.constructed_from_pretrained_config() and (
                hasattr(cls, "smart_convert") or hasattr(cls, "register_load_torch_hook")
            ):
                return from_pretrained_v3(
                    cls,
                    pretrained_model_name_or_path,
                    *args,
                    from_hf_hub=from_hf_hub,
                    subfolder=subfolder,
                    paddle_dtype=paddle_dtype,
                    from_diffusers=from_diffusers,
                    variant=variant,
                    **kwargs,
                )
        except Exception:
            pass

        dtype = kwargs.pop("dtype", paddle_dtype)
        if isinstance(dtype, paddle.dtype):
            dtype = str(dtype).replace("paddle.", "")
        return raw_from_pretrained(
            cls,
            pretrained_model_name_or_path,
            *args,
            from_hf_hub=from_hf_hub,
            subfolder=subfolder,
            dtype=dtype,
            **kwargs,
        )

    PretrainedModel.from_pretrained = from_pretrained

    if is_safetensors_available():
        from safetensors.numpy import save_file as safetensors_numpy_save_file

        if is_torch_available():
            from safetensors.torch import save_file as safetensors_torch_save_file

    if is_torch_available():
        import torch

    def save_pretrained_v3(
        self: PretrainedModel,
        save_directory: str,
        is_main_process: bool = True,
        save_function: Callable = None,
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        to_diffusers: Optional[bool] = None,
    ):
        from ..models.modeling_pytorch_paddle_utils import (
            convert_paddle_state_dict_to_pytorch,
        )
        from ..models.modeling_utils import convert_state_dict

        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS

        if to_diffusers and safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        model_to_save = self._layers if isinstance(self, paddle.DataParallel) else self
        if is_main_process:
            try:
                model_to_save.config.dtype = str(model_to_save._dtype).split(".")[-1]
            except:
                model_to_save.config.dtype = "float32"
            # Attach architecture to the config
            model_to_save.config.architectures = [model_to_save.__class__.__name__]

            model_to_save.config.save_pretrained(save_directory)

        state_dict = model_to_save.state_dict()
        # save ignore lora_weights
        fn = lambda k: ".lora_" in k or ".alpha" in k
        state_dict = {k: v for k, v in state_dict.items() if not fn(k)}

        # choose save_function
        if save_function is None:
            if to_diffusers:
                if safe_serialization:
                    if is_torch_available():
                        save_function = safetensors_torch_save_file
                        state_dict = convert_state_dict(state_dict, framework="torch")
                    else:
                        save_function = safetensors_numpy_save_file
                        state_dict = convert_state_dict(state_dict, framework="numpy")
                    weights_name = _add_variant("model.safetensors", variant)
                else:
                    if not is_torch_available():
                        raise ImportError(
                            "`to_diffusers=True` with `safe_serialization=False` requires the `torch library: `pip install torch`."
                        )
                    save_function = torch.save
                    weights_name = _add_variant("pytorch_model.bin", variant)
                    state_dict = convert_state_dict(state_dict, framework="torch")

                state_dict = convert_paddle_state_dict_to_pytorch(state_dict, model_to_save)
            else:
                save_function = paddle.save
                weights_name = _add_variant("model_state.pdparams", variant)

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weights_name))

        logger.info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")

    def save_pretrained(
        self,
        save_dir: str,
        is_main_process: bool = True,
        state_dict=None,
        save_function: Callable = None,
        max_shard_size="10GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        to_diffusers: Optional[bool] = None,
        *args,
        **kwargs,
    ):
        if self.constructed_from_pretrained_config() and hasattr(self, "smart_convert"):
            return save_pretrained_v3(
                self,
                save_dir,
                is_main_process=is_main_process,
                save_function=save_function,
                safe_serialization=safe_serialization,
                variant=variant,
                to_diffusers=to_diffusers,
            )
        return raw_save_pretrained(
            self,
            save_dir=save_dir,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            *args,
            **kwargs,
        )

    PretrainedModel.save_pretrained = save_pretrained

    from paddlenlp.transformers import (
        BertModel,
        BitBackbone,
        ClapTextModelWithProjection,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModel,
        CLIPVisionModelWithProjection,
        DPTForDepthEstimation,
        SpeechT5HifiGan,
        T5EncoderModel,
    )

    if not hasattr(T5EncoderModel, "_keep_in_fp32_modules"):
        T5EncoderModel._keep_in_fp32_modules = ["wo"]

    from ..models.modeling_pytorch_paddle_utils import (
        convert_pytorch_state_dict_to_paddle_class_method,
    )
    from ..pipelines.alt_diffusion.modeling_roberta_series import (
        RobertaSeriesModelWithTransformation,
    )
    from ..pipelines.deepfloyd_if.safety_checker import IFSafetyChecker
    from ..pipelines.latent_diffusion.pipeline_latent_diffusion import LDMBertModel
    from ..pipelines.paint_by_example.image_encoder import PaintByExampleImageEncoder
    from ..pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from ..pipelines.stable_diffusion_safe.safety_checker import (
        SafeStableDiffusionSafetyChecker,
    )

    @classmethod
    def clip_smart_convert(cls, state_dict, pd_model):
        new_model_state = {}
        name_mapping_dict = {
            ".encoder.": ".transformer.",
            ".layer_norm": ".norm",
            ".mlp.": ".",
            ".fc1.": ".linear1.",
            ".fc2.": ".linear2.",
            ".final_layer_norm.": ".ln_final.",
            ".embeddings.": ".",
            ".position_embedding.": ".positional_embedding.",
            ".patch_embedding.": ".conv1.",
            "visual_projection.weight": "vision_projection",
            "text_projection.weight": "text_projection",
            ".pre_layrnorm.": ".ln_pre.",
            ".post_layernorm.": ".ln_post.",
        }
        ignore_value = [
            "position_ids",
        ]
        if cls in [PaintByExampleImageEncoder]:
            # ignore mapper. prefix, we will use convert_pytorch_state_dict_to_paddle to convert mapper.xxxx state_dict
            ignore_value.append("mapper.")
        elif cls in [IFSafetyChecker]:
            pass
        else:
            name_mapping_dict.update({".vision_model.": "."})

        donot_transpose = ["embeddings", "norm", "concept_embeds", "special_care_embeds"]
        if not hasattr(cls, "paddle_torch_name_mapping"):
            cls.paddle_torch_name_mapping = {}
        for name, value in state_dict.items():
            torch_name = name
            # step1: ignore position_ids
            if any(i in name for i in ignore_value):
                continue
            # step2: transpose nn.Linear weight
            if value.ndim == 2 and not any(i in name for i in donot_transpose):
                value = value.T
            # step3: hf_name -> ppnlp_name mapping
            for hf_name, ppnlp_name in name_mapping_dict.items():
                name = name.replace(hf_name, ppnlp_name)
            # step4: 0d tensor -> 1d tensor
            if name == "logit_scale" and value.ndim == 1:
                value = value.reshape((1,))
            # step5: safety_checker need prefix "clip."
            if "vision_model" in name and cls in [StableDiffusionSafetyChecker, SafeStableDiffusionSafetyChecker]:
                name = "clip." + name
            new_model_state[name] = value

            cls.paddle_torch_name_mapping[name] = torch_name

        if cls in [PaintByExampleImageEncoder]:
            # convert mapper
            mappersd = cls.smart_convert(state_dict, pd_model, sub_layer="mapper.")
            new_model_state.update(mappersd)

        return new_model_state

    @classmethod
    def bert_smart_convert(cls, state_dict, pd_model):
        new_model_state = {}
        name_mapping_dict = {
            # about embeddings
            "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
            "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
            # about encoder layer
            "encoder.layer": "encoder.layers",
            "attention.self.query": "self_attn.q_proj",
            "attention.self.key": "self_attn.k_proj",
            "attention.self.value": "self_attn.v_proj",
            "attention.output.dense": "self_attn.out_proj",
            "attention.output.LayerNorm.weight": "norm1.weight",
            "attention.output.LayerNorm.bias": "norm1.bias",
            "intermediate.dense": "linear1",
            "output.dense": "linear2",
            "output.LayerNorm.weight": "norm2.weight",
            "output.LayerNorm.bias": "norm2.bias",
            # about cls predictions ignore
            "cls.predictions.transform.dense": "cls.predictions.transform",
            "cls.predictions.decoder.weight": "cls.predictions.decoder_weight",
            "cls.predictions.transform.LayerNorm.weight": "cls.predictions.layer_norm.weight",
            "cls.predictions.transform.LayerNorm.bias": "cls.predictions.layer_norm.bias",
            "cls.predictions.bias": "cls.predictions.decoder_bias",
        }
        ignore_value = ["position_ids"]
        donot_transpose = ["embeddings", "norm"]
        if not hasattr(cls, "paddle_torch_name_mapping"):
            cls.paddle_torch_name_mapping = {}
        for name, value in state_dict.items():
            torch_name = name
            # step1: ignore position_ids
            if any(i in name for i in ignore_value):
                continue
            # step2: transpose nn.Linear weight
            if value.ndim == 2 and not any(i in name for i in donot_transpose):
                value = value.T
            # step3: hf_name -> ppnlp_name mapping
            for hf_name, ppnlp_name in name_mapping_dict.items():
                name = name.replace(hf_name, ppnlp_name)
            new_model_state[name] = value
            cls.paddle_torch_name_mapping[name] = torch_name

        return new_model_state

    @classmethod
    def ldmbert_smart_convert(cls, state_dict, pd_model):
        transformers2ppnlp = {
            "model.embed_tokens.weight": "embeddings.word_embeddings.weight",
            "model.embed_positions.weight": "embeddings.position_embeddings.weight",
            "model.layer_norm.": "final_layer_norm.",
            "model.layers": "encoder.layers",
            ".self_attn_layer_norm.": ".norm1.",
            ".final_layer_norm.": ".norm2.",
            ".fc1.": ".linear1.",
            ".fc2.": ".linear2.",
        }
        ignore_value = ["to_logits"]
        donot_transpose = ["embed_tokens", "embed_positions", "norm"]
        new_model_state = {}
        if not hasattr(cls, "paddle_torch_name_mapping"):
            cls.paddle_torch_name_mapping = {}
        for name, value in state_dict.items():
            torch_name = name
            # step1: ignore to_logits
            if any(i in name for i in ignore_value):
                continue
            # step2: transpose nn.Linear weight
            if value.ndim == 2 and not any(i in name for i in donot_transpose):
                value = value.T
            # step3: hf_name -> ppnlp_name mapping
            for hf_name, ppnlp_name in transformers2ppnlp.items():
                name = name.replace(hf_name, ppnlp_name)
            new_model_state[name] = value
            cls.paddle_torch_name_mapping[name] = torch_name

        return new_model_state

    LDMBertModel.smart_convert = ldmbert_smart_convert
    for cls_ in [
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModel,
        CLIPVisionModelWithProjection,
        StableDiffusionSafetyChecker,
        SafeStableDiffusionSafetyChecker,
        PaintByExampleImageEncoder,
        IFSafetyChecker,
    ]:
        setattr(cls_, "smart_convert", clip_smart_convert)

    for cls_ in [BertModel, RobertaSeriesModelWithTransformation]:
        setattr(cls_, "smart_convert", bert_smart_convert)

    if bool(os.getenv("USE_TORCH_LINEAR", False)):
        # NEW TRANSFORMERS CLIP MODEL
        from ..pipelines.stable_diffusion.hf_clip_model import (
            HFCLIPModel,
            HFCLIPTextModel,
            HFCLIPTextModelWithProjection,
            HFCLIPVisionModel,
            HFCLIPVisionModelWithProjection,
        )

        TRANSFORMERS_CLIP_MODEL = [
            HFCLIPModel,
            HFCLIPTextModel,
            HFCLIPTextModelWithProjection,
            HFCLIPVisionModel,
            HFCLIPVisionModelWithProjection,
        ]
    else:
        TRANSFORMERS_CLIP_MODEL = []
    for cls_ in [
        DPTForDepthEstimation,
        BitBackbone,
        SpeechT5HifiGan,
        ClapTextModelWithProjection,
        T5EncoderModel,
    ] + TRANSFORMERS_CLIP_MODEL:
        setattr(cls_, "smart_convert", convert_pytorch_state_dict_to_paddle_class_method)

    # TODO remove this when we updage ImageProcessingMixin
    # patch get_image_processor_dict support subfolder.

    IMAGE_PROCESSOR_NAME = "preprocessor_config.json"
    from paddlenlp.transformers.feature_extraction_utils import FeatureExtractionMixin
    from paddlenlp.transformers.image_processing_utils import ImageProcessingMixin

    @classmethod
    def get_image_processor_dict(cls, pretrained_model_name_or_path, **kwargs):
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        user_agent = kwargs.pop("user_agent", None)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        resolved_image_processor_file = _get_model_file(
            pretrained_model_name_or_path,
            weights_name=IMAGE_PROCESSOR_NAME,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            from_hf_hub=from_hf_hub,
        )
        try:
            # Load image_processor dict
            with open(resolved_image_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            image_processor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_image_processor_file}' is not a valid JSON file."
            )
        # use ppdiffusers logger, not ppnlp_logger
        logger.info(
            f"loading configuration file {resolved_image_processor_file} from cache at {resolved_image_processor_file}"
        )

        return image_processor_dict, kwargs

    ImageProcessingMixin.get_image_processor_dict = get_image_processor_dict
    FeatureExtractionMixin.get_feature_extractor_dict = get_image_processor_dict

    # patch T5LayerFF, we will remove this in the near future.
    from paddlenlp.transformers.t5.modeling import T5LayerFF

    def new_forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        # make sure FP32 + FP16 = FP32
        hidden_states = self.dropout(forwarded_states) + hidden_states
        return hidden_states

    T5LayerFF.forward = new_forward
