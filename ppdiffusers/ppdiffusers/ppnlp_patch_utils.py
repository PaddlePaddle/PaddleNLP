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

from .utils import (
    DIFFUSERS_CACHE,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
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
    smart_load,
)

logger = get_logger(__name__)

__all__ = []


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
                setattr(c_, nm, MethodType(nf, c_))
            else:
                setattr(c_, nm, property(nf) if as_prop else nf)
        # Avoid clobbering existing functions
        return globals().get(nm, builtins.__dict__.get(nm, None))

    return _inner


if is_paddle_available():
    import paddle
    import paddle.nn as nn

    paddle.long = paddle.int64
    paddle.int = paddle.int32
    paddle.double = paddle.float64
    paddle.half = paddle.float16
    paddle.from_numpy = paddle.to_tensor
    paddle.Tensor.half = lambda x: paddle.cast(x, paddle.float16)
    paddle.Tensor.float = lambda x: paddle.cast(x, paddle.float32)
    paddle.Tensor.double = lambda x: paddle.cast(x, paddle.float64)
    paddle.Tensor.int = lambda x: paddle.cast(x, paddle.int32)
    paddle.Tensor.long = lambda x: paddle.cast(x, paddle.int64)
    paddle.Tensor.bool = lambda x: paddle.cast(x, paddle.bool)
    paddle.Tensor.bfloat16 = lambda x: paddle.cast(x, paddle.bfloat16)
    paddle.Tensor.clamp = paddle.clip
    paddle.clamp = paddle.clip

    def view_pt(x, *shape: builtins.int, name=None):
        return paddle.reshape(x, shape=shape, name=name)

    paddle.view = view_pt
    paddle.Tensor.view = view_pt
    setattr(paddle.Tensor, "data", property(lambda x: x))
    paddle.Tensor.data_ptr = lambda x: x.value().get_tensor()._ptr()

    def permute_pt(x, *perm: builtins.int, name=None):
        return paddle.transpose(x, perm=perm, name=name)

    paddle.permute = permute_pt
    paddle.Tensor.permute = permute_pt
    paddle.cat = paddle.concat
    paddle.Tensor.softmax = nn.functional.softmax

    # patch repeat_interleave
    raw_repeat_interleave = paddle.repeat_interleave

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

    def size_pt(self, i=None):
        if i is None:
            return self.shape
        return self.shape[i]

    paddle.Tensor.size = size_pt
    paddle.Tensor.contiguous = lambda x: x

    # must return self!
    @patch_to(nn.Layer)
    def eval(self):
        # Layer-level setting
        self.training = False
        for layer in self.sublayers():
            layer.training = False
        return self

    @patch_to(nn)
    def Parameter(data: paddle.Tensor, requires_grad=True):
        tensor = paddle.create_parameter(data.shape, dtype=data.dtype, default_initializer=nn.initializer.Assign(data))
        if not requires_grad:
            tensor.stop_gradient = True
        return tensor

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

    @patch_to(nn.Layer)
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

    @patch_to(nn.Layer)
    def register_load_state_dict_pre_hook(self, hook, with_module=False):
        handle = HookRemoveHelper(self.load_state_dict_pre_hooks)
        self.load_state_dict_pre_hooks[handle._hook_id] = _WrappedHook(hook, self if with_module else None)
        return handle

    raw_set_state_dict = nn.Layer.set_state_dict

    @patch_to(nn.Layer)
    def set_state_dict(self, state_dict, use_structured_name: bool = True):
        for hook in self.load_state_dict_pre_hooks.values():
            hook(state_dict)
        return raw_set_state_dict(self, state_dict, use_structured_name=use_structured_name)

    nn.Layer.load_dict = nn.Layer.set_state_dict
    nn.Layer.set_dict = nn.Layer.set_state_dict

    raw_init = nn.Layer.__init__

    @patch_to(nn.Layer)
    def __init__(self, name_scope=None, dtype="float32"):
        raw_init(self, name_scope=name_scope, dtype=dtype)
        self.load_state_dict_pre_hooks = OrderedDict()


if is_paddle_available() and is_paddlenlp_available():
    # set logger level warning
    import paddle

    import paddlenlp.transformers
    from paddlenlp import __version__
    from paddlenlp.transformers import PretrainedConfig, PretrainedModel
    from paddlenlp.utils.log import logger as ppnlp_logger

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
            training=False,
            attention_op="cutlass",
        ):
            if attention_op is None or attention_op == "cutlass" or training:
                if scale is None:
                    scale = 1 / math.sqrt(query.shape[-1])
                # support fp32, fp16, bfp16
                output = memory_efficient_attention(
                    query,
                    key,
                    value,
                    attn_mask,
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
                raise ValueError("ppxformers's attention_op shoulde be in ['cutlass', 'flash']")
            return output

        paddle.nn.functional.scaled_dot_product_attention_ = scaled_dot_product_attention_

    @patch_to(nn.Layer, as_prop=True)
    def dtype(parameter: nn.Layer) -> paddle.dtype:
        try:
            return next(parameter.named_parameters())[1].dtype
        except StopIteration:
            return parameter._dtype

    @patch_to(PretrainedModel, as_prop=True)
    def device(self):
        try:
            return next(self.named_parameters())[1].place
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
    def from_pretrained_v3(cls, pretrained_model_name_or_path, from_hf_hub: bool = False, *args, **kwargs):
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_diffusers = kwargs.pop("from_diffusers", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        paddle_dtype = kwargs.pop("paddle_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)

        user_agent = {
            "ppdiffusers": __version__,
            "file_type": "model",
            "framework": "paddle",
        }

        config = None
        # 1. get the PretrainedConfig to init model
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            # must from hf hub
            if from_hf_hub:
                if subfolder is not None:
                    kwargs["subfolder"] = subfolder
            else:
                if subfolder is not None:
                    config_path = os.path.join(config_path, subfolder)

            config = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=False,
                force_download=force_download,
                from_hf_hub=from_hf_hub,
                **kwargs,
            )
        assert config is not None

        model = cls(config)
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

        # convert weights
        if from_diffusers and hasattr(cls, "smart_convert"):
            state_dict = cls.smart_convert(state_dict, model)

        loaded_state_dict_keys = list(state_dict.keys())

        dtype = set(v.dtype for v in state_dict.values())
        if len(dtype) > 1 and paddle.float32 not in dtype:
            raise ValueError(
                f"The weights of the model file {model_file} have a mixture of incompatible dtypes {dtype}. Please"
                f" make sure that {model_file} weights have only one dtype."
            )
        elif len(dtype) > 1 and paddle.float32 in dtype:
            dtype = paddle.float32
        else:
            dtype = dtype.pop()
        model = model.to(dtype=dtype)

        model, missing_keys, unexpected_keys, mismatched_keys = cls._load_pretrained_model(
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

        if paddle_dtype is not None and not isinstance(paddle_dtype, paddle.dtype):
            raise ValueError(
                f"{paddle_dtype} needs to be of type `paddle.dtype`, e.g. `paddle.float16`, but is {type(paddle_dtype)}."
            )
        elif paddle_dtype is not None:
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

    # PretrainedModel.from_pretrained is classmethod
    raw_from_pretrained = PretrainedModel.from_pretrained.__func__
    raw_save_pretrained = PretrainedModel.save_pretrained

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, from_hf_hub=False, subfolder=None, **kwargs):
        if cls.constructed_from_pretrained_config() and hasattr(cls, "smart_convert"):
            return from_pretrained_v3(
                cls, pretrained_model_name_or_path, from_hf_hub=from_hf_hub, subfolder=subfolder, *args, **kwargs
            )
        return raw_from_pretrained(
            cls, pretrained_model_name_or_path, *args, from_hf_hub=from_hf_hub, subfolder=subfolder, **kwargs
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
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        to_diffusers: Optional[bool] = None,
    ):
        from .models.modeling_pytorch_paddle_utils import (
            convert_paddle_state_dict_to_pytorch,
        )
        from .models.modeling_utils import convert_state_dict

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
        save_function: Callable = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        to_diffusers: Optional[bool] = None,
    ):
        if self.constructed_from_pretrained_config() and hasattr(self, "paddle_torch_name_mapping"):
            return save_pretrained_v3(
                self,
                save_dir,
                is_main_process=is_main_process,
                save_function=save_function,
                safe_serialization=safe_serialization,
                variant=variant,
                to_diffusers=to_diffusers,
            )
        return raw_save_pretrained(self, save_dir)

    PretrainedModel.save_pretrained = save_pretrained

    from paddlenlp.transformers import (
        BertModel,
        BitBackbone,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModel,
        CLIPVisionModelWithProjection,
        DPTForDepthEstimation,
    )

    # logger.set_level("WARNING")
    from .models.modeling_pytorch_paddle_utils import (
        convert_pytorch_state_dict_to_paddle,
    )
    from .pipelines.alt_diffusion.modeling_roberta_series import (
        RobertaSeriesModelWithTransformation,
    )
    from .pipelines.latent_diffusion.pipeline_latent_diffusion import LDMBertModel
    from .pipelines.paint_by_example.image_encoder import PaintByExampleImageEncoder
    from .pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from .pipelines.stable_diffusion_safe.safety_checker import (
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
        else:
            name_mapping_dict.update({".vision_model.": "."})

        donot_transpose = ["embeddings", "norm", "concept_embeds", "special_care_embeds"]

        paddle_torch_name_mapping = {}
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

            paddle_torch_name_mapping[name] = torch_name

        cls.paddle_torch_name_mapping = paddle_torch_name_mapping
        if cls in [PaintByExampleImageEncoder]:
            # convert mapper
            mappersd = convert_pytorch_state_dict_to_paddle(state_dict, pd_model, sub_layer="mapper.")
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
        paddle_torch_name_mapping = {}

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
            paddle_torch_name_mapping[name] = torch_name

        cls.paddle_torch_name_mapping = paddle_torch_name_mapping

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
        paddle_torch_name_mapping = {}
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
            paddle_torch_name_mapping[name] = torch_name

        cls.paddle_torch_name_mapping = paddle_torch_name_mapping

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
    ]:
        setattr(cls_, "smart_convert", clip_smart_convert)

    for cls_ in [BertModel, RobertaSeriesModelWithTransformation]:
        setattr(cls_, "smart_convert", bert_smart_convert)

    for cls_ in [DPTForDepthEstimation, BitBackbone]:
        setattr(cls_, "smart_convert", convert_pytorch_state_dict_to_paddle)

    # patch get_image_processor_dict support subfolder.

    IMAGE_PROCESSOR_NAME = "preprocessor_config.json"
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

        ppnlp_logger.info(
            f"loading configuration file {resolved_image_processor_file} from cache at {resolved_image_processor_file}"
        )

        return image_processor_dict, kwargs

    ImageProcessingMixin.get_image_processor_dict = get_image_processor_dict
