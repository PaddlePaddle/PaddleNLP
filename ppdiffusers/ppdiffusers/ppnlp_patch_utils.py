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

import builtins
import contextlib
import copy
import functools
import time
from types import FunctionType, MethodType
from typing import Any, Dict, List, Optional, Tuple

from .utils import is_paddle_available, is_paddlenlp_available


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
    from paddle.fluid import framework

    @contextlib.contextmanager
    def device_scope(device="cpu"):
        new_device = framework._get_paddle_place(device.replace("cuda", "gpu"))
        old_device = framework._current_expected_place()
        if str(new_device) == str(old_device):
            yield
        else:
            try:
                framework._set_expected_place(new_device)
                yield
            finally:
                framework._set_expected_place(old_device)

    paddle.device_scope = device_scope

    class RNGStatesTracker:
        def __init__(self):
            self.states_ = {}

        def reset(self):
            self.states_ = {}

        def remove(self, generator_name=None):
            if generator_name is not None:
                del self.states_[generator_name]

        def manual_seed(self, seed, generator_name=None):
            if generator_name is None:
                generator_name = str(time.time())
            if generator_name in self.states_:
                raise ValueError("state {} already exists".format(generator_name))
            orig_rng_state = paddle.get_cuda_rng_state()
            paddle.seed(seed)
            self.states_[generator_name] = paddle.get_cuda_rng_state()
            paddle.set_cuda_rng_state(orig_rng_state)
            return generator_name

        @contextlib.contextmanager
        def rng_state(self, generator_name=None):
            if generator_name is not None:
                if generator_name not in self.states_:
                    raise ValueError("state {} does not exist".format(generator_name))
                orig_cuda_rng_state = paddle.get_cuda_rng_state()
                paddle.set_cuda_rng_state(self.states_[generator_name])
                try:
                    yield
                finally:
                    self.states_[generator_name] = paddle.get_cuda_rng_state()
                    paddle.set_cuda_rng_state(orig_cuda_rng_state)
            else:
                yield

    RNG_STATE_TRACKER = RNGStatesTracker()

    def get_rng_state_tracker(*args, **kwargs):
        return RNG_STATE_TRACKER

    paddle.Generator = get_rng_state_tracker
    randn = paddle.randn

    def randn_pt(shape, dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        if generator is None:
            return randn(shape, dtype=dtype, name=name)
        else:
            with get_rng_state_tracker().rng_state(generator):
                return randn(shape, dtype=dtype, name=name)

    paddle.randn = randn_pt

    rand = paddle.rand

    def rand_pt(shape, dtype=None, name=None, **kwargs):
        generator = kwargs.get("generator", None)
        if generator is None:
            return randn(shape, dtype=dtype, name=name)
        else:
            with get_rng_state_tracker().rng_state(generator):
                return rand(shape, dtype=dtype, name=name)

    paddle.rand = rand_pt

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


if is_paddle_available() and is_paddlenlp_available():
    import paddle

    import paddlenlp.transformers
    from paddlenlp.transformers import PretrainedModel

    @patch_to(PretrainedModel, as_prop=True)
    def dtype(self):
        try:
            return next(self.named_parameters())[1].dtype
        except StopIteration:
            return paddle.get_default_dtype()

    @patch_to(PretrainedModel, as_prop=True)
    def device(self):
        try:
            return next(self.named_parameters())[1].place
        except StopIteration:
            return paddle.get_device()

    try:
        from paddlenlp.transformers import (
            CLIPTextModelWithProjection,
            CLIPVisionModelWithProjection,
        )
    except ImportError:
        # patch model
        from dataclasses import dataclass

        from paddlenlp.transformers import (
            CLIPPretrainedModel,
            TextTransformer,
            VisionTransformer,
            register_base_model,
        )
        from paddlenlp.transformers.model_outputs import ModelOutput

        @dataclass
        class CLIPVisionModelOutput(ModelOutput):
            image_embeds: Optional[paddle.Tensor] = None
            last_hidden_state: paddle.Tensor = None
            hidden_states: Optional[Tuple[paddle.Tensor]] = None
            attentions: Optional[Tuple[paddle.Tensor]] = None

        @dataclass
        class CLIPTextModelOutput(ModelOutput):
            text_embeds: Optional[paddle.Tensor] = None
            last_hidden_state: paddle.Tensor = None
            hidden_states: Optional[Tuple[paddle.Tensor]] = None
            attentions: Optional[Tuple[paddle.Tensor]] = None

        @register_base_model
        class CLIPTextModelWithProjection(CLIPPretrainedModel):
            base_model_class = None

            def __init__(
                self,
                max_text_length=77,
                text_embed_dim=512,
                text_heads=8,
                text_layers=12,
                vocab_size=49408,
                text_hidden_act="quick_gelu",
                initializer_range=0.02,
                initializer_factor=1.0,
                projection_dim=512,
                **kwargs
            ):
                super().__init__()
                self.initializer_range = initializer_range
                self.initializer_factor = initializer_factor
                self.text_embed_dim = text_embed_dim
                self.text_layers = text_layers
                self.text_model = TextTransformer(
                    context_length=max_text_length,
                    transformer_width=text_embed_dim,
                    transformer_heads=text_heads,
                    transformer_layers=text_layers,
                    vocab_size=vocab_size,
                    activation=text_hidden_act,
                    normalize_before=True,
                )
                self.text_projection = paddle.create_parameter(
                    (text_embed_dim, projection_dim), paddle.get_default_dtype()
                )
                self.apply(self._init_weights)

            def get_input_embeddings(self) -> nn.Layer:
                return self.text_model.token_embedding

            def set_input_embeddings(self, value):
                self.text_model.token_embedding = value

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            ):
                text_outputs = self.text_model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                pooled_output = text_outputs[1]
                text_embeds = paddle.matmul(pooled_output, self.text_projection)

                if not return_dict:
                    outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
                    return tuple(output for output in outputs if output is not None)

                return CLIPTextModelOutput(
                    text_embeds=text_embeds,
                    last_hidden_state=text_outputs.last_hidden_state,
                    hidden_states=text_outputs.hidden_states,
                    attentions=text_outputs.attentions,
                )

        @register_base_model
        class CLIPVisionModelWithProjection(CLIPPretrainedModel):
            base_model_class = None

            def __init__(
                self,
                image_resolution=224,
                vision_patch_size=32,
                vision_embed_dim=768,
                vision_layers=12,
                vision_heads=12,
                vision_hidden_act="quick_gelu",
                vision_mlp_ratio=4,
                initializer_range=0.02,
                initializer_factor=1.0,
                projection_dim=512,
                **kwargs
            ):
                super().__init__()
                self.initializer_range = initializer_range
                self.initializer_factor = initializer_factor
                self.vision_embed_dim = vision_embed_dim
                self.vision_layers = vision_layers

                if vision_heads is None:
                    vision_heads = vision_embed_dim // 64
                self.vision_model = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_embed_dim,
                    layers=vision_layers,
                    heads=vision_heads,
                    activation=vision_hidden_act,
                    mlp_ratio=vision_mlp_ratio,
                    normalize_before=True,
                )
                self.vision_projection = paddle.create_parameter(
                    (vision_embed_dim, projection_dim), paddle.get_default_dtype()
                )
                self.apply(self._init_weights)

            def get_input_embeddings(self) -> nn.Layer:
                return self.vision_model.conv1

            def forward(
                self,
                pixel_values=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            ):
                vision_outputs = self.vision_model(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                pooled_output = vision_outputs[1]  # pooled_output
                image_embeds = paddle.matmul(pooled_output, self.vision_projection)
                if not return_dict:
                    outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
                    return tuple(output for output in outputs if output is not None)

                return CLIPVisionModelOutput(
                    image_embeds=image_embeds,
                    last_hidden_state=vision_outputs.last_hidden_state,
                    hidden_states=vision_outputs.hidden_states,
                    attentions=vision_outputs.attentions,
                )

        CLIPTextModelWithProjection.base_model_class = CLIPTextModelWithProjection
        CLIPVisionModelWithProjection.base_model_class = CLIPVisionModelWithProjection

        paddlenlp.transformers.CLIPTextModelWithProjection = CLIPTextModelWithProjection
        paddlenlp.transformers.CLIPVisionModelWithProjection = CLIPVisionModelWithProjection

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
        from paddlenlp.utils.log import logger

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

    raw_forward = BertModel.forward

    @patch_to(BertModel)
    def forward(
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
        return raw_forward(
            self,
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            past_key_values,
            use_cache,
            output_hidden_states,
            output_attentions,
            return_dict,
        )
