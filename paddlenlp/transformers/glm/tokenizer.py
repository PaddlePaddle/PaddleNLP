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

import json
import os
from shutil import copyfile
from typing import List, Optional, Tuple

import paddle
import paddle.nn.functional as F
import sentencepiece as spm
from scipy.linalg import block_diag

from ...utils.log import logger
from .. import BertTokenizer, GPTTokenizer
from ..tokenizer_utils import PretrainedTokenizer
from ..tokenizer_utils_base import BatchEncoding


class GLMTokenizerMixin:
    """
    BOS and EOS tokens are used for autoregressive blank filling.
    """

    @property
    def sop_token(self) -> Optional[str]:
        return "<|startofpiece|>"

    @property
    def sop_token_id(self) -> Optional[int]:
        return self.convert_tokens_to_ids(self.sop_token)

    @property
    def eop_token(self) -> Optional[str]:
        return "<|endofpiece|>"

    @property
    def eop_token_id(self) -> Optional[int]:
        return self.convert_tokens_to_ids(self.eop_token)

    @property
    def gmask_token_id(self) -> int:
        return self.convert_tokens_to_ids("[gMASK]")

    @property
    def smask_token_id(self) -> int:
        return self.convert_tokens_to_ids("[sMASK]")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id, self.smask_token_id, self.gmask_token_id]

    def _build_input_for_multiple_choice(self, context, choices):
        context_id = context["input_ids"]
        if isinstance(context_id, paddle.Tensor):
            context_id = context_id.tolist()

        division = len(context_id)
        mask_position = context_id.index(self.mask_token_id)
        token = paddle.to_tensor(context_id, dtype="int64")
        attention_mask = [context["attention_mask"].expand([division, -1])]
        position_id = paddle.arange(division, dtype="int64")
        block_position_id = paddle.zeros([division], dtype="int64")

        choice_ids, choice_indices = [], []

        for choice_str in choices:
            choice = paddle.to_tensor(
                self(choice_str, add_special_tokens=False, padding=False)["input_ids"],
                dtype="int64",
            )
            choice_ids.append(choice)

            choice_indices.append(paddle.arange(len(token), len(token) + len(choice), dtype="int64"))
            attention_mask.append(paddle.tril(paddle.ones([len(choice), len(choice)], dtype="int64")))

            token = paddle.concat([token, paddle.to_tensor([self.sop_token_id], dtype="int64"), choice[:-1]])
            position_id = paddle.concat([position_id, paddle.to_tensor([mask_position] * len(choice), dtype="int64")])
            block_position_id = paddle.concat([block_position_id, paddle.arange(1, len(choice) + 1, dtype="int64")])

        attention_mask = paddle.to_tensor(block_diag(*[x.tolist() for x in attention_mask]))
        attention_mask[division:, :division] = context["attention_mask"].unsqueeze(0)

        return {
            "input_ids": token,
            "position_ids": paddle.stack([position_id, block_position_id]),
            "attention_mask": attention_mask,
            "choice_ids": choice_ids,
            "choice_indices": choice_indices,
        }

    def _pad_batch(self, tokens, position_ids, attention_mask, max_seq_length):
        pad_length = max_seq_length - len(tokens)
        attention_mask = F.pad(attention_mask, [0, pad_length, 0, pad_length], mode="constant", value=0)
        tokens = paddle.concat([tokens, paddle.zeros([pad_length], dtype="int64")])
        if pad_length > 0:
            position_ids = paddle.concat([position_ids, position_ids[..., -1:].expand([-1, pad_length])], axis=-1)
        return tokens, position_ids, attention_mask

    def _collate(self, samples):
        TILE = 1
        length_to_pad = (max(map(lambda spl: len(spl["input_ids"]), samples)) + TILE - 1) // TILE * TILE

        token_batch, position_id_batch, attention_mask_batch = [], [], []
        choices_batch, choice_target_ids_batch = [], []

        for sample in samples:
            token, position_id, attention_mask = self._pad_batch(
                sample["input_ids"], sample["position_ids"], sample["attention_mask"], length_to_pad
            )
            token_batch.append(token)
            position_id_batch.append(position_id)
            attention_mask_batch.append(attention_mask)
            choices_batch.append(sample["choice_ids"])
            choice_target_ids_batch.append(sample["choice_indices"])
        return BatchEncoding(
            {
                "input_ids": paddle.stack(token_batch),
                "position_ids": paddle.stack(position_id_batch),
                "attention_mask": paddle.stack(attention_mask_batch).unsqueeze(1),
                "choice_ids": choices_batch,
                "choice_indices": choice_target_ids_batch,
            }
        )

    def build_inputs_for_multiple_choice(self, model_input: BatchEncoding, choices, max_length=None):
        samples = [{key: value[i] for key, value in model_input.items()} for i in range(len(model_input["input_ids"]))]
        samples = [self._build_input_for_multiple_choice(sample, choice) for sample, choice in zip(samples, choices)]
        inputs = self._collate(samples)
        return BatchEncoding(inputs)

    def build_inputs_for_generation(
        self,
        model_input: BatchEncoding,
        max_gen_length=512,
        targets=None,
        padding=False,
        is_train=False,
    ):
        mask_ids = self.mask_token_ids
        input_ids = model_input.input_ids
        batch_size, seq_length = input_ids.shape[:2]
        position_id, block_position_id = list(range(seq_length)), [0 for _ in range(seq_length)]
        position_ids, block_position_ids = [], []
        labels = None
        loss_mask = None
        if targets is not None:
            is_batched = isinstance(targets, (list, tuple))
            targets = self(targets, add_special_tokens=False, padding=False).input_ids
            if not is_batched:
                targets = [targets]
            assert len(targets) == len(input_ids)
            targets = [(target + [self.eop_token_id])[:max_gen_length] for target in targets]
            if not padding:
                max_gen_length = max(map(len, targets))
            targets = [[self.sop_token_id] + target for target in targets]
            labels = [target[1:] for target in targets]
            targets = [target + [self.pad_token_id] * (max_gen_length + 1 - len(target)) for target in targets]
            labels = [label + [self.pad_token_id] * (max_gen_length - len(label)) for label in labels]
            targets = paddle.to_tensor(targets, dtype=input_ids.dtype)
            loss_mask = paddle.logical_and(targets != self.pad_token_id, targets != self.eop_token_id).astype("int64")
            labels = paddle.to_tensor(labels, dtype=input_ids.dtype)
            labels = paddle.concat([paddle.zeros([batch_size, seq_length], dtype=labels.dtype), labels], axis=1)

        for i in range(batch_size):
            mask_positions = []
            for mask_id in mask_ids:
                mask_positions += (input_ids[i] == mask_id).nonzero(as_tuple=True)[0].squeeze(1).tolist()
            if not mask_positions:
                raise ValueError("Cannot find mask token in the input.")
            mask_positions.sort()
            mask_pos = mask_positions[0]
            position_ids.append(position_id + [mask_pos] * max_gen_length)
            block_position_ids.append(block_position_id + list(range(1, max_gen_length + 1)))
        position_ids = paddle.to_tensor(position_ids, dtype=input_ids.dtype)
        block_position_ids = paddle.to_tensor(block_position_ids, dtype=input_ids.dtype)
        position_ids = paddle.stack([position_ids, block_position_ids], axis=1)

        attention_mask = model_input.attention_mask
        attention_mask = attention_mask.unsqueeze(1).expand([-1, seq_length + max_gen_length, -1])
        generation_attention_mask = (
            paddle.concat(
                [
                    paddle.zeros([seq_length, max_gen_length], dtype=attention_mask.dtype),
                    paddle.tril(paddle.ones([max_gen_length, max_gen_length], dtype=attention_mask.dtype)),
                ],
                axis=0,
            )
            .unsqueeze(0)
            .expand([batch_size, -1, -1])
        )
        attention_mask = paddle.concat([attention_mask, generation_attention_mask], axis=2).unsqueeze(1)

        if targets is None:
            input_ids = paddle.concat(
                [input_ids, paddle.full([batch_size, 1], self.sop_token_id, dtype=input_ids.dtype)], axis=-1
            )
        else:
            loss_mask = paddle.concat([paddle.zeros_like(input_ids), loss_mask], axis=1)
            input_ids = paddle.concat([input_ids, targets[:, :-1]], axis=1)
            loss_mask = loss_mask[:, : len(input_ids[0])]

        batch = {"input_ids": input_ids, "position_ids": position_ids}
        if labels is None:
            batch["attention_mask"] = attention_mask
        else:
            batch["attention_mask"] = attention_mask
            batch["loss_mask"] = loss_mask
            batch["label_ids"] = labels
        return BatchEncoding(batch)


class GLMChineseTokenizer(PretrainedTokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    resource_files_names = {"model_file": "cog-pretrain.model", "added_tokens_file": "added_tokens.json"}
    truncation_side: str = "left"
    pretrained_init_configuration = {
        "glm-large-chinese": {"do_lower_case": True},
        "glm-10b-chinese": {"do_lower_case": True},
    }
    cog_model_link = "https://paddlenlp.bj.bcebos.com/models/transformers/glm/cog-pretrain.model"
    added_tokens_link = "https://paddlenlp.bj.bcebos.com/models/transformers/glm/glm-chinese-added-tokens.json"
    pretrained_resource_files_map = {
        "model_file": {
            "glm-large-chinese": cog_model_link,
            "glm-10b-chinese": cog_model_link,
        },
        "added_tokens_file": {
            "glm-large-chinese": added_tokens_link,
            "glm-10b-chinese": added_tokens_link,
        },
    }
    max_model_input_sizes = {"glm-10b-chinese": 1024, "glm-large-chinese": 1024}

    def __init__(
        self,
        model_file,
        cls_token="[CLS]",
        sep_token="[SEP]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        pad_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        **kwargs
    ):
        super().__init__(
            cls_token=cls_token,
            sep_token=sep_token,
            unk_token=unk_token,
            mask_token=mask_token,
            pad_token=pad_token,
            eos_token=eos_token,
            **kwargs,
        )
        self._model_file = model_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_file)

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.sp_model.Encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.Decode(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is not None:
            logger.warning("Support single input text and the second one is ignored.")
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        return cls + token_ids_0 + eos


class GLMGPT2Tokenizer(GPTTokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"
    pretrained_init_configuration = {
        "glm-2b": {},
        "glm-10b": {},
    }
    added_tokens_link = "https://paddlenlp.bj.bcebos.com/models/transformers/glm/glm-added-tokens.json"
    pretrained_resource_files_map = {
        "vocab_file": {
            "glm-2b": "https://paddlenlp.bj.bcebos.com/models/transformers/glm/glm-2b-vocab.json",
            "glm-10b": "https://paddlenlp.bj.bcebos.com/models/transformers/glm/glm-10b-vocab.json",
        },
        "merges_file": {
            "glm-2b": "https://paddlenlp.bj.bcebos.com/models/transformers/glm/glm-2b-merges.txt",
            "glm-10b": "https://paddlenlp.bj.bcebos.com/models/transformers/glm/glm-10b-merges.txt",
        },
        "added_tokens_file": {
            "glm-2b": added_tokens_link,
            "glm-10b": added_tokens_link,
        },
    }
    max_model_input_sizes = {
        "glm-2b": 1024,
        "glm-10b": 1024,
    }

    def __init__(
        self,
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        pad_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        **kwargs
    ):
        super().__init__(cls_token=cls_token, sep_token=sep_token, pad_token=pad_token, eos_token=eos_token, **kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        if token_ids_1 is not None:
            logger.warning("Support single input text and the second one is ignored.")
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        return cls + token_ids_0 + eos


class GLMBertTokenizer(BertTokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"
    pretrained_init_configuration = {
        "glm-515m": {"do_lower_case": True},
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "glm-515m": "https://paddlenlp.bj.bcebos.com/models/transformers/glm/glm-515m-vocab.txt",
        },
    }
    max_model_input_sizes = {
        "glm-515m": 512,
    }


class GLMTokenizer:
    """
    GLMTokenizer is a generic tokenizer class that will be instantiated as GLMChineseTokenizer,
    GLMGPT2Tokenizer or GLMBertTokenizer when created with GLMTokenizer.from_pretrained() class method.
    """

    bert_model_names = GLMBertTokenizer.pretrained_init_configuration.keys()
    chinese_model_names = GLMChineseTokenizer.pretrained_init_configuration.keys()
    gpt2_model_names = GLMGPT2Tokenizer.pretrained_init_configuration.keys()
    tokenizer_config_file = "tokenizer_config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        # From built-in pretrained models
        if pretrained_model_name_or_path in cls.bert_model_names:
            return GLMBertTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        elif pretrained_model_name_or_path in cls.chinese_model_names:
            return GLMChineseTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        elif pretrained_model_name_or_path in cls.gpt2_model_names:
            return GLMGPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        # From local dir path
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, cls.tokenizer_config_file)
            with open(config_file, "r", encoding="utf-8") as fp:
                tokenizer_config = json.load(fp)
            config_tokenizer_class = tokenizer_config.get("tokenizer_class")
            if config_tokenizer_class == "GLMChineseTokenizer":
                tokenizer_class = GLMChineseTokenizer
            elif config_tokenizer_class == "GLMGPT2Tokenizer":
                tokenizer_class = GLMGPT2Tokenizer
            elif config_tokenizer_class == "GLMBertTokenizer":
                tokenizer_class = GLMBertTokenizer
            else:
                raise NotImplementedError("Not implemented tokenizer type:", config_tokenizer_class)
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        # TODO: Assuming from community-contributed pretrained models
