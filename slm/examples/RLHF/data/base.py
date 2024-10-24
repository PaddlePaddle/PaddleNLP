# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
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
"""Base dataset class."""

from __future__ import annotations

import abc
import bisect
import copy
import os
import warnings
from fractions import Fraction
from typing import Any, Callable, ClassVar, Collection, Iterable, Iterator, List
from weakref import WeakValueDictionary

import numpy as np
import paddle
from paddle.io import Dataset, IterableDataset, Subset
from paddle.io.dataloader.collate import default_collate_fn
from tqdm import tqdm
from typing_extensions import NotRequired  # Python 3.11+
from typing_extensions import TypedDict  # Python 3.10+

from paddlenlp.transformers.tokenizer_utils import (
    PaddingStrategy,
    PretrainedTokenizerBase,
    TruncationStrategy,
)

# from safe_rlhf.utils import is_main_process


__all__ = [
    "RawDataset",
    "RawSample",
    "TokenizedDataset",
    "CollatorBase",
    "parse_dataset",
]

IGNORE_INDEX: int = -100
PROMPT_BEGIN: str = "BEGINNING OF CONVERSATION: "
PROMPT_USER: str = "USER: {input} "
PROMPT_ASSISTANT: str = "ASSISTANT:"  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT


def format_prompt(
    input: str | list[str],  # pylint: disable=redefined-builtin
    eos_token: str,
) -> str:
    if isinstance(input, str):
        input = [input]
    elif not isinstance(input, list):
        raise ValueError(f"Unsupported type of `input`: {type(input)}. Expected: str or list[str].")

    if len(input) % 2 != 1:
        raise ValueError(
            "The length of `input` must be odd, while `input` must end at the user question.",
        )

    buffer = [PROMPT_BEGIN]
    for i, line in enumerate(input):
        if i % 2 == 0:
            # User input
            buffer.extend((PROMPT_USER.format(input=line), PROMPT_ASSISTANT))
        else:
            # Assistant response
            buffer.extend((line, eos_token))

    return "".join(buffer)


def left_padding(sequences, padding_value=0):
    arrs = [np.asarray(seq) for seq in sequences]
    max_length = max([len(seq) for seq in sequences])
    bs = len(sequences)
    data = np.full([bs, max_length], padding_value, dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        data[i, -len(arr) :] = arr
    return data


def right_padding(sequences, padding_value=0):
    arrs = [np.asarray(seq) for seq in sequences]
    max_length = max([len(seq) for seq in sequences])
    bs = len(sequences)
    data = np.full([bs, max_length], padding_value, dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        data[i, : len(arr)] = arr
    return data


class ConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn(
            "cummulative_sizes attribute is renamed to " "cumulative_sizes", DeprecationWarning, stacklevel=2
        )
        return self.cumulative_sizes


def parse_dataset(string: str) -> tuple[str, dict[str, Any]]:
    """Parse dataset path and its proportion and optionally additional arguments from a string.

    Args:
        string (str): Dataset string in the format of ``dataset_name[:proportion[:dataset_path]]``.
    """
    if string.count(":") > 2:
        raise ValueError(
            f"Invalid dataset string `{string}`, "
            "should be in the format of `dataset_name[:proportion[:dataset_path]]`.",
        )
    name, colon, proportion = string.partition(":")
    if not colon:
        return name, {"proportion": 1.0}
    proportion, colon, path = proportion.partition(":")
    if "/" in proportion:
        left, right = proportion.split("/")
        left, right = left.strip(), right.strip()
        if right == "-1":
            proportion = Fraction(int(left), -1)
            if proportion == 0:
                proportion = 0.0
        else:
            proportion = float(left) / float(right)
    elif proportion != "":
        proportion = float(proportion)
    else:
        proportion = 1.0
    if not colon:
        return name, {"proportion": proportion}
    if not path:
        raise ValueError(f"Invalid dataset path `{path}`.")
    path = os.path.expanduser(path)
    return name, {"proportion": proportion, "path": path}


class RawSample(TypedDict, total=False):
    """Raw sample type.

    For SupervisedDataset, should provide (input, answer) or (dialogue).
    For PreferenceDataset, should provide (input, answer, other_answer, better).
    For SafetyPreferenceDataset, should provide (input, answer, other_answer, safer, is_safe, is_other_safe).
    For PromptOnlyDataset, should provide (input).

    When input is a list, it would be processed as a dialogue.
    """

    # Texts
    input: NotRequired[str | list[str]]  # either `input` or `dialogue` should be provided
    """User input text."""
    answer: NotRequired[str]
    """Assistant answer text."""
    other_answer: NotRequired[str]
    """Other assistant answer text via resampling."""
    dialogue: NotRequired[list[str]]  # either `input` or `dialogue` should be provided
    """Dialogue history."""

    # Flags
    better: NotRequired[bool]
    """Whether ``answer`` is better than ``other_answer``."""
    safer: NotRequired[bool]
    """Whether ``answer`` is safer than ``other_answer``."""
    is_safe: NotRequired[bool]
    """Whether ``answer`` is safe."""
    is_other_safe: NotRequired[bool]
    """Whether ``other_answer`` is safe."""


class RawDataset(Dataset):
    """Dataset that provides raw text samples."""

    NAME: ClassVar[str]
    """Name of the dataset."""
    ALIASES: ClassVar[Collection[str]]
    """Name aliases of the dataset."""
    __REGISTRY: ClassVar[WeakValueDictionary[str, RawDataset]] = WeakValueDictionary()
    """Registry of all subclasses."""
    __ALIAS_NAME_MAPPING: ClassVar[dict[str, str]] = {}
    """Mapping from aliases to names."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        name = getattr(cls, "NAME", None)
        if name is None:
            return

        if not isinstance(name, str):
            raise ValueError("`NAME` should be a string.")
        if not name:
            raise ValueError("`NAME` should not be empty.")
        if name in cls.__REGISTRY or name in cls.__ALIAS_NAME_MAPPING:
            raise ValueError(f"Duplicate dataset name `{name}`.")
        cls.__REGISTRY[name] = cls

        aliases = set(cls.__dict__.get("ALIASES", ()))  # do not inherit aliases from parent class
        for alias in aliases:
            if not isinstance(alias, str):
                raise ValueError("`ALIASES` should be a set of strings.")
            if not alias:
                raise ValueError("`ALIASES` should not contain empty strings.")
            if alias in cls.__REGISTRY or alias in cls.__ALIAS_NAME_MAPPING:
                raise ValueError(f"Duplicate dataset alias `{alias}`.")
        cls.__ALIAS_NAME_MAPPING.update(dict.fromkeys(aliases, name))

    @staticmethod
    def load(name: str, /, *args: Any, **kwargs: Any) -> RawDataset:  # noqa: E999
        """Load a raw dataset by name."""
        normalized_name = RawDataset.__ALIAS_NAME_MAPPING.get(name, name)
        try:
            cls = RawDataset.__REGISTRY[normalized_name]
        except KeyError as ex:
            raise ValueError(
                f"Unknown dataset name `{name}`. "
                "You should implement a raw dataset with that name and import it in your code. "
                f'Available datasets are: {", ".join(sorted(RawDataset.__REGISTRY.keys()))}.',
            ) from ex
        return cls(*args, **kwargs)

    make = load  # alias

    @abc.abstractmethod
    def __getitem__(self, index: int) -> RawSample:
        """Get a data sample by index."""
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[RawSample]:
        """Iterate over the dataset."""
        return (self[index] for index in range(len(self)))

    def split_train_test(
        self,
        split_ratio: float = 0.2,
        shuffle: bool = True,
        seed: int = 42,
    ) -> tuple[Dataset[RawSample], Dataset[RawSample]]:
        """Split the dataset into train dataset and test dataset by split ratio."""
        num_samples = len(self)
        indices = list(range(num_samples))
        num_test_samples = round(num_samples * split_ratio)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        train_dataset = Subset(self, indices[:-num_test_samples])
        test_dataset = Subset(self, indices[-num_test_samples:])
        return train_dataset, test_dataset


class TokenizedDataset(Dataset):
    """Dataset that provides tokenized samples."""

    raw_datasets: list[RawDataset]
    """Raw datasets."""
    tokenizer: PretrainedTokenizerBase
    """Tokenizer."""
    rawdata: list[RawSample]
    """Merged raw data samples."""
    data: list[dict[str, paddle.Tensor]]
    """Tokenized data samples."""

    _SENTINEL: Any = object()

    def __init__(  # pylint: disable=too-many-branches
        self,
        dataset_names_and_attributes: dict[str, float | dict[str, Any]] | Iterable[tuple[str, float | dict[str, Any]]],
        tokenizer: PretrainedTokenizerBase,
        lazy_tokenization: bool = True,
        seed: int = 42,
    ) -> None:
        if not isinstance(dataset_names_and_attributes, dict):
            dataset_names_and_attributes = tuple(dataset_names_and_attributes)
            dataset_names = [name for name, _ in dataset_names_and_attributes]
            if len(dataset_names) != len(set(dataset_names)):
                raise ValueError(
                    f"Dataset names should be unique, but got {dataset_names}.",
                )

        super().__init__()
        self.dataset_names_and_proportion: dict[str, float | Fraction] = {}
        self.raw_datasets = []
        for name, attributes in dict(dataset_names_and_attributes).items():
            if isinstance(attributes, float):
                kwargs = {"proportion": attributes}
            elif isinstance(attributes, dict):
                kwargs = dict(attributes)  # copy
            else:
                raise TypeError(
                    f"Dataset `{name}` attributes should be a float or a dict, " f"got {type(attributes).__name__}.",
                )
            proportion = kwargs.pop("proportion", 1.0)
            if isinstance(proportion, Fraction):
                if not (proportion < 0 and proportion.denominator == 1):
                    raise ValueError(
                        f"Dataset `{name}` proportion should be a negative integer "
                        f"represents `num_samples / -1`, got {proportion}.",
                    )
            else:
                proportion = float(proportion)
                if proportion < 0.0:
                    raise ValueError(
                        f"Dataset `{name}` proportion should be no less than 0.0, " f"got {proportion}.",
                    )
            if proportion == 0.0:
                continue
            self.dataset_names_and_proportion[name] = proportion
            self.raw_datasets.append(RawDataset.load(name, **kwargs))

        self.tokenizer = tokenizer
        self.seed = seed

        merged_rawdata = self._merge_raw_datasets(seed=seed)
        self.rawdata = [merged_rawdata[i] for i in range(len(merged_rawdata))]
        if lazy_tokenization:
            self.data = [self._SENTINEL for _ in range(len(self.rawdata))]
        else:
            self.data = list(
                map(
                    self.preprocess,
                    tqdm(
                        self.rawdata,
                        desc="Preprocessing raw dataset...",
                        disable=False,  # not is_main_process(),
                    ),
                ),
            )

    def __getitem__(self, index: int) -> dict[str, paddle.Tensor]:
        """Get a tokenized data sample by index."""
        data = self.data[index]
        if data is self._SENTINEL:
            raw_sample = self.rawdata[index]
            data = self.preprocess(raw_sample)
            self.data[index] = data
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.data)

    @abc.abstractmethod
    def preprocess(self, raw_sample: RawSample) -> dict[str, paddle.Tensor]:
        """Pre-process a raw sample into a tokenized sample."""
        raise NotImplementedError

    def _merge_raw_datasets(self, seed: int | None = None) -> Dataset[RawSample]:
        """Merge multiple raw datasets into one dataset."""
        if seed is None:
            seed = self.seed

        num_datasets = len(self.raw_datasets)
        datasets = []
        for raw_dataset, seed_seq in zip(
            self.raw_datasets,
            np.random.SeedSequence(seed).spawn(num_datasets),
        ):
            num_raw_samples = len(raw_dataset)
            proportion = self.dataset_names_and_proportion[raw_dataset.NAME]
            if isinstance(proportion, Fraction):
                num_selected_samples = abs(int(proportion))
            else:
                num_selected_samples = round(num_raw_samples * proportion)

            indices = []
            while num_selected_samples >= num_raw_samples > 0:
                indices.extend(range(num_raw_samples))
                num_selected_samples -= num_raw_samples
            if num_selected_samples > 0:
                rng = np.random.default_rng(seed_seq)
                indices.extend(
                    rng.choice(
                        len(raw_dataset),
                        size=num_selected_samples,
                        replace=False,
                    ).tolist(),
                )
            datasets.append(Subset(raw_dataset, indices))

        if num_datasets == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool | str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: int | None = None,
    ) -> paddle.Tensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors="np",
        )["input_ids"][0]

    def get_collator(self) -> Callable[[list[dict[str, paddle.Tensor]]], dict[str, paddle.Tensor]]:
        """Get a collator function for the dataset."""
        return default_collate_fn

    def resize(self, size: int) -> None:
        """Resize the dataset to the given size."""
        if size < 0:
            raise ValueError(f"Invalid dataset size: {size}")
        old_size = len(self)
        if size == old_size:
            return
        if size > old_size:
            num_replicates = (size + old_size - 1) // old_size
            self.data = self.data * num_replicates
            self.rawdata = self.rawdata * num_replicates
        self.data = self.data[:size]
        self.rawdata = self.rawdata[:size]

    def split_train_test(
        self,
        split_ratio: float = 0.2,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> tuple[Dataset[dict[str, paddle.Tensor]], Dataset[dict[str, paddle.Tensor]]]:
        """Split the dataset into train dataset and test dataset by split ratio."""
        if seed is None:
            seed = self.seed

        num_samples = len(self)
        indices = list(range(num_samples))
        num_test_samples = round(num_samples * split_ratio)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        memo = {id(attr): attr for attr in self.__dict__.values()}
        train_dataset = copy.deepcopy(self, memo=memo.copy())
        test_dataset = copy.deepcopy(self, memo=memo.copy())
        train_dataset.data = [self.data[i] for i in indices[:-num_test_samples]]
        train_dataset.rawdata = [self.rawdata[i] for i in indices[:-num_test_samples]]
        test_dataset.data = [self.data[i] for i in indices[-num_test_samples:]]
        test_dataset.rawdata = [self.rawdata[i] for i in indices[-num_test_samples:]]

        return train_dataset, test_dataset


class CollatorBase(metaclass=abc.ABCMeta):
    pad_token_id: int  # The id of the padding token for the tokenizer.

    def __init__(self, pad_token_id: int) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id

    @abc.abstractmethod
    def __call__(self, samples: list[dict[str, paddle.Tensor]]) -> dict[str, paddle.Tensor]:
        """Collate a list of samples into a batch."""
        raise NotImplementedError
