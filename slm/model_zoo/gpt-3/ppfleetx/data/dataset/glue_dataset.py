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

import os

import numpy as np
import paddle
from ppfleetx.data.tokenizers import GPTTokenizer
from ppfleetx.utils.download import cached_path
from ppfleetx.utils.file import parse_csv, unzip

__all__ = ["CoLA", "SST2", "MNLI", "QNLI", "RTE", "WNLI", "MRPC", "QQP", "STSB"]
"""

Single-Sentence Tasks:
* CoLA
* SST-2


Similarity and Paraphrase Tasks:
* MRPC
* STS-B
* QQP


Inference Tasks:
* MNLI
* QNLI
* RTE
* WNLI
"""


class CoLA(paddle.io.Dataset):
    """The Corpus of Linguistic Acceptability consists of English
    acceptability judgments drawn from books and journal articles on
    linguistic theory. Each example is a sequence of words annotated
    with whether it is a grammatical English sentence."""

    # ref https://pytorch.org/text/stable/_modules/torchtext/datasets/cola.html#CoLA

    URL = "https://nyu-mll.github.io/CoLA/cola_public_1.1.zip"
    MD5 = "9f6d88c3558ec424cd9d66ea03589aba"

    NUM_LINES = {
        "train": 8551,
        "dev": 527,
        "test": 516,
    }

    _PATH = "cola_public_1.1.zip"

    DATASET_NAME = "CoLA"

    _EXTRACTED_FILES = {
        "train": os.path.join("raw", "in_domain_train.tsv"),
        "dev": os.path.join("raw", "in_domain_dev.tsv"),
        "test": os.path.join("raw", "out_of_domain_dev.tsv"),
    }

    def __init__(self, root, split, max_length=128):

        self.root = root
        self.split = split
        if os.path.exists(self.root):
            assert os.path.isdir(self.root)
        else:
            zip_path = cached_path(self.URL, cache_dir=os.path.abspath(self.root))
            unzip(zip_path, mode="r", out_dir=os.path.join(self.root, ".."), delete=True)

        self.path = os.path.join(self.root, self._EXTRACTED_FILES[split])
        assert os.path.exists(self.path), f"{self.path} is not exists!"
        self.max_length = max_length

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        assert split in ["train", "dev", "test"]

        def _filter_res(x):
            return len(x) == 4

        def _modify_res(x):
            return (x[3], int(x[1]))

        self.samples = parse_csv(
            self.path, skip_lines=1, delimiter="\t", map_funcs=_modify_res, filter_funcs=_filter_res
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoded_inputs = self.tokenizer(
            sample[0],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        input_ids = encoded_inputs["input_ids"]
        input_ids = paddle.to_tensor(input_ids)
        if self.split != "test":
            return input_ids, sample[1]
        else:
            return input_ids

    def __len__(self):
        return len(self.samples)

    @property
    def class_num(self):
        return 2


class SST2(paddle.io.Dataset):
    """The Stanford Sentiment Treebank consists of sentences from movie reviews and
    human annotations of their sentiment. The task is to predict the sentiment of a
    given sentence. We use the two-way (positive/negative) class split, and use only
    sentence-level labels."""

    # ref https://pytorch.org/text/stable/_modules/torchtext/datasets/sst2.html#SST2

    URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
    MD5 = "9f81648d4199384278b86e315dac217c"

    NUM_LINES = {
        "train": 67349,
        "dev": 872,
        "test": 1821,
    }

    _PATH = "SST-2.zip"

    DATASET_NAME = "SST2"

    _EXTRACTED_FILES = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv",
    }

    def __init__(self, root, split, max_length=128):

        self.root = root
        self.split = split
        if os.path.exists(self.root):
            assert os.path.isdir(self.root)
        else:
            zip_path = cached_path(self.URL, cache_dir=os.path.abspath(self.root))
            unzip(zip_path, mode="r", out_dir=os.path.join(self.root, ".."), delete=True)

        self.path = os.path.join(self.root, self._EXTRACTED_FILES[split])
        assert os.path.exists(self.path), f"{self.path} is not exists!"
        self.max_length = max_length

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        assert split in ["train", "dev", "test"]

        # test split for SST2 doesn't have labels
        if split == "test":

            def _modify_test_res(t):
                return (t[1].strip(),)

            self.samples = parse_csv(self.path, skip_lines=1, delimiter="\t", map_funcs=_modify_test_res)
        else:

            def _modify_res(t):
                return (t[0].strip(), int(t[1]))

            self.samples = parse_csv(self.path, skip_lines=1, delimiter="\t", map_funcs=_modify_res)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoded_inputs = self.tokenizer(
            sample[0],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        input_ids = encoded_inputs["input_ids"]
        input_ids = paddle.to_tensor(input_ids)
        if self.split != "test":
            return input_ids, sample[1]
        else:
            return input_ids

    def __len__(self):
        return len(self.samples)

    @property
    def class_num(self):
        return 2


class MNLI(paddle.io.Dataset):
    """The Multi-Genre Natural Language Inference Corpus is a crowdsourced
    collection of sentence pairs with textual entailment annotations. Given a premise sentence
    and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis
    (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are
    gathered from ten different sources, including transcribed speech, fiction, and government reports.
    We use the standard test set, for which we obtained private labels from the authors, and evaluate
    on both the matched (in-domain) and mismatched (cross-domain) section. We also use and recommend
    the SNLI corpus as 550k examples of auxiliary training data."""

    # ref https://pytorch.org/text/stable/_modules/torchtext/datasets/mnli.html#MNLI

    URL = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
    MD5 = "0f70aaf66293b3c088a864891db51353"

    NUM_LINES = {
        "train": 392702,
        "dev_matched": 9815,
        "dev_mismatched": 9832,
    }

    _PATH = "multinli_1.0.zip"

    DATASET_NAME = "MNLI"

    _EXTRACTED_FILES = {
        "train": "multinli_1.0_train.txt",
        "dev_matched": "multinli_1.0_dev_matched.txt",
        "dev_mismatched": "multinli_1.0_dev_mismatched.txt",
    }

    LABEL_TO_INT = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def __init__(self, root, split, max_length=128):

        self.root = root
        self.split = split
        if os.path.exists(self.root):
            assert os.path.isdir(self.root)
        else:
            zip_path = cached_path(self.URL, cache_dir=os.path.abspath(self.root))
            unzip(zip_path, mode="r", out_dir=os.path.join(self.root, ".."), delete=True)

        self.path = os.path.join(self.root, self._EXTRACTED_FILES[split])
        assert os.path.exists(self.path), f"{self.path} is not exists!"
        self.max_length = max_length

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        assert split in ["train", "dev_matched", "dev_mismatched"]

        def _filter_res(x):
            return x[0] in self.LABEL_TO_INT

        def _modify_res(x):
            return (x[5], x[6], self.LABEL_TO_INT[x[0]])

        self.samples = parse_csv(
            self.path, skip_lines=1, delimiter="\t", map_funcs=_modify_res, filter_funcs=_filter_res
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoded_inputs = self.tokenizer(
            sample[0],
            text_pair=sample[1],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        input_ids = encoded_inputs["input_ids"]
        input_ids = paddle.to_tensor(input_ids)
        return input_ids, sample[2]

    def __len__(self):
        return len(self.samples)

    @property
    def class_num(self):
        return 3


class QNLI(paddle.io.Dataset):
    """The Stanford Question Answering Dataset is a question-answering
    dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn
    from Wikipedia) contains the answer to the corresponding question (written by an annotator). We
    convert the task into sentence pair classification by forming a pair between each question and each
    sentence in the corresponding context, and filtering out pairs with low lexical overlap between the
    question and the context sentence. The task is to determine whether the context sentence contains
    the answer to the question. This modified version of the original task removes the requirement that
    the model select the exact answer, but also removes the simplifying assumptions that the answer
    is always present in the input and that lexical overlap is a reliable cue."""

    # ref https://pytorch.org/text/stable/_modules/torchtext/datasets/qnli.html#QNLI

    URL = "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip"
    MD5 = "b4efd6554440de1712e9b54e14760e82"

    NUM_LINES = {
        "train": 104743,
        "dev": 5463,
        "test": 5463,
    }

    _PATH = "QNLIv2.zip"

    DATASET_NAME = "QNLI"

    _EXTRACTED_FILES = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv",
    }

    MAP_LABELS = {"entailment": 0, "not_entailment": 1}

    def __init__(self, root, split, max_length=128):

        self.root = root
        self.split = split
        if os.path.exists(self.root):
            assert os.path.isdir(self.root)
        else:
            zip_path = cached_path(self.URL, cache_dir=os.path.abspath(self.root))
            unzip(zip_path, mode="r", out_dir=os.path.join(self.root, ".."), delete=True)

        self.path = os.path.join(self.root, self._EXTRACTED_FILES[split])
        assert os.path.exists(self.path), f"{self.path} is not exists!"
        self.max_length = max_length

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        assert split in ["train", "dev", "test"]

        def _modify_res(x):
            if split == "test":
                # test split for QNLI doesn't have labels
                return (x[1], x[2])
            else:
                return (x[1], x[2], self.MAP_LABELS[x[3]])

        self.samples = parse_csv(self.path, skip_lines=1, delimiter="\t", map_funcs=_modify_res)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoded_inputs = self.tokenizer(
            sample[0],
            text_pair=sample[1],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        input_ids = encoded_inputs["input_ids"]
        input_ids = paddle.to_tensor(input_ids)
        if self.split != "test":
            return input_ids, sample[2]
        else:
            return input_ids

    def __len__(self):
        return len(self.samples)

    @property
    def class_num(self):
        return 2


class RTE(paddle.io.Dataset):
    """The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual
    entailment challenges. We combine the data from RTE1 (Dagan et al., 2006), RTE2 (Bar Haim
    et al., 2006), RTE3 (Giampiccolo et al., 2007), and RTE5 (Bentivogli et al., 2009).4 Examples are
    constructed based on news and Wikipedia text. We convert all datasets to a two-class split, where
    for three-class datasets we collapse neutral and contradiction into not entailment, for consistency."""

    # ref https://pytorch.org/text/stable/_modules/torchtext/datasets/rte.html#RTE

    URL = "https://dl.fbaipublicfiles.com/glue/data/RTE.zip"
    MD5 = "bef554d0cafd4ab6743488101c638539"

    NUM_LINES = {
        "train": 67349,
        "dev": 872,
        "test": 1821,
    }

    _PATH = "RTE.zip"

    DATASET_NAME = "RTE"

    _EXTRACTED_FILES = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv",
    }

    MAP_LABELS = {"entailment": 0, "not_entailment": 1}

    def __init__(self, root, split, max_length=128):

        self.root = root
        self.split = split
        if os.path.exists(self.root):
            assert os.path.isdir(self.root)
        else:
            zip_path = cached_path(self.URL, cache_dir=os.path.abspath(self.root))
            unzip(zip_path, mode="r", out_dir=os.path.join(self.root, ".."), delete=True)

        self.path = os.path.join(self.root, self._EXTRACTED_FILES[split])
        assert os.path.exists(self.path), f"{self.path} is not exists!"
        self.max_length = max_length

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        assert split in ["train", "dev", "test"]

        def _modify_res(x):
            if split == "test":
                # test split for RTE doesn't have labels
                return (x[1], x[2])
            else:
                return (x[1], x[2], self.MAP_LABELS[x[3]])

        self.samples = parse_csv(self.path, skip_lines=1, delimiter="\t", map_funcs=_modify_res)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoded_inputs = self.tokenizer(
            sample[0],
            text_pair=sample[1],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        input_ids = encoded_inputs["input_ids"]
        input_ids = paddle.to_tensor(input_ids)
        if self.split != "test":
            return input_ids, sample[2]
        else:
            return input_ids

    def __len__(self):
        return len(self.samples)

    @property
    def class_num(self):
        return 2


class WNLI(paddle.io.Dataset):
    """The Winograd Schema Challenge (Levesque et al., 2011) is a reading comprehension task
    in which a system must read a sentence with a pronoun and select the referent of that pronoun from
    a list of choices. The examples are manually constructed to foil simple statistical methods: Each
    one is contingent on contextual information provided by a single word or phrase in the sentence.
    To convert the problem into sentence pair classification, we construct sentence pairs by replacing
    the ambiguous pronoun with each possible referent. The task is to predict if the sentence with the
    pronoun substituted is entailed by the original sentence. We use a small evaluation set consisting of
    new examples derived from fiction books that was shared privately by the authors of the original
    corpus. While the included training set is balanced between two classes, the test set is imbalanced
    between them (65% not entailment). Also, due to a data quirk, the development set is adversarial:
    hypotheses are sometimes shared between training and development examples, so if a model memorizes the
    training examples, they will predict the wrong label on corresponding development set
    example. As with QNLI, each example is evaluated separately, so there is not a systematic correspondence
    between a model's score on this task and its score on the unconverted original task. We
    call converted dataset WNLI (Winograd NLI)."""

    # ref https://pytorch.org/text/stable/_modules/torchtext/datasets/wnli.html#WNLI

    URL = "https://dl.fbaipublicfiles.com/glue/data/WNLI.zip"
    MD5 = "a1b4bd2861017d302d29e42139657a42"

    NUM_LINES = {
        "train": 635,
        "dev": 71,
        "test": 146,
    }

    _PATH = "WNLI.zip"

    DATASET_NAME = "WNLI"

    _EXTRACTED_FILES = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv",
    }

    def __init__(self, root, split, max_length=128):

        self.root = root
        self.split = split
        if os.path.exists(self.root):
            assert os.path.isdir(self.root)
        else:
            zip_path = cached_path(self.URL, cache_dir=os.path.abspath(self.root))
            unzip(zip_path, mode="r", out_dir=os.path.join(self.root, ".."), delete=True)

        self.path = os.path.join(self.root, self._EXTRACTED_FILES[split])
        assert os.path.exists(self.path), f"{self.path} is not exists!"
        self.max_length = max_length

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        assert split in ["train", "dev", "test"]

        def _modify_res(x):
            if split == "test":
                # test split for WNLI doesn't have labels
                return (x[1], x[2])
            else:
                return (x[1], x[2], int(x[3]))

        self.samples = parse_csv(self.path, skip_lines=1, delimiter="\t", map_funcs=_modify_res)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoded_inputs = self.tokenizer(
            sample[0],
            text_pair=sample[1],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        input_ids = encoded_inputs["input_ids"]
        input_ids = paddle.to_tensor(input_ids)
        if self.split != "test":
            return input_ids, sample[2]
        else:
            return input_ids

    def __len__(self):
        return len(self.samples)

    @property
    def class_num(self):
        return 2


class MRPC(paddle.io.Dataset):
    """The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of
    sentence pairs automatically extracted from online news sources, with human annotations
    for whether the sentences in the pair are semantically equivalent."""

    # ref https://pytorch.org/text/stable/_modules/torchtext/datasets/mrpc.html#MRPC

    URL = {
        "train": "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt",
        "test": "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt",
    }

    MD5 = {
        "train": "793daf7b6224281e75fe61c1f80afe35",
        "test": "e437fdddb92535b820fe8852e2df8a49",
    }

    NUM_LINES = {
        "train": 4076,
        "test": 1725,
    }

    DATASET_NAME = "MRPC"

    _EXTRACTED_FILES = {
        "train": "msr_paraphrase_train.txt",
        "test": "msr_paraphrase_test.txt",
    }

    def __init__(self, root, split, max_length=128):

        self.root = root
        self.split = split
        if os.path.exists(self.root):
            assert os.path.isdir(self.root)
        cached_path(self.URL[split], cache_dir=os.path.abspath(self.root))

        self.path = os.path.join(self.root, self._EXTRACTED_FILES[split])
        assert os.path.exists(self.path), f"{self.path} is not exists!"
        self.max_length = max_length

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        assert split in ["train", "test"]

        def _modify_res(x):
            return (x[3], x[4], int(x[0]))

        self.samples = parse_csv(self.path, skip_lines=1, delimiter="\t", map_funcs=_modify_res)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoded_inputs = self.tokenizer(
            sample[0],
            text_pair=sample[1],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        input_ids = encoded_inputs["input_ids"]
        input_ids = paddle.to_tensor(input_ids)
        return input_ids, sample[2]

    def __len__(self):
        return len(self.samples)

    @property
    def class_num(self):
        return 2


class QQP(paddle.io.Dataset):
    """The Quora Question Pairs2 dataset is a collection of question pairs from the
    community question-answering website Quora. The task is to determine whether a
    pair of questions are semantically equivalent."""

    # ref https://huggingface.co/datasets/glue/blob/main/glue.py#L212-L239

    URL = "https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip"
    MD5 = "884bf26e39c783d757acc510a2a516ef"

    NUM_LINES = {
        "train": 363846,
        "dev": 40430,
        "test": 390961,
    }

    _PATH = "QQP-clean.zip"

    DATASET_NAME = "QQP"

    _EXTRACTED_FILES = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv",
    }

    MAP_LABELS = {"not_duplicate": 0, "duplicate": 1}

    def __init__(self, root, split, max_length=128):

        self.root = root
        self.split = split
        if os.path.exists(self.root):
            assert os.path.isdir(self.root)
        else:
            zip_path = cached_path(self.URL, cache_dir=os.path.abspath(self.root))
            unzip(zip_path, mode="r", out_dir=os.path.join(self.root, ".."), delete=True)

        self.path = os.path.join(self.root, self._EXTRACTED_FILES[split])
        assert os.path.exists(self.path), f"{self.path} is not exists!"
        self.max_length = max_length

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        assert split in ["train", "dev", "test"]

        def _modify_res(x):
            if split == "test":
                # test split for QQP doesn't have labels
                return (x[1], x[2])
            else:
                return (x[3], x[4], int(x[5]))

        self.samples = parse_csv(self.path, skip_lines=1, delimiter="\t", map_funcs=_modify_res)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoded_inputs = self.tokenizer(
            sample[0],
            text_pair=sample[1],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        input_ids = encoded_inputs["input_ids"]
        input_ids = paddle.to_tensor(input_ids)
        if self.split != "test":
            return input_ids, sample[2]
        else:
            return input_ids

    def __len__(self):
        return len(self.samples)

    @property
    def class_num(self):
        return 2


class STSB(paddle.io.Dataset):
    """The Semantic Textual Similarity Benchmark (Cer et al., 2017) is a collection of
    sentence pairs drawn from news headlines, video and image captions, and natural
    language inference data. Each pair is human-annotated with a similarity score
    from 1 to 5."""

    # ref https://huggingface.co/datasets/glue/blob/main/glue.py#L240-L267

    URL = "https://dl.fbaipublicfiles.com/glue/data/STS-B.zip"
    MD5 = "d573676be38f1a075a5702b90ceab3de"

    NUM_LINES = {
        "train": 5749,
        "dev": 1500,
        "test": 1379,
    }

    _PATH = "STS-B.zip"

    DATASET_NAME = "STSB"

    _EXTRACTED_FILES = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv",
    }

    def __init__(self, root, split, max_length=128):

        self.root = root
        self.split = split
        if os.path.exists(self.root):
            assert os.path.isdir(self.root)
        else:
            zip_path = cached_path(self.URL, cache_dir=os.path.abspath(self.root))
            unzip(zip_path, mode="r", out_dir=os.path.join(self.root, ".."), delete=True)

        self.path = os.path.join(self.root, self._EXTRACTED_FILES[split])
        assert os.path.exists(self.path), f"{self.path} is not exists!"
        self.max_length = max_length

        self.tokenizer = GPTTokenizer.from_pretrained("gpt2")

        assert split in ["train", "dev", "test"]

        def _modify_res(x):
            if split == "test":
                # test split for STSB doesn't have labels
                return (x[7], x[8])
            else:
                return (x[7], x[8], float(x[9]))

        self.samples = parse_csv(self.path, skip_lines=1, delimiter="\t", map_funcs=_modify_res)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoded_inputs = self.tokenizer(
            sample[0],
            text_pair=sample[1],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        input_ids = encoded_inputs["input_ids"]
        input_ids = paddle.to_tensor(input_ids)
        if self.split != "test":
            # Note(GuoxiaWang): We need return shape [1] value,
            # so that we can attain a batched label with shape [batchsize, 1].
            # Because the logits shape is [batchsize, 1], and feed into MSE loss.
            return input_ids, np.array([sample[2]], dtype=np.float32)
        else:
            return input_ids

    def __len__(self):
        return len(self.samples)

    @property
    def class_num(self):
        return 2
