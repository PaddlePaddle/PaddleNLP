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

from dataclasses import dataclass, field
from typing import Optional
import copy
import yaml
import os.path as osp

from paddlenlp.data import Stack, Tuple, Pad

TASKS = [
    "SequenceClassification",
    "TokenClassification",
    "QuestionAnswering",
]

config = yaml.load(open(osp.join(osp.abspath("."), "./config.yml"), 'r'),
                   Loader=yaml.FullLoader)
default_args = config["DefaultArgs"]

ALL_DATASETS = {}

for task_type in TASKS:
    task = config[task_type]
    for data_name in task.keys():
        new_args = task[data_name]
        new_args = {} if new_args is None else new_args
        final_args = copy.deepcopy(default_args)
        final_args.update(new_args)
        final_args["model"] = "AutoModelFor{}".format(task_type)
        ALL_DATASETS[data_name] = final_args


class Dict(object):

    def __init__(self, fn):
        assert isinstance(fn, (dict)), 'Input pattern not understood. The input of Dict must be a dict with key of input column name and value of collate_fn ' \
                                   'Received fn=%s' % (str(fn))

        self._fn = fn

        for col_name, ele_fn in self._fn.items():
            assert callable(
                ele_fn
            ), 'Batchify functions must be callable! type(fn[%d]) = %s' % (
                col_name, str(type(ele_fn)))

    def __call__(self, data):

        ret = {}
        if len(data) <= 0:
            return ret

        for col_name, ele_fn in self._fn.items():
            # skip unused col_name, such as labels in test mode.
            if col_name not in data[0].keys():
                continue
            result = ele_fn([ele[col_name] for ele in data])
            ret[col_name] = result

        return ret


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset: str = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        })

    max_seq_length: int = field(
        default=128,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    # Additional configs for QA task.
    doc_stride: int = field(
        default=128,
        metadata={
            "help":
            "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )

    n_best_size: int = field(
        default=20,
        metadata={
            "help":
            "The total number of n-best predictions to generate in the nbest_predictions.json output file."
        },
    )

    max_query_length: int = field(
        default=64,
        metadata={"help": "Max query length."},
    )

    max_answer_length: int = field(
        default=30,
        metadata={"help": "Max answer length."},
    )

    do_lower_case: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to lower case the input text. Should be True for uncased models and False for cased models."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help":
            "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        })
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the dataset cache."},
    )
    export_model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory to store the exported inference model."
        },
    )
