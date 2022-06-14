# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sklearn
from scipy.stats import pearsonr, spearmanr
import collections
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score

import json
import pickle
import random

import paddle

from paddlenlp.transformers import (
    CosineDecayWithWarmup,
    LinearDecayWithWarmup,
    PolyDecayWithWarmup,
)


def accuracy(targets, predictions):
    return {"accuracy": 100 * accuracy_score(targets, predictions)}


def sklearn_metrics_wrapper(metric_str,
                            metric_dict_str=None,
                            metric_post_process_fn=None,
                            **metric_fn_kwargs):

    def fn(targets, predictions):
        if metric_str == "matthews_corrcoef":
            metric_fn = matthews_corrcoef
        else:
            metric_fn = getattr(sklearn.metrics, metric_str)
        metric_val = metric_fn(targets, predictions, **metric_fn_kwargs)
        if metric_post_process_fn is not None:
            metric_val = metric_post_process_fn(metric_val)
        return {metric_dict_str or metric_str: metric_val}

    return fn


def f1_score_with_invalid(targets, predictions):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    invalid_idx_mask = np.logical_and(predictions != 0, predictions != 1)
    predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
    return {"f1": 100 * f1_score(targets, predictions)}


def pearson_corrcoef(targets, predictions):
    return {"pearson_corrcoef": 100 * pearsonr(targets, predictions)[0]}


def spearman_corrcoef(targets, predictions):
    return {"spearman_corrcoef": 100 * spearmanr(targets, predictions)[0]}


GLUE_METRICS = collections.OrderedDict([
    (
        "cola",
        [
            sklearn_metrics_wrapper("matthews_corrcoef",
                                    metric_post_process_fn=lambda x: 100 * x)
        ],
    ),
    ("sst-2", [accuracy]),
    ("mrpc", [f1_score_with_invalid, accuracy]),
    ("sts-b", [pearson_corrcoef, spearman_corrcoef]),
    ("qqp", [f1_score_with_invalid, accuracy]),
    ("mnli", [accuracy]),
    ("qnli", [accuracy]),
    ("rte", [accuracy]),
    ("wnli", [accuracy]),
    ("ax", []),  # Only test set available.
])

scheduler_type2cls = {
    "linear": LinearDecayWithWarmup,
    "cosine": CosineDecayWithWarmup,
    "poly": PolyDecayWithWarmup,
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def get_writer(args):
    if args.writer_type == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.logdir)
    elif args.writer_type == "tensorboard":
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=args.logdir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer


def get_scheduler(
    learning_rate,
    scheduler_type,
    num_warmup_steps=None,
    num_training_steps=None,
    **scheduler_kwargs,
):
    if scheduler_type not in scheduler_type2cls.keys():
        data = " ".join(scheduler_type2cls.keys())
        raise ValueError(f"scheduler_type must be choson from {data}")

    if num_warmup_steps is None:
        raise ValueError(
            f"requires `num_warmup_steps`, please provide that argument.")

    if num_training_steps is None:
        raise ValueError(
            f"requires `num_training_steps`, please provide that argument.")

    return scheduler_type2cls[scheduler_type](
        learning_rate=learning_rate,
        total_steps=num_training_steps,
        warmup=num_warmup_steps,
        **scheduler_kwargs,
    )


def save_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as w:
        w.write(json.dumps(data, ensure_ascii=False, indent=4) + "\n")


def save_pickle(data, file_path):
    with open(str(file_path), "wb") as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    with open(str(input_file), "rb") as f:
        data = pickle.load(f)
    return data
