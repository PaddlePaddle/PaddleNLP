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

import inspect
import os
import pickle
import random
import re
import shutil
from pathlib import Path

import numpy as np
import paddle
import sklearn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import TensorParallel
from paddle.distributed.sharding import group_sharded_parallel
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from paddlenlp.transformers import (
    CosineDecayWithWarmup,
    LinearDecayWithWarmup,
    PolyDecayWithWarmup,
)
from paddlenlp.utils.log import logger

try:
    from paddle.fluid.dygraph.parallel import sync_params_buffers
except ImportError:
    from paddle.distributed.parallel import sync_params_buffers


def accuracy(targets, predictions):
    return {"accuracy": 100 * accuracy_score(targets, predictions)}


def sklearn_metrics_wrapper(metric_str, metric_dict_str=None, metric_post_process_fn=None, **metric_fn_kwargs):
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


def save_pickle(data, file_path):
    with open(str(file_path), "wb") as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    with open(str(input_file), "rb") as f:
        data = pickle.load(f)
    return data


PREFIX_CHECKPOINT_DIR = "model_state"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\_mp_(\d+)" + ".pdparams$")


def left_padding(inputs, pad_id, padding="longest"):
    assert "input_ids" in inputs, "input_ids should be in inputs!"
    max_length = 0
    for ids in inputs["input_ids"]:
        max_length = max(max_length, len(ids))

    def extend_max_lenth(value, max_length, to_pad_id):
        return [to_pad_id] * (max_length - len(value)) + value

    def extend_filed(name, max_length, to_pad_id):
        values = inputs[name]
        res = []
        for index, value in enumerate(values):
            res.append(extend_max_lenth(value, max_length, to_pad_id))
        inputs[name] = res

    extend_filed("input_ids", max_length, pad_id)
    if "attention_mask" in inputs:
        extend_filed("attention_mask", max_length, 0)
    if "position_ids" in inputs:
        extend_filed("position_ids", max_length, 0)

    return inputs


def get_model_parallel_paramerters(folder):
    content = os.listdir(folder)
    if "model_state.pdparams" in content:
        return [os.path.join(folder, "model_state.pdparams")]

    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isfile(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        raise ValueError("No checkpoint found within folder {}".format(folder))

    return [
        os.path.join(folder, v) for v in sorted(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0]))
    ]


def convert_example(
    example,
    tokenizer,
    decoder_start_token_id,
    max_source_length,
    max_target_length,
    is_train=True,
):
    """
    Convert an example into necessary features.
    """
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    context = example["context"]
    question = example["question"]
    try:
        answer = example["answers"][0]
    except Exception:
        print(example["context"])
        print(example["question"])
        print(example["answers"])
        print(example["answer_starts"])
        print(example["is_impossible"])

    input_seq = f"answer: {answer} context: {context} </s>"
    output_seq = f"question: {question} </s>"

    outputs = tokenizer(
        output_seq,
        max_seq_len=max_target_length,
        truncation_strategy="longest_first",
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    inputs = tokenizer(
        input_seq,
        max_seq_len=max_source_length,
        truncation_strategy="longest_first",
        return_attention_mask=True,
        return_length=False,
    )

    final = {}
    for k in outputs.keys():
        final[k] = inputs[k] + outputs[k]
        if k == "input_ids":
            final["labels"] = [tokenizer.pad_token_id] * len(inputs["input_ids"]) + outputs[k]

    return final


def _sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint", use_mtime=False):
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

    return checkpoints_sorted


def _rotate_checkpoints(save_total_limit, use_mtime=False, output_dir=None) -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
    # we don't do to allow resuming.

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint)


def all_gather(v, group=None):
    if paddle.distributed.get_world_size() <= 1:
        return v.item()
    ret = []
    paddle.distributed.all_gather(ret, v, group=group)
    concat = paddle.concat(ret, axis=0)
    return concat.mean().item()


def is_dp_group_support_in_group_sharded_parallel():
    return "dp_group" in set(inspect.signature(paddle.distributed.sharding.group_sharded_parallel).parameters.keys())


def wrap_sharding_2_3(model, optimizer, scaler, dist_config):
    """_summary_
    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        scaler (_type_): _description_
        dist_config (_type_): _description_
    Returns:
        _type_: _description_
    """
    # group = fleet.get_hybrid_communicate_group().get_sharding_parallel_group()
    # level = "p_g_os" if dist_config.sharding_stage == 3 else "os_g"
    # return group_sharded_parallel(
    #     model=model, optimizer=optimizer, level=level, scaler=scaler, group=group, offload=dist_config.sharding_offload,

    # )

    hcg = fleet.get_hybrid_communicate_group()
    dp_group = hcg.get_data_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()

    # sync params (broadcast) buffers in dp group
    if (
        not is_dp_group_support_in_group_sharded_parallel()
        and dist_config.dp_degree > 1
        and dist_config.sharding_stage == 2
    ):
        sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])

    if dist_config.dp_degree > 1 and dist_config.sharding_stage == 3:
        sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])

    if dist_config.mp_degree > 1:
        assert dist_config.sharding_stage == 2, "only support mp + sharding stage2 hybrid parallel now."
        model = TensorParallel(model, hcg, strategy=None)

    level = "p_g_os" if dist_config.sharding_stage == 3 else "os_g"
    # origin_model = model

    extra_kwargs = {}
    if is_dp_group_support_in_group_sharded_parallel():
        extra_kwargs["dp_group"] = dp_group if dp_group.nranks > 1 else None
        extra_kwargs["exclude_layer"] = ["GroupNorm"]

    model, optimizer, scaler = group_sharded_parallel(
        model=model,
        optimizer=optimizer,
        level=level,
        scaler=scaler,
        group=sharding_group,
        offload=dist_config.sharding_offload,
        # dp_group=dp_group if dp_group.nranks > 1 else None,
        **extra_kwargs,
    )

    # if dist_config.sharding.reduce_overlap:
    #     model._set_reduce_overlap(dist_config.sharding.reduce_overlap)

    # if dist_config.sharding.broadcast_overlap:
    #     optimizer._set_broadcast_overlap(
    #         dist_config.sharding.broadcast_overlap,
    #         layers=origin_model,
    #         num_groups=2)

    return model, optimizer, scaler
