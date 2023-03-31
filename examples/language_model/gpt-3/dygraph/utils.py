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

import copy
import inspect
import os
import re
import shutil
from pathlib import Path

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import TensorParallel
from paddle.distributed.sharding import group_sharded_parallel

from paddlenlp.utils.log import logger

try:
    from paddle.fluid.dygraph.parallel import sync_params_buffers
except ImportError:
    from paddle.distributed.parallel import sync_params_buffers

import evaluate
import nltk

from paddlenlp.metrics import BLEU

__all__ = [
    "merge_model_parallel",
    "_rotate_checkpoints",
    "all_gather",
    "wrap_sharding_2_3",
    "is_dp_group_support_in_group_sharded_parallel",
]

PREFIX_CHECKPOINT_DIR = "model_state"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\.tp(\d+)" + ".pdparams$")


def use_hybrid_parallel():
    try:
        hcg = fleet.get_hybrid_communicate_group()
        return hcg
    except:
        return None


def optimizer_name_suffix():
    hcg = use_hybrid_parallel()
    if hcg is not None:
        name = []
        if hcg.get_model_parallel_world_size() > 1:
            name.append(f"tp{hcg.get_model_parallel_rank():0>2d}")
        if hcg.get_pipe_parallel_world_size() > 1:
            name.append(f"pp{hcg.get_stage_id():0>2d}")
        if hcg.get_sharding_parallel_world_size() > 1:
            name.append(f"shard{hcg.get_sharding_parallel_rank():0>2d}")

        return "_".join(name)
    else:
        return None


def weight_name_suffix():
    hcg = use_hybrid_parallel()
    if hcg is not None:
        name = []
        if hcg.get_model_parallel_world_size() > 1:
            name.append(f"tp{hcg.get_model_parallel_rank():0>2d}")
        if hcg.get_pipe_parallel_world_size() > 1:
            name.append(f"pp{hcg.get_stage_id():0>2d}")
        return "_".join(name)
    else:
        return None


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


def MergedKeys(num_layers):
    res = {}
    Column = [
        "gpt.decoder.layers.0.linear1.bias",
        "gpt.decoder.layers.0.linear1.weight",
        "gpt.decoder.layers.0.self_attn.qkv_proj.bias",
        "gpt.decoder.layers.0.self_attn.qkv_proj.weight",
    ]

    Row = [
        "gpt.embeddings.word_embeddings.weight",
        # 'gpt.decoder.layers.0.self_attn.out_proj.bias',
        "gpt.decoder.layers.0.self_attn.out_proj.weight",
        # 'gpt.decoder.layers.0.linear2.bias',
        "gpt.decoder.layers.0.linear2.weight",
    ]
    for v in Column:
        if "layers.0." in v:
            for i in range(num_layers):
                res[v.replace("layers.0.", f"layers.{i}.")] = "col"
        else:
            res[v] = "col"
    for v in Row:
        if "layers.0." in v:
            for i in range(num_layers):
                res[v.replace("layers.0.", f"layers.{i}.")] = "row"
        else:
            res[v] = "row"

    return res


def merge_rows(values):
    return np.concatenate(values, axis=0)


def merge_column(values):
    return np.concatenate(values, axis=-1)


def merge_model_parallel(model_path, config, as_float32=True):
    final_weight = None
    weights_path = get_model_parallel_paramerters(model_path)
    if len(weights_path) == 1:
        final_weight = paddle.load(weights_path[0], return_numpy=True)
    else:
        weights_list = []
        for path in weights_path:
            weights_list.append(paddle.load(path, return_numpy=True))

        final_weight = copy.deepcopy(weights_list[0])
        merged_keys = MergedKeys(config.num_hidden_layers)

        for k, func_name in merged_keys.items():
            func = merge_column if "col" == func_name else merge_rows
            final_weight[k] = func([weight[k] for weight in weights_list])

    if as_float32:
        for k in final_weight.keys():
            final_weight[k] = final_weight[k].astype("float32")

    return final_weight


def convert_example(
    example,
    tokenizer,
    decoder_start_token_id,
    max_source_length,
    max_target_length,
    ignore_pad_token_for_loss=True,
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
        # pad_to_max_seq_len=True,
        truncation_strategy="longest_first",
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    inputs = tokenizer(
        input_seq,
        max_seq_len=max_source_length,
        # pad_to_max_seq_len=True,
        truncation_strategy="longest_first",
        return_attention_mask=True,
        return_length=False,
    )

    final = {}
    for k in outputs.keys():
        final[k] = inputs[k] + outputs[k]
        if k == "input_ids":
            final["labels"] = [tokenizer.pad_token_id] * len(inputs["input_ids"]) + outputs[k]

    # input_ids [0, 1, 2, 3, 4, PAD, PAD]
    # attn_mask [1, 1, 1, 1, 1, 0, 0]
    # labels    [1, 2, 3, 4, -100, -100, -100]
    # labels    [PAD, PAD, 3, 4, -100, -100, -100]
    # loss_mask [1, 1, 1, 1, 0, 0, 0]

    return final
    # output_ids = [decoder_start_token_id] + outputs["input_ids"][:-1]

    # if ignore_pad_token_for_loss:
    #     # Replace all tokenizer.pad_token_id in the outputs by -100 when we want to ignore padding in the loss.
    #     outputs["input_ids"] = [(l if l != tokenizer.pad_token_id else -100) for l in outputs["input_ids"]]

    # if is_train:
    #     inputs = tokenizer(
    #         input_seq,
    #         max_seq_len=max_source_length,
    #         pad_to_max_seq_len=True,
    #         truncation_strategy="longest_first",
    #         return_attention_mask=True,
    #         return_length=False,
    #     )
    #     return inputs["input_ids"], inputs["attention_mask"], output_ids, outputs["input_ids"]
    # else:
    #     inputs = tokenizer(
    #         input_seq,
    #         max_seq_len=max_source_length,
    #         pad_to_max_seq_len=True,
    #         truncation_strategy="longest_first",
    #         return_attention_mask=True,
    #         return_length=True,
    #     )
    #     return inputs["input_ids"], inputs["attention_mask"], inputs["length"], output_ids, outputs["input_ids"]


def compute_metrics(preds, labels, tokenizer, ignore_pad_token_for_loss=True):
    def compute_bleu(predictions, references, rouge_types=None, use_stemmer=True):
        bleu1 = BLEU(n_size=1)
        bleu2 = BLEU(n_size=2)
        bleu3 = BLEU(n_size=3)
        bleu4 = BLEU(n_size=4)
        assert len(predictions) == len(references)
        for i in range(len(predictions)):
            bleu1.add_inst(predictions[i], [references[i]])
            bleu2.add_inst(predictions[i], [references[i]])
            bleu3.add_inst(predictions[i], [references[i]])
            bleu4.add_inst(predictions[i], [references[i]])
        result = {
            "BLEU-1": bleu1.score() * 100,
            "BLEU-2": bleu2.score() * 100,
            "BLEU-3": bleu3.score() * 100,
            "BLEU-4": bleu4.score() * 100,
        }
        return result

    def compute_bleu_hf(predictions, references, rouge_types=None, use_stemmer=True):
        predictions = [" ".join(prediction) for prediction in predictions]
        references = [[" ".join(reference)] for reference in references]

        bleu = evaluate.load("bleu")
        assert len(predictions) == len(references)
        bleu1_results = bleu.compute(predictions=predictions, references=references, max_order=1)
        bleu2_results = bleu.compute(predictions=predictions, references=references, max_order=2)
        bleu3_results = bleu.compute(predictions=predictions, references=references, max_order=3)
        bleu4_results = bleu.compute(predictions=predictions, references=references, max_order=4)

        result = {
            "BLEU-1": bleu1_results["bleu"] * 100,
            "BLEU-2": bleu2_results["bleu"] * 100,
            "BLEU-3": bleu3_results["bleu"] * 100,
            "BLEU-4": bleu4_results["bleu"] * 100,
        }
        return result

    def post_process_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = [pred.strip("question:") for pred in preds]
        labels = [label.strip("question:") for label in labels]
        # spreds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        #  expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        preds = [pred.split() for pred in preds]
        labels = [label.split() for label in labels]

        return preds, labels

    def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
        """
        Post-process the decoded sequence.
        """
        eos_pos = len(seq) - 1
        for i, idx in enumerate(seq):
            if idx == eos_idx:
                eos_pos = i
                break
        seq = [idx for idx in seq[: eos_pos + 1] if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)]
        return seq

    if ignore_pad_token_for_loss:
        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds, decoded_labels = [], []
    for pred, label in zip(preds, labels):
        pred_id = post_process_seq(pred, tokenizer.bos_token_id, tokenizer.eos_token_id)
        label_id = post_process_seq(label, tokenizer.bos_token_id, tokenizer.eos_token_id)
        decoded_preds.append(tokenizer.decode(pred_id))
        decoded_labels.append(tokenizer.decode(label_id))
    decoded_preds, decoded_labels = post_process_text(decoded_preds, decoded_labels)
    # bleu_result = compute_bleu(decoded_preds, decoded_labels)
    bleu_result = compute_bleu_hf(decoded_preds, decoded_labels)
    return bleu_result, decoded_preds, decoded_labels


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
