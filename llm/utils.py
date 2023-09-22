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
from __future__ import annotations

import glob
import math
import os
import struct
from typing import Dict, Optional

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from sklearn.metrics import accuracy_score

from paddlenlp.datasets import InTokensIterableDataset
from paddlenlp.trainer import Trainer, TrainerCallback
from paddlenlp.trainer.trainer_utils import IterableDatasetShard, has_length
from paddlenlp.utils.log import logger


def compute_metrics(eval_preds):

    flattened_preds = np.array(eval_preds.predictions).flatten()
    flattened_labels = np.array(eval_preds.label_ids).flatten()
    filtered_preds = flattened_preds[flattened_labels != -100]
    filtered_labels = flattened_labels[flattened_labels != -100]
    accuracy = accuracy_score(y_true=filtered_labels, y_pred=filtered_preds)
    return {
        "accuracy": accuracy,
    }


def get_prefix_tuning_params(model):
    if model.base_model_prefix == "chatglm":
        from paddlenlp.peft.prefix import chatglm_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = chatglm_postprocess_past_key_value
        multi_query_group_num = None
    elif model.base_model_prefix == "chatglm_v2":
        from paddlenlp.peft.prefix import chatglm_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_layers
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = chatglm_postprocess_past_key_value
        multi_query_group_num = model.config.multi_query_group_num
    elif model.base_model_prefix == "bloom":
        from paddlenlp.peft.prefix import bloom_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.n_layer
        hidden_size = model.config.n_embed
        postprocess_past_key_value = bloom_postprocess_past_key_value
        multi_query_group_num = None
    elif model.base_model_prefix == "llama":
        from paddlenlp.peft.prefix import llama_postprocess_past_key_value

        num_attention_heads = model.config.n_head
        num_hidden_layers = model.config.n_layer
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = llama_postprocess_past_key_value
        multi_query_group_num = None
    elif model.base_model_prefix == "qwen":
        from paddlenlp.peft.prefix import qwen_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = qwen_postprocess_past_key_value
        multi_query_group_num = None
    else:
        raise ValueError(f"Unknown base_model_prefix: {model.base_model_prefix}. ")
    return dict(
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        postprocess_past_key_value=postprocess_past_key_value,
        multi_query_group_num=multi_query_group_num,
    )


def get_lora_target_modules(model):
    # Not yet support RowParallelLinear
    if model.base_model_prefix == "chatglm":
        target_modules = [".*query_key_value.*", ".*dense.*", ".*dense_h_to_4h.*", ".*dense_4h_to_h.*"]
    elif model.base_model_prefix == "chatglm_v2":
        target_modules = [
            ".*query.*",
            ".*key.*",
            ".*value.*",
            ".*dense.*",
            ".*dense_h_to_4h.*",
            ".*dense_4h_to_h.*",
        ]
    elif model.base_model_prefix == "bloom":
        target_modules = [".*query_key_value.*", ".*dense.*", ".*dense_h_to_4h.*", ".*dense_4h_to_h.*"]
    elif model.base_model_prefix == "llama":
        target_modules = [
            ".*q_proj.*",
            ".*v_proj.*",
            ".*k_proj.*",
            ".*o_proj.*",
            ".*gate_proj.*",
            ".*down_proj.*",
            ".*up_proj.*",
        ]
    elif model.base_model_prefix == "opt":
        target_modules = [
            ".*project_in.*",
            ".*project_out.*",
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*qkv_proj.*",
            ".*out_proj.*",
            ".*linear1.*",
            ".*linear2.*",
        ]
    elif model.base_model_prefix == "qwen":
        target_modules = [
            ".*attn.c_attn.*",
            ".*attn.c_proj.*",
            ".*mlp.w1.*",
            ".*mlp.w2.*",
            ".*mlp.c_proj.*",
        ]
    else:
        raise ValueError(f"Unknown base_model_prefix: {model.base_model_prefix}.")
    return target_modules


class InTokensIterDatasetCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles early stopping.

    """

    def on_step_end(self, args, state, control, **kwargs):
        train_dataloader = kwargs["train_dataloader"]
        if isinstance(train_dataloader.dataset, InTokensIterableDataset):
            dataset = train_dataloader.dataset
        elif isinstance(train_dataloader.dataset, IterableDatasetShard) and isinstance(
            train_dataloader.dataset.dataset, InTokensIterableDataset
        ):
            dataset = train_dataloader.dataset.dataset
        else:
            raise ValueError(
                "Unexpected dataset format: InTokensIterDatasetCallback expectes `paddlenlp.datasets.InTokensIterableDataset`"
            )
        if state.trial_params is None:
            state.trial_params = {}
        state.trial_params["intokens_global_step"] = dataset.intokens_global_step


class CausalLMTrainer(Trainer):
    def __init__(self, do_generation: bool, gen_args, data_args, **kwargs):
        super().__init__(**kwargs)
        self.do_generation = do_generation
        self.gen_args = gen_args
        self.data_args = data_args

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        elif not self.do_generation:
            loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            # argmax here to avoid gather all logits, which is too memory-consuming.
            # keepdim in order to maintain the same shape as logits
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            return (loss, logits.argmax(axis=-1, keepdim=True), labels)

        loss = None

        model.eval()
        with paddle.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
                position_ids=inputs["position_ids"] if "position_ids" in inputs else None,
                max_length=max(self.data_args.max_length - inputs["input_ids"].shape[-1], 1),
                decode_strategy="sampling",
                top_k=self.gen_args.top_k,
                top_p=self.gen_args.top_p,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )[0]
            all_preds = []
            for pred_tokens in generated_tokens:
                pred_tokens = pred_tokens[pred_tokens != self.tokenizer.pad_token_id].tolist()
                all_preds.append(pred_tokens)
            max_pred_length = max([len(x) for x in all_preds])
            for index, preds in enumerate(all_preds):
                all_preds[index] = preds + [-100] * (max_pred_length - len(preds))
            all_preds = paddle.to_tensor(all_preds)

            if "labels" in inputs:
                all_labels = paddle.to_tensor(inputs["labels"])
            else:
                all_labels = None

        return (loss, all_preds, all_labels)

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        if "loss" in logs:
            logs["ppl"] = np.exp(logs["loss"])
        if "eval_loss" in logs:
            logs["eval_ppl"] = np.exp(logs["eval_loss"])

        super(CausalLMTrainer, self).log(logs, **kwargs)

    def get_ptq_dataloader(self, ptq_ds):
        if self.args.world_size <= 1:
            ptq_sampler = BatchSampler(
                dataset=ptq_ds,
                shuffle=True,
                batch_size=self.args.per_device_train_batch_size,
                drop_last=self.args.dataloader_drop_last,
            )
        else:
            ptq_sampler = DistributedBatchSampler(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                num_replicas=self.args.dataset_world_size,
                rank=self.args.dataset_rank,
                drop_last=self.args.dataloader_drop_last,
            )
        ptq_dataloader = DataLoader(
            ptq_ds,
            batch_sampler=ptq_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )
        return ptq_dataloader

    def ptq_loop(
        self,
        dataloader: DataLoader,
        description: str,
        max_eval_iters: Optional[int] = -1,
    ):
        if isinstance(dataloader, paddle.io.DataLoader):
            batch_size = dataloader.batch_sampler.batch_size
        else:
            raise ValueError("Only support for paddle.io.DataLoader")

        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            if max_eval_iters > 0:
                logger.info(f"  Total {description} steps = {max_eval_iters}")
            else:
                logger.info(f"  Total {description} steps = {len(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
            if max_eval_iters > 0:
                logger.info(f"  Total {description} steps = {max_eval_iters}")

        logger.info(f"  Pre device batch size = {batch_size}")
        logger.info(f"  Total Batch size = {batch_size * self.args.dataset_world_size}")
        self.model.eval()
        with paddle.no_grad():
            for step, inputs in enumerate(dataloader):
                self.prediction_step(model=self.model, inputs=inputs, prediction_loss_only=True, ignore_keys=None)
                if max_eval_iters > 0 and step >= max_eval_iters - 1:
                    break


def get_infer_model_path(input_dir, model_prefix):
    if dist.get_world_size() > 1:
        local_rank = dist.ParallelEnv().dev_id
        return os.path.join(input_dir, "rank_{}".format(local_rank), model_prefix)
    else:
        return os.path.join(input_dir, model_prefix)


def generate_rank_mapping(output_filename):
    ring_id = -1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        ring_id = model_parallel_group.id
    except Exception:
        pass

    if ring_id == -1:
        return

    world_size = dist.get_world_size()
    with open(output_filename, "w") as f:
        f.write("[ring_id -> ranks]\n")
        f.write(",".join(map(str, [0] + list(range(world_size)))) + "\n")
        f.write(",".join(map(str, [ring_id] + list(range(world_size)))) + "\n")

        f.write("[rank -> ring_ids]\n")
        for i in range(world_size):
            f.write("{},0,{}\n".format(i, ring_id))


def deserialize_from_file(fp):
    x_type = fp.read(1)
    x_type_out = struct.unpack("c", x_type)[0]
    # data
    data_list = []
    if x_type_out == b"0":
        data = fp.read(4)
        data_out = struct.unpack("f", data)[0]
        while data:
            data_out = struct.unpack("f", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    elif x_type_out == b"1":
        data = fp.read(8)
        while data:
            data_out = struct.unpack("l", data)[0]
            data_list.append(data_out)
            data = fp.read(8)
    elif x_type_out == b"2":
        data = fp.read(4)
        while data:
            data_out = struct.unpack("i", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    else:
        print("type error")
    data_arr = np.array(data_list)
    return data_arr


def get_alibi_slopes(num_heads):
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = np.arange(1, 1 + closest_power_of_2)
    slopes = np.power(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = np.arange(1, 1 + 2 * num_remaining_heads, 2)
        slopes = np.concatante([slopes, np.power(extra_base, extra_powers)], axis=0)

    return slopes.astype("float32")


def pad_batch_data(insts, pad_id=0, return_seq_len=False, pad_style="right"):
    """Pad sequences to the max sequence length in batch."""
    max_len = max(map(len, insts))
    if pad_style == "left":
        inst_data = np.array([[pad_id] * (max_len - len(inst)) + list(inst) for inst in insts])
    else:
        inst_data = np.array([list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts])

    if return_seq_len:
        seq_len = np.array([len(inst) for inst in insts])
        return inst_data.astype("int64").reshape([-1, max_len]), seq_len
    else:
        return inst_data.astype("int64").reshape([-1, max_len])


def dybatch_preprocess(
    tokenizer,
    texts: list[str],
    src_length: int,
    max_length: int,
    architectures: str,
    top_p: float,
    temperature: float,
    pre_caches_length: int = 0,
    benchmark: bool = False,
):
    """Pre-process generation inputs."""
    inputs = {}
    if "chatglm" in architectures:
        input_ids = []
        position_ids = []

        for text in texts:
            tokens = tokenizer(text, return_tensors="np", padding=True, max_length=src_length)
            input_ids.append(tokens["input_ids"][0])
            position_ids.append(tokens["position_ids"][0])

        pad_token_id = tokenizer([tokenizer.pad_token], return_tensors="np")["input_ids"][0][0]
        inputs["input_ids"], seq_len = pad_batch_data(input_ids, pad_id=pad_token_id, return_seq_len=True)
        bs = inputs["input_ids"].shape[0]
        max_len = max(map(len, input_ids))

        inst_data_pos = []
        for i in range(len(position_ids)):
            inst_data_pos.append(np.array([list(inst) + [0] * (max_len - len(inst)) for inst in position_ids[i]]))
        inputs["position_ids"] = paddle.to_tensor(np.array(inst_data_pos))
    elif "gpt" in architectures:
        input_ids = []
        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            tokens = tokenizer(
                text,
                return_tensors="np",
                padding=False,
                max_length=src_length,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            input_ids.append(tokens["input_ids"][0])

        pad_token_id = tokenizer([tokenizer.pad_token], return_tensors="np")["input_ids"][0][-1]
        inputs["input_ids"], seq_len = pad_batch_data(input_ids, pad_id=pad_token_id, return_seq_len=True)
        bs = inputs["input_ids"].shape[0]
        max_len = max(map(len, input_ids))

        position_ids = paddle.arange(sum(seq_len), dtype="int64")
        pre_len = seq_len[0]
        for length in seq_len[1:]:
            position_ids[pre_len : length + pre_len] = position_ids[pre_len : length + pre_len] - pre_len
            pre_len += length
        inputs["position_ids"] = position_ids
    else:
        input_ids = []
        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            tokens = tokenizer(
                text,
                return_tensors="np",
                padding=False,
                max_length=src_length,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            input_ids.append(tokens["input_ids"][0])

        pad_token_id = tokenizer([tokenizer.pad_token], return_tensors="np")["input_ids"][0][-1]
        inputs["input_ids"], seq_len = pad_batch_data(input_ids, pad_id=pad_token_id, return_seq_len=True)
        bs = inputs["input_ids"].shape[0]
        max_len = max(map(len, input_ids))

        position_ids = paddle.zeros(shape=[bs, max_length + src_length], dtype="int64")

        for i in range(bs):
            position_ids[i, pre_caches_length : pre_caches_length + seq_len[i]] = paddle.arange(seq_len[i])
        inputs["position_ids"] = position_ids

    tgt_ids = [input[-1:] for input in input_ids]
    tgt_pos = []
    for i, valid_len in enumerate(map(len, input_ids)):
        tgt_pos.append(valid_len - 1)

    step_idx = [
        0,
    ] * bs
    tgt_pos = np.array(tgt_pos).astype("int64")
    inputs["eos_token_id"] = (
        np.array(
            [
                tokenizer.eos_token_id,
            ]
            * bs
        )
        .reshape(-1, 1)
        .astype("int64")
    )
    inputs["top_p"] = (
        np.array(
            [
                top_p,
            ]
            * bs
        )
        .reshape(-1, 1)
        .astype("float32")
    )
    inputs["temperature"] = (
        np.array(
            [
                temperature,
            ]
            * bs
        )
        .reshape(-1, 1)
        .astype("float32")
    )
    inputs["seq_len_encoder"] = seq_len.astype("int32").reshape(-1, 1)
    inputs["seq_len_decoder"] = (seq_len + pre_caches_length).astype("int32").reshape(-1, 1)
    inputs["step_idx"] = np.array(step_idx).astype("int64").reshape(-1, 1)
    inputs["tgt_ids"] = np.array(tgt_ids).astype("int64").reshape(-1, 1)
    inputs["tgt_pos"] = tgt_pos.reshape(-1, 1)
    inputs["max_length"] = np.array(max_length - pre_caches_length).astype("int64").reshape((-1, 1))
    inputs["min_length"] = (
        np.array(
            [
                1
                if not benchmark
                else max_length
                - pre_caches_length,  # Note(Zhengzekang): When in benchmark mode, we need to set a fixed decode length.
            ]
            * bs
        )
        .astype("int64")
        .reshape((-1, 1))
    )
    inputs["penalty_score"] = (
        np.array(
            [
                1.0,
            ]
            * bs
        )
        .astype("float32")
        .reshape((-1, 1))
    )
    inputs["frequency_score"] = (
        np.array(
            [
                0.0,
            ]
            * bs
        )
        .astype("float32")
        .reshape((-1, 1))
    )
    inputs["presence_score"] = (
        np.array(
            [
                0.0,
            ]
            * bs
        )
        .astype("float32")
        .reshape((-1, 1))
    )
    inputs["stop_flags"] = (
        np.array(
            [
                0,
            ]
            * bs
        )
        .astype("bool")
        .reshape((-1, 1))
    )
    inputs["stop_nums"] = np.array([bs]).astype("int64")
    return inputs


def load_real_time_tokens():
    tokens = []
    files = glob.glob(os.path.join("./real_time_save.*"))
    for j in range(1, len(files) + 1):
        filename = "./real_time_save.temp_ids_rank_0_step_{}".format(j)
        if not os.path.exists(filename):
            break
        fp = open(filename, "rb+")
        fp.read(1)
        data_list = deserialize_from_file(fp)
        fp.close()
        tokens.append(np.array(data_list).reshape(-1, 1))
    os.system("rm -f ./real_time_save.temp_ids_rank_*")
    tokens = np.concatenate(tokens, axis=1)
    return tokens
