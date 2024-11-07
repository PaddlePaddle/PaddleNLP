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
from typing import List, Optional

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet.base.topology as tp
import paddle.incubate.multiprocessing as mp
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler
from sklearn.metrics import accuracy_score

from paddlenlp.datasets import ZeroPaddingIterableDataset
from paddlenlp.generation import GenerationConfig
from paddlenlp.trainer import TrainerCallback
from paddlenlp.trainer.trainer_utils import IterableDatasetShard, ShardingOption
from paddlenlp.transformers import (
    AutoTokenizer,
    ChatGLMv2Tokenizer,
    LlamaForCausalLMPipe,
    PretrainedConfig,
    Qwen2ForCausalLMPipe,
)
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
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
        multi_query_group_num = model.config.multi_query_group_num  # num_key_value_heads
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
    elif model.base_model_prefix == "mistral":
        from paddlenlp.peft.prefix import mistral_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = mistral_postprocess_past_key_value
        multi_query_group_num = model.config.num_key_value_heads
    elif model.base_model_prefix == "qwen":
        from paddlenlp.peft.prefix import qwen_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = qwen_postprocess_past_key_value
        multi_query_group_num = None
    elif model.base_model_prefix == "qwen2":
        from paddlenlp.peft.prefix import qwen_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = qwen_postprocess_past_key_value
        multi_query_group_num = model.config.num_key_value_heads  # num_key_value_heads
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
    elif model.base_model_prefix == "gpt":
        target_modules = [
            ".*qkv_proj.*",
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*linear1.*",
            ".*linear2.*",
            ".*out_proj.*",
        ]
    elif model.base_model_prefix == "bloom":
        target_modules = [".*query_key_value.*", ".*dense.*", ".*dense_h_to_4h.*", ".*dense_4h_to_h.*"]
    elif model.base_model_prefix in ["llama", "jamba"] or isinstance(model, LlamaForCausalLMPipe):
        target_modules = [
            ".*q_proj.*",
            ".*v_proj.*",
            ".*k_proj.*",
            ".*o_proj.*",
            ".*qkv_proj.*",
            ".*gate_proj.*",
            ".*down_proj.*",
            ".*up_proj.*",
            ".*gate_up_fused_proj.*",
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
    elif model.base_model_prefix == "qwen2" or isinstance(model, Qwen2ForCausalLMPipe):
        target_modules = [
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*o_proj.*",
            ".*gate_proj.*",
            ".*down_proj.*",
            ".*up_proj.*",
        ]
    elif model.base_model_prefix == "mixtral":
        target_modules = [
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*o_proj.*",
            # ".*gate.*", # TODO(DrownFish19): Does the gate weight require training?
            ".*w1.*",
            ".*w2.*",
            ".*w3.*",
        ]
    elif model.base_model_prefix == "mistral":
        target_modules = [
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*o_proj.*",
            ".*gate.*",
            ".*w1.*",
            ".*w2.*",
            ".*w3.*",
        ]
    elif model.base_model_prefix == "qwen2_moe":
        target_modules = [
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*o_proj.*",
            # ".*gate.*", # TODO(DrownFish19): Does the gate weight require training?
            ".*gate_proj.*",
            ".*up_proj.*",
            ".*down_proj.*",
        ]
    elif model.base_model_prefix == "yuan":
        target_modules = [
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*o_proj.*",
            ".*gate_proj.*",
            ".*up_proj.*",
            ".*down_proj.*",
        ]
    else:
        raise ValueError(f"Unknown base_model_prefix: {model.base_model_prefix}.")
    return target_modules


class ZeroPaddingIterDatasetCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles early stopping.

    """

    def on_step_end(self, args, state, control, **kwargs):
        train_dataloader = kwargs["train_dataloader"]
        if isinstance(train_dataloader.dataset, ZeroPaddingIterableDataset):
            dataset = train_dataloader.dataset
        elif isinstance(train_dataloader.dataset, IterableDatasetShard) and isinstance(
            train_dataloader.dataset.dataset, ZeroPaddingIterableDataset
        ):
            dataset = train_dataloader.dataset.dataset
        else:
            raise ValueError(
                "Unexpected dataset format: ZeroPaddingIterDatasetCallback expectes `paddlenlp.datasets.ZeroPaddingIterableDataset`"
            )
        if state.trial_params is None:
            state.trial_params = {}
        state.trial_params["zero_padding_global_step"] = dataset.zero_padding_global_step


def get_infer_model_path(input_dir, model_prefix):
    if dist.get_world_size() > 1:
        local_rank = dist.get_rank()
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
        slopes = np.concatenate([slopes, np.power(extra_base, extra_powers)], axis=0)

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
    eos_token_id: int | list[list[int]],
    pre_caches_length: int = 0,
    benchmark: bool = False,
):
    """Pre-process generation inputs."""
    inputs = {}
    if "chatglmforcausallm" == architectures.lower():
        input_ids = []
        position_ids = []

        for text in texts:
            tokens = tokenizer(
                text,
                return_tensors="np",
                padding=True,
                max_length=src_length,
                # if use chat_template, it will not add special_tokens
                add_special_tokens=tokenizer.chat_template is None or isinstance(tokenizer, ChatGLMv2Tokenizer),
            )
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
                add_special_tokens=tokenizer.chat_template is None or isinstance(tokenizer, ChatGLMv2Tokenizer),
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

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    inputs["eos_token_id"] = np.array(eos_token_id * bs).reshape(-1, 1).astype("int64")

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


def init_chat_template(
    tokenizer: PretrainedTokenizer, model_name_or_path: str, chat_template_file: Optional[str] = None
):
    """init chat template for the given tokenizer.

        If is None, it will not use `chat_template.json`;
        If is equal with `model_name_or_path`, it will use the default loading;
        If is directory, it will find the `chat_template.json` under the directory;
        If is file, it will load it.

    Args:
        tokenizer (PretrainedTokenizer): the instance of tokenizer
        model_name_or_path (str): _description_
        chat_template_file (Optional[str], optional): _description_. Defaults to None.
    """
    # 1. use the default chat_template file
    if chat_template_file is None:
        return

    if str(chat_template_file).lower() == "none":
        # delete the chat_template from tokenizer if not use chat_template.
        # why do this: it will load the `chat_template.json` file by default
        tokenizer.chat_template = None
        return

    # it will load the `chat_template.json` file by default, so do nothing
    if chat_template_file == model_name_or_path:
        if tokenizer.chat_template is None:
            logger.warning(f"there is not `chat_template.json` file in the `{model_name_or_path}`")
        return

    if os.path.isdir(chat_template_file):
        local_chat_template_file_path = os.path.join(chat_template_file, "chat_template.json")
        if os.path.exists(local_chat_template_file_path):
            chat_template_file = local_chat_template_file_path
        else:
            logger.warning(f"there is not `chat_template.json` file in the `{model_name_or_path}`")
            return

    if not os.path.exists(chat_template_file):
        logger.warning(f"there is not `chat_template.json` file from path<`{model_name_or_path}`>")
        return

    logger.info(f"loading `chat_template.json` from `{chat_template_file}`")
    tokenizer.init_chat_template(chat_template_file)


def get_model_max_position_embeddings(config: PretrainedConfig) -> Optional[int]:
    names = [
        "max_position_embeddings",  # most of models
        "max_sequence_length",  # GLM model
        "seq_length",  # llama model
    ]
    for name in names:
        max_length = config.get(name, None)
        if max_length is not None:
            return max_length
    return None


def read_res(model_name_or_path: str, tensor_queue: mp.Queue, result_queue: mp.Queue, done_event: mp.Event):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    paddle.device.set_device("cpu")
    paddle.disable_static()
    outputs = []
    output_tensor = tensor_queue.get(timeout=1)
    done_event.set()
    logger.info("Start read result message")
    logger.info(f"Current path is {os.getcwd()}")

    from paddlenlp_ops import get_output

    while True:
        get_output(output_tensor, 0, True)
        if int(output_tensor[0, 0]) == -2:  # read none
            continue
        bsz = int(output_tensor[1, 0])
        output_numpy = output_tensor[2 : bsz + 2].numpy()
        output_numpy[output_numpy == -1] = tokenizer.eos_token_id
        outputs.append(output_numpy)
        if int(output_tensor[0, 0]) == -1:
            break
    output = np.concatenate(outputs, axis=1).tolist()
    seqs = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for i, (out, seq) in enumerate(zip(output, seqs)):
        result_queue.put([i, out, seq])

    logger.info("Finish read result message")


def get_rotary_position_embedding(position_ids, head_dim, rope_theta=10000.0, rope_scaling: dict = None):
    """
    Pre-calculate rotary position embedding for position_ids.

    Args:
        position_ids: [1, S]
        head_dim: D

    Returns:
        rot_emb: [2, 1, S, 1, D], cos + sin
    """
    bsz, max_seq_len = position_ids.shape[:2]
    rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, head_dim), dtype="float32")
    inv_freq = rope_theta ** (-paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)

    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", None)
        if rope_type is not None and rope_type == "llama3":
            factor = rope_scaling.get("factor", 8.0)
            low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
            original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", 8192)

            low_freq_wavelen = original_max_position_embeddings / low_freq_factor
            high_freq_wavelen = original_max_position_embeddings / high_freq_factor
            new_freqs = []
            for freq in inv_freq:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / factor)
                else:
                    assert low_freq_wavelen != high_freq_wavelen
                    smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
                        high_freq_factor - low_freq_factor
                    )
                    new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
            inv_freq = paddle.to_tensor(new_freqs, dtype=inv_freq.dtype)

    # shape: [B, S, D/2]
    freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
    # shape: [B, S, 1, D]
    emb = paddle.concat([freqs, freqs], axis=-1).reshape((bsz, max_seq_len, 1, head_dim))

    rot_emb[0] = paddle.cos(emb)
    rot_emb[1] = paddle.sin(emb)
    return rot_emb


def init_dist_env():
    tensor_parallel_degree = paddle.distributed.get_world_size()
    tensor_parallel_rank = paddle.distributed.get_rank()

    if tensor_parallel_degree > 1:
        # refer to: https://github.com/PaddlePaddle/Paddle/blob/4abea956ee852ce52791a1e08fa92ed4d3be150d/python/paddle/distributed/fleet/fleet.py#L298C23-L298C45
        hcg = tp._HYBRID_PARALLEL_GROUP
        if hcg is None:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": tensor_parallel_degree,
                "pp_degree": 1,
                "sharding_degree": 1,
            }
            fleet.init(is_collective=True, strategy=strategy)
            hcg = fleet.get_hybrid_communicate_group()

        tensor_parallel_rank = hcg.get_model_parallel_rank()
    return tensor_parallel_rank, tensor_parallel_degree


def get_eos_token_id(
    tokenizer: PretrainedTokenizer, generation_config: Optional[GenerationConfig] = None
) -> List[List[int]]:
    """get eos_token_id from generation_config or tokenizer

    Returns:
        List[int]: eos_token_id to stop the generation
    """
    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)

    if generation_config is not None and generation_config.eos_token_id is not None:
        if isinstance(generation_config.eos_token_id, int):
            eos_token_ids.append(generation_config.eos_token_id)
        else:
            eos_token_ids.extend(generation_config.eos_token_id)

    eos_token_ids_dict = {str(item): item for item in eos_token_ids}
    return list(eos_token_ids_dict.values())


def wrap_loraga_model(model, training_args):
    sharding = None
    if len(training_args.sharding) > 0:
        if training_args.local_rank == -1:
            raise ValueError("Using sharding only works in distributed training.")
        sharding = True

    in_pipeline_parallel_mode = training_args.pipeline_parallel_degree > 1
    in_sharding_parallel_mode = sharding is not None
    in_tensor_parallel_mode = training_args.tensor_parallel_degree > 1
    in_sep_parallel_mode = training_args.sep_parallel_degree > 1
    in_cp_parallel_mode = training_args.context_parallel_degree > 1

    # Multi-gpu training
    if training_args.world_size > 1 and (not training_args.use_hybrid_parallel):
        # MOE use DDP to broadcaset parameters.
        ddp_kwargs = {}
        if training_args.ddp_find_unused_parameters is not None:
            ddp_kwargs["find_unused_parameters"] = training_args.ddp_find_unused_parameters
        elif isinstance(model, PretrainedModel):
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
            ddp_kwargs["find_unused_parameters"] = not any(
                hasattr(m, "enable_recompute") and m.enable_recompute for m in model.sublayers(include_self=True)
            )
        else:
            ddp_kwargs["find_unused_parameters"] = True
        model = paddle.DataParallel(model, **ddp_kwargs)

    # No pipeline mode, sharding only
    if not in_pipeline_parallel_mode and in_sharding_parallel_mode:
        # Sharded DDP!
        if training_args.tensor_parallel_degree > 1:
            hcg = fleet.get_hybrid_communicate_group()
            assert (
                ShardingOption.SHARD_GRAD_OP in training_args.sharding
                or ShardingOption.SHARD_OP in training_args.sharding
            ), "Only support tensor parallel + sharding stage1/stage2 hybrid parallel now."
            model = paddle.distributed.fleet.meta_parallel.TensorParallel(model, hcg, strategy=None)
        if ShardingOption.SHARD_OP in training_args.sharding:
            model = fleet.distributed_model(model)

    if (
        not in_pipeline_parallel_mode
        and not in_sharding_parallel_mode
        and (in_tensor_parallel_mode or in_sep_parallel_mode or in_cp_parallel_mode)
    ):
        model = fleet.distributed_model(model)

    return model


def get_loraga_dataloader(train_dataset, data_collator, training_args):
    from paddlenlp.data import DistDataLoader

    def _is_iterable_dataset(dataset):
        return isinstance(dataset, paddle.io.IterableDataset)

    def _is_iterable_dataset_distributed(dataset):
        # For distributed dataloaer.
        is_iterable_dataset_tensor = paddle.to_tensor(is_iterable_dataset(dataset)).astype("int32").reshape([1])
        if dist.get_world_size() > 1:
            dist.all_reduce(is_iterable_dataset_tensor, op=dist.ReduceOp.MAX)
        if is_iterable_dataset_tensor.item() == 1:
            return True
        return False

    if training_args.distributed_dataloader:
        is_iterable_dataset = _is_iterable_dataset_distributed(train_dataset)
    else:
        is_iterable_dataset = _is_iterable_dataset(train_dataset)

    # if is_datasets_available() and train_dataset is not None and isinstance(train_dataset, datasets.Dataset):
    #     train_dataset = self._remove_unused_columns(train_dataset, description="training")
    _DataLoader = DistDataLoader if training_args.distributed_dataloader else DataLoader

    if is_iterable_dataset:  # For iterable dataset
        if training_args.dataset_world_size > 1 and train_dataset is not None:
            train_dataset = IterableDatasetShard(
                train_dataset,
                batch_size=training_args.per_device_train_batch_size,
                drop_last=training_args.dataloader_drop_last,
                num_processes=training_args.dataset_world_size,
                process_index=training_args.dataset_rank,
            )

        if training_args.distributed_dataloader:
            logger.info("Training using DistDataLoader.")
            additional_configs = {"is_iterable_dataset": True}
        else:
            additional_configs = {}
        return _DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            collate_fn=data_collator,
            num_workers=training_args.dataloader_num_workers,
            **additional_configs,
        )
    else:
        train_sampler = get_loraga_train_sampler(train_dataset, training_args)
        if training_args.distributed_dataloader:
            logger.info("Training using DistDataLoader.")
        return _DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=training_args.dataloader_num_workers,
        )


def get_loraga_train_sampler(train_dataset, training_args) -> Optional[paddle.io.Sampler]:
    if training_args.world_size <= 1:
        return paddle.io.BatchSampler(
            dataset=train_dataset,
            shuffle=True,
            batch_size=training_args.per_device_train_batch_size,
            drop_last=training_args.dataloader_drop_last,
        )

    return DistributedBatchSampler(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        num_replicas=training_args.dataset_world_size,
        rank=training_args.dataset_rank,
        drop_last=training_args.dataloader_drop_last,
    )


def estimate_gradient(model, train_ds, data_collator, training_args, loraga_init_iters=32):
    """Estimate the gradient of the model on the given dataset"""

    import time

    start_time = time.time()
    logger.info("Estimating gradient for LoraGA")
    split_mappings = model._get_tensor_parallel_mappings(config=model.config, is_split=False)
    model = wrap_loraga_model(model, training_args)
    model.train()
    gradient_dict = {}
    logger.info(f"Initilization iterions for LoraGA: {loraga_init_iters}")
    dataloader = get_loraga_dataloader(train_ds, data_collator, training_args)
    iters = 0
    for batch in dataloader:
        iters += 1
        batch = {k: paddle.to_tensor(v) for k, v in batch.items()}
        # Do not support pipeline parallel by now
        loss, logits = model(**batch)
        # log_memory_usage()
        loss.backward()
        # log_memory_usage()
        # Record gradients
        for grad_name, param in model.named_parameters():
            # 经过tp和sharding包裹后的模型可能以若干个_layer.开头，这里需要去掉
            grad_name = grad_name.split("_layers.")[-1]
            if not param.stop_gradient and param.grad is not None:
                if grad_name not in gradient_dict:
                    gradient_dict[grad_name] = param.grad.clone()
                else:
                    gradient_dict[grad_name] += param.grad
                param.clear_gradient(False)  # release gradient memory

        if iters == loraga_init_iters:
            break

    for grad_name, param in gradient_dict.items():
        # 暂时不支持pp!
        # tp
        if training_args.tensor_parallel_degree > 1:
            if grad_name.split("gpt.")[-1] in split_mappings:
                # 有的模型可能不以gpt.开头？
                merge_func = split_mappings[grad_name.split("gpt.")[-1]]
                hcg = fleet.get_hybrid_communicate_group()
                model_parallel_group = hcg.get_model_parallel_group()
                output_tensors = []
                dist.all_gather(output_tensors, gradient_dict[grad_name], group=model_parallel_group)
                output_tensors = [t if len(t.shape) > 0 else t.reshape_([-1]) for t in output_tensors]
                gradient_dict[grad_name] = paddle.to_tensor(merge_func(output_tensors))
        # sharding
        if training_args.sharding_parallel_degree > 1:
            hcg = fleet.get_hybrid_communicate_group()
            sharding_parallel_group = hcg.get_sharding_parallel_group()
            if sharding_parallel_group.nranks > 1:
                dist.all_reduce(gradient_dict[grad_name], op=dist.ReduceOp.SUM, group=sharding_parallel_group)
                gradient_dict[grad_name] /= sharding_parallel_group.nranks
        # dp
        if training_args.data_parallel_degree > 1:
            hcg = fleet.get_hybrid_communicate_group()
            data_parallel_group = hcg.get_data_parallel_group()
            if data_parallel_group.nranks > 1:
                dist.all_reduce(gradient_dict[grad_name], op=dist.ReduceOp.SUM, group=data_parallel_group)
                gradient_dict[grad_name] /= data_parallel_group.nranks
        gradient_dict[grad_name] /= loraga_init_iters
    paddle.device.cuda.empty_cache()

    logger.info("Gradient Approximation execution time: {} seconds".format(time.time() - start_time))
    return gradient_dict


def loraga_reinit(model, gradient_dict, stable_gamma, training_args, **kwargs):
    """Re-initialize the weights of the model using the estimated gradients"""
    from tqdm import tqdm

    for name, module in tqdm(
        model.named_sublayers(),
        desc="Reinitializing Lora",
        total=len(list(model.named_sublayers())),
    ):
        from paddlenlp.peft.lora.lora_layers import (
            ColumnParallelLoRALinear,
            ColumnSequenceParallelLoRALinear,
            LoRALinear,
            RowParallelLoRALinear,
            RowSequenceParallelLoRALinear,
        )

        lora_split_mapping = None
        if (
            isinstance(module, LoRALinear)
            or isinstance(module, RowSequenceParallelLoRALinear)
            or isinstance(module, ColumnSequenceParallelLoRALinear)
            or isinstance(module, RowParallelLoRALinear)
            or isinstance(module, ColumnParallelLoRALinear)
        ):
            is_tp = training_args.tensor_parallel_degree > 1
            if is_tp:
                lora_split_mapping = model._get_tensor_parallel_mappings(model.config)
            loraga_reinit_modules(name, module, gradient_dict, stable_gamma, is_tp, lora_split_mapping, **kwargs)


def loraga_reinit_modules(name, module, gradient_dict, stable_gamma, is_tp=False, lora_split_mapping=None, **kwargs):
    with paddle.no_grad():
        lora_r = module.r
        grad_name = ".".join(name.split(".")[1:]) + ".weight"
        loraA_name = ".".join(name.split(".")[1:]) + ".lora_A"
        loraB_name = ".".join(name.split(".")[1:]) + ".lora_B"
        grads = gradient_dict[grad_name]

        U, S, V = paddle.linalg.svd_lowrank(grads.astype("float32"), q=4 * lora_r, niter=4)

        V = V.T
        A = U[:, lora_r : 2 * lora_r]
        B = V[:lora_r, :]
        m, n = grads.shape  # m: feature_out, n: feature_in
        # If stable_gamma is not -1, scale the matrices A and B by the square root of the stable_gamma
        if stable_gamma != -1:
            A = A * m**0.25 / stable_gamma**0.5
            B = B * m**0.25 / stable_gamma**0.5
        else:
            A = A / module.scaling
            B = B / module.scaling
        if is_tp:
            if module.lora_A.is_distributed and lora_split_mapping:
                split_function = lora_split_mapping[loraA_name]
                A = paddle.to_tensor(split_function(A))
            if module.lora_B.is_distributed and lora_split_mapping:
                split_function = lora_split_mapping[loraB_name]
                B = paddle.to_tensor(split_function(B))
        module.lora_A.set_value(A.astype(module.lora_A.dtype))
        module.lora_B.set_value(B.astype(module.lora_B.dtype))
        offset = module.lora_A @ module.lora_B
        module.weight.data -= module.scaling * offset
