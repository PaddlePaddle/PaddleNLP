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

import glob
import os
import struct

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet


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


def dybatch_preprocess(tokenizer, texts, config, args):
    """Pre-process generation inputs."""
    input_ids = []
    for text in texts:
        tokens = tokenizer(
            text,
            return_tensors="np",
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids.append(tokens["input_ids"][0])

    inputs = {}
    pad_token_id = tokenizer([tokenizer.pad_token], return_tensors="np")["input_ids"][0][-1]
    inputs["input_ids"], seq_len = pad_batch_data(input_ids, pad_id=pad_token_id, return_seq_len=True)
    bs = inputs["input_ids"].shape[0]
    max_len = max(map(len, input_ids))

    position_ids = paddle.zeros(shape=[bs, max_len], dtype="int64")
    for i in range(bs):
        position_ids[i, : seq_len[i]] = paddle.arange(seq_len[i])
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
                0.0,
            ]
            * bs
        )
        .reshape(-1, 1)
        .astype("float32")
    )
    inputs["temperature"] = (
        np.array(
            [
                1.0,
            ]
            * bs
        )
        .reshape(-1, 1)
        .astype("float32")
    )
    inputs["seq_len_encoder"] = np.array(seq_len).astype("int32").reshape(-1, 1)
    inputs["seq_len_decoder"] = np.array(seq_len).astype("int32").reshape(-1, 1)
    inputs["step_idx"] = np.array(step_idx).astype("int64").reshape(-1, 1)
    inputs["tgt_ids"] = np.array(tgt_ids).astype("int64").reshape(-1, 1)
    inputs["tgt_pos"] = tgt_pos.reshape(-1, 1)
    inputs["max_length"] = (
        np.array(
            [
                args.tgt_length,
            ]
            * bs
        )
        .astype("int64")
        .reshape((-1, 1))
    )
    inputs["min_length"] = (
        np.array(
            [
                2,
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
    for j in range(1, len(files)):
        filename = "./real_time_save.temp_ids_rank_0_step_{}".format(j)
        if not os.path.exists(filename):
            break
        fp = open(filename, "rb+")
        fp.read(1)
        data_list = deserialize_from_file(fp)
        fp.close()
        tokens.append(np.array(data_list).reshape(-1, 1))
    os.system(f"rm -f ./real_time_save.temp_ids_rank_*")
    tokens = np.concatenate(tokens, axis=1)
    return tokens
