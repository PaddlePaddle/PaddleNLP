# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle


def check_preference_data(data):

    if isinstance(data["src"], str):
        data["src"] = [data["src"]]
    if isinstance(data["tgt"], str):
        data["tgt"] = [data["tgt"]]
    if len(data["src"]) != len(data["tgt"]) + 1:
        raise ValueError(
            "The number of src and tgt should differ by 1, but got {} and {}".format(
                len(data["src"]), len(data["tgt"])
            )
        )
    if (len(data["response"]) != 2) or (len(data["response"]) != len(data["sort"])):
        raise ValueError(
            "The number of response and sort should be 2, but got {} and {}".format(
                len(data["response"]), len(data["sort"])
            )
        )
    if len(data["response"][0]) == 0 or len(data["response"][1]) == 0:
        raise ValueError(f"The response should not be empty, buut got {data}.")
    if data["sort"][0] == data["sort"][1]:
        raise ValueError("The two sort should be different.")

    return data


def preprocess_preference_data(data, tokenizer, data_args, model_args):
    """Convert raw format example to Example."""
    # 1. Check data format
    data = check_preference_data(data)

    if data["sort"][0] > data["sort"][1]:
        chosen = data["response"][0]
        rejected = data["response"][1]
    else:
        chosen = data["response"][1]
        rejected = data["response"][0]

    chosen_token_ids = tokenizer(chosen)["input_ids"] + [tokenizer.eos_token_id]
    rejected_token_ids = tokenizer(rejected)["input_ids"] + [tokenizer.eos_token_id]
    prompt_tokens_ids = tokenizer(data["src"][-1], add_special_tokens=True)["input_ids"]

    for idx in range(len(data["tgt"])):
        src_token_ids = tokenizer(data["src"][-idx - 1], add_special_tokens=True)["input_ids"]
        tgt_token_ids = tokenizer(data["tgt"][-idx])["input_ids"] + [tokenizer.eos_token_id]
        prompt_tokens_ids = src_token_ids + tgt_token_ids + prompt_tokens_ids

    if len(prompt_tokens_ids) + len(rejected_token_ids) + len(chosen_token_ids) > data_args.max_seq_len:
        prompt_tokens_ids = prompt_tokens_ids[-data_args.max_prompt_len :]
        if len(prompt_tokens_ids) + len(rejected_token_ids) + len(chosen_token_ids) > data_args.max_seq_len:
            max_response_len = data_args.max_seq_len - len(prompt_tokens_ids)
            # 按比例截断
            max_chosen_len = int(
                len(chosen_token_ids) / (len(chosen_token_ids) + len(rejected_token_ids)) * max_response_len
            )
            max_rejected_len = max_response_len - max_chosen_len
            chosen_token_ids = chosen_token_ids[:max_chosen_len]
            rejected_token_ids = rejected_token_ids[:max_rejected_len]
    input_ids = prompt_tokens_ids + chosen_token_ids + rejected_token_ids
    prompt_len, chosen_len, rejected_len, seq_len = (
        len(prompt_tokens_ids),
        len(chosen_token_ids),
        len(rejected_token_ids),
        len(input_ids),
    )
    position_ids = (
        list(range(prompt_len))  # prompt
        + list(range(prompt_len, prompt_len + chosen_len))  # chosen
        + list(range(prompt_len, prompt_len + rejected_len))  # rejected
    )
    # response index
    response_indexs = [prompt_len + chosen_len - 1, seq_len - 1]
    output_dict = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "response_indexs": response_indexs,
    }

    # attention mask
    if model_args.flash_mask:
        output_dict["attn_mask_startend_row_indices"] = (
            [seq_len] * prompt_len + [prompt_len + chosen_len] * chosen_len + [seq_len] * rejected_len
        )
    else:
        attention_mask = np.tri(seq_len, seq_len, dtype=bool)
        attention_mask[(prompt_len + chosen_len) :, prompt_len : (prompt_len + chosen_len)] = False
        output_dict["attention_mask"] = attention_mask
    return output_dict


def preference_collate_fn(batch, max_seq_len=None, pad_token_id=0):
    """Convert batch data into tensor."""
    if max_seq_len is None:
        raise ValueError("max_seq_len is None.")

    input_dict = {
        "input_ids": [],
        "position_ids": [],
        "response_indexs": [],
    }
    sequence = batch[0]
    if "attn_mask_startend_row_indices" in sequence:
        input_dict["attn_mask_startend_row_indices"] = []
        use_attn_mask_startend_row_indices = True
    elif "attention_mask" in sequence:
        input_dict["attention_mask"] = []
        use_attn_mask_startend_row_indices = False
    else:
        raise ValueError("attention_mask and attn_mask_startend_row_indices are both None.")

    for i, sequence in enumerate(batch):
        difference = max_seq_len - len(sequence["input_ids"])

        input_dict["input_ids"].append(sequence["input_ids"] + [pad_token_id] * difference)
        input_dict["position_ids"].append(sequence["position_ids"] + [0] * difference)
        if use_attn_mask_startend_row_indices:
            input_dict["attn_mask_startend_row_indices"].append(
                [
                    sequence["attn_mask_startend_row_indices"]
                    + [sequence["attn_mask_startend_row_indices"][-1]] * difference
                ]
            )
        else:
            input_dict["attention_mask"].append(
                np.pad(
                    sequence["attention_mask"],
                    pad_width=((0, 0), (0, difference), (0, difference)),
                    mode="constant",
                    constant_values=False,
                )
            )

        for ri in sequence["response_indexs"]:
            input_dict["response_indexs"].append(
                [
                    i,  # bs
                    ri[0],  # chosen_response_start_index
                    ri[1],  # rejeted_response_start_index
                ]
            )
    for key in input_dict:
        if key == "attention_mask":
            input_dict[key] = np.array(input_dict[key], dtype=bool)
        elif key == "attn_mask_startend_row_indices":
            input_dict[key] = np.array(input_dict[key], dtype=np.int32)
        else:
            input_dict[key] = np.array(input_dict[key])
    return input_dict


def check_process_data(data):
    """
    "src" : ["prompt"],
    "tgt" : [],
    "responses" : ["step_1", ..., "step_k"]
    "labels" : ["label_1", ..., "label_k"]
    """

    if isinstance(data["src"], str):
        data["src"] = [data["src"]]
    if isinstance(data["tgt"], str):
        data["tgt"] = [data["tgt"]]
    if len(data["src"]) != len(data["tgt"]) + 1:
        raise ValueError(
            "The number of src and tgt should differ by 1, but got {} and {}".format(
                len(data["src"]), len(data["tgt"])
            )
        )
    if len(data["responses"]) != len(data["labels"]):
        raise ValueError(
            "The number of responses and labels should be equal, but got {} and {}".format(
                len(data["responses"]), len(data["labels"])
            )
        )
    if "" in data["responses"] or "" in data["labels"]:
        raise ValueError(f"Any step in the responses or labels should not be empty, but got {data}.")

    return data


def preprocess_process_data(data, tokenizer, data_args, model_args):
    """Convert raw format example to Example."""
    # Check data format
    data = check_process_data(data)

    placeholder_token_id = tokenizer(model_args.placeholder_token, add_special_tokens=False)["input_ids"]

    placeholder_token_id = placeholder_token_id[-1]

    prompt_token_ids = tokenizer(data["src"][-1], add_special_tokens=False)["input_ids"]
    for idx in range(len(data["tgt"])):
        src_token_ids = tokenizer(data["src"][-idx - 1], add_special_tokens=False)["input_ids"]
        tgt_token_ids = tokenizer(data["tgt"][-idx], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        prompt_token_ids = src_token_ids + tgt_token_ids + prompt_token_ids

    response_token_ids = tokenizer(
        f" {model_args.placeholder_token}\n".join(data["responses"]) + f" {model_args.placeholder_token}",
        add_special_tokens=False,
    )["input_ids"]

    # NOTE: Truncation may leads to incompleteness of the last CoT step, however, the prm will not predict the
    # corresponding reward either. So it is ok then.
    if len(prompt_token_ids) + len(response_token_ids) > data_args.max_seq_len:
        prompt_token_ids = prompt_token_ids[-data_args.max_prompt_len :]
        if len(prompt_token_ids) + len(response_token_ids) > data_args.max_seq_len:
            max_response_len = data_args.max_seq_len - len(prompt_token_ids)
            response_token_ids = response_token_ids[-max_response_len:]

    input_ids = paddle.to_tensor(prompt_token_ids + response_token_ids)

    label_token_ids = []
    for local_label in data["labels"]:
        if local_label not in model_args.reward_tokens:
            raise ValueError(
                f"The label {local_label} should be in reward tokens {model_args.reward_tokens}, got {data}."
            )
        label_token_ids.append(tokenizer(local_label, add_special_tokens=False)["input_ids"][-1])
    labels = paddle.full_like(input_ids, -100, dtype=input_ids.dtype)

    indices = paddle.nonzero(input_ids == placeholder_token_id).flatten()
    for idx, replacement_value in zip(indices, label_token_ids):
        labels[idx] = replacement_value

    prompt_len, seq_len = (len(prompt_token_ids), len(input_ids))
    position_ids = list(range(prompt_len)) + list(range(prompt_len, seq_len))

    output_dict = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
    }

    if model_args.flash_mask:
        output_dict["attn_mask_startend_row_indices"] = [seq_len] * seq_len
    else:
        attention_mask = np.tri(seq_len, seq_len, dtype=bool)
        output_dict["attention_mask"] = attention_mask

    return output_dict


def zero_padding_process_collate_fn(batch, max_seq_len=None, pad_token_id=0):
    """Convert batch data into tensor."""
    if max_seq_len is None:
        raise ValueError("max_seq_len is None.")

    input_dict = {
        "input_ids": [],
        "position_ids": [],
        "labels": [],
    }
    sequence = batch[0]
    if "attn_mask_startend_row_indices" in sequence:
        input_dict["attn_mask_startend_row_indices"] = []
        use_attn_mask_startend_row_indices = True
    elif "attention_mask" in sequence:
        input_dict["attention_mask"] = []
        use_attn_mask_startend_row_indices = False
    else:
        raise ValueError("attention_mask and attn_mask_startend_row_indices are both None.")

    for i, sequence in enumerate(batch):
        difference = max_seq_len - len(sequence["input_ids"])

        input_dict["input_ids"].append(sequence["input_ids"] + [pad_token_id] * difference)
        input_dict["position_ids"].append(sequence["position_ids"] + [0] * difference)
        input_dict["labels"].append(sequence["labels"] + [-100] * difference)
        if use_attn_mask_startend_row_indices:
            input_dict["attn_mask_startend_row_indices"].append(
                [
                    sequence["attn_mask_startend_row_indices"]
                    + [sequence["attn_mask_startend_row_indices"][-1]] * difference
                ]
            )
        else:
            input_dict["attention_mask"].append(
                np.pad(
                    sequence["attention_mask"],
                    pad_width=((0, 0), (0, difference), (0, difference)),
                    mode="constant",
                    constant_values=False,
                )
            )

    for key in input_dict:
        if key == "attention_mask":
            input_dict[key] = np.array(input_dict[key], dtype=bool)
        elif key == "attn_mask_startend_row_indices":
            input_dict[key] = np.array(input_dict[key], dtype=np.int32)
        else:
            input_dict[key] = np.array(input_dict[key])
    return input_dict


def process_collate_fn(batch, pad_token_id=0):
    """Convert batch data into tensor."""
    max_seq_len = max([len(sequence["input_ids"]) for sequence in batch])

    input_dict = {
        "input_ids": [],
        "position_ids": [],
        "labels": [],
    }
    sequence = batch[0]
    if "attn_mask_startend_row_indices" in sequence:
        input_dict["attn_mask_startend_row_indices"] = []
        use_attn_mask_startend_row_indices = True
    elif "attention_mask" in sequence:
        input_dict["attention_mask"] = []
        use_attn_mask_startend_row_indices = False
    else:
        raise ValueError("attention_mask and attn_mask_startend_row_indices are both None.")

    for i, sequence in enumerate(batch):
        difference = max_seq_len - len(sequence["input_ids"])

        # input_ids: Tensor(seqL, ); position_ids: list, len(seqL); labels: Tensor(seqL, )
        input_dict["input_ids"].append(sequence["input_ids"].tolist() + [pad_token_id] * difference)
        input_dict["position_ids"].append(sequence["position_ids"] + [0] * difference)
        input_dict["labels"].append(sequence["labels"].tolist() + [-100] * difference)
        if use_attn_mask_startend_row_indices:
            input_dict["attn_mask_startend_row_indices"].append(
                [
                    sequence["attn_mask_startend_row_indices"]
                    + [sequence["attn_mask_startend_row_indices"][-1]] * difference
                ]
            )
        else:
            input_dict["attention_mask"].append(
                np.pad(
                    sequence["attention_mask"],
                    pad_width=((0, difference), (0, difference)),
                    mode="constant",
                    constant_values=False,
                )
            )

    for key in input_dict:
        if key == "attention_mask":
            input_dict[key] = np.array(input_dict[key], dtype=bool)
        elif key == "attn_mask_startend_row_indices":
            input_dict[key] = np.array(input_dict[key], dtype=np.int32)
        else:
            input_dict[key] = np.array(input_dict[key])
    return input_dict
