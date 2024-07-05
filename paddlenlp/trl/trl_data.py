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
        raise ValueError("The response should not be empty, buut got {data}.")
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
    chosen_encode_tokens = []
    for idx in range(len(data["src"])):
        if idx < len(data["tgt"]):
            if tokenizer.chat_template is not None:
                chosen_encode_tokens.append(
                    [
                        data["src"][idx].strip(),
                        data["tgt"][idx].strip(),
                    ]
                )
            else:
                chosen_encode_tokens.append(
                    [
                        tokenizer.encode(data["src"][idx].strip(), add_special_tokens=True)["input_ids"],
                        tokenizer.encode(data["tgt"][idx].strip(), add_special_tokens=False)["input_ids"]
                        + [tokenizer.eos_token_id],
                    ]
                )
        else:
            if tokenizer.chat_template is not None:
                chosen_encode_tokens.append(
                    [
                        data["src"][idx].strip(),
                        chosen.strip(),
                    ]
                )
            else:
                chosen_encode_tokens.append(
                    [
                        tokenizer.encode(data["src"][idx].strip(), add_special_tokens=True)["input_ids"],
                        tokenizer.encode(chosen.strip(), add_special_tokens=False)["input_ids"]
                        + [tokenizer.eos_token_id],
                    ]
                )
    if tokenizer.chat_template is not None:
        chat_input_list = chosen_encode_tokens
        chosen_encode_tokens = tokenizer.encode_chat_inputs(chat_input_list)["conversations"]
        # convert to rejected chosen_encode_tokens
        chat_input_list[-1][-1] = rejected.strip()
        rejected_encode_tokens = tokenizer.encode_chat_inputs(chat_input_list)["conversations"]

        """Post process sequence: tokenization & truncation."""
        tokens_prompt = chosen_encode_tokens[-1][0][:-1]
        eos_token_id = chosen_encode_tokens[-1][-1][-1]
        tokens_chosen = chosen_encode_tokens[-1][0][-1:] + chosen_encode_tokens[-1][-1][:-1]
        tokens_rejected = chosen_encode_tokens[-1][0][-1:] + rejected_encode_tokens[-1][-1][:-1]
    else:
        eos_token_id = tokenizer.eos_token_id
        tokens_prompt = chosen_encode_tokens[-1][0][:-1]
        tokens_chosen = (
            chosen_encode_tokens[-1][0][-1:] + tokenizer.encode(chosen.strip(), add_special_tokens=False)["input_ids"]
        )
        tokens_rejected = (
            chosen_encode_tokens[-1][0][-1:]
            + tokenizer.encode(rejected.strip(), add_special_tokens=False)["input_ids"]
        )

    if len(tokens_prompt) + len(tokens_chosen) + len(tokens_rejected) > data_args.max_seq_len:
        # truncate prompt
        tokens_prompt = tokens_prompt[-data_args.max_prompt_len :]
        if (len(tokens_prompt) + len(tokens_chosen) + len(tokens_rejected)) > data_args.max_seq_len:
            max_response_len = data_args.max_seq_len - len(tokens_prompt)
            # 按比例截断
            max_chosen_len = int(len(tokens_chosen) / (len(tokens_chosen) + len(tokens_rejected)) * max_response_len)
            max_rejected_len = max_response_len - max_chosen_len
            tokens_chosen = tokens_chosen[:max_chosen_len]
            tokens_rejected = tokens_rejected[:max_rejected_len]

    cur_len = len(tokens_prompt) + len(tokens_chosen) + len(tokens_rejected)
    turn_index = len(chosen_encode_tokens) - 2

    # append former dialog contents
    while turn_index >= 0:
        tokens_src = chosen_encode_tokens[turn_index][0]
        tokens_target = chosen_encode_tokens[turn_index][1]
        turn_index -= 1

        if len(tokens_src) + len(tokens_target) > data_args.max_seq_len - cur_len:
            break
        tokens_prompt = tokens_src + tokens_target + tokens_prompt
        cur_len += len(tokens_src) + len(tokens_target)

    input_ids = tokens_prompt + tokens_chosen + tokens_rejected
    prompt_len = len(tokens_prompt)
    chosen_len = len(tokens_chosen)
    rejected_len = len(tokens_rejected)
    seq_len = len(input_ids)
    # make position ids & labels

    position_ids = (
        list(range(prompt_len))  # prompt
        + list(range(prompt_len, prompt_len + chosen_len))  # chosen
        + list(range(prompt_len, prompt_len + rejected_len))  # rejected
    )
    chosen_labels = [0] * prompt_len + tokens_chosen[1:] + [eos_token_id] + [0] * rejected_len
    rejected_labels = [0] * prompt_len + [0] * chosen_len + tokens_rejected[1:] + [eos_token_id]

    # response index
    response_indexs = [prompt_len, prompt_len + chosen_len, seq_len]
    output_dict = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "chosen_labels": chosen_labels,
        "rejected_labels": rejected_labels,
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


def preference_collate_fn(batch, max_seq_len=None):
    """Convert batch data into tensor."""
    if max_seq_len is None:
        raise ValueError("max_seq_len is None.")

    input_dict = {
        "input_ids": [],
        "position_ids": [],
        "chosen_labels": [],
        "rejected_labels": [],
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

        input_dict["input_ids"].append(sequence["input_ids"] + [0] * difference)
        input_dict["position_ids"].append(sequence["position_ids"] + [0] * difference)
        input_dict["chosen_labels"].append(sequence["chosen_labels"] + [0] * difference)
        input_dict["rejected_labels"].append(sequence["rejected_labels"] + [0] * difference)
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
                    ri[2],  # rejeted_response_end_index + 1
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
