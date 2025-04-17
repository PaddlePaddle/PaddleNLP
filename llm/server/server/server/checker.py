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


def check_basic_params(req_dict):
    """
    checks input requests for basic parameters

    Args:
        req_dict (dict): request parameters

    Returns:
        list[str]: if error, return a list of error messages; return an empty list otherwise
    """

    error_msg = []
    bools = ("text" in req_dict, "input_ids" in req_dict, "messages" in req_dict)
    if sum(bools) == 0:
        error_msg.append("The input parameters should contain either `text`, `input_ids` or `messages`")
    else:
        if "text" in req_dict:
            if not isinstance(req_dict["text"], str):
                error_msg.append("The `text` in input parameters must be a string")
            elif req_dict["text"] == "":
                error_msg.append("The `text` in input parameters cannot be empty")
        if "system" in req_dict and not isinstance(req_dict["system"], str):
            error_msg.append("The `system` in input parameters must be a string")
        if "input_ids" in req_dict and not isinstance(req_dict["input_ids"], list):
            error_msg.append("The `input_ids` in input parameters must be a list")
        if "messages" in req_dict:
            msg_len = len(req_dict["messages"])
            if not all("content" in item for item in req_dict["messages"]):
                error_msg.append("The item in messages must include `content`")

    if "req_id" not in req_dict:
        error_msg.append("The input parameters should contain `req_id`.")

    if "min_dec_len" in req_dict and \
        (not isinstance(req_dict["min_dec_len"], int) or req_dict["min_dec_len"] < 1):
        error_msg.append("The `min_dec_len` must be an integer and greater than 0")

    keys = ("max_dec_len", "seq_len", "max_tokens")
    for key in keys:
        if key in req_dict and (not isinstance(req_dict[key], int) or req_dict[key] < 1):
            error_msg.append(f"The `{key}` must be an integer and greater than 0")
    if "seq_len" in req_dict and "max_dec_len" not in req_dict:
        req_dict["max_dec_len"] = req_dict["seq_len"]
    if "max_tokens" in req_dict and "max_dec_len" not in req_dict:
        req_dict["max_dec_len"] = req_dict["max_tokens"]

    keys = ("topp", "top_p")
    if sum([key in req_dict for key in keys]) > 1:
        error_msg.append(f"Only one of {keys} should be set")
    else:
        for key in keys:
            if key in req_dict and not 0 <= req_dict[key] <= 1:
                error_msg.append(f"The `{key}` must be in [0, 1]")
        if "top_p" in req_dict and "topp" not in req_dict:
            req_dict["topp"] = req_dict["top_p"]

    if "temperature" in req_dict and not 0 <= req_dict["temperature"]:
        error_msg.append(f"The `temperature` must be >= 0")

    if "eos_token_ids" in req_dict:
        if isinstance(req_dict["eos_token_ids"], int):
            req_dict["eos_token_ids"] = [req_dict["eos_token_ids"]]
        elif isinstance(req_dict["eos_token_ids"], tuple):
            req_dict["eos_token_ids"] = list(req_dict["eos_token_ids"])
        if not isinstance(req_dict["eos_token_ids"], list):
            error_msg.append("The `eos_token_ids` must be an list")
        elif len(req_dict["eos_token_ids"]) > 1:
            error_msg.append("The length of `eos_token_ids` must be 1 if you set it")

    keys = ("infer_seed", "seed")
    if sum([key in req_dict for key in keys]) > 1:
        error_msg.append(f"Only one of {keys} should be set")
    else:
        if "seed" in req_dict and "infer_seed" not in req_dict:
            req_dict["infer_seed"] = req_dict["seed"]

    if "stream" in req_dict and not isinstance(req_dict["stream"], bool):
        error_msg.append("The `stream` must be a boolean")

    if "response_type" in req_dict and (req_dict["response_type"].lower() not in ("fastdeploy", "openai")):
        error_msg.append("The `response_type` must be either `fastdeploy` or `openai`.")

    return error_msg


def add_default_params(req_dict):
    """
    add default params to req_dict

    Args:
        req_dict (dict): input dict

    Returns:
        dict: req_dict with default params
    """
    assert isinstance(req_dict, dict), "The `req_dict` must be a dict."
    if "min_dec_len" not in req_dict:
        req_dict["min_dec_len"] = 1
    if "topp" not in req_dict:
        req_dict["topp"] = 0.7
    if "temperature" not in req_dict:
        req_dict["temperature"] = 0.95
    if "penalty_score" not in req_dict:
        req_dict["penalty_score"] = 1.0
    if "frequency_score" not in req_dict:
        req_dict["frequency_score"] = 0.0
    if "presence_score" not in req_dict:
        req_dict["presence_score"] = 0.0
    return req_dict
