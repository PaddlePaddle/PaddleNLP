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

import json
import logging
import os
import time
from typing import Dict, List, Sequence, Union

import numpy as np
import paddle
import requests

from paddlenlp.transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def load_data(file_path: str, mode: str) -> Dict:
    """
    Load data from a JSON file in StructuredIndexer and return it as a dictionary.

    Args:
        file_path (str): The path to the JSON file to be loaded.
        mode (str): A string indicating the mode (e.g., "read", "load") for logging purposes.

    Returns:
        Dict: A dictionary containing the data loaded from the JSON file.

    Raises:
        ValueError: If the provided path is not a file or if the file is not a JSON file.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"{file_path} is not a file")
    if not file_path.endswith(".json"):
        raise ValueError(f"File {file_path} is not a json file")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    logging.info(f"{mode} file {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def load_model(model_name_or_path: str):
    """
    Load a model and its tokenizer from a specified path or model name.

    Args:
        model_name_or_path (str): The path to the model or the name of the model to be loaded.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the loaded model and its tokenizer.

    Raises:
        RuntimeError: If the model fails to load from the specified path or model name.
    """
    device = "gpu" if paddle.device.cuda.device_count() >= 1 else "cpu"
    paddle.device.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype="float16")
    except Exception:
        try:
            model = AutoModel.from_pretrained(model_name_or_path, dtype="float16")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_name_or_path}: {e}")
    model.eval()
    logging.info(f"Model {model_name_or_path} Loaded to {device}")
    logging.debug(f"{model.config}")
    return model, tokenizer


def encode(
    sentences: Sequence[str], tokenizer, model, convert_to_numpy: bool = True
) -> Union[np.ndarray, paddle.Tensor]:
    """
    Encode a sequence of sentences into embeddings using a specified model and tokenizer.

    Args:
        sentences (Sequence[str]): A sequence of sentences to be encoded.
        tokenizer: The tokenizer used to preprocess the sentences.
        model: The model used to generate embeddings.
        convert_to_numpy (bool, optional): Whether to convert the embeddings to a numpy array. Defaults to True.

    Returns:
        Union[np.ndarray, paddle.Tensor]: The embeddings of the sentences, either as a numpy array or a PaddlePaddle tensor.
    """
    model.eval()
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pd")
    with paddle.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, (0)]
    sentence_embeddings = paddle.nn.functional.normalize(x=sentence_embeddings, p=2, axis=1)
    if convert_to_numpy and isinstance(sentence_embeddings, paddle.Tensor):
        sentence_embeddings_np: np.ndarray = sentence_embeddings.cpu().numpy()
        return sentence_embeddings_np
    return sentence_embeddings


def get_response(messages: List[Dict], tokenizer, model, max_new_tokens=1024) -> str:
    """
    Generate a response using a specified model and tokenizer based on the input messages.

    Args:
        messages (List[Dict]): A list of dictionaries containing the input messages.
        tokenizer: The tokenizer used to preprocess the input messages.
        model: The model used to generate the response.
        max_new_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 1024.

    Returns:
        str: The generated response.
    """
    # logging.debug(messages[0]['content'])
    inputs = tokenizer(messages[0]["content"], return_tensors="pd")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response: List[str] = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
    assert isinstance(response, list)
    response: str = response[0]
    # logging.debug(f"response =\n{response}")
    return response.strip()


def get_messages(
    messages: List[Dict],
    model_name: str = "Qwen2.5-72B-Instruct-GPTQ-Int4",
    url: str = None,
    api_key: str = None,
    temperature: float = None,
    n: int = 1,
    max_tokens: int = None,
    max_new_tokens: int = None,
    stream: bool = False,
    debug: bool = False,
) -> List[str]:
    """
    Send a request to a remote model API to generate responses based on the input messages.
    """
    headers = {"Content-Type": "application/json"}
    if model_name.startswith("gpt"):
        if api_key is None:
            raise ValueError("api_key is None")
        headers["Authorization"] = f"Bearer {api_key}"

    data = {
        "model": model_name,
        "messages": messages,
        "n": n,
        "stream": stream,
    }
    if temperature is not None:
        data["temperature"] = temperature
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if max_new_tokens is not None:
        data["max_new_tokens"] = max_new_tokens

    if debug:
        print(f"data={data}")

    message = ""
    if not isinstance(message, Dict) or "choices" not in message:
        repeat_index = 0
        while not isinstance(message, Dict) or "choices" not in message:
            if repeat_index > 5:
                raise ConnectionError(f"{url} Error\nmessage=\n{message}")
            if debug:
                print(f"message=\n{message}")
            time.sleep(5)
            response = requests.post(url, json=data, headers=headers)
            if debug:
                print(f"response=\n{response.text}")
            message = json.loads(response.text)
            repeat_index += 1
        if not isinstance(message, Dict) or "choices" not in message:
            raise ConnectionError(f"{url} Error\nmessage=\n{message}")
    if len(message["choices"]) != n:
        raise ValueError(f"{model_name} response num error")
    return [message["choices"][i]["message"]["content"] for i in range(n)]
