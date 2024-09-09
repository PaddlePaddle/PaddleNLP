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

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import sseclient
from pipelines.nodes.prompt.invocation_layer.base import PromptModelInvocationLayer
from pipelines.nodes.prompt.invocation_layer.handlers import (
    DefaultTokenStreamingHandler,
    TokenStreamingHandler,
)

logger = logging.getLogger(__name__)

OPENAI_TIMEOUT = 30


def openai_request(
    url: str,
    headers: Dict,
    payload: Dict,
    timeout: Union[float, Tuple[float, float]] = OPENAI_TIMEOUT,
    read_response: Optional[bool] = True,
    **kwargs,
):
    """Make a request to the OpenAI API given a `url`, `headers`, `payload`, and `timeout`.

    :param url: The URL of the OpenAI API.
    :param headers: Dictionary of HTTP Headers to send with the :class:`Request`.
    :param payload: The payload to send with the request.
    :param timeout: The timeout length of the request. The default is 30s.
    :param read_response: Whether to read the response as JSON. The default is True.
    """

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload), timeout=timeout, **kwargs)
    if read_response:
        json_response = json.loads(response.text)

    if response.status_code != 200:
        openai_error: Exception
        if response.status_code == 429:
            openai_error = Exception(f"API rate limit exceeded: {response.text}")
        elif response.status_code == 401:
            openai_error = Exception(f"API key is invalid: {response.text}")
        else:
            openai_error = Exception(
                f"OpenAI returned an error.\n"
                f"Status code: {response.status_code}\n"
                f"Response body: {response.text}",
                status_code=response.status_code,
            )
        raise openai_error
    if read_response:
        return json_response
    else:
        return response


def _check_openai_finish_reason(result: Dict, payload: Dict) -> None:
    """Check the `finish_reason` the answers returned by OpenAI completions endpoint.
    If the `finish_reason` is `length` or `content_filter`, log a warning to the user.

    :param result: The result returned from the OpenAI API.
    :param payload: The payload sent to the OpenAI API.
    """
    number_of_truncated_completions = sum(1 for ans in result["choices"] if ans["finish_reason"] == "length")
    if number_of_truncated_completions > 0:
        logger.warning(
            "%s out of the %s completions have been truncated before reaching a natural stopping point. "
            "Increase the max_tokens parameter to allow for longer completions.",
            number_of_truncated_completions,
            payload["n"],
        )

    number_of_content_filtered_completions = sum(
        1 for ans in result["choices"] if ans["finish_reason"] == "content_filter"
    )
    if number_of_content_filtered_completions > 0:
        logger.warning(
            "%s out of the %s completions have omitted content due to a flag from OpenAI content filters.",
            number_of_truncated_completions,
            payload["n"],
        )


class OpenAIInvocationLayer(PromptModelInvocationLayer):
    """
    PromptModelInvocationLayer implementation for OpenAI's GPT-3 InstructGPT models. Invocations are made using REST API.
    See [OpenAI GPT-3](https://platform.openai.com/docs/models/gpt-3) for more details.

    Note: kwargs other than init parameter names are ignored to enable reflective construction of the class
    as many variants of PromptModelInvocationLayer are possible and they may have different parameters.
    """

    def __init__(
        self,
        api_key: str,
        model_name_or_path: str = "text-davinci-003",
        max_length: Optional[int] = 100,
        api_base: str = "https://api.openai.com/v1",
        **kwargs
    ):
        """
         Creates an instance of OpenAIInvocationLayer for OpenAI's GPT-3 InstructGPT models.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum number of tokens the output text can have.
        :param api_key: The OpenAI API key.
        :param api_base: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param kwargs: Additional keyword arguments passed to the underlying model. Due to reflective construction of
        all PromptModelInvocationLayer instances, this instance of OpenAIInvocationLayer might receive some unrelated
        kwargs. Only the kwargs relevant to OpenAIInvocationLayer are considered. The list of OpenAI-relevant
        kwargs includes: suffix, temperature, top_p, presence_penalty, frequency_penalty, best_of, n, max_tokens,
        logit_bias, stop, echo, and logprobs. For more details about these kwargs, see OpenAI
        [documentation](https://platform.openai.com/docs/api-reference/completions/create).
        """
        super().__init__(model_name_or_path)
        if not isinstance(api_key, str) or len(api_key) == 0:
            raise Exception(f"api_key {api_key} must be a valid OpenAI key. Visit https://openai.com/api/ to get one.")
        self.api_key = api_key
        self.api_base = api_base

        # 16 is the default length for answers from OpenAI shown in the docs
        # here, https://platform.openai.com/docs/api-reference/completions/create.
        # max_length must be set otherwise OpenAIInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or 16

        # Due to reflective construction of all invocation layers we might receive some
        # unknown kwargs, so we need to take only the relevant.
        # For more details refer to OpenAI documentation
        self.model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "suffix",
                "max_tokens",
                "temperature",
                "top_p",
                "n",
                "logprobs",
                "echo",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "best_of",
                "logit_bias",
                "stream",
                "stream_handler",
            ]
            if key in kwargs
        }

        max_tokens_limit = 4096
        self.max_tokens_limit = max_tokens_limit

    @property
    def url(self) -> str:
        return f"{self.api_base}/completions"

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def invoke(self, *args, **kwargs):
        """
        Invokes a prompt on the model. It takes in a prompt and returns a list of responses using a REST invocation.

        :return: The responses are being returned.

        Note: Only kwargs relevant to OpenAI are passed to OpenAI rest API. Others kwargs are ignored.
        For more details, see OpenAI [documentation](https://platform.openai.com/docs/api-reference/completions/create).
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                f"No prompt provided. Model {self.model_name_or_path} requires prompt."
                f"Make sure to provide prompt in kwargs."
            )

        kwargs_with_defaults = self.model_input_kwargs
        if kwargs:
            # we use keyword stop_words but OpenAI uses stop
            if "stop_words" in kwargs:
                kwargs["stop"] = kwargs.pop("stop_words")
            if "top_k" in kwargs:
                top_k = kwargs.pop("top_k")
                kwargs["n"] = top_k
                kwargs["best_of"] = top_k
            kwargs_with_defaults.update(kwargs)

        # either stream is True (will use default handler) or stream_handler is provided
        stream = (
            kwargs_with_defaults.get("stream", False) or kwargs_with_defaults.get("stream_handler", None) is not None
        )
        payload = {
            "model": self.model_name_or_path,
            "prompt": prompt,
            "suffix": kwargs_with_defaults.get("suffix", None),
            "max_tokens": kwargs_with_defaults.get("max_tokens", self.max_length),
            "temperature": kwargs_with_defaults.get("temperature", 0.7),
            "top_p": kwargs_with_defaults.get("top_p", 1),
            "n": kwargs_with_defaults.get("n", 1),
            "stream": stream,
            "logprobs": kwargs_with_defaults.get("logprobs", None),
            "echo": kwargs_with_defaults.get("echo", False),
            "stop": kwargs_with_defaults.get("stop", None),
            "presence_penalty": kwargs_with_defaults.get("presence_penalty", 0),
            "frequency_penalty": kwargs_with_defaults.get("frequency_penalty", 0),
            "best_of": kwargs_with_defaults.get("best_of", 1),
            "logit_bias": kwargs_with_defaults.get("logit_bias", {}),
        }
        if not stream:
            res = openai_request(url=self.url, headers=self.headers, payload=payload)
            _check_openai_finish_reason(result=res, payload=payload)
            responses = [ans["text"].strip() for ans in res["choices"]]
            return responses
        else:
            response = openai_request(
                url=self.url, headers=self.headers, payload=payload, read_response=False, stream=True
            )
            handler: TokenStreamingHandler = kwargs_with_defaults.pop("stream_handler", DefaultTokenStreamingHandler())
            return self._process_streaming_response(response=response, stream_handler=handler)

    def _process_streaming_response(self, response, stream_handler: TokenStreamingHandler):
        client = sseclient.SSEClient(response)
        tokens: List[str] = []
        try:
            for event in client.events():
                if event.data != TokenStreamingHandler.DONE_MARKER:
                    event_data = json.loads(event.data)
                    token: str = self._extract_token(event_data)
                    if token:
                        tokens.append(stream_handler(token, event_data=event_data["choices"]))
        finally:
            client.close()
        return ["".join(tokens)]  # return a list of strings just like non-streaming

    def _extract_token(self, event_data: Dict[str, Any]):
        return event_data["choices"][0]["text"]

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Ensure that the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.

        :param prompt: Prompt text to be sent to the generative model.
        """
        n_prompt_tokens = len(prompt)
        n_answer_tokens = self.max_length
        if (n_prompt_tokens + n_answer_tokens) <= self.max_tokens_limit:
            return prompt

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
            "answer length (%s tokens) fit within the max token limit (%s tokens). "
            "Reduce the length of the prompt to prevent it from being cut off.",
            n_prompt_tokens,
            self.max_tokens_limit - n_answer_tokens,
            n_answer_tokens,
            self.max_tokens_limit,
        )
        return prompt

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        valid_model = any(m for m in ["ada", "babbage", "davinci", "curie"] if m in model_name_or_path)
        return valid_model and kwargs.get("azure_base_url") is None
