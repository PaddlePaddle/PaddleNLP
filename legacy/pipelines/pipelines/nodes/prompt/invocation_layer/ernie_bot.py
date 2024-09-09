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

import logging
import os
from typing import Dict, List, Union

from pipelines.nodes.llm.ernie_bot import ErnieBot
from pipelines.nodes.prompt.invocation_layer import PromptModelInvocationLayer

logger = logging.getLogger(__name__)


class ErnieBotInvocationLayer(ErnieBot, PromptModelInvocationLayer):
    """
    A subclass of the PromptModelInvocationLayer class. It loads a pre-trained model from Taskflow and
    passes a prepared prompt into that model.

    Note: kwargs other than init parameter names are ignored to enable reflective construction of the class,
    as many variants of PromptModelInvocationLayer are possible and they may have different parameters.
    """

    def __init__(
        self,
        api_key=None,
        secret_key=None,
        **kwargs,
    ):
        """
        Creates an instance of ChatGLMInvocationLayer used to invoke local ChatGLM models.

        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum number of tokens the output text can have.
        :param use_auth_token: The token to use as HTTP bearer authorization for remote files.
        :param use_gpu: Whether to use GPU for inference.
        :param device: The device to use for inference.
        :param kwargs: Additional keyword arguments passed to the underlying model. Due to reflective construction of
        all PromptModelInvocationLayer instances, this instance of ChatGLMInvocationLayer might receive some unrelated
        kwargs. Only kwargs relevant to the ChatGLMInvocationLayer are considered.
        The model_max_length is used to specify the custom sequence length for the underlying pipeline.
        """
        super().__init__(api_key, secret_key)

        self.kwargs = kwargs

    def invoke(self, *args, **kwargs):

        prompt = kwargs.pop("prompt")
        if not prompt:
            raise ValueError(
                f"No prompt provided. Model {self.model_name_or_path} requires prompt."
                f"Make sure to provide prompt in kwargs."
            )
        output = self.predict(prompt)
        if "stop_words" in kwargs and kwargs["stop_words"] is not None:
            # split text by stop words
            result = output["result"].split(kwargs["stop_words"][0])[0]
            generated_texts = [result]
        else:
            generated_texts = [output["result"]]
        return generated_texts

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Ensure that the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.

        :param prompt: Prompt text to be sent to the generative model.
        """
        # TODO: support truncation
        return prompt

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        if os.path.exists(model_name_or_path):
            return True
        return model_name_or_path in ["ernie-bot"]
