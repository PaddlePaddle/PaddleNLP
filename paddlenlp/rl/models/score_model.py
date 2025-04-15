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

from typing import Any, Optional

import paddle
from paddle import nn

from ...transformers import PretrainedConfig, PretrainedModel
from ...transformers.auto import AutoModel
from .score_model_utils import ScoreModelMixin, ScoreModelOutput


class AutoModelForScore(ScoreModelMixin, PretrainedModel):
    _keys_to_ignore_on_load_missing = ["lm_head.weight"]

    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None:
        """
        Initializes a `AutoModelForScore` model.

        Args:
            config (PretrainedConfig): Model configuration class with all the parameters of the model.
            kwargs (Any, optional): Additional keyword arguments passed along to the `__init__` of the parent class.
                This is necessary because of how `transformers.AutoModelWithHead` is designed. Defaults to `None`.

        Raises:
            TypeError: If the config is not an instance of `PretrainedConfig`.
        """
        super().__init__(config)
        self.model = AutoModel.from_config(config)

        # config.architectures = [self.__class__.__name__]
        self.init_score_head(config, hidden_size=config.hidden_size, **kwargs)

    def get_input_embeddings(self) -> Optional[nn.Embedding]:
        """
        Returns the nn.Embedding object for input embeddings, which converts input sequences into embedding vectors.
        If the model does not use embeddings, it returns None.

        Returns:
            Optional[nn.Embedding]: The nn.Embedding object for input embeddings, or None if embeddings are not used.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """
        Set the input embeddings to be used for the model.

        Args:
            value (nn.Embedding): The embedding layer to use.

        Returns:
            NoneType: No return value is returned. Instead, the input embeddings are updated in-place.
        """
        self.model.embed_tokens = value

    def get_decoder(self) -> PretrainedModel:
        """
        Retrieve the decoder model.

        Returns:
            PretrainedModel: The decoder model, which is an instance of PaddlePaddle's PretrainedModel class.
        """
        return self.model

    def set_decoder(self, decoder: PretrainedModel) -> None:
        """
        Sets the decoder for text generation.

        Args:
            decoder (PretrainedModel): A pre-trained model object that serves as a valid decoder.

        Returns:
            None: No return value.
        """
        self.model = decoder

    def forward(
        self,
        input_ids: paddle.Tensor,
        attention_mask: paddle.Tensor,
        position_ids: paddle.Tensor | None = None,
        past_key_values: list[paddle.Tensor] | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor] | ScoreModelOutput:
        """
        Forward pass of the sentence.

        Args:
            input_ids (paddle.Tensor):
                IDs of the input sequences, with shape (batch_size, sequence_length).
            attention_mask (paddle.Tensor):
                Mask used to distinguish padding and non-padding elements, with shape (batch_size, sequence_length), values are 0 or 1.
            position_ids (paddle.Tensor, optional):
                Position IDs corresponding to input_ids, with shape (batch_size, sequence_length), default is None.
            past_key_values (list[paddle.Tensor], optional):
                Contains all preprocessed keys and values, default is None.
            inputs_embeds (paddle.Tensor, optional):
                Embeddings of the input sequences, with shape (batch_size, sequence_length, embedding_dimension), default is None.
            use_cache (bool, optional):
                Whether to use caching, default is None.
            output_attentions (bool, optional):
                Whether to return attention tensors, default is None.
            output_hidden_states (bool, optional):
                Whether to return hidden states, default is None.
            return_dict (bool, optional):
                Whether to return results in dictionary format, default is None.

        Returns:
            tuple[paddle.Tensor, paddle.Tensor] or ScoreModelOutput:
                If `return_dict` is True, returns a tuple of ScoreModelOutput type containing two elements: score and additional information; otherwise, returns a tuple containing the score and additional information.
        Raises:
            AssertionError:
                Raised when `attention_mask` is not None.
        """
        assert attention_mask is not None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # size = (B, L, E)
        return self.get_score(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
        )
