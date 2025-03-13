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


from paddlenlp.transformers import LlamaForCausalLM, PretrainedConfig

from .ppo_model_utils import PolicyOutput, RLHFPPOMixedLoss, RLHFValueLoss, ValueOutput
from .score_model import LlamaModelForScore


# TODO(guosheng): create Mixin and make model classes using metaclass.
class LlamaPolicyModel(LlamaForCausalLM):
    def __init__(self, config: PretrainedConfig, **kwargs):
        """
        Initializes a RLHFPPOMixedLossWrapper instance.

        Args:
            config (PretrainedConfig): The model configuration used for initialization.
            kwargs (Dict[str, Any], optional): Additional keyword arguments passed along. Defaults to {}.
        """
        super().__init__(config)
        self.loss_fn = RLHFPPOMixedLoss(config, **kwargs)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
        log_probs=None,
        advantages=None,
        sequence_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Returns a tuple containing:
        1. the loss, calculated as the sum of the cross entropy for each token and the KL divergence between the
           policy distribution and the uniform distribution. If `advantages` are provided, the loss will be
           augmented with the additional term -E[log P(a|x)] where x is the input and a is the action.
        2. the model's output as a tuple of:
            - the last layer's output of shape `(batch_size, sequence_length, config.vocab_size)`
            - the cache used in inference for next chunk.
            - the decoder's attention weights for each layer.
        """
        outputs = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs[0]
        loss = None
        if labels is not None or advantages is not None:
            loss = self.loss_fn(
                logits,
                (labels, input_ids, log_probs, advantages, sequence_mask),
            )
        if not return_dict:
            return (loss,) + outputs if loss is not None else outputs

        return PolicyOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaValueModel(LlamaModelForScore):
    def __init__(self, config, **kwargs):
        """
        Initializes the RLHFValueLossWrapper instance.

        Args:
            config (DictConfig): Config dict for the model.
            **kwargs (Any, optional): Keyword arguments to be passed to the parent class. Defaults to None.

        Returns:
            None.
        """
        super().__init__(config, **kwargs)
        self.loss_fn = RLHFValueLoss(config, **kwargs)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
        old_values=None,
        returns=None,
        sequence_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Returns:
        Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
            if `return_dict` is False, a tuple of tensors is returned, containing:
                - the loss, if it is not None;
                - the reward values;
                - the rewards;
                - the past key values;
                - the hidden states;
                - the attentions.
            if `return_dict` is True, a [`ValueOutput`] is returned, containing:
                - the loss, if it is not None;
                - the reward values;
                - the rewards;
                - the past key values;
                - the hidden states;
                - the attentions.
        """
        outputs = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        reward_values, rewards = outputs
        loss = None
        if returns is not None:
            loss = self.loss_fn(reward_values, old_values, returns, sequence_mask)
        if not return_dict:
            return (loss,) + outputs if loss is not None else outputs

        return ValueOutput(
            loss=loss,
            value=reward_values,
            reward=rewards,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
