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

from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

from paddlenlp.transformers import GPTLMHeadModel, GPTTokenizer


class ClipCaptionModel(nn.Layer):
    def __init__(self, prefix_length=77, hidden_dim=64):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        eos = "<|EOS|>"
        special_tokens_dict = {"eos_token": eos}
        base_tokenizer = GPTTokenizer.from_pretrained("gpt2-en")
        base_tokenizer.add_special_tokens(special_tokens_dict)
        self.gpt = GPTLMHeadModel.from_pretrained("gpt2-en", eos_token_id=base_tokenizer.eos_token_id)
        self.gpt.resize_token_embeddings(len(base_tokenizer))

        self.hidden_dim = hidden_dim
        self.encode_prefix = nn.Linear(768, hidden_dim) if hidden_dim is not None else nn.Identity()
        self.decode_prefix = nn.Linear(hidden_dim, 768) if hidden_dim is not None else nn.Identity()

    def get_dummy_token(self, batch_size: int) -> paddle.Tensor:
        return paddle.zeros([batch_size, self.prefix_length], dtype=paddle.int64)

    def encode_decode_prefix(self, prefix):
        return self.decode_prefix(self.encode_prefix(prefix))

    def forward(
        self,
        tokens: paddle.Tensor,
        prefix: paddle.Tensor,
        mask: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
    ):
        embedding_text = self.gpt.gpt.embeddings.word_embeddings(tokens)
        hidden = self.encode_prefix(prefix)
        prefix = self.decode_prefix(hidden)
        embedding_cat = paddle.concat((prefix, embedding_text), axis=1)

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0])
            labels = paddle.concat((dummy_token, tokens), axis=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)

        if self.hidden_dim is not None:
            return out, hidden
        else:
            return out


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embedding=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = "<|EOS|>",
):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)["input_ids"][0]
    tokens = None
    scores = None

    seq_lengths = paddle.ones([beam_size])
    is_stopped = paddle.zeros([beam_size], dtype=paddle.bool)
    with paddle.no_grad():
        if embedding is not None:
            generated = embedding
        else:
            if tokens is None:
                tokens = paddle.to_tensor(tokenizer.encode(prompt)["input_ids"])
                tokens = tokens.unsqueeze(0)
                generated = model.gpt.gpt.embeddings.word_embeddings(tokens)

        for i in range(entry_length):
            model.gpt.lm_head.decoder_weight = model.gpt.embeddings.word_embeddings.weight
            logits = model.gpt(inputs_embeds=generated)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = F.softmax(logits, axis=-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand([beam_size, *generated.shape[1:]])
                next_tokens, scores = next_tokens.transpose([1, 0]), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand([beam_size, *tokens.shape[1:]])
                    tokens = paddle.concat((tokens, next_tokens), axis=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.reshape([-1]).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = paddle.concat((tokens, next_tokens), axis=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = paddle.cast(is_stopped, "int32")
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.gpt.embeddings.word_embeddings(next_tokens.squeeze()).reshape(
                [generated.shape[0], 1, -1]
            )
            generated = paddle.concat((generated, next_token_embed), axis=1)
            is_stopped = paddle.cast(is_stopped, "int32") + paddle.cast(
                next_tokens.equal(stop_token_index).squeeze(), "int32"
            )
            is_stopped = paddle.cast(is_stopped, "bool")
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)], skip_special_tokens=True)
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    model.train()
    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embedding=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = "<|EOS|>",
):
    model.eval()
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)["input_ids"][0]
    filter_value = -float("Inf")

    with paddle.no_grad():
        for entry_idx in range(entry_count):
            if embedding is not None:
                generated = embedding
            else:
                if tokens is None:
                    tokens = paddle.to_tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0)

                generated = model.gpt.gpt.embeddings.word_embeddings(tokens)

            for i in range(entry_length):
                model.gpt.lm_head.decoder_weight = model.gpt.embeddings.word_embeddings.weight
                logits = model.gpt(inputs_embeds=generated)
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits = paddle.sort(logits, descending=True)
                sorted_indices = paddle.argsort(logits, descending=True)
                cumulative_probs = paddle.cumsum(F.softmax(sorted_logits, axis=-1), axis=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = paddle.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.gpt.embeddings.word_embeddings(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = paddle.concat((tokens, next_token), axis=1)
                generated = paddle.concat((generated, next_token_embed), axis=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


class CaptionDecoder(object):
    def __init__(self, pretrained_path, hidden_dim=64):
        super(CaptionDecoder, self).__init__()
        if hidden_dim < 0:
            hidden_dim = None
        # tokenizer initialize
        eos = "<|EOS|>"
        special_tokens_dict = {"eos_token": eos}
        self.tokenizer = GPTTokenizer.from_pretrained("gpt2-en")
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # model initialize
        self.caption_model = ClipCaptionModel(prefix_length=77, hidden_dim=hidden_dim)

        ckpt = paddle.load(pretrained_path)
        state_dict = OrderedDict()
        for k, v in ckpt.items():
            new_k = k[7:]  # remove 'module.'
            state_dict[new_k] = v
        self.caption_model.set_state_dict(state_dict)

    def encode_prefix(self, features):
        return self.caption_model.encode_prefix(features)

    def generate_captions(self, features):  # the low dimension representation of clip feature
        """
        generate captions given features
        Args:
            features : (tensor([B x L x D]))
        Return:
            generated_captions: (list([L]))
        """
        features = paddle.split(features, 1, axis=0)
        generated_captions = []
        use_beam_search = True
        with paddle.no_grad():
            for feature in features:
                feature = self.caption_model.decode_prefix(feature)  # back to the clip feature
                if use_beam_search:
                    generated_captions.append(generate_beam(self.caption_model, self.tokenizer, embedding=feature)[0])
                else:
                    generated_captions.append(generate2(self.caption_model, self.tokenizer, embedding=feature))
        return generated_captions
