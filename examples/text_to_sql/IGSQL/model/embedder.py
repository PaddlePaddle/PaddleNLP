#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
""" Embedder for tokens. """

import paddle

import data_util.snippets as snippet_handler
import data_util.vocabulary as vocabulary_handler


class Embedder(paddle.nn.Layer):
    """ Embeds tokens. """

    def __init__(self,
                 embedding_size,
                 name="",
                 initializer=None,
                 vocabulary=None,
                 num_tokens=-1,
                 anonymizer=None,
                 freeze=False,
                 use_unk=True):
        super().__init__()

        if vocabulary:
            assert num_tokens < 0, "Specified a vocabulary but also set number of tokens to " + \
                str(num_tokens)
            self.in_vocabulary = lambda token: token in vocabulary.tokens
            self.vocab_token_lookup = lambda token: vocabulary.token_to_id(token
                                                                           )
            if use_unk:
                self.unknown_token_id = vocabulary.token_to_id(
                    vocabulary_handler.UNK_TOK)
            else:
                self.unknown_token_id = -1
            self.vocabulary_size = len(vocabulary)
        else:

            def check_vocab(index):
                """ Makes sure the index is in the vocabulary."""
                assert index < num_tokens, "Passed token ID " + \
                    str(index) + "; expecting something less than " + str(num_tokens)
                return index < num_tokens

            self.in_vocabulary = check_vocab
            self.vocab_token_lookup = lambda x: x
            self.unknown_token_id = num_tokens  # Deliberately throws an error here,
            # But should crash before this
            self.vocabulary_size = num_tokens

        self.anonymizer = anonymizer

        emb_name = name + "-tokens"
        print("Creating token embedder called " + emb_name + " of size " +
              str(self.vocabulary_size) + " x " + str(embedding_size))

        if initializer is not None:
            self.token_embedding_matrix = paddle.nn.Embedding(
                initializer.shape[0], initializer.shape[1])
            self.token_embedding_matrix.weight.set_value(initializer)
        else:
            initializer = paddle.nn.initializer.Uniform(low=-0.1, high=0.1)
            init_tensor = paddle.ParamAttr(initializer=initializer,
                                           trainable=True)
            self.token_embedding_matrix = paddle.nn.Embedding(
                self.vocabulary_size, embedding_size, weight_attr=initializer)

    def forward(self, token):
        assert isinstance(token, int) or not snippet_handler.is_snippet(
            token
        ), "embedder should only be called on flat tokens; use snippet_bow if you are trying to encode snippets"

        if self.in_vocabulary(token):
            index_list = paddle.to_tensor(self.vocab_token_lookup(token),
                                          'int64')
            return self.token_embedding_matrix(index_list).squeeze()
        else:
            index_list = paddle.to_tensor(self.unknown_token_id, 'int64')
            return self.token_embedding_matrix(index_list).squeeze()


def bow_snippets(token, snippets, output_embedder, input_schema):
    """ Bag of words embedding for snippets"""
    assert snippet_handler.is_snippet(token) and snippets

    snippet_sequence = []
    for snippet in snippets:
        if snippet.name == token:
            snippet_sequence = snippet.sequence
            break
    assert snippet_sequence

    if input_schema:
        snippet_embeddings = []
        for output_token in snippet_sequence:
            assert output_embedder.in_vocabulary(
                output_token) or input_schema.in_vocabulary(output_token,
                                                            surface_form=True)
            if output_embedder.in_vocabulary(output_token):
                snippet_embeddings.append(output_embedder(output_token))
            else:
                snippet_embeddings.append(
                    input_schema.column_name_embedder(output_token,
                                                      surface_form=True))
    else:
        snippet_embeddings = [
            output_embedder(subtoken) for subtoken in snippet_sequence
        ]

    snippet_embeddings = paddle.stack(
        snippet_embeddings, axis=0)  # len(snippet_sequence) x emb_size
    return paddle.mean(snippet_embeddings, axis=0)  # emb_size
