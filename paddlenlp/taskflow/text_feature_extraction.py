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

from typing import Optional

import numpy as np
import paddle

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.transformers import AutoModel, AutoTokenizer, ErnieDualEncoder

from ..utils.log import logger
from .task import Task
from .utils import dygraph_mode_guard, static_mode_guard

ENCODER_TYPE = {
    "rocketqa-zh-dureader-query-encoder": "query",
    "rocketqa-zh-dureader-para-encoder": "paragraph",
    "rocketqa-zh-base-query-encoder": "query",
    "rocketqa-zh-base-para-encoder": "paragraph",
    "rocketqa-zh-medium-query-encoder": "query",
    "rocketqa-zh-medium-para-encoder": "paragraph",
    "rocketqa-zh-mini-query-encoder": "query",
    "rocketqa-zh-mini-para-encoder": "paragraph",
    "rocketqa-zh-micro-query-encoder": "query",
    "rocketqa-zh-micro-para-encoder": "paragraph",
    "rocketqa-zh-nano-query-encoder": "query",
    "rocketqa-zh-nano-para-encoder": "paragraph",
    "rocketqav2-en-marco-query-encoder": "query",
    "rocketqav2-en-marco-para-encoder": "paragraph",
    "ernie-search-base-dual-encoder-marco-en": "query_paragraph",
}


usage = r"""
            from paddlenlp import Taskflow
            import paddle.nn.functional as F
            # Text feature_extraction with rocketqa-zh-base-query-encoder
            text_encoder = Taskflow("feature_extraction", model='rocketqa-zh-base-query-encoder')
            text_embeds = text_encoder(['春天适合种什么花？','谁有狂三这张高清的?'])
            text_features1 = text_embeds["features"]
            print(text_features1)
            '''
            Tensor(shape=[2, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                [[ 0.27640465, -0.13405125,  0.00612330, ..., -0.15600294,
                    -0.18932408, -0.03029604],
                    [-0.12041329, -0.07424965,  0.07895312, ..., -0.17068857,
                    0.04485796, -0.18887770]])
            '''
            text_embeds = text_encoder('春天适合种什么菜？')
            text_features2 = text_embeds["features"]
            print(text_features2)
            '''
            Tensor(shape=[1, 768], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                [[ 0.32578075, -0.02398480, -0.18929179, -0.18639392, -0.04062131,
                    0.06708499, -0.04631376, -0.41177100, -0.23074438, -0.23627219,
                ......
            '''
            probs = F.cosine_similarity(text_features1, text_features2)
            print(probs)
            '''
            Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                [0.86455142, 0.41222256])
            '''
         """


class TextFeatureExtractionTask(Task):

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "config": "config.json",
        "vocab_file": "vocab.txt",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json",
    }

    resource_files_urls = {
        "rocketqa-zh-dureader-query-encoder": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-dureader-query-encoder/model_state.pdparams",
                "6125930530fd55ed715b0595e65789aa",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-dureader-query-encoder/config.json",
                "efc1280069bb22b5bd06dc44b780bc6a",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-dureader-query-encoder/vocab.txt",
                "062f696cad47bb62da86d8ae187b0ef4",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-dureader-query-encoder/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-dureader-query-encoder/tokenizer_config.json",
                "3a50349b8514e744fed72e59baca51b5",
            ],
        },
        "rocketqa-zh-base-query-encoder": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-base-query-encoder/model_state.pdparams",
                "3bb1a7870792146c6dd2fa47a45e15cc",
            ],
            "config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-base-query-encoder/config.json",
                "be88115dd8a00e9de6b44f8c9a055e1a",
            ],
            "vocab_file": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-base-query-encoder/vocab.txt",
                "1c1c1f4fd93c5bed3b4eebec4de976a8",
            ],
            "special_tokens_map": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-base-query-encoder/special_tokens_map.json",
                "8b3fb1023167bb4ab9d70708eb05f6ec",
            ],
            "tokenizer_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/feature_extraction/rocketqa-zh-base-query-encoder/tokenizer_config.json",
                "be86466f6769fde498690269d099ea7c",
            ],
        },
    }

    def __init__(
        self,
        task: str = None,
        model: str = None,
        batch_size: int = 1,
        max_seq_len: int = 128,
        _static_mode: bool = True,
        return_tensors: str = "pd",
        reinitialize: bool = False,
        share_parameters: bool = False,
        is_paragraph: bool = False,
        output_emb_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(task=task, model=model, **kwargs)
        self._seed = None
        self.export_type = "text"
        self._batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.model = model
        self._static_mode = _static_mode
        self.return_tensors = return_tensors

        self.reinitialize = reinitialize
        self.share_parameters = share_parameters
        self.output_emb_size = output_emb_size
        self.is_paragraph = is_paragraph
        self._check_para_encoder()
        # self._check_task_files()
        self._check_predictor_type()
        self._construct_tokenizer()
        # self._get_inference_model()
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)

    def _check_para_encoder(self):
        if self.model in ENCODER_TYPE:
            if ENCODER_TYPE[self.model] == "paragraph":
                self.is_paragraph = True
            else:
                self.is_paragraph = False
        else:
            self.is_paragraph = False

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        # self._model = ErnieDualEncoder(self._task_path)
        self._model = ErnieDualEncoder(
            query_model_name_or_path=self.model,
            output_emb_size=self.output_emb_size,
            reinitialize=self.reinitialize,
            share_parameters=self.share_parameters,
        )
        self._model.eval()

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        # Fix windows dtype bug
        if self._static_mode:
            self._collator = DataCollatorWithPadding(self._tokenizer, return_tensors="np")
        else:
            self._collator = DataCollatorWithPadding(self._tokenizer, return_tensors="pd")

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
        ]

    def _batchify(self, data, batch_size):
        """
        Generate input batches.
        """

        def _parse_batch(batch_examples):
            if self.is_paragraph:
                # The input of the passage encoder is [CLS][SEP]...[SEP].
                tokenized_inputs = self._tokenizer(
                    text=[""] * len(batch_examples),
                    text_pair=batch_examples,
                    padding="max_length",
                    truncation=True,
                    max_seq_len=self.max_seq_len,
                )
            else:
                tokenized_inputs = self._tokenizer(
                    text=[""] * len(batch_examples),
                    text_pair=batch_examples,
                    padding="max_length",
                    truncation=True,
                    max_seq_len=self.max_seq_len,
                )
            return tokenized_inputs

        # Separates data into some batches.
        one_batch = []
        for example in data:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield _parse_batch(one_batch)
                one_batch = []
        if one_batch:
            yield _parse_batch(one_batch)

    def _preprocess(self, inputs):
        """
        Transform the raw inputs to the model inputs, two steps involved:
           1) Transform the raw text/image to token ids/pixel_values.
           2) Generate the other model inputs from the raw text/image and token ids/pixel_values.
        """
        inputs = self._check_input_text(inputs)
        batches = self._batchify(inputs, self._batch_size)
        outputs = {"batches": batches, "inputs": inputs}
        return outputs

    def _run_model(self, inputs, **kwargs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """
        all_feats = []
        if self._static_mode:
            with static_mode_guard():
                for batch_inputs in inputs["batches"]:
                    batch_inputs = self._collator(batch_inputs)
                    if self._predictor_type == "paddle-inference":
                        if "input_ids" in batch_inputs:
                            self.input_handles[0].copy_from_cpu(batch_inputs["input_ids"])
                            self.input_handles[1].copy_from_cpu(batch_inputs["token_type_ids"])
                            self.predictor.run()
                            text_features = self.output_handle[0].copy_to_cpu()
                            all_feats.append(text_features)
                    else:
                        # onnx mode
                        if "input_ids" in batch_inputs:
                            input_dict = {}
                            input_dict["input_ids"] = batch_inputs["input_ids"]
                            input_dict["token_type_ids"] = batch_inputs["token_type_ids"]
                            text_features = self.predictor.run(None, input_dict)[0].tolist()
                            all_feats.append(text_features)

        else:
            with dygraph_mode_guard():
                for batch_inputs in inputs["batches"]:
                    batch_inputs = self._collator(batch_inputs)
                    text_features = self._model.get_pooled_embedding(
                        input_ids=batch_inputs["input_ids"], token_type_ids=batch_inputs["token_type_ids"]
                    )
                    all_feats.append(text_features.numpy())
        inputs.update({"features": all_feats})
        return inputs

    def _postprocess(self, inputs):
        inputs["features"] = np.concatenate(inputs["features"], axis=0)
        if self.return_tensors == "pd":
            inputs["features"] = paddle.to_tensor(inputs["features"])
        return inputs

    def _convert_dygraph_to_static(self):
        """
        Convert the dygraph model to static model.
        """
        assert (
            self._model is not None
        ), "The dygraph model must be created before converting the dygraph model to static model."
        assert (
            self._input_spec is not None
        ), "The input spec must be created before converting the dygraph model to static model."
        logger.info("Converting to the inference model cost a little time.")

        static_model = paddle.jit.to_static(self._model.get_pooled_embedding, input_spec=self._input_spec)
        paddle.jit.save(static_model, self.inference_model_path)
        logger.info("The inference model save in the path:{}".format(self.inference_model_path))


def text_length(text):
    # {key: value} case
    if isinstance(text, dict):
        return len(next(iter(text.values())))
    # Object has no len() method
    elif not hasattr(text, "__len__"):
        return 1
    # Empty string or list of ints
    elif len(text) == 0 or isinstance(text[0], int):
        return len(text)
    # Sum of length of individual strings
    else:
        return sum([len(t) for t in text])


class SentenceFeatureExtractionTask(Task):

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "config": "config.json",
        "vocab_file": "vocab.txt",
        "special_tokens_map": "special_tokens_map.json",
        "tokenizer_config": "tokenizer_config.json",
    }

    def __init__(
        self,
        task: str = None,
        model: str = None,
        batch_size: int = 1,
        max_seq_len: int = 512,
        _static_mode: bool = True,
        return_tensors: str = "pd",
        pooling_mode: str = "cls_token",
        **kwargs
    ):
        super().__init__(
            task=task,
            model=model,
            pooling_mode=pooling_mode,
            **kwargs,
        )
        self._seed = None
        self.export_type = "text"
        self._batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.model = model
        self._static_mode = _static_mode
        self.return_tensors = return_tensors
        self.pooling_mode = pooling_mode
        self._check_predictor_type()
        self._construct_tokenizer()
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        self._model = AutoModel.from_pretrained(self.model)
        self._model.eval()

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pad_token_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)
        # Fix windows dtype bug
        if self._static_mode:
            self._collator = DataCollatorWithPadding(self._tokenizer, return_tensors="np")
        else:
            self._collator = DataCollatorWithPadding(self._tokenizer, return_tensors="pd")

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
        ]

    def _batchify(self, data, batch_size):
        """
        Generate input batches.
        """

        def _parse_batch(batch_examples, max_seq_len=None):
            if isinstance(batch_examples[0], str):
                to_tokenize = [batch_examples]
            else:
                batch1, batch2 = [], []
                for text_tuple in batch_examples:
                    batch1.append(text_tuple[0])
                    batch2.append(text_tuple[1])
                to_tokenize = [batch1, batch2]
            to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]
            if max_seq_len is None:
                max_seq_len = self.max_seq_len
            tokenized_inputs = self._tokenizer(
                to_tokenize[0],
                padding=True,
                truncation="longest_first",
                max_seq_len=max_seq_len,
            )
            return tokenized_inputs

        # Seperates data into some batches.
        one_batch = []
        self.length_sorted_idx = np.argsort([-text_length(sen) for sen in data])
        sentences_sorted = [data[idx] for idx in self.length_sorted_idx]

        for example in range(len(sentences_sorted)):
            one_batch.append(sentences_sorted[example])
            if len(one_batch) == batch_size:
                yield _parse_batch(one_batch)
                one_batch = []
        if one_batch:
            yield _parse_batch(one_batch)

    def _preprocess(self, inputs):
        """
        Transform the raw inputs to the model inputs, two steps involved:
           1) Transform the raw text/image to token ids/pixel_values.
           2) Generate the other model inputs from the raw text/image and token ids/pixel_values.
        """
        inputs = self._check_input_text(inputs)
        batches = self._batchify(inputs, self._batch_size)
        outputs = {"batches": batches, "inputs": inputs}
        return outputs

    def _run_model(self, inputs, **kwargs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """
        pooling_mode = kwargs.get("pooling_mode", None)
        if pooling_mode is None:
            pooling_mode = self.pooling_mode
        all_feats = []
        if self._static_mode:
            with static_mode_guard():
                for batch_inputs in inputs["batches"]:
                    batch_inputs = self._collator(batch_inputs)
                    if self._predictor_type == "paddle-inference":
                        if "input_ids" in batch_inputs:
                            self.input_handles[0].copy_from_cpu(batch_inputs["input_ids"])
                            self.input_handles[1].copy_from_cpu(batch_inputs["token_type_ids"])
                            self.predictor.run()
                            token_embeddings = self.output_handle[0].copy_to_cpu()
                            if pooling_mode == "max_tokens":
                                attention_mask = (batch_inputs["input_ids"] != self.pad_token_id).astype(
                                    token_embeddings.dtype
                                )
                                input_mask_expanded = np.expand_dims(attention_mask, -1).repeat(
                                    token_embeddings.shape[-1], axis=-1
                                )
                                token_embeddings[input_mask_expanded == 0] = -1e9
                                max_over_time = np.max(token_embeddings, 1)
                                all_feats.append(max_over_time)
                            elif pooling_mode == "mean_tokens" or pooling_mode == "mean_sqrt_len_tokens":
                                attention_mask = (batch_inputs["input_ids"] != self.pad_token_id).astype(
                                    token_embeddings.dtype
                                )
                                input_mask_expanded = np.expand_dims(attention_mask, -1).repeat(
                                    token_embeddings.shape[-1], axis=-1
                                )
                                sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
                                sum_mask = input_mask_expanded.sum(1)
                                sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=np.max(sum_mask))
                                if pooling_mode == "mean_tokens":
                                    all_feats.append(sum_embeddings / sum_mask)
                                elif pooling_mode == "mean_sqrt_len_tokens":
                                    all_feats.append(sum_embeddings / np.sqrt(sum_mask))
                            else:
                                cls_token = token_embeddings[:, 0]
                                all_feats.append(cls_token)
                    else:
                        # onnx mode
                        if "input_ids" in batch_inputs:
                            input_dict = {}
                            input_dict["input_ids"] = batch_inputs["input_ids"]
                            input_dict["token_type_ids"] = batch_inputs["token_type_ids"]
                            token_embeddings = self.predictor.run(None, input_dict)[0]
                            if pooling_mode == "max_tokens":
                                attention_mask = (batch_inputs["input_ids"] != self.pad_token_id).astype(
                                    token_embeddings.dtype
                                )
                                input_mask_expanded = np.expand_dims(attention_mask, -1).repeat(
                                    token_embeddings.shape[-1], axis=-1
                                )
                                token_embeddings[input_mask_expanded == 0] = -1e9
                                max_over_time = np.max(token_embeddings, 1)
                                all_feats.append(max_over_time)
                            elif pooling_mode == "mean_tokens" or pooling_mode == "mean_sqrt_len_tokens":
                                attention_mask = (batch_inputs["input_ids"] != self.pad_token_id).astype(
                                    token_embeddings.dtype
                                )
                                input_mask_expanded = np.expand_dims(attention_mask, -1).repeat(
                                    token_embeddings.shape[-1], axis=-1
                                )
                                sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 1)
                                sum_mask = input_mask_expanded.sum(1)
                                sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=np.max(sum_mask))
                                if pooling_mode == "mean_tokens":
                                    all_feats.append(sum_embeddings / sum_mask)
                                elif pooling_mode == "mean_sqrt_len_tokens":
                                    all_feats.append(sum_embeddings / np.sqrt(sum_mask))
                            else:
                                cls_token = token_embeddings[:, 0]
                                all_feats.append(cls_token)
        else:
            with dygraph_mode_guard():
                for batch_inputs in inputs["batches"]:
                    batch_inputs = self._collator(batch_inputs)
                    token_embeddings = self._model(input_ids=batch_inputs["input_ids"])[0]
                    if pooling_mode == "max_tokens":
                        attention_mask = (batch_inputs["input_ids"] != self.pad_token_id).astype(
                            self._model.pooler.dense.weight.dtype
                        )
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.shape)
                        token_embeddings[input_mask_expanded == 0] = -1e9
                        max_over_time = paddle.max(token_embeddings, 1)
                        all_feats.append(max_over_time)

                    elif pooling_mode == "mean_tokens" or pooling_mode == "mean_sqrt_len_tokens":
                        attention_mask = (batch_inputs["input_ids"] != self.pad_token_id).astype(
                            self._model.pooler.dense.weight.dtype
                        )
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.shape)
                        sum_embeddings = paddle.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = input_mask_expanded.sum(1)
                        sum_mask = paddle.clip(sum_mask, min=1e-9)
                        if pooling_mode == "mean_tokens":
                            all_feats.append(sum_embeddings / sum_mask)
                        elif pooling_mode == "mean_sqrt_len_tokens":
                            all_feats.append(sum_embeddings / paddle.sqrt(sum_mask))
                    else:
                        cls_token = token_embeddings[:, 0]
                        all_feats.append(cls_token)
        inputs.update({"features": all_feats})
        return inputs

    def _postprocess(self, inputs):
        inputs["features"] = np.concatenate(inputs["features"], axis=0)
        inputs["features"] = [inputs["features"][idx] for idx in np.argsort(self.length_sorted_idx)]

        if self.return_tensors == "pd":
            inputs["features"] = paddle.to_tensor(inputs["features"])
        return inputs

    def _convert_dygraph_to_static(self):
        """
        Convert the dygraph model to static model.
        """
        assert (
            self._model is not None
        ), "The dygraph model must be created before converting the dygraph model to static model."
        assert (
            self._input_spec is not None
        ), "The input spec must be created before converting the dygraph model to static model."
        logger.info("Converting to the inference model cost a little time.")

        static_model = paddle.jit.to_static(self._model, input_spec=self._input_spec)
        paddle.jit.save(static_model, self.inference_model_path)
        logger.info("The inference model save in the path:{}".format(self.inference_model_path))
