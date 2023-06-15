# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from paddlenlp.transformers import AutoModel, AutoTokenizer

from ..data import Pad, Tuple
from ..transformers import ErnieCrossEncoder, ErnieTokenizer
from ..utils.log import logger
from .task import Task
from .utils import static_mode_guard

usage = r"""
         from paddlenlp import Taskflow

         similarity = Taskflow("text_similarity")
         similarity([["世界上什么东西最小", "世界上什么东西最小？"]])
         '''
         [{'text1': '世界上什么东西最小', 'text2': '世界上什么东西最小？', 'similarity': 0.992725}]
         '''

         similarity = Taskflow("text_similarity", batch_size=2)
         similarity([["光眼睛大就好看吗", "眼睛好看吗？"], ["小蝌蚪找妈妈怎么样", "小蝌蚪找妈妈是谁画的"]])
         '''
         [{'text1': '光眼睛大就好看吗', 'text2': '眼睛好看吗？', 'similarity': 0.74502707}, {'text1': '小蝌蚪找妈妈怎么样', 'text2': '小蝌蚪找妈妈是谁画的', 'similarity': 0.8192149}]
         '''
         """
MATCH_TYPE = {
    "rocketqa-zh-dureader-cross-encoder": "matching",
    "rocketqa-base-cross-encoder": "matching",
    "rocketqa-medium-cross-encoder": "matching",
    "rocketqa-mini-cross-encoder": "matching",
    "rocketqa-micro-cross-encoder": "matching",
    "rocketqa-nano-cross-encoder": "matching",
    "rocketqav2-en-marco-cross-encoder": "matching_v2",
    "ernie-search-large-cross-encoder-marco-en": "matching_v3",
}


class TextSimilarityTask(Task):
    """
    Text similarity task using SimBERT to predict the similarity of sentence pair.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json",
    }
    resource_files_urls = {
        "simbert-base-chinese": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/text_similarity/simbert-base-chinese/model_state.pdparams",
                "27d9ef240c2e8e736bdfefea52af2542",
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/taskflow/text_similarity/simbert-base-chinese/model_config.json",
                "1254bbd7598457a9dad0afcb2e24b70c",
            ],
        },
        "rocketqa-zh-dureader-cross-encoder": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-zh-dureader-cross-encoder/model_state.pdparams",
                "88bc3e1a64992a1bdfe4044ecba13bc7",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-zh-dureader-cross-encoder/model_config.json",
                "b69083c2895e8f68e1a10467b384daab",
            ],
        },
        "rocketqa-base-cross-encoder": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-base-cross-encoder/model_state.pdparams",
                "6d845a492a2695e62f2be79f8017be92",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-base-cross-encoder/model_config.json",
                "18ce260ede18bc3cb28dcb2e7df23b1a",
            ],
        },
        "rocketqa-medium-cross-encoder": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-medium-cross-encoder/model_state.pdparams",
                "4b929f4fc11a1df8f59fdf2784e23fa7",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-medium-cross-encoder/model_config.json",
                "10997db96bc86e29cd113e1bf58989d7",
            ],
        },
        "rocketqa-mini-cross-encoder": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-mini-cross-encoder/model_state.pdparams",
                "c411111df990132fb88c070d8b8cf3f7",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-mini-cross-encoder/model_config.json",
                "271e6d779acbe8e8acdd596b1c835546",
            ],
        },
        "rocketqa-micro-cross-encoder": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-micro-cross-encoder/model_state.pdparams",
                "3d643ff7d6029c8ceab5653680167dc0",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-micro-cross-encoder/model_config.json",
                "b32d1a932d8c367fab2a6216459dd0a7",
            ],
        },
        "rocketqa-nano-cross-encoder": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-nano-cross-encoder/model_state.pdparams",
                "4c1d36e5e94f5af09f665fc7ad0be140",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-nano-cross-encoder/model_config.json",
                "dcff14cd671e1064be2c5d63734098bb",
            ],
        },
        "rocketqav2-en-marco-cross-encoder": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqav2-en-marco-cross-encoder/model_state.pdparams",
                "a5afc77b6a63fc32a1beca3010f40f32",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqav2-en-marco-cross-encoder/config.json",
                "8f5d5c71c8a891b68d0402a13e38b6f9",
            ],
        },
        "ernie-search-large-cross-encoder-marco-en": {
            "model_state": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/ernie-search-large-cross-encoder-marco-en/model_state.pdparams",
                "fdf29f7de0f7fe570740d343c96165e5",
            ],
            "model_config": [
                "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/ernie-search-large-cross-encoder-marco-en/config.json",
                "28bad2c7b36fa148fa75a8dc5b690485",
            ],
        },
        "__internal_testing__/tiny-random-bert": {
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/models/community/__internal_testing__/tiny-random-bert/model_state.pdparams",
                "8d8814d589c21bf083fdb35de6c11a57",
            ],
            "model_config": [
                "https://bj.bcebos.com/paddlenlp/models/community/__internal_testing__/tiny-random-bert/config.json",
                "37e28e2359f330f64fc82beff1967a1e",
            ],
        },
    }

    def __init__(self, task, model, batch_size=1, max_length=384, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = True
        self._check_predictor_type()
        if not self.from_hf_hub:
            self._check_task_files()
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)
        self._construct_tokenizer(model)
        self._batch_size = batch_size
        self._max_length = max_length
        self._usage = usage
        self.model_name = model

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

        if "rocketqav2-en" in model or "ernie-search" in model:
            self._model = ErnieCrossEncoder(self._task_path, num_classes=1, reinitialize=True)
        elif "rocketqa" in model:
            self._model = ErnieCrossEncoder(self._task_path, num_classes=2)
        else:
            self._model = AutoModel.from_pretrained(self._task_path, pool_act="linear")
        self._model.eval()

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        if "rocketqa" in model or "ernie-search" in model:
            self._tokenizer = ErnieTokenizer.from_pretrained(model)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model)

    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if not all([isinstance(i, list) and i and all(i) and len(i) == 2 for i in inputs]):
            raise TypeError("Invalid input format.")
        return inputs

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)

        examples = []
        for data in inputs:
            text1, text2 = data[0], data[1]
            if "rocketqa" in self.model_name or "ernie-search" in self.model_name:
                # Todo: wugaosheng, Add erine-search encoding support
                encoded_inputs = self._tokenizer(text=text1, text_pair=text2, max_length=self._max_length)
                ids = encoded_inputs["input_ids"]
                segment_ids = encoded_inputs["token_type_ids"]
                examples.append((ids, segment_ids))
            else:
                text1_encoded_inputs = self._tokenizer(text=text1, max_length=self._max_length)
                text1_input_ids = text1_encoded_inputs["input_ids"]
                text1_token_type_ids = text1_encoded_inputs["token_type_ids"]

                text2_encoded_inputs = self._tokenizer(text=text2, max_length=self._max_length)
                text2_input_ids = text2_encoded_inputs["input_ids"]
                text2_token_type_ids = text2_encoded_inputs["token_type_ids"]

                examples.append((text1_input_ids, text1_token_type_ids, text2_input_ids, text2_token_type_ids))

        batches = [examples[idx : idx + self._batch_size] for idx in range(0, len(examples), self._batch_size)]
        if "rocketqa" in self.model_name or "ernie-search" in self.model_name:
            batchify_fn = lambda samples, fn=Tuple(  # noqa: E731
                Pad(axis=0, pad_val=self._tokenizer.pad_token_id, dtype="int64"),  # input ids
                Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id, dtype="int64"),  # token type ids
            ): [data for data in fn(samples)]
        else:
            batchify_fn = lambda samples, fn=Tuple(  # noqa: E731
                Pad(axis=0, pad_val=self._tokenizer.pad_token_id, dtype="int64"),  # text1_input_ids
                Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id, dtype="int64"),  # text1_token_type_ids
                Pad(axis=0, pad_val=self._tokenizer.pad_token_id, dtype="int64"),  # text2_input_ids
                Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id, dtype="int64"),  # text2_token_type_ids
            ): [data for data in fn(samples)]

        outputs = {}
        outputs["data_loader"] = batches
        outputs["text"] = inputs
        self._batchify_fn = batchify_fn
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function.
        """
        results = []
        if "rocketqa" in self.model_name or "ernie-search" in self.model_name:
            with static_mode_guard():
                for batch in inputs["data_loader"]:

                    if self._predictor_type == "paddle-inference":
                        input_ids, segment_ids = self._batchify_fn(batch)
                        self.input_handles[0].copy_from_cpu(input_ids)
                        self.input_handles[1].copy_from_cpu(segment_ids)
                        self.predictor.run()
                        scores = self.output_handle[0].copy_to_cpu().tolist()
                        results.extend(scores)
                    else:
                        # onnx mode
                        input_dict = {}
                        input_ids, segment_ids = self._batchify_fn(batch)
                        input_dict["input_ids"] = input_ids
                        input_dict["token_type_ids"] = segment_ids
                        scores = self.predictor.run(None, input_dict)[0].tolist()
                        results.extend(scores)
        else:
            with static_mode_guard():
                for batch in inputs["data_loader"]:
                    text1_ids, text1_segment_ids, text2_ids, text2_segment_ids = self._batchify_fn(batch)
                    self.input_handles[0].copy_from_cpu(text1_ids)
                    self.input_handles[1].copy_from_cpu(text1_segment_ids)
                    self.predictor.run()
                    vecs_text1 = self.output_handle[1].copy_to_cpu()

                    self.input_handles[0].copy_from_cpu(text2_ids)
                    self.input_handles[1].copy_from_cpu(text2_segment_ids)
                    self.predictor.run()
                    vecs_text2 = self.output_handle[1].copy_to_cpu()

                    vecs_text1 = vecs_text1 / (vecs_text1**2).sum(axis=1, keepdims=True) ** 0.5
                    vecs_text2 = vecs_text2 / (vecs_text2**2).sum(axis=1, keepdims=True) ** 0.5
                    similarity = (vecs_text1 * vecs_text2).sum(axis=1)
                    results.extend(similarity)
        inputs["result"] = results
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        final_results = []
        for text, similarity in zip(inputs["text"], inputs["result"]):
            result = {}
            result["text1"] = text[0]
            result["text2"] = text[1]
            # The numpy.float32 can not be converted to the json format
            if isinstance(similarity, list):
                result["similarity"] = float(similarity[0])
            else:
                result["similarity"] = float(similarity)
            final_results.append(result)
        return final_results

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
        if self.model in MATCH_TYPE:
            if MATCH_TYPE[self.model] == "matching":
                static_model = paddle.jit.to_static(self._model.matching, input_spec=self._input_spec)
            elif MATCH_TYPE[self.model] == "matching_v2":
                static_model = paddle.jit.to_static(self._model.matching_v2, input_spec=self._input_spec)
            elif MATCH_TYPE[self.model] == "matching_v3":
                static_model = paddle.jit.to_static(self._model.matching_v3, input_spec=self._input_spec)
        else:
            static_model = paddle.jit.to_static(self._model, input_spec=self._input_spec)

        paddle.jit.save(static_model, self.inference_model_path)
        logger.info("The inference model save in the path:{}".format(self.inference_model_path))
