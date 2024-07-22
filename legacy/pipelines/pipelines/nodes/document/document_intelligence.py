# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import collections
import logging
import math
import os
from multiprocessing import cpu_count
from typing import List

import paddle
from pipelines.nodes.base import BaseComponent

from paddlenlp.taskflow.utils import (
    ImageReader,
    download_file,
    find_answer_pos,
    get_doc_pred,
    sort_res,
)
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.env import PPNLP_HOME

logger = logging.getLogger(__name__)

URLS = {
    "docprompt": [
        "https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/docprompt/docprompt_params.tar",
        "8eae8148981731f230b328076c5a08bf",
    ],
}


class DocPrompter(BaseComponent):
    """
    DocPrompter: extract prompt's answers from the document input.
    """

    return_no_answers: bool
    outgoing_edges = 1
    query_count = 0
    query_time = 0

    def __init__(
        self,
        topn: int = 1,
        use_gpu: bool = True,
        task_path: str = None,
        model: str = "docprompt",
        device_id: int = 0,
        num_threads: int = None,
        lang: str = "ch",
        batch_size: int = 1,
    ):
        """
        Init Document Prompter.
        :param topn: return top n answers.
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param task_path: Custom model path if using custom model parameters.
        :param model: Choose model name.
        :param device_id: Choose gpu device id.
        :param num_threads: Number of processing threads.
        :param lang: Choose langugae.
        :param batch_size: Number of samples the model receives in one batch for inference.
                           Memory consumption is much lower in inference mode. Recommendation: Increase the batch size
                           to a value so only a single batch is used.
        """
        self._use_gpu = False if paddle.get_device() == "cpu" else use_gpu
        self.model = model
        self._device_id = device_id
        self._num_threads = num_threads if num_threads else math.ceil(cpu_count() / 2)
        self._topn = topn
        self._lang = lang
        self._batch_size = batch_size
        if task_path is None:
            self._task_path = os.path.join(PPNLP_HOME, "pipelines", "document_intelligence", self.model)
        else:
            self._task_path = task_path

        download_file(self._task_path, "docprompt_params.tar", URLS[self.model][0], URLS[self.model][1])
        self._get_inference_model()
        self._tokenizer = AutoTokenizer.from_pretrained("ernie-layoutx-base-uncased")
        self._reader = ImageReader(super_rel_pos=False, tokenizer=self._tokenizer)

    def _get_inference_model(self):
        inference_model_path = os.path.join(self._task_path, "static", "inference")
        self._static_model_file = inference_model_path + ".pdmodel"
        self._static_params_file = inference_model_path + ".pdiparams"
        self._config = paddle.inference.Config(self._static_model_file, self._static_params_file)
        self._prepare_static_mode()

    def _prepare_static_mode(self):
        """
        Construct the input data and predictor in the PaddlePaddele static mode.
        """
        if paddle.get_device() == "cpu":
            self._config.disable_gpu()
            self._config.enable_mkldnn()
        else:
            self._config.enable_use_gpu(100, self._device_id)
            self._config.delete_pass("embedding_eltwise_layernorm_fuse_pass")
        self._config.set_cpu_math_library_num_threads(self._num_threads)
        self._config.switch_use_feed_fetch_ops(False)
        self._config.disable_glog_info()
        self._config.enable_memory_optim()
        self._config.switch_ir_optim(False)
        self.predictor = paddle.inference.create_predictor(self._config)
        self.input_names = [name for name in self.predictor.get_input_names()]
        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]
        self.output_handle = [self.predictor.get_output_handle(name) for name in self.predictor.get_output_names()]

    def _run_model(self, inputs: List[dict]):
        """
        Run docprompt model.
        """
        all_predictions_list = []
        for example in inputs:
            ocr_result = example["ocr_result"]
            doc_path = example["doc"]
            prompt = example["prompt"]
            ocr_type = example["ocr_type"]

            if not ocr_result:
                all_predictions = [
                    {"prompt": p, "result": [{"value": "", "prob": 0.0, "start": -1, "end": -1}]} for p in prompt
                ]
                all_boxes = {}
            else:
                data_loader = self._reader.data_generator(ocr_result, doc_path, prompt, self._batch_size, ocr_type)

                RawResult = collections.namedtuple("RawResult", ["unique_id", "seq_logits"])

                all_results = []
                for data in data_loader:
                    for idx in range(len(self.input_names)):
                        self.input_handles[idx].copy_from_cpu(data[idx])
                    self.predictor.run()
                    outputs = [output_handle.copy_to_cpu() for output_handle in self.output_handle]
                    unique_ids, seq_logits = outputs

                    for idx in range(len(unique_ids)):
                        all_results.append(
                            RawResult(
                                unique_id=int(unique_ids[idx]),
                                seq_logits=seq_logits[idx],
                            )
                        )

                all_examples = self._reader.examples["infer"]
                all_features = self._reader.features["infer"]
                all_key_probs = [1 for _ in all_examples]

                example_index_to_features = collections.defaultdict(list)

                for feature in all_features:
                    example_index_to_features[feature.qas_id].append(feature)

                unique_id_to_result = {}
                for result in all_results:
                    unique_id_to_result[result.unique_id] = result

                all_predictions = []
                all_boxes = {}
                for (example_index, example) in enumerate(all_examples):
                    example_doc_tokens = example.doc_tokens
                    example_qas_id = example.qas_id
                    page_id = example_qas_id.split("_")[0]
                    if page_id not in all_boxes:
                        all_boxes[page_id] = example.ori_boxes
                    example_query = example.keys[0]
                    features = example_index_to_features[example_qas_id]

                    preds = []
                    # keep track of the minimum score of null start+end of position 0
                    for feature in features:
                        if feature.unique_id not in unique_id_to_result:
                            continue
                        result = unique_id_to_result[feature.unique_id]

                        # find preds
                        ans_pos = find_answer_pos(result.seq_logits, feature)
                        preds.extend(
                            get_doc_pred(
                                result, ans_pos, example, self._tokenizer, feature, True, all_key_probs, example_index
                            )
                        )

                    if not preds:
                        preds.append({"value": "", "prob": 0.0, "start": -1, "end": -1})
                    else:
                        preds = sort_res(example_query, preds, example_doc_tokens, all_boxes[page_id], self._lang)[
                            : self._topn
                        ]
                    all_predictions.append({"prompt": example_query, "result": preds})
            all_predictions_list.append(all_predictions)
        return all_predictions_list

    def run(self, example: dict):
        results = self._run_model([example])
        output = {"results": results}
        return output, "output_1"
