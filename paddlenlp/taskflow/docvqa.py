# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import os
import collections

from ..transformers import AutoTokenizer
from .utils import download_file, ExtractReader, get_dvqa_pred, find_answer_pos
from .task import Task

usage = r"""
            from paddlenlp import Taskflow
            docvqa = Taskflow("document_intelligence")
            # Types of image: A string containing a local path to an image
            docvqa([{"image": "./invoice.jpg", "question": ["发票号码是多少?", "校验码是多少?"]}])
            # Types of image: A string containing a http link pointing to an image
            docvqa({"image": "https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/invoice.jpg", "question": ["发票号码是多少?", "校验码是多少?"]})
            '''
            [{'question': '发票号码是多少?', 'answer': [{'value': 'No44527206', 'prob': 0.96}]}, {'question': '校验码的后六位是多少?', 'answer': [{'value': '01107 555427109891646', 'prob': 0.99}]}]
            '''
         """

URLS = {
    "ernie-layoutx-large-pruned": [
        "https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/ernie-layoutx/ernie-layoutx-large-pruned_params.tar",
        "fe72df2168caa83815bd8939155105b7"
    ],
}


class DocVQATask(Task):
    """
    The document intelligence model, give the querys and predict the answers. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._batch_size = kwargs.get("batch_size", 1)
        self._topn = kwargs.get("topn", 1)
        self._use_gpu = kwargs.get("use_gpu", False)

        try:
            from paddleocr import PaddleOCR
        except:
            raise ImportError(
                "Please install the dependencies first, pip install paddleocr --upgrade"
            )

        self._ocr = PaddleOCR(use_angle_cls=True,
                              show_log=False,
                              use_gpu=self._use_gpu)
        self._usage = usage
        download_file(self._task_path, "ernie-layoutx-large-pruned_params.tar",
                      URLS[self.model][0], URLS[self.model][1])
        self._get_inference_model()
        self._construct_tokenizer()
        self._reader = ExtractReader(super_rel_pos=False,
                                     tokenizer=self._tokenizer,
                                     random_seed=1)

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(
            "ernie-layoutx-base-uncased")

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)

        img_path = inputs["image"]
        if not os.path.exists(img_path):
            download_file("./", img_path.rsplit("/", 1)[-1], img_path)
            img_path = img_path.rsplit("/", 1)[-1]
        question = inputs["question"]

        ocr_result = self._ocr.ocr(img_path, cls=True)
        data_loader = self._reader.data_generator(ocr_result, img_path,
                                                  question, self._batch_size)

        infer_cache = {}
        infer_cache["data_loader"] = data_loader
        return infer_cache

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """
        infer_cache = inputs
        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "seq_logits"])

        all_results = []
        for data in infer_cache['data_loader']:
            for idx in range(len(self.input_names)):
                self.input_handles[idx].copy_from_cpu(data[idx])
            self.predictor.run()
            outputs = [
                output_handle.copy_to_cpu()
                for output_handle in self.output_handle
            ]
            unique_ids, seq_logits = outputs

            for idx in range(len(unique_ids)):
                all_results.append(
                    RawResult(
                        unique_id=int(unique_ids[idx]),
                        seq_logits=seq_logits[idx],
                    ))
        infer_cache['results'] = all_results
        return infer_cache

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        all_results = inputs['results']
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

        for (example_index, example) in enumerate(all_examples):
            example_qas_id = example.qas_id
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
                    get_dvqa_pred(result, ans_pos, example, self._tokenizer,
                                  feature, True, all_key_probs, example_index))

            if not preds:
                preds.append({'value': '', 'prob': 0., 'start': -1, 'end': -1})
            else:
                preds = sorted(preds,
                               key=lambda x: x["prob"])[::-1][:self._topn]
            all_predictions.append({"question": example_query, "answer": preds})
        return all_predictions

    def _check_input_text(self, inputs):
        inputs = inputs[0]
        if isinstance(inputs, dict):
            if "image" not in inputs.keys():
                raise TypeError(
                    "Invalid inputs, the inputs should contain the image file path or image url."
                    .format(type(inputs[0])))
        else:
            raise TypeError(
                "Invalid inputs, input for document question answering should be dict, but type of {} found!"
                .format(type(inputs)))
        return inputs

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        pass

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        pass
