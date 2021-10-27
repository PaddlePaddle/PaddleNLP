# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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

import json
from collections import OrderedDict
from typing import Optional, List

import numpy as np
import paddle

from ..datasets import MapDataset, load_dataset
from ..data import Stack, Pad, Tuple
from ..transformers import ErnieCtmNptagModel, ErnieCtmTokenizer
from .utils import download_file, add_docstrings, static_mode_guard, dygraph_mode_guard
from .utils import BurkhardKellerTree
from .task import Task


URLS = {
    "name_category_map.json":[
        "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_ctm/name_category_map.json",
        "40e8adaaadb567e90978c5389d595b48",
    ],
}


usage = r"""
           from paddlenlp import Taskflow 

           nptag = Taskflow("noun_phrase_tagging")
           nptag("糖醋排骨")
           '''
           [{'text': '糖醋排骨', 'cls_label': '菜品'}]
           '''
           nptag(["糖醋排骨", "红曲霉菌"])
           '''
           [{'text': '糖醋排骨', 'cls_label': '菜品'}, {'text': '红曲霉菌', 'cls_label': '微生物'}]
           '''

           nptag = Taskflow("noun_phrase_tagging", linking=True)
           nptag(["糖醋排骨", "红曲霉菌"])
           '''
           [{'text': '糖醋排骨', 'cls_label': '菜品', 'category': '饮食类_菜品'}, {'text': '红曲霉菌', 'cls_label': '微生物', 'category': '生物类_微生物'}]
           '''
         """


@add_docstrings(usage)
class NPTagTask(Task):
    """
    Noun phrase tagging task that convert the noun phrase to POS tag.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        batch_size(int): Numbers of examples a batch.
        linking(bool): Returns the categories. The fine-grained labels (cls_label) will link with the coarse-grained labels (category).
    """
    def __init__(self,
                 task,
                 model,
                 batch_size=1,
                 linking=False,
                 **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._usage = usage
        self._static_mode = True
        self._batch_size = batch_size
        self._linking = linking
        self._construct_tokenizer(model)
        self._name_dict = None
        name_dict_path = download_file(self._task_path,
                                    "name_category_map.json",
                                    URLS["name_category_map.json"][0],
                                    URLS["name_category_map.json"][1])
        self._construct_dict_map(name_dict_path)
        if self._static_mode:
            self._get_inference_model()
        else:
            self._construct_model(model)

    def _construct_dict_map(self, name_dict_path):
        """
        Construct dict map for the predictor.
        """
        with open(name_dict_path, encoding="utf-8") as fp:
            self._name_dict = json.load(fp)
        self._tree = BurkhardKellerTree()
        for k in self._name_dict:
            self._tree.add(k)
        self._cls_vocabs = OrderedDict()
        for k in self._name_dict:
            for c in k:
                if c not in self._cls_vocabs:
                    self._cls_vocabs[c] = len(self._cls_vocabs)
        self._cls_vocabs["[PAD]"] = len(self._cls_vocabs)
        self._id_vocabs = dict(zip(self._cls_vocabs.values(), self._cls_vocabs.keys()))
        self._vocab_ids = self._tokenizer.vocab.to_indices(list(self._cls_vocabs.keys()))

    def _decode(self, pred_ids):
        tokens = [self._id_vocabs[i] for i in pred_ids]
        valid_token = []
        for token in tokens:
            if token == "[PAD]":
                break
            valid_token.append(token)
        return "".join(valid_token)

    def _search(self, scores_can, pred_ids_can, depth, path, score):
        if depth >= 5:
            return [(path, score)]
        res = []
        for i in range(len(pred_ids_can[0])):
            tmp_res = self._search(
                scores_can, pred_ids_can, depth + 1, path + [pred_ids_can[depth][i]],
                score + scores_can[depth][i]
            )
            res.extend(tmp_res)
        return res

    def _find_topk(self, a, k, axis=-1, largest=True, sorted=True):
        if axis is None:
            axis_size = a.size
        else:
            axis_size = a.shape[axis]
        assert 1 <= k <= axis_size

        a = np.asanyarray(a)
        if largest:
            index_array = np.argpartition(a, axis_size-k, axis=axis)
            topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
        else:
            index_array = np.argpartition(a, k-1, axis=axis)
            topk_indices = np.take(index_array, np.arange(k), axis=axis)
        topk_values = np.take_along_axis(a, topk_indices, axis=axis)
        if sorted:
            sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
            if largest:
                sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
            sorted_topk_values = np.take_along_axis(
                topk_values, sorted_indices_in_topk, axis=axis)
            sorted_topk_indices = np.take_along_axis(
                topk_indices, sorted_indices_in_topk, axis=axis)
            return sorted_topk_values, sorted_topk_indices
        return topk_values, topk_indices

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64",
                name="input_ids"), # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64",
                name="token_type_ids"), # segment_ids
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        model_instance = ErnieCtmNptagModel.from_pretrained(model)
        model_instance.eval()
        self._model = model_instance

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        tokenizer_instance = ErnieCtmTokenizer.from_pretrained(model)
        self._tokenizer = tokenizer_instance

    def _preprocess(self, inputs):
        """
        Create the dataset and dataloader for the predict.
        """
        inputs = self._check_input_text(inputs)
        self._cls_seq_length = 5
        num_workers = self.kwargs[
            'num_workers'] if 'num_workers' in self.kwargs else 0
        lazy_load = self.kwargs[
            'lazy_load'] if 'lazy_load' in self.kwargs else False

        # Prompt template: input_text + "是" + "[MASK]" * cls_seq_length
        prompt_template = ["是"] + ["[MASK]"] * self._cls_seq_length
        prompt_template_ids = self._tokenizer.vocab.to_indices(prompt_template)

        def read(inputs):
            for text in inputs:
                input_id = self._tokenizer.vocab.to_indices(list(text))
                input_id += prompt_template_ids
                input_id = self._tokenizer.build_inputs_with_special_tokens(input_id)
                token_type_id = [self._tokenizer.pad_token_type_id] * len(input_id)
                yield input_id, token_type_id

        infer_ds = load_dataset(read, inputs=inputs, lazy=lazy_load)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id), # input_ids
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id) # token_type_ids
        ): fn(samples)

        infer_data_loader = paddle.io.DataLoader(
            infer_ds,
            collate_fn=batchify_fn,
            num_workers=num_workers,
            batch_size=self._batch_size,
            shuffle=False,
            return_list=True)
        
        outputs = {}
        outputs['data_loader'] = infer_data_loader
        outputs['texts'] = inputs
        return outputs

    def _run_model(self, inputs):
        all_scores_can = []
        all_preds_can = []
        pred_ids = []

        for batch in inputs['data_loader']:
            input_ids, token_type_ids = batch
            self.input_handles[0].copy_from_cpu(input_ids.numpy())
            self.input_handles[1].copy_from_cpu(token_type_ids.numpy())
            self.predictor.run()
            logits = self.output_handle[0].copy_to_cpu()
            logits = logits.squeeze()[-(self._cls_seq_length + 1): -1, self._vocab_ids]
            # Find topk candidates of scores and predicted indices.
            scores_can, pred_ids_can = self._find_topk(logits, k=4, axis=-1)
            all_scores_can.extend([scores_can.tolist()])
            all_preds_can.extend([pred_ids_can.tolist()])
            pred_ids.extend([pred_ids_can[:, 0].tolist()])                

        inputs['all_scores_can'] = all_scores_can
        inputs['all_preds_can'] = all_preds_can
        inputs['pred_ids'] = pred_ids
        return inputs
    
    def _postprocess(self, inputs):
        results = []

        for i in range(len(inputs['texts'])):
            cls_label = self._decode(inputs['pred_ids'][i])

            result = {
                'text': inputs['texts'][i],
                'cls_label': cls_label,
            }

            if cls_label not in self._name_dict:
                scores_can = inputs['all_scores_can'][i]
                pred_ids_can = inputs['all_preds_can'][i]
                labels_can = self._search(scores_can, pred_ids_can, 0, [], 0)
                labels_can.sort(key=lambda d: -d[1])
                for labels in labels_can:
                    cls_label_can = self._decode(labels[0])
                    if cls_label_can in self._name_dict:
                        result['cls_label'] = cls_label_can
                        break
                    else:
                        labels_can = self._tree.search_similar_word(cls_label)
                        result['cls_label'] = labels_can[0][0]

            if self._linking:
                result['category'] = self._name_dict[result['cls_label']]
            results.append(result)
        return results