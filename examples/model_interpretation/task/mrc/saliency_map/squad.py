#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# !/usr/bin/env python3
import collections
import json
import os
import numpy as np

from paddlenlp.datasets import DatasetBuilder


class Similarity(DatasetBuilder):
    # similarity test 21.10.3
    def _read(self, filename):
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line_split = line.strip().split('\t')
                assert len(line_split) == 3
                yield {
                    'text_a': line_split[0],
                    'text_b': line_split[1],
                    'label': line_split[2]
                }


class RCInterpret(DatasetBuilder):
    # interpret 21.9.24
    def _read(self, filename):
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                example_dic = json.loads(line)
                id = example_dic['id']
                title = example_dic['title']
                context = example_dic['context']
                question = example_dic['question']
                if 'sent_token' in example_dic:
                    sent_token = example_dic['sent_token']
                    yield {
                        'id': id,
                        'title': title,
                        'context': context,
                        'question': question,
                        'sent_token': sent_token
                    }
                else:
                    yield {
                        'id': id,
                        'title': title,
                        'context': context,
                        'question': question
                    }


class DuReaderChecklist(DatasetBuilder):

    def _read(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            input_data = json.load(f)["data"]

        for entry in input_data:
            # title = entry.get("title", "").strip()
            for paragraph in entry["paragraphs"]:
                context = paragraph["context"].strip()
                title = paragraph.get("title", "").strip()
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question = qa["question"].strip()
                    answer_starts = []
                    answers = []
                    is_impossible = False

                    if "is_impossible" in qa.keys():
                        is_impossible = qa["is_impossible"]

                    answer_starts = [
                        answer["answer_start"]
                        for answer in qa.get("answers", [])
                    ]
                    answers = [
                        answer["text"].strip()
                        for answer in qa.get("answers", [])
                    ]

                    yield {
                        'id': qas_id,
                        'title': title,
                        'context': context,
                        'question': question,
                        'answers': answers,
                        'answer_starts': answer_starts,
                        'is_impossible': is_impossible
                    }


def compute_prediction_checklist(examples,
                                 features,
                                 predictions,
                                 version_2_with_negative: bool = False,
                                 n_best_size: int = 20,
                                 max_answer_length: int = 30,
                                 cls_threshold: float = 0.5):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).

            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
    """

    assert len(
        predictions
    ) == 3, "`predictions` should be a tuple with two elements (start_logits, end_logits, cls_logits)."
    all_start_logits, all_end_logits, all_cls_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), "Number of predictions should be equal to number of features."  # 样本数

    # Build a map example to its corresponding features.
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(
            i
        )  # feature: dict(keys: 'input_ids', 'token_type_ids', 'offset_mapping', 'overflow_to_sample', 'example_id')

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_feature_index = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    all_cls_predictions = []

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example['id']]

        # if len(feature_indices) > 1:
        #     print('example_index: %s' % example_index)

        min_null_prediction = None
        prelim_predictions = []
        score_answerable = -1
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            cls_logits = all_cls_logits[feature_index]
            input_ids = features[feature_index]['input_ids']
            # This is what will allow us to map some the positions in our logits to span of texts in the original context.
            offset_mapping = features[feature_index][
                "offset_mapping"]  # list[tuple(2)], list长度与input_ids, start_logits, end_logits相同

            # if len(feature_indices) > 1:
            #     print('offset_mapping: %s' % offset_mapping)

            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None)

            exp_answerable_scores = np.exp(cls_logits - np.max(cls_logits))
            feature_answerable_score = exp_answerable_scores / exp_answerable_scores.sum(
            )
            if feature_answerable_score[-1] > score_answerable:
                score_answerable = feature_answerable_score[-1]
                answerable_probs = feature_answerable_score

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction[
                    "score"] > feature_null_score:
                min_null_prediction = {
                    "feature_index": (0, 0),
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(
                start_logits)[-1:-n_best_size -
                              1:-1].tolist()  # list(n_best_size) 从大到小
            end_indexes = np.argsort(
                end_logits)[-1:-n_best_size -
                            1:-1].tolist()  # list(n_best_size) 从大到小
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or  # CLS、Question和第一个SEP的位置
                            offset_mapping[end_index] is None or
                            offset_mapping[start_index] == (0,
                                                            0) or  # 第二个SEP的位置
                            offset_mapping[end_index] == (0, 0)):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(
                            str(start_index), False):
                        continue
                    prelim_predictions.append({
                        "feature_index": (start_index, end_index),
                        "offsets": (offset_mapping[start_index][0],
                                    offset_mapping[end_index][1]),
                        "score":
                        start_logits[start_index] + end_logits[end_index],
                        "start_logit":
                        start_logits[start_index],
                        "end_logit":
                        end_logits[end_index],
                    })
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            pred_cls_label = np.argmax(np.array(answerable_probs))
            all_cls_predictions.append([
                example['id'], pred_cls_label, answerable_probs[0],
                answerable_probs[1]
            ])


# Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions,
                             key=lambda x: x["score"],
                             reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0)
                                               for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            # offsets = pred.pop("offsets")
            offsets = pred["offsets"]
            pred["text"] = context[offsets[0]:offsets[1]] if context[
                offsets[0]:offsets[1]] != "" else "no answer"

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1
                                     and predictions[0]["text"] == "no answer"):
            predictions.insert(
                0, {
                    "feature_index": (0, 0),
                    "offsets": (0, 0),
                    "text": "no answer",
                    "start_logit": 0.0,
                    "end_logit": 0.0,
                    "score": 0.0
                })

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
            all_feature_index[example["id"]] = predictions[0]['feature_index']
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "no answer":
                i += 1
            best_non_null_pred = predictions[i]

            if answerable_probs[1] < cls_threshold:
                all_predictions[example['id']] = "no answer"
            else:
                all_predictions[example['id']] = best_non_null_pred['text']
            all_feature_index[example["id"]] = predictions[i]['feature_index']

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [{
            k: (float(v) if isinstance(v, (np.float16, np.float32,
                                           np.float64)) else v)
            for k, v in pred.items()
        } for pred in predictions]

    return all_predictions, all_nbest_json, all_cls_predictions, all_feature_index


def compute_prediction(examples,
                       features,
                       predictions,
                       version_2_with_negative=False,
                       n_best_size=20,
                       max_answer_length=30,
                       null_score_diff_threshold=0.0):
    """
    Post-processes the predictions of a question-answering model to convert 
    them to answers that are substrings of the original contexts. This is 
    the base postprocessing functions for models that only return start and 
    end logits.

    Args:
        examples (list): List of raw squad-style data (see `run_squad.py 
            <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/
            machine_reading_comprehension/SQuAD/run_squad.py>`__ for more 
            information).
        features (list): List of processed squad-style features (see 
            `run_squad.py <https://github.com/PaddlePaddle/PaddleNLP/blob/
            develop/examples/machine_reading_comprehension/SQuAD/run_squad.py>`__
            for more information).
        predictions (tuple): The predictions of the model. Should be a tuple
            of two list containing the start logits and the end logits.
        version_2_with_negative (bool, optional): Whether the dataset contains
            examples with no answers. Defaults to False.
        n_best_size (int, optional): The total number of candidate predictions
            to generate. Defaults to 20.
        max_answer_length (int, optional): The maximum length of predicted answer.
            Defaults to 20.
        null_score_diff_threshold (float, optional): The threshold used to select
            the null answer. Only useful when `version_2_with_negative` is True.
            Defaults to 0.0.
    
    Returns:
        A tuple of three dictionaries containing final selected answer, all n_best 
        answers along with their probability and scores, and the score_diff of each 
        example.
    """
    assert len(
        predictions
    ) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), "Number of predictions should be equal to number of features."

    # Build a map example to its corresponding features.
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    all_feature_index = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example['id']]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction[
                    "score"] > feature_null_score:
                min_null_prediction = {
                    "feature_index": (0, 0),
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1:-n_best_size -
                                                     1:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size -
                                                 1:-1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                            or offset_mapping[start_index] == (0, 0)
                            or offset_mapping[end_index] == (0, 0)):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(
                            str(start_index), False):
                        continue
                    prelim_predictions.append({
                        "feature_index": (start_index, end_index),
                        "offsets": (offset_mapping[start_index][0],
                                    offset_mapping[end_index][1]),
                        "score":
                        start_logits[start_index] + end_logits[end_index],
                        "start_logit":
                        start_logits[start_index],
                        "end_logit":
                        end_logits[end_index],
                    })
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions,
                             key=lambda x: x["score"],
                             reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0)
                                               for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]:offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1
                                     and predictions[0]["text"] == ""):
            predictions.insert(
                0, {
                    "feature_index": (0, 0),
                    "text": "empty",
                    "start_logit": 0.0,
                    "end_logit": 0.0,
                    "score": 0.0
                })

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
            all_feature_index[example["id"]] = predictions[0]['feature_index']
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred[
                "start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(
                score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]
            all_feature_index[example["id"]] = predictions[i]['feature_index']

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [{
            k: (float(v) if isinstance(v, (np.float16, np.float32,
                                           np.float64)) else v)
            for k, v in pred.items()
        } for pred in predictions]

    return all_predictions, all_nbest_json, scores_diff_json, all_feature_index
