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

import numpy as np
import collections
import sys
import paddle
from paddle.utils import try_import
from paddlenlp.metrics.dureader import get_final_text, _compute_softmax, _get_best_indexes

# Metric for ERNIE-DOCs


class F1(object):

    def __init__(self, positive_label=1):
        self.positive_label = positive_label
        self.reset()

    def compute(self, preds, labels):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        elif isinstance(preds, list):
            preds = np.array(preds, dtype='float32')
        if isinstance(labels, list):
            labels = np.array(labels, dtype='int64')
        elif isinstance(labels, paddle.Tensor):
            labels = labels.numpy()
        preds = np.argmax(preds, axis=1)
        tp = ((preds == labels) & (labels == self.positive_label)).sum()
        fn = ((preds != labels) & (labels == self.positive_label)).sum()
        fp = ((preds != labels) & (preds == self.positive_label)).sum()
        return tp, fp, fn

    def update(self, statistic):
        tp, fp, fn = statistic
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def accumulate(self):
        recall = self.tp / (self.tp + self.fn)
        precision = self.tp / (self.tp + self.fp)
        f1 = 2 * recall * precision / (recall + precision)
        return f1

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0


class EM_AND_F1(object):

    def __init__(self):
        self.nltk = try_import('nltk')
        self.re = try_import('re')

    def _mixed_segmentation(self, in_str, rm_punc=False):
        """mixed_segmentation"""
        in_str = in_str.lower().strip()
        segs_out = []
        temp_str = ""
        sp_char = [
            '-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。',
            '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、', '「',
            '」', '（', '）', '－', '～', '『', '』'
        ]
        for char in in_str:
            if rm_punc and char in sp_char:
                continue
            pattern = '[\\u4e00-\\u9fa5]'
            if self.re.search(pattern, char) or char in sp_char:
                if temp_str != "":
                    ss = self.nltk.word_tokenize(temp_str)
                    segs_out.extend(ss)
                    temp_str = ""
                segs_out.append(char)
            else:
                temp_str += char

        # Handling last part
        if temp_str != "":
            ss = self.nltk.word_tokenize(temp_str)
            segs_out.extend(ss)

        return segs_out

    # Remove punctuation
    def _remove_punctuation(self, in_str):
        """remove_punctuation"""
        in_str = in_str.lower().strip()
        sp_char = [
            '-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。',
            '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、', '「',
            '」', '（', '）', '－', '～', '『', '』'
        ]
        out_segs = []
        for char in in_str:
            if char in sp_char:
                continue
            else:
                out_segs.append(char)
        return ''.join(out_segs)

    # Find longest common string
    def _find_lcs(self, s1, s2):
        m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
        mmax = 0
        p = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i + 1][j + 1] = m[i][j] + 1
                    if m[i + 1][j + 1] > mmax:
                        mmax = m[i + 1][j + 1]
                        p = i + 1
        return s1[p - mmax:p], mmax

    def _calc_f1_score(self, answers, prediction):
        f1_scores = []
        for ans in answers:
            ans_segs = self._mixed_segmentation(ans, rm_punc=True)
            prediction_segs = self._mixed_segmentation(prediction, rm_punc=True)
            lcs, lcs_len = self._find_lcs(ans_segs, prediction_segs)
            if lcs_len == 0:
                f1_scores.append(0)
                continue
            precision = 1.0 * lcs_len / len(prediction_segs)
            recall = 1.0 * lcs_len / len(ans_segs)
            f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1)
        return max(f1_scores)

    def _calc_em_score(self, answers, prediction):
        em = 0
        for ans in answers:
            ans_ = self._remove_punctuation(ans)
            prediction_ = self._remove_punctuation(prediction)
            if ans_ == prediction_:
                em = 1
                break
        return em

    def __call__(self, prediction, ground_truth):
        f1 = 0
        em = 0
        total_count = 0
        skip_count = 0
        for instance in ground_truth:
            total_count += 1
            query_id = instance['id']
            query_text = instance['question'].strip()
            answers = instance["answers"]
            if query_id not in prediction:
                sys.stderr.write('Unanswered question: {}\n'.format(query_id))
                skip_count += 1
                continue
            preds = str(prediction[query_id])
            f1 += self._calc_f1_score(answers, preds)
            em += self._calc_em_score(answers, preds)

        f1_score = 100.0 * f1 / total_count
        em_score = 100.0 * em / total_count

        avg_score = (f1_score + em_score) * 0.5
        return em_score, f1_score, avg_score, total_count


def compute_qa_predictions(all_examples, all_features, all_results, n_best_size,
                           max_answer_length, do_lower_case, tokenizer,
                           verbose):
    """Write final predictions to the json file and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index", "start_index", "end_index", "start_logit",
            "end_logit"
        ])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # Keep track of the minimum score of null start+end of position 0
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.qid]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(prelim_predictions,
                                    key=lambda x: (x.start_logit + x.end_logit),
                                    reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index +
                                                              1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end +
                                                                 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, tokenizer,
                                            verbose)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(text=final_text,
                                 start_logit=pred.start_logit,
                                 end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json
    return all_predictions, all_nbest_json
