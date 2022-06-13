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
import numpy as np
import nltk
from rouge_score import rouge_scorer, scoring


def convert_example(example,
                    text_column,
                    summary_column,
                    tokenizer,
                    decoder_start_token_id,
                    max_source_length,
                    max_target_length,
                    ignore_pad_token_for_loss=True,
                    is_train=True):
    """
    Convert a example into necessary features.
    """
    inputs = example[text_column]
    targets = example[summary_column]
    labels = tokenizer(targets,
                       max_length=max_target_length,
                       padding='max_length',
                       truncation=True)
    decoder_input_ids = [decoder_start_token_id] + labels["input_ids"][:-1]
    if ignore_pad_token_for_loss:
        labels["input_ids"] = [(l if l != tokenizer.pad_token_id else -100)
                               for l in labels["input_ids"]]
    if is_train:
        model_inputs = tokenizer(inputs,
                                 max_length=max_source_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_attention_mask=True,
                                 return_length=False)
        return model_inputs["input_ids"], model_inputs[
            "attention_mask"], decoder_input_ids, labels["input_ids"]
    else:
        model_inputs = tokenizer(inputs,
                                 max_length=max_source_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_attention_mask=True,
                                 return_length=True)
        return model_inputs["input_ids"], model_inputs["attention_mask"], \
        model_inputs["length"], decoder_input_ids, labels["input_ids"]


def compute_metrics(preds, labels, tokenizer, ignore_pad_token_for_loss=True):

    def compute_rouge(predictions,
                      references,
                      rouge_types=None,
                      use_stemmer=True):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeLsum"]

        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types,
                                          use_stemmer=use_stemmer)
        aggregator = scoring.BootstrapAggregator()

        for ref, pred in zip(references, predictions):
            score = scorer.score(ref, pred)
            aggregator.add_scores(score)
        result = aggregator.aggregate()
        result = {
            key: round(value.mid.fmeasure * 100, 4)
            for key, value in result.items()
        }
        return result

    def post_process_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def post_process_seq(seq,
                         bos_idx,
                         eos_idx,
                         output_bos=False,
                         output_eos=False):
        """
        Post-process the decoded sequence.
        """
        eos_pos = len(seq) - 1
        for i, idx in enumerate(seq):
            if idx == eos_idx:
                eos_pos = i
                break
        seq = [
            idx for idx in seq[:eos_pos + 1]
            if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
        ]
        return seq

    if ignore_pad_token_for_loss:
        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds, decoded_labels = [], []
    for pred, label in zip(preds, labels):
        pred_id = post_process_seq(pred, tokenizer.bos_token_id,
                                   tokenizer.eos_token_id)
        label_id = post_process_seq(label, tokenizer.bos_token_id,
                                    tokenizer.eos_token_id)
        decoded_preds.append(tokenizer.convert_ids_to_string(pred_id))
        decoded_labels.append(tokenizer.convert_ids_to_string(label_id))
    decoded_preds, decoded_labels = post_process_text(decoded_preds,
                                                      decoded_labels)
    rouge_result = compute_rouge(decoded_preds, decoded_labels)
    return rouge_result, decoded_preds
