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
import evaluate
import nltk
import numpy as np

from paddlenlp.metrics import BLEU


def convert_example(
    example,
    tokenizer,
    decoder_start_token_id,
    max_source_length,
    max_target_length,
    ignore_pad_token_for_loss=True,
    is_train=True,
):
    """
    Convert an example into necessary features.
    """
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    context = example["context"]
    question = example["question"]
    try:
        answer = example["answers"][0]
    except:
        print(example["context"])
        print(example["question"])
        print(example["answers"])
        print(example["answer_starts"])
        print(example["is_impossible"])

    input_seq = f"answer: {answer} context: {context} </s>"
    output_seq = f"question: {question} </s>"

    outputs = tokenizer(
        output_seq,
        max_seq_len=max_target_length,
        pad_to_max_seq_len=True,
        truncation_strategy="longest_first",
    )

    output_ids = [decoder_start_token_id] + outputs["input_ids"][:-1]

    if ignore_pad_token_for_loss:
        # Replace all tokenizer.pad_token_id in the outputs by -100 when we want to ignore padding in the loss.
        outputs["input_ids"] = [(l if l != tokenizer.pad_token_id else -100) for l in outputs["input_ids"]]

    if is_train:
        inputs = tokenizer(
            input_seq,
            max_seq_len=max_source_length,
            pad_to_max_seq_len=True,
            truncation_strategy="longest_first",
            return_attention_mask=True,
            return_length=False,
        )
        return inputs["input_ids"], inputs["attention_mask"], output_ids, outputs["input_ids"]
    else:
        inputs = tokenizer(
            input_seq,
            max_seq_len=max_source_length,
            pad_to_max_seq_len=True,
            truncation_strategy="longest_first",
            return_attention_mask=True,
            return_length=True,
        )
        return inputs["input_ids"], inputs["attention_mask"], inputs["length"], output_ids, outputs["input_ids"]


def compute_metrics(preds, labels, tokenizer, ignore_pad_token_for_loss=True):
    def compute_bleu(predictions, references, rouge_types=None, use_stemmer=True):
        bleu1 = BLEU(n_size=1)
        bleu2 = BLEU(n_size=2)
        bleu3 = BLEU(n_size=3)
        bleu4 = BLEU(n_size=4)
        assert len(predictions) == len(references)
        for i in range(len(predictions)):
            bleu1.add_inst(predictions[i], [references[i]])
            bleu2.add_inst(predictions[i], [references[i]])
            bleu3.add_inst(predictions[i], [references[i]])
            bleu4.add_inst(predictions[i], [references[i]])
        result = {
            "BLEU-1": bleu1.score() * 100,
            "BLEU-2": bleu2.score() * 100,
            "BLEU-3": bleu3.score() * 100,
            "BLEU-4": bleu4.score() * 100,
        }
        return result

    def compute_bleu_hf(predictions, references, rouge_types=None, use_stemmer=True):
        predictions = [" ".join(prediction) for prediction in predictions]
        references = [[" ".join(reference)] for reference in references]

        bleu = evaluate.load("bleu")
        assert len(predictions) == len(references)
        bleu1_results = bleu.compute(predictions=predictions, references=references, max_order=1)
        bleu2_results = bleu.compute(predictions=predictions, references=references, max_order=2)
        bleu3_results = bleu.compute(predictions=predictions, references=references, max_order=3)
        bleu4_results = bleu.compute(predictions=predictions, references=references, max_order=4)

        result = {
            "BLEU-1": bleu1_results["bleu"] * 100,
            "BLEU-2": bleu2_results["bleu"] * 100,
            "BLEU-3": bleu3_results["bleu"] * 100,
            "BLEU-4": bleu4_results["bleu"] * 100,
        }
        return result

    def post_process_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = [pred.strip("question:") for pred in preds]
        labels = [label.strip("question:") for label in labels]
        labels = [label.strip() for label in labels]

        #  expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        preds = [pred.split() for pred in preds]
        labels = [label.split() for label in labels]

        return preds, labels

    def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
        """
        Post-process the decoded sequence.
        """
        eos_pos = len(seq) - 1
        for i, idx in enumerate(seq):
            if idx == eos_idx:
                eos_pos = i
                break
        seq = [idx for idx in seq[: eos_pos + 1] if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)]
        return seq

    if ignore_pad_token_for_loss:
        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds, decoded_labels = [], []
    for pred, label in zip(preds, labels):
        pred_id = post_process_seq(pred, tokenizer.bos_token_id, tokenizer.eos_token_id)
        label_id = post_process_seq(label, tokenizer.bos_token_id, tokenizer.eos_token_id)
        decoded_preds.append(tokenizer.decode(pred_id))
        decoded_labels.append(tokenizer.decode(label_id))
    decoded_preds, decoded_labels = post_process_text(decoded_preds, decoded_labels)
    # bleu_result = compute_bleu(decoded_preds, decoded_labels)
    bleu_result = compute_bleu_hf(decoded_preds, decoded_labels)
    return bleu_result, decoded_preds, decoded_labels
