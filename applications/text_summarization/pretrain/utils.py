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


import contextlib
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import nn
from paddle.io import DataLoader
from rouge import Rouge

from paddlenlp.metrics import BLEU
from paddlenlp.trainer import Seq2SeqTrainer
from paddlenlp.utils.log import logger

rouge = Rouge()


class FakeAbstractCollator:
    def __init__(self, tokenizer, stopwords_dict, max_enc_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_enc_length
        self.stopwords_dict = stopwords_dict

    def __call__(self, samples):
        labels = []
        attn_mask = []
        decoder_attn_mask = []
        source_inputs = []

        for text in samples:
            texts = text["content"]
            text = text_segmentate(texts)

            if len(text) < 2:
                continue
            sentence_id_vec, source, target, source_idxs, target_idxs = pseudo_summary_f1(
                text, self.stopwords_dict, self.tokenizer, self.max_seq_length, "rouge-l"
            )
            source_idxs, target_idxs = get_input_mask(sentence_id_vec, target_idxs)
            if len(source_idxs) > self.max_seq_length:
                if 2 not in source_idxs[self.max_seq_length - 1 :]:
                    source_idxs = source_idxs[: self.max_seq_length]
                    source_idxs[-1] = self.tokenizer.eos_token_id
                    sys.stderr.write("Warning split long line: " + source + "\n")
                else:
                    continue

            source_idxs, attention_mask = padding_to_maxlength(
                source_idxs, self.max_seq_length, self.tokenizer.pad_token_id
            )
            label, target_attention_mask = padding_to_maxlength(
                target_idxs, self.max_seq_length, self.tokenizer.pad_token_id
            )
            source_inputs.append(source_idxs)
            attn_mask.append(attention_mask)
            decoder_attn_mask.append(target_attention_mask)
            labels.append(label)
        labels = paddle.to_tensor(labels)
        decode_input_idxs = shift_tokens_right(labels, self.tokenizer.pad_token_id, self.tokenizer.pad_token_id)
        end_token_index = paddle.where(labels == self.tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx, end_idx + 1 :] = -100

        return {
            "input_ids": paddle.to_tensor(source_inputs),
            "attention_mask": paddle.to_tensor(attn_mask),
            "labels": labels,
            "decoder_input_ids": decode_input_idxs,
            "decoder_attention_mask": paddle.to_tensor(decoder_attn_mask),
        }


def load_stopwords(stopwords_path):
    stopwords_dict = {}
    with open(stopwords_path, "r") as rf:
        for line in rf:
            line = line.strip()
            if line not in stopwords_dict:
                stopwords_dict[line] = 0
            else:
                pass
    return stopwords_dict


def text_segmentate(text):
    en_seg_pattern = "((?:\\!|\\?|\\.|\\n)+(?:\\s)+)"
    ch_seg_pattern = "((?:？|！|。|\\n)+)"
    try:
        text = re.sub(en_seg_pattern, r"\1[SEP]", text)
    except Exception as e:
        print("input: ", text)
        raise e
    text = re.sub(ch_seg_pattern, r"\1[SEP]", text)
    text_list = text.split("[SEP]")
    text_list = list(filter(lambda x: len(x) != 0, text_list))
    return text_list


def gather_join(texts, idxs):
    return "".join([texts[i] for i in idxs])


def gather_join_f1(texts_token, idsx):
    join_texts = []
    for id in idsx:
        join_texts.extend(texts_token[id])
    return join_texts


def compute_rouge(source, target):
    source, target = " ".join(source), " ".join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            "rouge-1": scores[0]["rouge-1"]["f"],
            "rouge-2": scores[0]["rouge-2"]["f"],
            "rouge-l": scores[0]["rouge-l"]["f"],
        }
    except ValueError:
        return {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
        }


def remove_stopwords(texts, stopwords_dict):
    for i, text in enumerate(texts):
        texts[i] = list(filter(lambda x: x not in stopwords_dict, text))
    return texts


def pseudo_summary_f1(texts, stopwords, tokenizer, max_length, rouge_strategy="rouge-l"):
    summary_rate = 0.25
    max_length = max_length - 1
    texts_tokens = []
    sentece_idxs_vec = []
    for text in texts:
        if len(texts) == 0:
            continue
        try:
            ids = tokenizer.encode(text.strip())["input_ids"][:-1]
        except ValueError:
            print("error, input : ", text)
            raise ValueError
        sentece_idxs_vec.append(ids)
        tokens = [tokenizer._convert_id_to_token(token) for token in ids]
        texts_tokens.append(tokens)

    texts_tokens_rm = remove_stopwords(texts_tokens, stopwords)
    source_idxs, target_idxs = list(range(len(texts))), []

    assert len(texts_tokens) == len(texts)
    while True:
        sims = []
        for i in source_idxs:
            new_source_idxs = [j for j in source_idxs if j != i]
            new_target_idxs = sorted(target_idxs + [i])
            new_source = gather_join_f1(texts_tokens_rm, new_source_idxs)
            new_target = gather_join_f1(texts_tokens_rm, new_target_idxs)
            sim = compute_rouge(new_source, new_target)[rouge_strategy]
            sims.append(sim)
        new_idx = source_idxs[np.argmax(sims)]
        del sims
        source_idxs.remove(new_idx)
        target_idxs = sorted(target_idxs + [new_idx])
        source = gather_join(texts, source_idxs)
        target = gather_join(texts, target_idxs)
        try:
            if len(source_idxs) == 1 or 1.0 * len(target) / len(source) > summary_rate:
                break
        except ZeroDivisionError:
            print(texts)
            print("source: ", source)
            print("target: ", target)

    if len(source) < len(target):
        source, target = target, source
        source_idxs, target_idxs = target_idxs, source_idxs

    return sentece_idxs_vec, source, target, source_idxs, target_idxs


def get_input_mask(sentence_id_vec, indexs):
    target_idxs = []
    input_idxs = []
    kMaskSentenceTokenId = 2
    kEosTokenId = 1
    mask_sentence_options_cumulative_prob = [0.9, 0.9, 1, 1]
    for index in indexs:
        target_idxs.extend(sentence_id_vec[index])
        choice = random.uniform(0, 1)
        if choice < mask_sentence_options_cumulative_prob[0]:
            sentence_id_vec[index] = [kMaskSentenceTokenId]
        elif choice < mask_sentence_options_cumulative_prob[1]:
            replace_id = random.randint(0, len(sentence_id_vec))
            sentence_id_vec[index] = sentence_id_vec[replace_id]
        elif choice < mask_sentence_options_cumulative_prob[2]:
            pass
        else:
            sentence_id_vec[index] = []

    target_idxs.append(kEosTokenId)
    for index, sentence_id in enumerate(sentence_id_vec):
        if len(sentence_id) == 0:
            continue
        input_idxs.extend(sentence_id_vec[index])

    input_idxs.append(kEosTokenId)
    return input_idxs, target_idxs


def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    shifted_input_ids = paddle.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = paddle.clone(input_ids[:, :-1])
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids = paddle.where(shifted_input_ids == -100, paddle.to_tensor(pad_token_id), shifted_input_ids)

    return shifted_input_ids


def padding_to_maxlength(ids, max_length, pad_id):
    cur_len = len(ids)
    len_diff = max_length - cur_len
    return ids + [pad_id] * len_diff, [1] * cur_len + [0] * len_diff


def convert_example(example, text_column, summary_column, tokenizer, max_source_length, max_target_length):
    """
    Convert a example into necessary features.
    """
    inputs = example[text_column]
    targets = example[summary_column]
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=False, truncation=True, return_attention_mask=True
    )
    labels = tokenizer(targets, max_length=max_target_length, padding=False, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_correct(logits, labels):
    y_pred = paddle.argmax(logits, axis=-1)
    y_pred = y_pred.reshape(
        [
            -1,
        ]
    )
    y_true = labels.reshape(
        [
            -1,
        ]
    )
    correct = paddle.sum(paddle.equal(y_pred, y_true).astype("float32")).item()
    return correct


def compute_metrics(preds, targets):
    assert len(preds) == len(targets), (
        "The length of pred_responses should be equal to the length of "
        "target_responses. But received {} and {}.".format(len(preds), len(targets))
    )
    rouge = Rouge()
    bleu4 = BLEU(n_size=4)
    scores = []
    for pred, target in zip(preds, targets):
        try:
            score = rouge.get_scores(" ".join(pred), " ".join(target))
            scores.append([score[0]["rouge-1"]["f"], score[0]["rouge-2"]["f"], score[0]["rouge-l"]["f"]])
        except ValueError:
            scores.append([0, 0, 0])
        bleu4.add_inst(pred, [target])
    rouge1 = np.mean([i[0] for i in scores])
    rouge2 = np.mean([i[1] for i in scores])
    rougel = np.mean([i[2] for i in scores])
    print("\n" + "*" * 15)
    print("The auto evaluation result is:")
    print("rouge-1:", round(rouge1, 4))
    print("rouge-2:", round(rouge2, 4))
    print("rouge-L:", round(rougel, 4))
    print("BLEU-4:", round(bleu4.score(), 4))
    return rougel


@contextlib.contextmanager
def main_process_first(desc="work"):
    if paddle.distributed.get_world_size() > 1:
        rank = paddle.distributed.get_rank()
        is_main_process = rank == 0
        main_process_desc = "main local process"

        try:
            if not is_main_process:
                # tell all replicas to wait
                logger.debug(f"{rank}: waiting for the {main_process_desc} to perform {desc}")
                paddle.distributed.barrier()
            yield
        finally:
            if is_main_process:
                # the wait is over
                logger.debug(f"{rank}: {main_process_desc} completed {desc}, releasing all replicas")
                paddle.distributed.barrier()
    else:
        yield


class PegasusTrainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        """
        Returns the training [`~paddle.io.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        train_sampler = self._get_train_sampler()

        stopwords_dict = load_stopwords(self.args.stop_words)
        train_batchify_fn = FakeAbstractCollator(self.tokenizer, stopwords_dict, self.args.max_source_length)

        return DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=train_batchify_fn,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.criterion is not None:
            if "labels" in inputs:
                labels = inputs.pop("labels")
            elif "start_positions" in inputs and "end_positions" in inputs:
                labels = (inputs.pop("start_positions"), inputs.pop("end_positions"))
            elif self.args.label_names is not None:
                labels = []
                for label in self.label_names:
                    labels.append(inputs.pop(label))
                labels = tuple(labels)
            elif "generator_labels" in inputs:
                labels = inputs["generator_labels"]
        else:
            labels = None

        outputs = model(**inputs)
        if self.criterion is not None:
            loss = self.criterion(outputs, labels)
            outputs = (loss, outputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        # pegasus output is lm_logits, new_cache, masked_lm_loss
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[2]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to evaluate.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[paddle.Tensor], Optional[paddle.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # different from hf returns: tuple[Tensor]: It is a tuple contains two elements: ids and scores.
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (
            gen_kwargs["max_new_tokens"] + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        with paddle.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    # pegasus output is lm_logits, new_cache, masked_lm_loss
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[2]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None

        return (loss, generated_tokens, labels)
