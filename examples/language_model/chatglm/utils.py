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
import json
import os
from typing import Dict

import numpy as np
import paddle
from predict_generation import Predictor, batchfy_text

from paddlenlp.trainer import Trainer


def save_infer_result(trainer, dev_ds, k=100, src_length=256, tgt_length=512):
    all_instructions = []
    all_answers = []
    all_output = []

    # top k instruction from dev_ds
    for i, ds in enumerate(dev_ds.data):
        if i == k:
            break
        all_instructions.append(ds["instruction"])
        all_answers.append(ds["output"])
    batch_texts = batchfy_text(all_instructions, trainer.args.per_device_eval_batch_size)
    predictor = Predictor(
        tokenizer=trainer.tokenizer, model=trainer.model, src_length=src_length, tgt_length=tgt_length
    )

    # infer results
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for i, (text, result) in enumerate(zip(texts, outputs["result"])):
            out = {
                "instruction": text,
                "answer": all_answers[bs * trainer.args.per_device_eval_batch_size + i],
                "output": result,
            }
            all_output.append(out)

    # save results
    if trainer.args.tensor_parallel_rank == 0:
        with open(os.path.join(trainer.args.output_dir, "infer_result.json"), "w") as f:
            for out in all_output:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


class ChatGLMTrainer(Trainer):
    def __init__(self, data_args, do_generation: bool, **kwargs):
        super().__init__(**kwargs)
        self.data_args = data_args
        self.do_generation = do_generation

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval", **gen_kwargs):
        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            self.data_args.generation_max_length
            if hasattr(self.data_args, "generation_max_length")
            else self.data_args.tgt_length
        )
        gen_kwargs["num_beams"] = self.data_args.num_beams
        self._gen_kwargs = gen_kwargs

        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix: str = "test", **gen_kwargs):
        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            self.data_args.generation_max_length
            if hasattr(self.data_args, "generation_max_length")
            else self.data_args.tgt_length
        )
        gen_kwargs["num_beams"] = self.data_args.num_beams
        self._gen_kwargs = gen_kwargs

        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        elif not self.do_generation:
            loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            # argmax here to avoid gather all logits, which is too memory-consuming.
            # keepdim in order to maintain the same shape as logits
            return (loss, logits[0][..., :-1, :].argmax(axis=-1, keepdim=True), labels[..., 1:])
        loss = None

        n_token_id = self.tokenizer.convert_tokens_to_ids("<n>")
        model.eval()
        with paddle.no_grad():
            generated_tokens = model.generate(
                **inputs,
                **self._gen_kwargs.copy(),
                decode_strategy="sampling",
                top_k=1,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.end_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )[0]
            all_preds = []
            for pred_tokens in generated_tokens:
                pred_tokens = pred_tokens[pred_tokens != self.tokenizer.pad_token_id]
                pred_tokens = pred_tokens[pred_tokens != n_token_id].tolist()
                all_preds.append(pred_tokens)
            max_pred_length = max([len(x) for x in all_preds])
            for index, preds in enumerate(all_preds):
                all_preds[index] = preds + [-100] * (max_pred_length - len(preds))
            all_preds = paddle.to_tensor(all_preds)

            if "labels" in inputs:
                all_labels = []
                for index, label in enumerate(inputs["labels"]):
                    label = label[:max_pred_length]
                    label[all_preds[index] == -100] = -100
                    all_labels.append(label)
                all_labels = paddle.to_tensor(all_labels)
            else:
                all_labels = None

        return (loss, all_preds, all_labels)

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        if "loss" in logs:
            logs["ppl"] = np.exp(logs["loss"])
        if "eval_loss" in logs:
            logs["eval_ppl"] = np.exp(logs["eval_loss"])

        super(ChatGLMTrainer, self).log(logs, **kwargs)
