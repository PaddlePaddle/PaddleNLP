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

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn
from paddle.optimizer.lr import LambdaDecay
from rouge import Rouge
from sklearn.metrics import accuracy_score

from paddlenlp.metrics import BLEU
from paddlenlp.trainer import PrinterCallback, ProgressCallback, Trainer
from paddlenlp.trainer.integrations import TrainerCallback
from paddlenlp.utils.log import logger


class AverageStatistical(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_cnt = 0
        self.time = 0

    def record(self, val, cnt=1):
        self.time += val
        self.total_cnt += cnt

    def get_average(self):
        if self.total_cnt == 0:
            return 0

        return self.time / self.total_cnt

    def get_average_per_sec(self):
        if self.time == 0.0:
            return 0.0

        return float(self.total_cnt) / self.time

    def get_total_cnt(self):
        return self.total_cnt

    def get_total_time(self):
        return self.time


class BenchmarkCallback(TrainerCallback):
    def __init__(self, benchmark=True, profiler_options=None):
        self.benchmark = benchmark
        self.profiler_options = profiler_options

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.gradient_accumulation_steps == 1 and not args.do_eval and not args.do_predict
        if self.benchmark:
            self.reader_cost_avg = AverageStatistical()

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            self.epoch_start = time.time()
            self.batch_start = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            self.reader_cost_avg.record(time.time() - self.batch_start)

    def on_step_end(self, args, state, control, **kwargs):
        if self.benchmark:
            self.batch_start = time.time()
            if control.should_log:
                self.maybe_log_save_evaluate_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.benchmark:
            if logs is not None and "interval_steps_per_second" in logs:
                self.batch_start = self.batch_start + (time.time() - self.maybe_log_save_evaluate_start)
                ips = logs["interval_steps_per_second"] * args.train_batch_size
                avg_batch_cost = 1 / logs["interval_steps_per_second"]
                max_mem_reserved_msg = ""
                max_mem_allocated_msg = ""
                if paddle.device.is_compiled_with_cuda():
                    max_mem_reserved_msg = (
                        f"max_mem_reserved: {paddle.device.cuda.max_memory_reserved() // (1024 ** 2)} MB,"
                    )
                    max_mem_allocated_msg = (
                        f"max_mem_allocated: {paddle.device.cuda.max_memory_allocated() // (1024 ** 2)} MB"
                    )
                logger.info(
                    "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, "
                    "avg_samples: %.5f, ips: %.5f sample/sec, %s %s"
                    % (
                        state.global_step,
                        state.max_steps,
                        logs["loss"],
                        self.reader_cost_avg.get_average(),
                        avg_batch_cost,
                        args.train_batch_size,
                        ips,
                        max_mem_reserved_msg,
                        max_mem_allocated_msg,
                    )
                )
                self.reader_cost_avg.reset()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.benchmark:
            train_epoch_cost = time.time() - self.epoch_start
            logger.info("train epoch: %d, epoch_cost: %.5f s" % (state.epoch, train_epoch_cost))


class LlamaTrainer(Trainer):
    def __init__(self, do_generation: bool, **kwargs):
        super().__init__(**kwargs)
        self.add_callback(BenchmarkCallback(benchmark=True, profiler_options=self.args.profiler_options))
        if self.args.disable_tqdm:
            self.pop_callback(PrinterCallback)
        else:
            self.pop_callback(ProgressCallback)
        self.do_generation = do_generation

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:

        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        elif not self.do_generation:
            loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            # argmax here to avoid gather all logits, which is too memory-consuming.
            # keepdim in order to maintain the same shape as logits
            return (loss, logits.argmax(axis=-1, keepdim=True), labels)

        model.eval()

        preds = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.args.tgt_length,
            min_length=0,
            use_cache=True,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
            decode_strategy="sampling",
        )[0]
        all_labels = []
        for label in inputs["labels"].numpy():
            label = [x for x in label[label != self.tokenizer.pad_token_id]]
            all_labels.append(label)
        max_label_length = max([len(x) for x in all_labels])
        for index, labels in enumerate(all_labels):
            all_labels[index] = labels + [-100] * (max_label_length - len(labels))

        return (None, paddle.to_tensor(preds), paddle.to_tensor(all_labels))

    def create_scheduler(self, num_training_steps: int):
        num_warmup_steps = (
            self.args.warmup_steps if self.args.warmup_steps > 0 else self.args.warmup_ratio * num_training_steps
        )

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                decay_step_ratio = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
                return 1.0 - (1.0 - self.args.lr_decay_ratio) * decay_step_ratio

        if self.lr_scheduler is None:
            self.lr_scheduler = LambdaDecay(self.args.learning_rate, lr_lambda, last_epoch=-1)
        return self.lr_scheduler

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        if "loss" in logs:
            logs["ppl"] = np.exp(logs["loss"])
        if "eval_loss" in logs:
            logs["eval_ppl"] = np.exp(logs["eval_loss"])

        super(LlamaTrainer, self).log(logs, **kwargs)


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

    rouge1 = round(rouge1, 4)
    rouge2 = round(rouge2, 4)
    rougel = round(rougel, 4)
    bleu4 = round(bleu4.score(), 4)
    return dict(
        rouge1=rouge1,
        rouge2=rouge2,
        rougel=rougel,
        bleu4=bleu4,
    )


def compute_metrics_not_do_generation(eval_preds):
    flattened_preds = np.array(eval_preds.predictions).flatten()
    flattened_labels = np.array(eval_preds.label_ids).flatten()
    filtered_preds = flattened_preds[flattened_labels != -100]
    filtered_labels = flattened_labels[flattened_labels != -100]
    accuracy = accuracy_score(y_true=filtered_labels, y_pred=filtered_preds)
    return {
        "accuracy": accuracy,
    }
