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

import os
import time

import paddle
from paddle.io import DataLoader

from paddlenlp.trainer import PrinterCallback, ProgressCallback, Trainer
from paddlenlp.trainer.integrations import (
    INTEGRATION_TO_CALLBACK,
    TrainerCallback,
    VisualDLCallback,
    rewrite_logs,
)
from paddlenlp.transformers.model_utils import _add_variant
from paddlenlp.utils import profiler
from paddlenlp.utils.log import logger
from ppdiffusers.training_utils import unwrap_model

from .text_image_pair_dataset import TextImagePair, worker_init_fn

PADDLE_WEIGHTS_NAME = "model_state.pdparams"
TRAINING_ARGS_NAME = "training_args.bin"


class VisualDLWithImageCallback(VisualDLCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if hasattr(model, "on_train_batch_end"):
            model.on_train_batch_end()
        if args.image_logging_steps > 0 and state.global_step % args.image_logging_steps == 0:
            control.should_log = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        # log image on each node
        inputs = kwargs.get("inputs", None)
        model = kwargs.get("model", None)
        image_logs = {}
        if (
            inputs is not None
            and model is not None
            and args.image_logging_steps > 0
            and state.global_step % args.image_logging_steps == 0
        ):
            max_batch = 4 if args.resolution > 256 else 8
            image_logs["reconstruction"] = model.decode_image(pixel_values=inputs["pixel_values"], max_batch=max_batch)
            image_logs["ddim-samples-1.0"] = model.log_image(
                input_ids=inputs["input_ids"],
                guidance_scale=1.0,
                height=args.resolution,
                width=args.resolution,
                max_batch=max_batch,
            )
            image_logs["ddim-samples-7.5"] = model.log_image(
                input_ids=inputs["input_ids"],
                guidance_scale=7.5,
                height=args.resolution,
                width=args.resolution,
                max_batch=max_batch,
            )

        if not state.is_world_process_zero:
            return

        if self.vdl_writer is None:
            self._init_summary_writer(args)

        base_learning_rate = logs.get("learning_rate", None)
        if base_learning_rate is not None:
            logs["unet_lr"] = base_learning_rate
            if model.train_text_encoder:
                if args.text_encoder_learning_rate != args.unet_learning_rate:
                    logs["unet_lr"] = base_learning_rate * args.unet_learning_rate
                    logs["text_encoder_lr"] = base_learning_rate * args.text_encoder_learning_rate
                else:
                    logs["text_encoder_lr"] = base_learning_rate

        if self.vdl_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.vdl_writer.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of VisualDL's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            # log images
            for k, v in image_logs.items():
                self.vdl_writer.add_image(k, v, state.global_step, dataformats="NHWC")
            self.vdl_writer.flush()


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
        if self.profiler_options is not None:
            profiler.add_profiler_step(self.profiler_options)

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
                logger.info(
                    "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sample/sec"
                    % (
                        state.global_step,
                        state.max_steps,
                        logs["loss"],
                        self.reader_cost_avg.get_average(),
                        avg_batch_cost,
                        args.train_batch_size,
                        ips,
                    )
                )
                self.reader_cost_avg.reset()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.benchmark:
            train_epoch_cost = time.time() - self.epoch_start
            logger.info("train epoch: %d, epoch_cost: %.5f s" % (state.epoch, train_epoch_cost))


# register visualdl_with_image
INTEGRATION_TO_CALLBACK.update({"custom_visualdl": VisualDLWithImageCallback})


class StableDiffusionTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.args.benchmark or self.args.profiler_options is not None:
            self.add_callback(
                BenchmarkCallback(benchmark=self.args.benchmark, profiler_options=self.args.profiler_options)
            )
            if self.args.benchmark:
                if self.args.disable_tqdm:
                    self.pop_callback(PrinterCallback)
                else:
                    self.pop_callback(ProgressCallback)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs)
        return loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, TextImagePair):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                num_workers=self.args.dataloader_num_workers,
                worker_init_fn=worker_init_fn,
            )
        else:
            return super().get_train_dataloader()

    def _save(self, output_dir=None, state_dict=None, merge_tensor_parallel=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.args.only_save_updated_model:
            unwraped_model = unwrap_model(self.model)
            logger.info(f"Saving unet checkpoint to {output_dir}/unet")
            unwraped_model.unet.save_pretrained(os.path.join(output_dir, "unet"))

            if unwraped_model.use_ema:
                logger.info(f"Saving ema unet checkpoint to {output_dir}/unet")
                with unwraped_model.ema_scope():
                    unwraped_model.unet.save_pretrained(os.path.join(output_dir, "unet"), variant="ema")

            if unwraped_model.train_text_encoder:
                logger.info(f"Saving text encoder checkpoint to {output_dir}/text_encoder")
                unwraped_model.text_encoder.save_pretrained(os.path.join(output_dir, "text_encoder"))
        else:
            logger.info(f"Saving model checkpoint to {output_dir}")
            if state_dict is None:
                state_dict = self.model.state_dict()
            paddle.save(
                state_dict,
                os.path.join(output_dir, _add_variant(PADDLE_WEIGHTS_NAME, self.args.weight_name_suffix)),
            )
            if self.args.should_save:
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
                paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
