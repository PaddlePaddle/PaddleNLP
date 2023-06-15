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
import os
import sys

import paddle
import paddle.amp.auto_cast as autocast
from paddle.io import DataLoader

from paddlenlp.trainer import Trainer
from paddlenlp.trainer.integrations import (
    INTEGRATION_TO_CALLBACK,
    VisualDLCallback,
    rewrite_logs,
)
from paddlenlp.utils.log import logger
from ppdiffusers.training_utils import unwrap_model

from .text_image_pair_dataset import TextImagePair, worker_init_fn


class VisualDLWithImageCallback(VisualDLCallback):
    def autocast_smart_context_manager(self, args):
        if args.fp16 or args.bf16:
            amp_dtype = "float16" if args.fp16 else "bfloat16"
            ctx_manager = autocast(
                True,
                custom_black_list=[
                    "reduce_sum",
                    "c_softmax_with_cross_entropy",
                ],
                level=args.fp16_opt_level,
                dtype=amp_dtype,
            )
        else:
            ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

        return ctx_manager

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if hasattr(model, "on_train_batch_end"):
            model.on_train_batch_end()
        if args.image_logging_steps > 0 and state.global_step % args.image_logging_steps == 0:
            control.should_log = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        # only on card 0
        if not state.is_world_process_zero:
            return
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
            with self.autocast_smart_context_manager(args):
                image_logs["reconstruction"] = model.decode_image(pixel_values=inputs["pixel_values"])
                image_logs["control"] = model.decode_control_image(adapter_cond=inputs["adapter_cond"])
                image_logs["ddim-samples-9.0"] = model.log_image(
                    input_ids=inputs["input_ids"],
                    adapter_cond=inputs["adapter_cond"],
                    guidance_scale=9.0,
                    height=args.resolution,
                    width=args.resolution,
                )

        if self.vdl_writer is None:
            self._init_summary_writer(args)

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


# register visualdl_with_image
INTEGRATION_TO_CALLBACK.update({"custom_visualdl": VisualDLWithImageCallback})


def collate_fn(examples):
    pixel_values = paddle.stack([paddle.to_tensor(example["pixel_values"]) for example in examples])
    input_ids = paddle.stack([paddle.to_tensor(example["input_ids"]) for example in examples])
    adapter_cond = paddle.stack([paddle.to_tensor(example["adapter_cond"]) for example in examples])

    batch = {"input_ids": input_ids, "pixel_values": pixel_values, "adapter_cond": adapter_cond}
    return batch


class AdapterLDMTrainer(Trainer):
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
                collate_fn=collate_fn,
            )
        else:
            return super().get_train_dataloader()

    def _save(self, output_dir=None, state_dict=None, merge_tensor_parallel=False):
        super()._save(output_dir=output_dir, state_dict=state_dict, merge_tensor_parallel=merge_tensor_parallel)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        unwrap_model(self.model).adapter.save_pretrained(os.path.join(output_dir, "adapter"))
