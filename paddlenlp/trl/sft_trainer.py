# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler

from paddlenlp.trainer import Trainer
from paddlenlp.trainer.trainer_utils import has_length
from paddlenlp.utils.log import logger

__all__ = ["SFTTrainer"]


class SFTTrainer(Trainer):
    def __init__(self, do_generation: bool, gen_args, data_args, **kwargs):
        super().__init__(**kwargs)
        self.do_generation = do_generation
        self.gen_args = gen_args
        self.data_args = data_args

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        if prediction_loss_only or self.args.pipeline_parallel_degree > 1:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        elif not self.do_generation:
            loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            # argmax here to avoid gather all logits, which is too memory-consuming.
            # keepdim in order to maintain the same shape as logits
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            # all gather logits when enabling tensor_parallel_output
            if self.args.tensor_parallel_degree > 1 and getattr(self.args, "tensor_parallel_output", False):
                hcg = fleet.get_hybrid_communicate_group()
                model_parallel_group = hcg.get_model_parallel_group()
                gathered_logits = []
                dist.all_gather(gathered_logits, logits, group=model_parallel_group)
                logits = paddle.concat(gathered_logits, axis=-1)
            return (loss, logits.argmax(axis=-1, keepdim=True), labels)

        loss = None

        model.eval()
        with paddle.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
                position_ids=inputs["position_ids"] if "position_ids" in inputs else None,
                max_length=max(self.data_args.max_length - inputs["input_ids"].shape[-1], 1),
                decode_strategy="sampling",
                top_k=self.gen_args.top_k,
                top_p=self.gen_args.top_p,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )[0]
            all_preds = []
            for pred_tokens in generated_tokens:
                pred_tokens = pred_tokens.numpy()
                pred_tokens = pred_tokens[pred_tokens != self.tokenizer.pad_token_id].tolist()
                all_preds.append(pred_tokens)
            max_pred_length = max([len(x) for x in all_preds])
            for index, preds in enumerate(all_preds):
                all_preds[index] = preds + [-100] * (max_pred_length - len(preds))
            all_preds = paddle.to_tensor(all_preds)

            if "labels" in inputs:
                all_labels = paddle.to_tensor(inputs["labels"])
            else:
                all_labels = None

        return (loss, all_preds, all_labels)

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        if "loss" in logs:
            logs["ppl"] = np.exp(logs["loss"])
        if "eval_loss" in logs:
            logs["eval_ppl"] = np.exp(logs["eval_loss"])

        super(SFTTrainer, self).log(logs, **kwargs)

    def get_ptq_dataloader(self, ptq_ds):
        if self.args.world_size <= 1:
            ptq_sampler = BatchSampler(
                dataset=ptq_ds,
                shuffle=True,
                batch_size=self.args.per_device_train_batch_size,
                drop_last=self.args.dataloader_drop_last,
            )
        else:
            ptq_sampler = DistributedBatchSampler(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                num_replicas=self.args.dataset_world_size,
                rank=self.args.dataset_rank,
                drop_last=self.args.dataloader_drop_last,
            )
        ptq_dataloader = DataLoader(
            ptq_ds,
            batch_sampler=ptq_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )
        return ptq_dataloader

    def ptq_loop(
        self,
        dataloader: DataLoader,
        description: str,
        max_eval_iters: Optional[int] = -1,
    ):
        if isinstance(dataloader, paddle.io.DataLoader):
            batch_size = dataloader.batch_sampler.batch_size
        else:
            raise ValueError("Only support for paddle.io.DataLoader")

        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            if max_eval_iters > 0:
                logger.info(f"  Total {description} steps = {max_eval_iters}")
            else:
                logger.info(f"  Total {description} steps = {len(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
            if max_eval_iters > 0:
                logger.info(f"  Total {description} steps = {max_eval_iters}")

        logger.info(f"  Pre device batch size = {batch_size}")
        logger.info(f"  Total Batch size = {batch_size * self.args.dataset_world_size}")
        self.model.eval()
        with paddle.no_grad():
            for step, inputs in enumerate(dataloader):
                self.prediction_step(model=self.model, inputs=inputs, prediction_loss_only=True, ignore_keys=None)
                if max_eval_iters > 0 and step >= max_eval_iters - 1:
                    break
