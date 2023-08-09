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

from typing import Dict, Optional

import numpy as np
import paddle
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from sklearn.metrics import accuracy_score

from paddlenlp.peft.prefix import (
    bloom_postprocess_past_key_value,
    chatglm_pad_attention_mask,
    chatglm_postprocess_past_key_value,
    llama_postprocess_past_key_value,
)
from paddlenlp.trainer import Trainer
from paddlenlp.trainer.trainer_utils import has_length
from paddlenlp.utils.log import logger


def compute_metrics(eval_preds):

    flattened_preds = np.array(eval_preds.predictions).flatten()
    flattened_labels = np.array(eval_preds.label_ids).flatten()
    filtered_preds = flattened_preds[flattened_labels != -100]
    filtered_labels = flattened_labels[flattened_labels != -100]
    accuracy = accuracy_score(y_true=filtered_labels, y_pred=filtered_preds)
    return {
        "accuracy": accuracy,
    }


def get_prefix_tuning_params(model):
    if model.base_model_prefix == "chatglm":
        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        pad_attention_mask = chatglm_pad_attention_mask
        postprocess_past_key_value = chatglm_postprocess_past_key_value
        multi_query_group_num = None
    elif model.base_model_prefix == "chatglm_v2":
        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_layers
        hidden_size = model.config.hidden_size
        pad_attention_mask = None
        postprocess_past_key_value = chatglm_postprocess_past_key_value
        multi_query_group_num = model.config.multi_query_group_num
    elif model.base_model_prefix == "bloom":
        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.n_layer
        hidden_size = model.config.n_embed
        pad_attention_mask = None
        postprocess_past_key_value = bloom_postprocess_past_key_value
        multi_query_group_num = None
    elif model.base_model_prefix == "llama":
        num_attention_heads = model.config.n_head
        num_hidden_layers = model.config.n_layer
        hidden_size = model.config.hidden_size
        pad_attention_mask = None
        postprocess_past_key_value = llama_postprocess_past_key_value
        multi_query_group_num = None
    else:
        raise ValueError(
            f"Unknown base_model_prefix: {model.base_model_prefix}. Supported base_model_prefix list: chatglm, bloom, llama."
        )
    return dict(
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        pad_attention_mask=pad_attention_mask,
        postprocess_past_key_value=postprocess_past_key_value,
        multi_query_group_num=multi_query_group_num,
    )


def get_lora_target_modules(model):
    # Not yet support RowParallelLinear
    if model.base_model_prefix == "chatglm":
        target_modules = [".*query_key_value.*", ".*dense.*", ".*dense_h_to_4h.*", ".*dense_4h_to_h.*"]
    elif model.base_model_prefix == "chatglm_v2":
        target_modules = [
            ".*query.*",
            ".*key.*",
            ".*value.*",
            ".*dense.*",
            ".*dense_h_to_4h.*",
            ".*dense_4h_to_h.*",
        ]
    elif model.base_model_prefix == "bloom":
        target_modules = [".*query_key_value.*", ".*dense.*", ".*dense_h_to_4h.*", ".*dense_4h_to_h.*"]
    elif model.base_model_prefix == "llama":
        target_modules = [
            ".*q_proj.*",
            ".*v_proj.*",
            ".*k_proj.*",
            ".*o_proj.*",
            ".*gate_proj.*",
            ".*down_proj.*",
            ".*up_proj.*",
        ]
    else:
        raise ValueError(
            f"Unknown base_model_prefix: {model.base_model_prefix}. Supported base_model_prefix list: chatglm, bloom, llama."
        )
    return target_modules


class CausalLMTrainer(Trainer):
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
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        elif not self.do_generation:
            loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
            # argmax here to avoid gather all logits, which is too memory-consuming.
            # keepdim in order to maintain the same shape as logits
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            return (loss, logits.argmax(axis=-1, keepdim=True), labels)

        loss = None

        model.eval()
        with paddle.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
                position_ids=inputs["position_ids"] if "position_ids" in inputs else None,
                max_length=self.data_args.tgt_length,
                decode_strategy=self.gen_args.decode_strategy,
                repetition_penalty=self.gen_args.repetition_penalty,
                top_k=self.gen_args.top_k,
                top_p=self.gen_args.top_p,
                num_beams=self.gen_args.num_beams,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )[0]
            all_preds = []
            for pred_tokens in generated_tokens:
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

        super(CausalLMTrainer, self).log(logs, **kwargs)

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

        logger.info(f"***** Running {description} *****")
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
        for step, inputs in enumerate(dataloader):
            self.prediction_step(model=self.model, inputs=inputs, prediction_loss_only=True, ignore_keys=None)
            if max_eval_iters > 0 and step >= max_eval_iters - 1:
                break
        logger.info(f"***** {description} done *****")
