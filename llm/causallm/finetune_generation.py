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
import sys
from functools import partial

import paddle
from argument import DataArgument, GenerateArgument, ModelArgument, QuantArgument
from data import get_convert_example, read_local_dataset
from utils import (
    CausalLMTrainer,
    compute_metrics,
    get_lora_target_modules,
    get_prefix_tuning_params,
)

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import BLEU, Rouge1, Rouge2, RougeL
from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from paddlenlp.utils.log import logger


def main():
    # Arguments
    parser = PdArgumentParser((GenerateArgument, QuantArgument, ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        gen_args, quant_args, model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        gen_args, quant_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(quant_args, "Quant")
    training_args.print_config(gen_args, "Generation")

    if sum([quant_args.do_ptq, quant_args.do_qat, quant_args.do_gptq, training_args.do_train]) > 1:
        raise ValueError(
            "--do_train, --do_ptq, --do_gptq and --do_qat cannot work at the same time. Please choose only one at a time"
        )

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Load model
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        elif training_args.bf16:
            dtype = "bfloat16"
        else:
            raise ValueError("Please specific dtype: --fp16 or --bf16")
    else:
        dtype = "float32"

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        tensor_parallel_output=False,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        dtype=dtype,
    )
    # Alreday shift label & logit in convert example
    # lm_shift_labels should be set before model initilization for some models(ex. llama)
    if hasattr(model_config, "lm_shift_labels"):
        model_config.lm_shift_labels = False
    if hasattr(model_config, "use_flash_attention"):
        model_config.use_flash_attention = model_args.use_flash_attention
    if hasattr(model_config, "max_position_embeddings"):
        if model_config.max_position_embeddings < data_args.src_length + data_args.tgt_length:
            raise ValueError(
                f"The src_length + tgt_length ({data_args.src_length + data_args.tgt_length}) must be smaller than max_position_embeddings({model_config.max_position_embeddings})."
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
    )

    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if model.base_model_prefix == "llama":
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"

    if data_args.dataset_name_or_path is None:
        raise ValueError(f"Please specific dataset name or path (got {data_args.dataset_name_or_path})")
    elif os.path.exists(os.path.join(data_args.dataset_name_or_path, "train.json")) and os.path.exists(
        os.path.join(data_args.dataset_name_or_path, "dev.json")
    ):
        train_ds = load_dataset(
            read_local_dataset, path=os.path.join(data_args.dataset_name_or_path, "train.json"), lazy=False
        )
        dev_ds = load_dataset(
            read_local_dataset, path=os.path.join(data_args.dataset_name_or_path, "dev.json"), lazy=False
        )
    else:
        if data_args.task_name is not None:
            train_ds, dev_ds = load_dataset(
                data_args.dataset_name_or_path, data_args.task_name, splits=["train", "dev"]
            )
        else:
            train_ds, dev_ds = load_dataset(data_args.dataset_name_or_path, splits=["train", "dev"])
    trans_func = partial(get_convert_example(model), tokenizer=tokenizer, data_args=data_args)
    if data_args.intokens:
        if model.base_model_prefix not in ["llama", "bloom"]:
            raise NotImplementedError("InTokens data stream is only implemented for LLaMA、 Bloom so far.")
    train_ds = train_ds.map(partial(trans_func, is_test=False, intokens=data_args.intokens))
    eval_intokens = data_args.intokens
    if data_args.intokens and data_args.eval_with_do_generation:
        logger.warning(
            "`intokens` conflicts with `eval_with_do_generation`. Setting intokens to False for the eval_dataset."
        )
        eval_intokens = False
    dev_ds = dev_ds.map(partial(trans_func, is_test=data_args.eval_with_do_generation, intokens=eval_intokens))
    if data_args.intokens:
        from paddlenlp.datasets import InTokensMapDataset

        logger.info("Creating InTokens Data Stream. This may take a few minutes.")
        train_ds = InTokensMapDataset(
            train_ds,
            tokenizer=tokenizer,
            max_length=data_args.intokens_max_length,
        )
        if eval_intokens:
            dev_ds = InTokensMapDataset(
                dev_ds,
                tokenizer=tokenizer,
                max_length=data_args.intokens_max_length,
            )

    if model_args.prefix_tuning:
        prefix_tuning_params = get_prefix_tuning_params(model)
        prefix_config = PrefixConfig(
            num_prefix_tokens=model_args.num_prefix_tokens,
            num_attention_heads=prefix_tuning_params["num_attention_heads"],
            num_hidden_layers=prefix_tuning_params["num_hidden_layers"],
            hidden_size=prefix_tuning_params["hidden_size"],
            multi_query_group_num=prefix_tuning_params["multi_query_group_num"],
            dtype=dtype,
        )
        model = PrefixModelForCausalLM(
            model=model,
            prefix_config=prefix_config,
            pad_attention_mask=prefix_tuning_params["pad_attention_mask"],
            postprocess_past_key_value=prefix_tuning_params["postprocess_past_key_value"],
        )
        model.mark_only_prefix_as_trainable()
        model.print_trainable_parameters()

    if model_args.lora:
        if model_args.lora_path is None:
            target_modules = get_lora_target_modules(model)
            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=2 * model_args.lora_rank,
                merge_weights=model_args.lora_merge_weights,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                dtype=dtype,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)
        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

    def compute_metrics_do_generation(eval_preds):
        rouge1 = Rouge1()
        rouge2 = Rouge2()
        rougel = RougeL()
        bleu4 = BLEU(n_size=4)

        predictions = [x[x != -100].tolist() for x in eval_preds.predictions]
        references = [x[x != -100].tolist() for x in eval_preds.label_ids]

        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        references = tokenizer.batch_decode(references, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if data_args.save_generation_output:
            with open(os.path.join(training_args.output_dir, "generated_output.json"), "w", encoding="utf-8") as f:
                for pred, ref in zip(predictions, references):
                    out = {"output": pred, "tgt": ref}
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")

        # for pred in predictions:
        rouge1_score = rouge1.score(predictions, references)
        rouge2_score = rouge2.score(predictions, references)
        for pred, ref in zip(predictions, references):
            rougel.add_inst(pred, [ref])
            bleu4.add_inst(pred, [ref])
        return {
            "rouge1": rouge1_score,
            "rouge2": rouge2_score,
            "rougel": rougel.score(),
            "bleu4": bleu4.score(),
        }

    # Create trainer
    trainer = CausalLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_do_generation if data_args.eval_with_do_generation else compute_metrics,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            max_length=data_args.src_length + data_args.tgt_length,
            padding=True,
            return_tensors="np",
        ),
        do_generation=data_args.eval_with_do_generation,
        gen_args=gen_args,
        data_args=data_args,
    )

    # Train
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # QAT
    if quant_args.do_qat:
        if training_args.tensor_parallel_degree > 1:
            raise NotImplementedError("Only support qat on single gpu.")
        from quant import create_qat_model

        trainer.model = create_qat_model(quant_args, trainer.model, dtype)
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("qat", train_result.metrics)
        trainer.save_metrics("qat", train_result.metrics)
        trainer.save_state()

    # PTQ
    if quant_args.do_ptq:
        if isinstance(model, LoRAModel):
            raise NotImplementedError(
                "PTQ strategy not supported for LoRA model. Please merge lora parameters to pretrain model first."
            )
        from quant import apply_ptq, apply_shift, apply_smooth, get_ptq_model_config

        trainer.model.eval()
        # Prepare ptq dataloader
        if os.path.exists(os.path.join(data_args.dataset_name_or_path, "ptq.json")):
            ptq_ds = load_dataset(
                read_local_dataset, path=os.path.join(data_args.dataset_name_or_path, "ptq.json"), lazy=False
            )
            ptq_ds = ptq_ds.map(partial(trans_func, is_test=False))
        else:
            ptq_ds = train_ds
            logger.info(
                f"Not found ptq.json in {data_args.dataset_name_or_path}. Set train dataset as PTQ calibration dataset."
            )
        ptq_dataloader = trainer.get_ptq_dataloader(ptq_ds)
        if quant_args.shift or quant_args.smooth:
            ptq_model_config = get_ptq_model_config(trainer.model)

        if quant_args.shift:
            apply_shift(quant_args, trainer, ptq_dataloader, ptq_model_config)

        if quant_args.smooth:
            apply_smooth(quant_args, trainer, ptq_dataloader, ptq_model_config)

        apply_ptq(quant_args, trainer, ptq_dataloader)

    if quant_args.do_gptq:
        if isinstance(model, LoRAModel):
            raise NotImplementedError(
                "PTQ strategy not supported for LoRA model. Please merge lora parameters to pretrain model first."
            )
        from quant import apply_gptq

        apply_gptq(quant_args, trainer, ptq_dataloader)

    # Evaluation dev set
    if training_args.do_eval:
        eval_result = trainer.evaluate(dev_ds)
        trainer.log_metrics("eval", eval_result)

    # Evaluation test set
    if training_args.do_predict:
        test_ds = load_dataset(
            read_local_dataset, path=os.path.join(data_args.dataset_name_or_path, "test.json"), lazy=False
        )
        test_ds = test_ds.map(partial(trans_func, is_test=data_args.eval_with_do_generation))
        eval_result = trainer.predict(test_ds).metrics
        trainer.log_metrics("test", eval_result)


if __name__ == "__main__":
    main()
