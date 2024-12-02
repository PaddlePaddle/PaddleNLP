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
from utils.argument import GenerateArgument
from utils.data import get_convert_example

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import (
    ZeroPaddingIterableDataset,
    ZeroPaddingMapDataset,
    load_dataset,
)
from paddlenlp.metrics import BLEU, Rouge1, Rouge2, RougeL
from paddlenlp.peft import LoRAModel
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.trainer.trainer_callback import TrainerState
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
    AutoTokenizer,
    Llama3Tokenizer,
    LlamaForCausalLM,
    LlamaForCausalLMPipe,
    LlamaTokenizer,
    Qwen2ForCausalLM,
    Qwen2ForCausalLMPipe,
    register_sequence_parallel_allreduce_hooks,
)
from paddlenlp.transformers.configuration_utils import LlmMetaConfig
from paddlenlp.trl import DataConfig, ModelConfig, QuantConfig, SFTConfig, SFTTrainer
from paddlenlp.trl.llm_utils import (
    ZeroPaddingIterDatasetCallback,
    compute_metrics,
    init_chat_template,
)
from paddlenlp.utils.log import logger
from paddlenlp.utils.tools import get_env_device

# Fine-tune Environment Variables to support sharding stage1 overlap optimization.
os.environ["USE_CASUAL_MASK"] = "False"

flash_mask_support_list = [LlamaForCausalLM, LlamaForCausalLMPipe, Qwen2ForCausalLM, Qwen2ForCausalLMPipe]


def main():
    parser = PdArgumentParser((GenerateArgument, QuantConfig, ModelConfig, DataConfig, SFTConfig))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        gen_args, quant_args, model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    else:
        gen_args, quant_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(quant_args, "Quant")
    training_args.print_config(gen_args, "Generation")

    if sum([quant_args.do_ptq, quant_args.do_qat, quant_args.do_gptq]) > 1:
        raise ValueError(
            "--do_ptq, --do_gptq and --do_qat cannot work at the same time. Please choose only one at a time"
        )

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    if get_env_device() == "xpu" and training_args.gradient_accumulation_steps > 1:
        try:
            from paddle_xpu.layers.nn.linear import LinearConfig  # noqa: F401

            LinearConfig.enable_accumulate_steps_opt()
            LinearConfig.set_accumulate_steps(training_args.gradient_accumulation_steps)
        except ImportError:
            # It's OK, not use accumulate_steps optimization
            pass

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
    quantization_config = dict(
        weight_quantize_algo=model_args.weight_quantize_algo,
        weight_blocksize=model_args.weight_blocksize,
        weight_double_quant=model_args.weight_double_quant,
        weight_double_quant_block_size=model_args.weight_double_quant_block_size,
    )

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        from_aistudio=model_args.from_aistudio,
        quantization_config=quantization_config,
    )

    LlmMetaConfig.set_llm_config(model_config, training_args)
    model_config.use_fast_layer_norm = model_args.use_fast_layer_norm

    # Config for model using dropout, such as GPT.
    if hasattr(model_config, "hidden_dropout_prob"):
        model_config.hidden_dropout_prob = model_args.hidden_dropout_prob
    if hasattr(model_config, "attention_probs_dropout_prob"):
        model_config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob
    if hasattr(model_config, "ignore_index"):
        model_config.ignore_index = -100

    if model_args.fuse_attention_qkv is not None:
        model_config.fuse_attention_qkv = model_args.fuse_attention_qkv
    if model_args.fuse_attention_ffn is not None:
        model_config.fuse_attention_ffn = model_args.fuse_attention_ffn

    model_config.seq_length = data_args.max_length

    logger.info(f"Final model config: {model_config}")

    model_class = AutoModelForCausalLM
    if training_args.pipeline_parallel_degree > 1:
        if data_args.eval_with_do_generation and training_args.do_eval:
            raise ValueError("Plese set eval_with_do_generation to false in pipeline parallel mode.")

        model_class = AutoModelForCausalLMPipe

    if model_args.continue_training and not training_args.autotuner_benchmark:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            from_aistudio=model_args.from_aistudio,
        )
    else:
        # NOTE(gongenlei): new add autotuner_benchmark
        model = model_class.from_config(model_config, dtype=dtype)

    if model_args.flash_mask and (not data_args.zero_padding or not model.config.use_flash_attention):
        logger.warning("`flash_mask` must use with zero padding and flash attention.")
        data_args.zero_padding = True
        model.config.use_flash_attention = True

    if model_args.flash_mask and not any(isinstance(model, cls) for cls in flash_mask_support_list):
        raise NotImplementedError(f"{model.__class__} not support flash mask.")

    if training_args.sequence_parallel:
        register_sequence_parallel_allreduce_hooks(
            model, training_args.gradient_accumulation_steps, training_args.fuse_sequence_parallel_allreduce
        )
    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, from_aistudio=model_args.from_aistudio)
    # init chat_template for tokenizer
    init_chat_template(tokenizer, model_args.model_name_or_path, data_args.chat_template)

    # if using chat_template, data_args.eval_with_do_generation must be false
    if tokenizer.chat_template is not None:
        data_args.eval_with_do_generation = False

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, Llama3Tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.dataset_name_or_path is None:
        raise ValueError(f"Please specific dataset name or path (got {data_args.dataset_name_or_path})")
    elif (
        os.path.exists(os.path.join(data_args.dataset_name_or_path, "train.json"))
        or os.path.exists(os.path.join(data_args.dataset_name_or_path, "dev.json"))
        or os.path.exists(os.path.join(data_args.dataset_name_or_path, "quant.json"))
    ):
        if quant_args.do_qat:
            train_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "train.json"),
                lazy=data_args.lazy,
            )[0]
        else:
            train_ds = None
        if training_args.do_eval:
            dev_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "dev.json"),
                lazy=data_args.lazy,
            )[0]
        else:
            dev_ds = None

        if os.path.exists(os.path.join(data_args.dataset_name_or_path, "quant.json")):
            ptq_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "quant.json"),
                lazy=data_args.lazy,
            )[0]
        elif os.path.exists(os.path.join(data_args.dataset_name_or_path, "train.json")):
            ptq_ds = load_dataset(
                "json",
                data_files=os.path.join(data_args.dataset_name_or_path, "train.json"),
                lazy=data_args.lazy,
            )[0]
            logger.info(
                f"Not found quant.json in {data_args.dataset_name_or_path}. Set train dataset as PTQ calibration dataset."
            )
        else:
            raise ValueError(f"Quant strategy requires quant.json or train.json in {data_args.dataset_name_or_path}")

    elif (
        os.path.exists(os.path.join(data_args.dataset_name_or_path, "train"))
        or os.path.exists(os.path.join(data_args.dataset_name_or_path, "dev"))
        or os.path.exists(os.path.join(data_args.dataset_name_or_path, "quant"))
    ):
        import glob

        if quant_args.do_qat:
            train_ds = load_dataset(
                "json",
                data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "train", "*.json")),
                lazy=data_args.lazy,
            )[0]
        else:
            train_ds = None
        if training_args.do_eval:
            dev_ds = load_dataset(
                "json",
                data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "dev", "*.json")),
                lazy=data_args.lazy,
            )[0]
        else:
            dev_ds = None

        if os.path.exists(os.path.join(data_args.dataset_name_or_path, "quant")):
            ptq_ds = load_dataset(
                "json",
                data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "quant", "*.json")),
                lazy=data_args.lazy,
            )[0]
        elif os.path.exists(os.path.join(data_args.dataset_name_or_path, "train")):
            ptq_ds = load_dataset(
                "json",
                data_files=glob.glob(os.path.join(data_args.dataset_name_or_path, "train", "*.json")),
                lazy=data_args.lazy,
            )[0]
            logger.info(
                f"Not found quant.json in {data_args.dataset_name_or_path}. Set train dataset as PTQ calibration dataset."
            )
        else:
            raise ValueError(f"Quant strategy requires quant or train folder in {data_args.dataset_name_or_path}")

    else:
        if quant_args.do_qat:
            train_ds = load_dataset(data_args.dataset_name_or_path, splits=["train"])[0]
        else:
            train_ds = None
        if training_args.do_eval:
            dev_ds = load_dataset(data_args.dataset_name_or_path, splits=["dev"])[0]
        else:
            dev_ds = None
        if quant_args.do_ptq or quant_args.do_gptq or quant_args.load_quant_model:
            ptq_ds = load_dataset(data_args.dataset_name_or_path, splits=["train"])[0]
            logger.info("Set train dataset as PTQ calibration dataset.")
        else:
            ptq_ds = None
    # TODO(ZHUI & sijunhe): Temporary implementation. Generalize this logic and move to Trainer later.
    if training_args.resume_from_checkpoint is not None and data_args.lazy:
        logger.info(
            f"Loading from '{training_args.resume_from_checkpoint}' with `lazy=True`, manually skipping dataset and setting `ignore_data_skip` to True."
        )
        training_args.ignore_data_skip = True
        state = TrainerState.load_from_json(os.path.join(training_args.resume_from_checkpoint, "trainer_state.json"))
        if state.trial_params is not None and "zero_padding_global_step" in state.trial_params:
            consumed_samples = state.trial_params["zero_padding_global_step"]
        else:
            consumed_samples = (
                state.global_step
                * training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.dataset_world_size
            )
        logger.info(
            f"Skipping the first {consumed_samples} samples to warmup the dataset from checkpoint '{training_args.resume_from_checkpoint}'."
        )
        train_ds = train_ds.skip(consumed_samples)

    if training_args.pipeline_parallel_degree > 1:
        from utils.data import convert_example_common

        trans_func = partial(convert_example_common, tokenizer=tokenizer, data_args=data_args)
    else:
        trans_func = partial(get_convert_example(model), tokenizer=tokenizer, data_args=data_args)

    train_ds = (
        train_ds.map(
            partial(trans_func, is_test=False, zero_padding=data_args.zero_padding, flash_mask=model_args.flash_mask)
        )
        if train_ds is not None
        else None
    )
    ptq_ds = (
        ptq_ds.map(
            partial(trans_func, is_test=False, zero_padding=data_args.zero_padding, flash_mask=model_args.flash_mask)
        )
        if ptq_ds is not None
        else None
    )
    eval_zero_padding = data_args.zero_padding
    if data_args.zero_padding and data_args.eval_with_do_generation:
        logger.warning(
            "`zero_padding` conflicts with `eval_with_do_generation`. Setting zero_padding to False for the eval_dataset."
        )
        eval_zero_padding = False
    dev_ds = (
        dev_ds.map(
            partial(
                trans_func,
                is_test=data_args.eval_with_do_generation,
                zero_padding=eval_zero_padding,
                flash_mask=model_args.flash_mask,
            )
        )
        if dev_ds is not None
        else None
    )
    if data_args.zero_padding:
        if data_args.lazy:
            intoken_dataset = ZeroPaddingIterableDataset
        else:
            intoken_dataset = ZeroPaddingMapDataset
        logger.info("Creating Zero Padding Data Stream. This may take a few minutes.")
        train_ds = (
            intoken_dataset(
                train_ds,
                tokenizer=tokenizer,
                max_length=data_args.max_length,
                greedy_zero_padding=data_args.greedy_zero_padding,
            )
            if train_ds is not None
            else None
        )
        ptq_ds = (
            intoken_dataset(
                ptq_ds,
                tokenizer=tokenizer,
                max_length=data_args.max_length,
                greedy_zero_padding=data_args.greedy_zero_padding,
            )
            if ptq_ds is not None
            else None
        )

        if eval_zero_padding:
            dev_ds = (
                intoken_dataset(
                    dev_ds,
                    tokenizer=tokenizer,
                    max_length=data_args.max_length,
                )
                if dev_ds is not None
                else None
            )

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

    if (
        training_args.pipeline_parallel_degree > 1
        or training_args.sequence_parallel
        or training_args.autotuner_benchmark
        or data_args.zero_padding
        or data_args.pad_to_max_length
    ):
        # NOTE(gongenlei): new add autotuner_benchmark
        max_length = data_args.max_length
        padding = "max_length"
    else:
        max_length = None
        padding = True

    if training_args.pipeline_parallel_degree > 1:
        metrics = None
    elif data_args.eval_with_do_generation:
        metrics = compute_metrics_do_generation
    else:
        metrics = compute_metrics

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            max_label_length=max_length,
            return_tensors="np",
            return_attention_mask=not model_args.flash_mask,
            pad_to_multiple_of=data_args.pad_to_multiple_of,
        ),
        do_generation=data_args.eval_with_do_generation,
        callbacks=[ZeroPaddingIterDatasetCallback()] if isinstance(train_ds, ZeroPaddingIterableDataset) else None,
        gen_args=gen_args,
        data_args=data_args,
    )
    trainable_parameters = [p for p in model.parameters() if not p.stop_gradient]
    trainer.set_optimizer_grouped_parameters(trainable_parameters)

    # QAT
    if quant_args.do_qat:
        from utils.quant import create_qat_model

        trainer.model = create_qat_model(quant_args, trainer.model, dtype)
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.log_metrics("qat", train_result.metrics)
        trainer.save_metrics("qat", train_result.metrics)
        trainer.save_state()

    # PTQ
    if quant_args.do_ptq:
        if isinstance(model, LoRAModel):
            raise NotImplementedError(
                "PTQ strategy not supported for LoRA model. Please merge lora parameters to pretrain model first."
            )
        from utils.quant import (
            apply_autoclip,
            apply_ptq,
            apply_shift,
            apply_smooth,
            get_ptq_model_config,
        )

        trainer.model.eval()
        trainer.model.config.quantization_config.quant_type = quant_args.quant_type
        trainer.model.config.quantization_config.smooth = quant_args.smooth
        trainer.model.config.quantization_config.shift = quant_args.shift
        trainer.model.config.quantization_config.shift_smooth_all_linears = (
            quant_args.smooth_all_linears or quant_args.shift_all_linears
        )
        ptq_dataloader = trainer.get_ptq_dataloader(ptq_ds)
        if quant_args.shift or quant_args.smooth:
            ptq_model_config = get_ptq_model_config(trainer.model)

        if quant_args.shift:
            apply_shift(quant_args, trainer, ptq_dataloader, ptq_model_config)

        if quant_args.smooth:
            apply_smooth(quant_args, trainer, ptq_dataloader, ptq_model_config)

        if quant_args.auto_clip:
            apply_autoclip(quant_args, trainer, ptq_dataloader)

        apply_ptq(quant_args, trainer, ptq_dataloader)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)

    if quant_args.do_gptq:
        if isinstance(model, LoRAModel):
            raise NotImplementedError(
                "PTQ strategy not supported for LoRA model. Please merge lora parameters to pretrain model first."
            )
        from utils.quant import apply_gptq

        ptq_dataloader = trainer.get_ptq_dataloader(ptq_ds)
        apply_gptq(quant_args, trainer, ptq_dataloader)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)

    # Evaluation test set
    if training_args.do_predict:
        test_ds = load_dataset(
            "json",
            data_files=os.path.join(data_args.dataset_name_or_path, "test.json"),
            lazy=data_args.lazy,
        )[0]

        test_ds = test_ds.map(partial(trans_func, is_test=data_args.eval_with_do_generation))
        if eval_zero_padding:
            test_ds = intoken_dataset(
                test_ds,
                tokenizer=tokenizer,
                max_length=data_args.max_length,
            )
        eval_result = trainer.predict(test_ds).metrics
        trainer.log_metrics("test", eval_result)

    if quant_args.load_quant_model and not quant_args.do_ptq:
        if isinstance(model, LoRAModel):
            raise NotImplementedError(
                "PTQ strategy not supported for LoRA model. Please merge lora parameters to pretrain model first."
            )
        from utils.quant import (
            apply_autoclip,
            apply_ptq,
            apply_shift,
            apply_smooth,
            get_ptq_model_config,
            load_quant_model,
        )

        trainer.model.eval()
        trainer.model.config.quantization_config.quant_type = quant_args.quant_type
        trainer.model.config.quantization_config.smooth = quant_args.smooth
        trainer.model.config.quantization_config.shift = quant_args.shift
        trainer.model.config.quantization_config.shift_smooth_all_linears = (
            quant_args.smooth_all_linears or quant_args.shift_all_linears
        )
        ptq_dataloader = trainer.get_ptq_dataloader(ptq_ds)
        if quant_args.shift or quant_args.smooth:
            ptq_model_config = get_ptq_model_config(trainer.model)

        if quant_args.shift:
            apply_shift(quant_args, trainer, ptq_dataloader, ptq_model_config)

        if quant_args.smooth:
            apply_smooth(quant_args, trainer, ptq_dataloader, ptq_model_config)

        load_quant_model(trainer.model, quant_args, training_args.output_dir)

    # Evaluation dev set
    if training_args.do_eval:

        logger.info("*** Evaluate result after ptq/qat/ etc.***")
        eval_result = trainer.evaluate(dev_ds)
        trainer.log_metrics("eval", eval_result)


if __name__ == "__main__":
    main()
