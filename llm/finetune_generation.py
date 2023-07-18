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

import os
from functools import partial

import paddle
from argument import DataArgument, GenerateArgument, ModelArgument, QuantArgument
from data import get_convert_example, read_local_dataset
from utils import (
    CausalLMTrainer,
    compute_metrics,
    get_lora_params,
    get_prefix_tuning_params,
    get_ptq_model_config,
)

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import BLEU, Rouge1, Rouge2, RougeL
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from paddlenlp.utils.log import logger


def main():
    # Arguments
    parser = PdArgumentParser((GenerateArgument, QuantArgument, ModelArgument, DataArgument, TrainingArguments))
    gen_args, quant_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(quant_args, "Quant")
    training_args.print_config(gen_args, "Generation")
    if quant_args.do_ptq and quant_args.do_qat:
        raise ValueError("PTQ and QAT can not work at the same time.")

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

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        lm_shift_labels=False,
    )

    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
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
    train_ds = train_ds.map(partial(trans_func, is_test=False))
    dev_ds = dev_ds.map(partial(trans_func, is_test=data_args.eval_with_do_generation))

    if model_args.prefix_tuning:
        from paddlenlp.peft import PrefixConfig, PrefixModelForCausalLM

        prefix_tuning_params = get_prefix_tuning_params(model)
        prefix_config = PrefixConfig(
            num_prefix_tokens=model_args.num_prefix_tokens,
            num_attention_heads=prefix_tuning_params[0],
            num_hidden_layers=prefix_tuning_params[1],
            hidden_size=prefix_tuning_params[2],
            dtype=dtype,
        )
        model = PrefixModelForCausalLM(
            model=model,
            prefix_config=prefix_config,
            pad_attention_mask=prefix_tuning_params[3],
            postprocess_past_key_value=prefix_tuning_params[4],
        )
        model.mark_only_prefix_as_trainable()
        model.print_trainable_parameters()

    if model_args.lora or model_args.lora_path is not None:
        from paddlenlp.peft import LoRAConfig, LoRAModel

        if model_args.lora_path is None:
            lora_params = get_lora_params(model, is_tp=training_args.tensor_parallel_degree > 1)
            lora_config = LoRAConfig(
                target_modules=lora_params[0],
                r=model_args.lora_rank,
                lora_alpha=2 * model_args.lora_rank,
                merge_weights=model_args.lora_merge_weights,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                dtype=dtype,
                head_dim=lora_params[1],
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)
        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

    if quant_args.do_qat:
        from paddle import nn
        from paddle.quantization import QAT, QuantConfig

        # FakeQuanterChannelWiseAbsMaxObserver not yet merge in Paddle develop
        from paddle.quantization.quanters import FakeQuanterChannelWiseAbsMaxObserver
        from paddle.quantization.quanters.abs_max import (
            FakeQuanterWithAbsMaxObserverLayer,
        )
        from paddleslim.quant.quanters import PACTQuanter

        from paddlenlp.peft.lora import LoRALinear
        from paddlenlp.peft.lora.lora_quant_layers import QuantedLoRALinear

        q_config = QuantConfig(activation=None, weight=None)
        q_config.add_qat_layer_mapping(LoRALinear, QuantedLoRALinear)
        if model_args.qat_type == "A8W8":
            activation = PACTQuanter(quanter=FakeQuanterWithAbsMaxObserverLayer, init_value=20, dtype=dtype)
            weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=8, dtype="float32")
        elif model_args.qat_type == "W4":
            activation = None
            weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=4, dtype="float32")
        elif model_args.qat_type == "A8W4":
            activation = PACTQuanter(quanter=FakeQuanterWithAbsMaxObserverLayer, init_value=20, dtype=dtype)
            weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=4, dtype="float32")
        else:
            raise ValueError("qat_type should be one of ['A8W8', 'W4', 'A8W4']")
        q_config.add_type_config(LoRALinear, weight=weight, activation=activation)
        q_config.add_type_config(nn.Linear, weight=weight, activation=activation)

        qat = QAT(q_config)
        model = qat.quantize(model, inplace=True)

    def compute_metrics_do_generation(eval_preds):
        rouge1 = Rouge1()
        rouge2 = Rouge2()
        rougel = RougeL()
        bleu4 = BLEU(n_size=4)

        predictions = [x[x != -100].tolist() for x in eval_preds.predictions]
        references = [x[x != -100].tolist() for x in eval_preds.label_ids]

        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        references = tokenizer.batch_decode(references, skip_special_tokens=True, clean_up_tokenization_spaces=False)

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

    # Train & QAT
    if training_args.do_train or quant_args.do_qat:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # PTQ
    if quant_args.do_ptq:
        from paddle.distributed.fleet.meta_parallel import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
        from paddle.quantization import PTQ, QuantConfig
        from paddleslim.quant.advanced import (
            EMASampler,
            MultiStepSampler,
            PieceWiseSearch,
            Shift,
            Smooth,
        )
        from paddleslim.quant.layers import (
            QuantizedColumnParallelLinear,
            QuantizedRowParallelLinear,
        )
        from paddleslim.quant.observers import (
            AbsMaxChannelWiseWeightObserver,
            AbsmaxObserver,
        )

        from paddlenlp.peft.lora.lora_quant_layers import QuantedLoRALinear

        trainer.model.eval()
        # prepare ptq dataloader
        if os.path.exists(os.path.join(data_args.dataset_name_or_path, "ptq.json")):
            ptq_ds = load_dataset(
                read_local_dataset, path=os.path.join(data_args.dataset_name_or_path, "ptq.json"), lazy=False
            )
        else:
            ptq_ds = train_ds
            logger.info(
                f"Not found ptq.json in {data_args.dataset_name_or_path}. Set train dataset to PTQ dataset path."
            )
        ptq_ds = ptq_ds.map(partial(trans_func, is_test=False))
        ptq_dataloader = trainer.get_ptq_dataloader(ptq_ds)
        ptq_model_config = get_ptq_model_config(trainer.model)

        if quant_args.shift:
            shift_sampler = EMASampler() if quant_args.shift_sampler == "ema" else None
            shift = Shift(
                model=trainer.model,
                model_config=ptq_model_config,
                sample_function=shift_sampler,
                shift_all_linears=quant_args.shift_all_linears,
            )

            trainer.ptq_loop(
                ptq_dataloader,
                description="Shift",
                max_eval_iters=quant_args.shift_step,
            )
            shift.update_weight()
            del shift, shift_sampler

        if quant_args.do_smooth:
            smooth_sampler = MultiStepSampler() if quant_args.smooth_sampler == "multi_step" else None
            if quant_args.smooth_piecewise_search:
                search_func = PieceWiseSearch(
                    k_piece=quant_args.smooth_k_piece,
                    bits_length=8,
                    search_piece=quant_args.smooth_search_piece,
                    search_alpha_min=0.2,
                    search_alpha_max=0.8,
                    search_scale_min=1.0,
                    search_scale_max=5.0,
                    weight_quant_method="abs_max_channel_wise",
                    act_quant_method="abs_max",
                )
            else:
                search_func = None
            smooth = Smooth(
                trainer.model,
                ptq_model_config,
                alpha=0.5,
                smooth_all_linears=quant_args.smooth_all_linears,
                sample_function=smooth_sampler,
                search_function=search_func,
            )
            trainer.ptq_loop(
                ptq_dataloader,
                description="Smooth",
                max_eval_iters=quant_args.smooth_step,
            )

            smooth.update_weight()
            del smooth, smooth_sampler, search_func

        q_config = QuantConfig(activation=None, weight=None)
        act_quanter = AbsmaxObserver()
        weight_quanter = AbsMaxChannelWiseWeightObserver()
        q_config.add_qat_layer_mapping(LoRALinear, QuantedLoRALinear)
        q_config.add_qat_layer_mapping(ColumnParallelLinear, QuantizedColumnParallelLinear)
        q_config.add_qat_layer_mapping(RowParallelLinear, QuantizedRowParallelLinear)
        q_config.add_type_config(
            [paddle.nn.Linear, ColumnParallelLinear, RowParallelLinear, QuantedLoRALinear],
            activation=act_quanter,
            weight=weight_quanter,
        )

        ptq = PTQ(q_config)
        trainer.model = ptq.quantize(trainer.model, inplace=True)
        trainer.ptq_loop(
            ptq_dataloader,
            description="PTQ",
            max_eval_iters=quant_args.ptq_step,
        )
        trainer.model = ptq.convert(trainer.model, inplace=True)

    # Evaluation
    if training_args.do_eval:
        eval_result = trainer.evaluate(dev_ds)
        trainer.log_metrics("eval", eval_result)

        test_ds = load_dataset(
            read_local_dataset, path=os.path.join(data_args.dataset_name_or_path, "dev.json"), lazy=False
        )
        trans_func = partial(get_convert_example(model), tokenizer=tokenizer, data_args=data_args)
        test_ds = test_ds.map(partial(trans_func, is_test=True))
        trainer.do_generation = True
        trainer.compute_metrics = compute_metrics_do_generation
        eval_result = trainer.evaluate(test_ds)
        trainer.log_metrics("test", eval_result)


if __name__ == "__main__":

    main()
