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

# Adapted from https://github.com/ymcui/Chinese-LLaMA-Alpaca and https://github.com/SJTU-LIT/ceval
import argparse
import json
import os
import time

import paddle
import pandas as pd
from model_evaluator import ModelEvaluator
from paddle.distributed import fleet

choices = ["A", "B", "C", "D"]
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddle.quantization import PTQ, QAT, QuantConfig
from paddleslim.quant.advanced import (  # Outliers_Search,; ReorderFFNWeight,
    GPTQ,
    AutoClip,
    AWQSearch,
    EMASampler,
    MultiStepSampler,
    PieceWiseSearch,
    Shift,
    Smooth,
)
from paddleslim.quant.advanced.utils import find_parent_layer_and_sub_name
from paddleslim.quant.layers import (
    QuantizedColumnParallelLinear,
    QuantizedRowParallelLinear,
)
from paddleslim.quant.observers import (
    AbsMaxChannelWiseWeightObserver,
    AbsmaxObserver,
    AVGObserver,
    KLObserver,
)


def run_ptq(args, evaluator):

    if args.model_name_or_path == "THUDM/chatglm-6b":
        model_config = {"fused_qkv": False, "parallel_ffn": True}
    elif args.model_name_or_path == "bigscience/bloomz-7b1-mt":
        model_config = {"fused_qkv": True, "parallel_ffn": False}
    elif "bloomz" in args.model_name_or_path:
        model_config = {"fused_qkv": True, "parallel_ffn": False}
    elif "lama" in args.model_name_or_path:
        model_config = {"fused_qkv": False, "parallel_ffn": True}
    elif "Baichuan2" in args.model_name_or_path:
        model_config = {"fused_qkv": False, "parallel_ffn": True}
    elif "qwen" in args.model_name_or_path:
        model_config = {"fused_qkv": False, "parallel_ffn": True}
    elif args.model_name_or_path == "idea-ccnl/ziya-llama-13b-v1":
        model_config = {"fused_qkv": False, "parallel_ffn": True}
    elif args.model_name_or_path == "THUDM/chatglm2-6b":
        model_config = {"fused_qkv": False, "skip_norm_list": ["rms_norm_56"], "parallel_ffn": False}
    evaluator.model.eval()

    if args.do_shift:  # from quant.py/apply_shift
        print(f"\n=======Start Shift=======")
        print("Shift iters:", args.shift_iters)
        if args.shift_sampler == "ema":
            shift_sampler = EMASampler()
        else:
            shift_sampler = None
        shift = Shift(
            evaluator.model, model_config, sample_function=shift_sampler, shift_all_linears=args.shift_all_linears
        )
        eval_loop(args, evaluator, args.shift_iters)
        shift.update_weight()
        del shift, shift_sampler
        print(f"=======Shift Done=======")
    # if args.do_reorder:
    #     print(f"\n=======Start reorder=======")
    #     if 'bloomz' in args.model_name_or_path:
    #         llama_ffn=False
    #     else:
    #         llama_ffn=True
    #     reorder = ReorderFFNWeight(evaluator.model, layer_prefix="mlp", llama_ffn=llama_ffn)
    #     eval_loop_reorder(args, evaluator, args.ptq_iters)
    #     reorder.update_weight()
    #     del reorder
    #     print(f"=======reorder Done=======")

    if args.do_smooth:  # from quant.py/apply_smooth
        print("---------------Start Smooth---------------")
        if args.use_pw_search:
            search_func = PieceWiseSearch(
                k_piece=args.k_piece,
                bits_length=8,
                search_piece=args.search_piece,
                search_alpha_min=0.2,
                search_alpha_max=0.8,
                search_scale_min=1.0,
                search_scale_max=5.0,
                weight_quant_method="abs_max_channel_wise",
                act_quant_method="abs_max",
            )
        elif args.do_awq:
            smooth_method = "awq"
            search_func = AWQSearch(
                n_grid=20, bits_length=4, weight_quant_method=args.weight_quant_method, group_size=args.group_size
            )
        else:
            search_func = None
        if args.smooth_sampler == "multi_step":
            smooth_sampler = MultiStepSampler()
        else:
            if args.do_shift:
                # search_func=Outliers_Search()
                args.smooth_iters = 2
                smooth_sampler = None
            else:
                smooth_sampler = None
        start_sample_step = 0

        smooth = Smooth(
            evaluator.model,
            model_config,
            alpha=0.5,
            smooth_all_linears=args.smooth_all_linears,
            sample_function=smooth_sampler,
            search_function=search_func,
            start_sample_step=start_sample_step,
            smooth_method="awq" if args.do_awq else "smoothquant",
        )
        eval_loop(args, evaluator, args.smooth_iters)
        print("---------------Smooth Done---------------")
        smooth.update_weight()
        del smooth, smooth_sampler, search_func
    if args.do_autoclip:
        print("-------------------Start AutoClip------------------")
        sampler = MultiStepSampler()
        auto_clip = AutoClip(
            evaluator.model,
            weight_bits=4,
            weight_quant_method=args.weight_quant_method,
            sample_function=sampler,
            n_grid=20,
            max_shrink=0.5,
            group_size=args.group_size,
        )
        eval_loop(args, evaluator, 8)
        auto_clip.auto_clip()
        del sampler, auto_clip
        print("***** AutoClip done *****")

    if args.do_ptq or args.do_int4:
        print(f"\n=======begin PTQ=======")
        print("PTQ iters:", args.ptq_iters)
        quant_bits = 4 if args.do_int4 else 8
        evaluator.prepare_ptq(quant_bits, args=args)

        eval_loop(args, evaluator, args.ptq_iters)
        evaluator.model = evaluator.ptq.convert(evaluator.model, inplace=True)
        print(f"\n=======PTQ Done !!!=======")

    if args.do_gptq:
        print(f"\n=======begin GPTQ=======")
        print("gptq iters:", args.gptq_iters)
        num_layer = 0
        model = evaluator.model
        for cur_name, cur_layer in model.named_sublayers():
            if type(cur_layer) in [paddle.nn.Linear, ColumnParallelLinear, RowParallelLinear]:
                num_layer += 1
                print("GPTQ layer", num_layer, cur_name)
                parent_layer, sub_name = find_parent_layer_and_sub_name(model, cur_name)
                cur_quant_layer = GPTQ(cur_layer)
                setattr(parent_layer, sub_name, cur_quant_layer)
                eval_loop(args, evaluator, args.gptq_iters)
                cur_quant_layer.fasterquant(percdamp=0.1, groupsize=-1, actorder=True)
                del cur_quant_layer
                setattr(parent_layer, sub_name, cur_layer)
    paddle.device.cuda.empty_cache()


def eval_loop_reorder(args, evaluator, target_iters):
    assert os.path.exists("subject_mapping.json"), "subject_mapping.json not found!"
    with open("subject_mapping.json") as f:
        subject_mapping = json.load(f)
    filenames = os.listdir("data/val")
    subject_list = [val_file.replace("_val.csv", "") for val_file in filenames]
    accuracy, summary = {}, {}

    run_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    output_dir = args.output_dir

    start_iters = 0
    subject_list = sorted(subject_list)
    for index, subject_name in enumerate(subject_list):
        print(
            f"{index/len(subject_list)} Inference starts at {run_date} on {args.model_name_or_path} with subject of {subject_name}!"
        )
        val_file_path = os.path.join("data/val", f"{subject_name}_val.csv")
        dev_file_path = os.path.join("data/dev", f"{subject_name}_dev.csv")
        test_file_path = os.path.join("data/test", f"{subject_name}_test.csv")

        val_df = pd.read_csv(val_file_path) if args.do_test is False else pd.read_csv(test_file_path)
        dev_df = pd.read_csv(dev_file_path) if args.few_shot else None
        if index <= 23:
            t = start_iters + 6
        else:
            t = start_iters + 4
        if index == len(subject_list) - 1:
            t = target_iters
        print("sampled", t - start_iters)
        start_iters = evaluator.ptq_subject(
            subject_name,
            val_df,
            dev_df,
            few_shot=args.few_shot,
            cot=args.cot,
            with_prompt=args.with_prompt,
            constrained_decoding=args.constrained_decoding,
            do_test=args.do_test,
            start_iters=start_iters,
            target_iters=t,
        )
        if start_iters >= target_iters:
            break
    print("Total iters:", start_iters)


def eval_loop(args, evaluator, target_iters):
    assert os.path.exists("subject_mapping.json"), "subject_mapping.json not found!"
    with open("subject_mapping.json") as f:
        subject_mapping = json.load(f)
    filenames = os.listdir("data/val")
    subject_list = [val_file.replace("_val.csv", "") for val_file in filenames]
    accuracy, summary = {}, {}

    run_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    output_dir = args.output_dir
    # paddle.flops(evaluator.model, input_size, custom_ops=None, print_detail=True)

    start_iters = 0
    subject_list = sorted(subject_list)
    for index, subject_name in enumerate(subject_list):
        print(
            f"{index/len(subject_list)} Inference starts at {run_date} on {args.model_name_or_path} with subject of {subject_name}!"
        )
        val_file_path = os.path.join("data/val", f"{subject_name}_val.csv")
        dev_file_path = os.path.join("data/dev", f"{subject_name}_dev.csv")
        test_file_path = os.path.join("data/test", f"{subject_name}_test.csv")

        val_df = pd.read_csv(val_file_path) if args.do_test is False else pd.read_csv(test_file_path)
        dev_df = pd.read_csv(dev_file_path) if args.few_shot else None
        if index <= 23:
            t = start_iters + 3
        else:
            t = start_iters + 2
        if index == len(subject_list) - 1:
            t = target_iters
        print("sampled", t - start_iters)
        start_iters = evaluator.ptq_subject(
            subject_name,
            val_df,
            dev_df,
            few_shot=args.few_shot,
            cot=args.cot,
            with_prompt=args.with_prompt,
            constrained_decoding=args.constrained_decoding,
            do_test=args.do_test,
            start_iters=start_iters,
            target_iters=t,
        )
        if start_iters >= target_iters:
            break
    print("Total iters:", start_iters)


def main(args, evaluator, take):
    assert os.path.exists("subject_mapping.json"), "subject_mapping.json not found!"
    with open("subject_mapping.json") as f:
        subject_mapping = json.load(f)
    filenames = os.listdir("data/val")
    subject_list = [val_file.replace("_val.csv", "") for val_file in filenames]
    accuracy, summary = {}, {}

    run_date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    output_dir = args.output_dir
    save_result_dir = os.path.join(output_dir, f"take{take}")
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir, exist_ok=True)

    all_answers = {}
    for index, subject_name in enumerate(subject_list):
        print(
            f"{index/len(subject_list)} Inference starts at {run_date} on {args.model_name_or_path} with subject of {subject_name}!"
        )
        val_file_path = os.path.join("data/val", f"{subject_name}_val.csv")
        dev_file_path = os.path.join("data/dev", f"{subject_name}_dev.csv")
        test_file_path = os.path.join("data/test", f"{subject_name}_test.csv")

        val_df = pd.read_csv(val_file_path) if args.do_test is False else pd.read_csv(test_file_path)
        dev_df = pd.read_csv(dev_file_path) if args.few_shot else None

        correct_ratio, answers = evaluator.eval_subject(
            subject_name,
            val_df,
            dev_df,
            save_result_dir=save_result_dir if args.do_save_csv else None,
            few_shot=args.few_shot,
            cot=args.cot,
            with_prompt=args.with_prompt,
            constrained_decoding=args.constrained_decoding,
            do_test=args.do_test,
        )
        print(f"Subject: {subject_name}")
        print(f"Acc: {correct_ratio}")
        accuracy[subject_name] = correct_ratio
        summary[subject_name] = {
            "score": correct_ratio,
            "num": len(val_df),
            "correct": correct_ratio * len(val_df) / 100,
        }
        all_answers[subject_name] = answers

    json.dump(all_answers, open(save_result_dir + "/submission.json", "w"), ensure_ascii=False, indent=4)
    print("Accuracy:")
    for k, v in accuracy.items():
        print(k, ": ", v)

    total_num = 0
    total_correct = 0
    summary["grouped"] = {
        "STEM": {"correct": 0.0, "num": 0},
        "Social Science": {"correct": 0.0, "num": 0},
        "Humanities": {"correct": 0.0, "num": 0},
        "Other": {"correct": 0.0, "num": 0},
    }
    for subj, info in subject_mapping.items():
        group = info[2]
        summary["grouped"][group]["num"] += summary[subj]["num"]
        summary["grouped"][group]["correct"] += summary[subj]["correct"]
    for group, info in summary["grouped"].items():
        info["score"] = info["correct"] / info["num"]
        total_num += info["num"]
        total_correct += info["correct"]
    summary["All"] = {"score": total_correct / total_num, "num": total_num, "correct": total_correct}

    json.dump(summary, open(save_result_dir + "/summary.json", "w"), ensure_ascii=False, indent=2)

    print(summary["All"])
    print("All done!")
    print(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--cot", choices=["False", "True"], default="False")
    parser.add_argument("--few_shot", choices=["False", "True"], default="True")
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--with_prompt", choices=["False", "True"], default="False")
    parser.add_argument("--constrained_decoding", choices=["False", "True"], default="True")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--n_times", default=1, type=int)
    parser.add_argument("--do_save_csv", choices=["False", "True"], default="False")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--do_test", choices=["False", "True"], default="False")
    parser.add_argument("--do_ptq", action="store_true")
    parser.add_argument("--ptq_iters", default=128, type=int)
    parser.add_argument("--do_shift", action="store_true")
    parser.add_argument("--shift_iters", default=2, type=int)
    parser.add_argument("--shift_all_linears", action="store_true")
    parser.add_argument("--shift_sampler", default=None)
    parser.add_argument("--do_smooth", action="store_true")
    parser.add_argument("--do_reorder", action="store_true")
    parser.add_argument("--smooth_all_linears", action="store_true")
    parser.add_argument("--smooth_sampler", default=None)
    parser.add_argument("--smooth_iters", default=128, type=int)
    parser.add_argument("--k_piece", default=1, type=int)
    parser.add_argument("--search_piece", action="store_true")
    parser.add_argument("--use_pw_search", action="store_true")
    parser.add_argument("--do_gptq", action="store_true")
    parser.add_argument("--gptq_iters", default=32, type=int)
    parser.add_argument("--do_int4", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", type=str)
    parser.add_argument("--tensor_parallel_degree", default=1, type=int)
    parser.add_argument("--do_awq", action="store_true")
    parser.add_argument("--do_autoclip", action="store_true")
    parser.add_argument("--weight_quant_method", default="groupwise")
    parser.add_argument("--group_size", default=128, type=int)

    args = parser.parse_args()

    args.cot = args.cot == "True"
    args.few_shot = args.few_shot == "True"
    args.with_prompt = args.with_prompt == "True"
    args.constrained_decoding = args.constrained_decoding == "True"
    args.do_test = args.do_test == "True"
    args.do_save_csv = args.do_save_csv == "True"
    if args.constrained_decoding is True:
        args.n_times = max(args.n_times, 1)
    print(args)

    if args.tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "mp_degree": args.tensor_parallel_degree,
        }
        # Set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": 1234}
        fleet.init(is_collective=True, strategy=strategy)

    evaluator = ModelEvaluator(
        choices=choices,
        k=args.ntrain,
        model_name_or_path=args.model_name_or_path,
        temperature=args.temperature,
        dtype=args.dtype,
        tensor_parallel_degree=args.tensor_parallel_degree,
    )

    if args.do_ptq or args.do_shift or args.do_smooth or args.do_gptq or args.do_int4 or args.do_awq:
        run_ptq(args, evaluator=evaluator)

    for i in range(args.n_times):
        main(args, evaluator=evaluator, take=i)
