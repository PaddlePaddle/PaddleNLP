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
import sys

runtime = "pd"
sys.path.insert(
    0,
    "/root/paddlejob/workspace/guosheng/share/safe-rlhf/"
    if runtime == "th"
    else "/root/paddlejob/workspace/guosheng/share/paddlenlp-rlhf/PaddleNLP/",
)

# os.environ["PYTHONPATH"] = (
#     "/root/paddlejob/workspace/guosheng/share/safe-rlhf/" if runtime == "th"
#     else "/root/paddlejob/workspace/guosheng/share/paddlenlp-rlhf/PaddleNLP/")


from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
import paddle

# import torch
from data import parse_dataset
from paddle.utils import map_structure
from tqdm import tqdm

from paddlenlp.trainer import (  # , get_last_checkpoint
    PdArgumentParser,
    TrainingArguments,
)

# torch.backends.cuda.matmul.allow_tf32 = False
# os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


model_path_dict = {
    "sft": {
        "th": "PKU-Alignment/alpaca-7b-reproduced",
        # "th":
        # "/root/paddlejob/workspace/guosheng/ppo-out/ppo-pdcpumul-distseed-newlen/",
        # "th":
        # "/root/paddlejob/workspace/guosheng/out/ppo-pd-mul-cpuseed-klcoe/",
        # "pd": "/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced",
        "pd": "/root/paddlejob/workspace/guosheng/checkpoints/llama_ppo_ckpts-test-fixclip/policy/checkpoint-219/",
        # "pd":
        # "/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced_safetensor"
    },
    "reward": {
        "th": "PKU-Alignment/beaver-7b-v1.0-reward",
        # "pd": "/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced",
        # "pd": "/root/paddlejob/workspace/guosheng/beaver-7b-v1.0-reward-saved"
        "pd": "/root/paddlejob/workspace/guosheng/beaver-7b-v1.0-reward-saved_safetensor",
    },
}

# launch would unset http_proxy
# export https_proxy=http://172.19.57.45:3128
# os.environ["http_proxy"] = "http://172.19.56.199:3128"
# os.environ["https_proxy"] = "http://172.19.56.199:3128"
# os.environ["http_proxy"] = "http://172.19.57.45:3128"
# os.environ["https_proxy"] = "http://172.19.57.45:3128"
os.environ["http_proxy"] = "http://10.162.37.16:8128"
os.environ["https_proxy"] = "http://10.162.37.16:8128"


@dataclass
class ModelArgument:
    actor_model_name_or_path: str = field(
        default=None, metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    reward_model_name_or_path: str = field(
        default=None, metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    reward_critic_model_name_or_path: str = field(
        default=None, metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    temperature: float = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."},
    )
    top_k: int = field(
        default=1,
        metadata={"help": "top_k"},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to`top_p` or higher are kept for generation."
        },
    )
    num_return_sequences: int = field(
        default=1,
        metadata={"help": "The number of independently computed returned sequences for each element in the batch."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."},
    )
    # seed: int = field(
    #     default=42,
    #     metadata={"help": "A seed for reproducible evaluation."},
    # )
    # runtime: str = field(
    #     default="th",
    #     metadata={
    #         "help":
    #         "Build-in pretrained model name or the path to local model."
    #     })


@dataclass
class DataArgument:
    eval_datasets: str = field(default=None, metadata={"help": "Dataset name(s) registered in the raw dataset."})
    max_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum length that model input tokens can have. When intokens is set to True, it's also the maximum length for InTokens data stream"
        },
    )

    @property
    def parsed_eval_datasets(self) -> Tuple[str, Dict[str, Any]]:
        """Parse dataset path and its proportion and optionally additional arguments from `eval_datasets`."""
        return [parse_dataset(string) for string in self.eval_datasets.split(",")]


def get_common_reward_model(model_args, training_args, model_cls):
    AutoConfig, LlamaModelForScore, AutoTokenizer, LlamaTokenizer, GenerationConfig = model_cls
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
        model_args.reward_model_name_or_path,
        tensor_parallel_output=False,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        dtype=dtype,
    )
    if hasattr(model_config, "use_flash_attention"):
        model_config.use_flash_attention = model_args.use_flash_attention
    reward_model = LlamaModelForScore.from_pretrained(
        model_args.reward_model_name_or_path,
        config=model_config,
        score_type="reward",
        do_normalize=False,
    )
    reward_model.eval()
    if hasattr(reward_model, "cuda"):
        reward_model.cuda()
    reward_tokenizer = AutoTokenizer.from_pretrained(
        model_args.reward_model_name_or_path, model_max_length=model_args.max_length, padding_side="right"
    )
    # save as safetensor for later speedup
    # reward_model.save_pretrained(model_args.reward_model_name_or_path +
    #                              "_safetensor",
    #                              safe_serialization=True)
    # reward_tokenizer.save_pretrained(model_args.reward_model_name_or_path +
    #                                  "_safetensor")
    return reward_model, reward_tokenizer, None


def get_th_reward_model(model_args, training_args):
    import safe_rlhf.models.score_model.llama.modeling_llama as modeling_llama
    import torch
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig, LlamaTokenizer

    device = torch.device("cuda", training_args.local_rank if training_args.local_rank > 0 else 0)
    torch.cuda.set_device(device)
    return get_common_reward_model(
        model_args,
        training_args,
        (AutoConfig, modeling_llama.LlamaModelForScore, AutoTokenizer, LlamaTokenizer, GenerationConfig),
    )


def get_pd_reward_model(model_args, training_args):
    import paddle

    from paddlenlp.generation import GenerationConfig
    from paddlenlp.transformers import (
        AutoConfig,
        AutoTokenizer,
        LlamaModelForScore,
        LlamaTokenizer,
    )

    paddle.set_device(training_args.device)
    return get_common_reward_model(
        model_args, training_args, (AutoConfig, LlamaModelForScore, AutoTokenizer, LlamaTokenizer, GenerationConfig)
    )


def get_common_model(model_args, training_args, model_cls):
    AutoConfig, LlamaForCausalLM, AutoTokenizer, LlamaTokenizer, GenerationConfig = model_cls
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
        model_args.actor_model_name_or_path,
        tensor_parallel_output=False,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        dtype=dtype,
    )
    if hasattr(model_config, "use_flash_attention"):
        model_config.use_flash_attention = model_args.use_flash_attention
    actor_model = LlamaForCausalLM.from_pretrained(
        model_args.actor_model_name_or_path,
        config=model_config,
    )
    actor_model.eval()
    if hasattr(actor_model, "cuda"):
        actor_model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.actor_model_name_or_path, model_max_length=model_args.max_length, padding_side="left"
    )
    if isinstance(tokenizer, LlamaTokenizer):
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        # to be consistent with PKU-Alignment/alpaca-7b-reproduced
        tokenizer.pad_token_id = 32000  # tokenizer.eos_token_id
    generation_config = GenerationConfig(
        max_length=model_args.max_length,
        num_return_sequences=model_args.num_return_sequences,
        temperature=model_args.temperature,
        top_p=model_args.top_p,
        # top_k=model_args.top_k,
        repetition_penalty=model_args.repetition_penalty,
        do_sample=True,
        trunc_input=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # print("=" * 20, "lm_head.weight after create", actor_model.lm_head.weight,
    #       actor_model.lm_head.weight.dtype)
    # save as safetensor for later speedup
    # actor_model.save_pretrained(
    #     model_args.actor_model_name_or_path + "_safetensor",
    #     safe_serialization=True)
    # tokenizer.save_pretrained(model_args.actor_model_name_or_path +
    #                           "_safetensor")
    return actor_model, tokenizer, generation_config


def get_pd_model(model_args, training_args):
    from paddlenlp.generation import GenerationConfig
    from paddlenlp.transformers import (
        AutoConfig,
        AutoTokenizer,
        LlamaForCausalLM,
        LlamaTokenizer,
    )

    paddle.set_device(training_args.device)
    return get_common_model(
        model_args, training_args, (AutoConfig, LlamaForCausalLM, AutoTokenizer, LlamaTokenizer, GenerationConfig)
    )


def get_th_model(model_args, training_args):
    import torch
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        GenerationConfig,
        LlamaForCausalLM,
        LlamaTokenizer,
    )

    device = torch.device("cuda", training_args.local_rank if training_args.local_rank > 0 else 0)
    torch.cuda.set_device(device)
    paddle.set_device("cpu")
    return get_common_model(
        model_args, training_args, (AutoConfig, LlamaForCausalLM, AutoTokenizer, LlamaTokenizer, GenerationConfig)
    )


def get_pd_dataloader(data_args, training_args, tokenizer, data_type=None):
    from paddle.io import DataLoader, DistributedBatchSampler

    if data_type is None:
        from data import PromptOnlyDataset

        dataset = PromptOnlyDataset(data_args.parsed_eval_datasets, tokenizer=tokenizer)
    else:
        import data

        dataset = getattr(data, data_type)(data_args.parsed_eval_datasets, tokenizer)

    if training_args.local_rank < 0:
        batch_sampler = paddle.io.BatchSampler(
            dataset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False,
        )
    else:
        batch_sampler = DistributedBatchSampler(
            dataset,
            num_replicas=training_args.dataset_world_size,
            rank=training_args.dataset_rank,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False,
        )

    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.get_collator(shift=True) if data_type == "SupervisedDataset" else dataset.get_collator(),
        batch_sampler=batch_sampler,
    )
    return dataloader


def get_th_dataloader(data_args, training_args, tokenizer, data_type=None):
    from torch.utils.data import DataLoader  # , DistributedSampler

    if data_type is None:
        from safe_rlhf.datasets import PromptOnlyDataset

        dataset = PromptOnlyDataset(data_args.parsed_eval_datasets, tokenizer)
    else:
        import safe_rlhf.datasets as data

        dataset = getattr(data, data_type)(data_args.parsed_eval_datasets, tokenizer)

    from paddle.io import BatchSampler, DistributedBatchSampler

    if training_args.local_rank < 0:
        # sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
        batch_sampler = BatchSampler(
            dataset=dataset, shuffle=False, batch_size=training_args.per_device_eval_batch_size, drop_last=False
        )
    else:
        # sampler = DistributedSampler(dataset, shuffle=False)
        batch_sampler = DistributedBatchSampler(
            dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=False,
            num_replicas=training_args.dataset_world_size,
            rank=training_args.dataset_rank,
            drop_last=training_args.dataloader_drop_last,
        )
    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.get_collator(),
        # sampler=sampler,
        # batch_size=training_args.per_device_eval_batch_size,
        batch_sampler=batch_sampler,
    )
    return dataloader


# input_ids = [[
#     32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#     32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#     32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#     32000, 32000, 32000, 32000, 32000, 1, 22815, 29940, 4214, 8079, 8707, 5348,
#     29903, 8098, 29901, 3148, 1001, 29901, 360, 1151, 29892, 4856, 5456, 383,
#     10051, 29954, 2891, 29903, 4822, 278, 10090, 310, 263, 23852, 2594, 319,
#     1799, 9047, 13566, 29901
# ],
#           [
#               1, 22815, 29940, 4214, 8079, 8707, 5348, 29903, 8098, 29901,
#               3148, 1001, 29901, 306, 29915, 29885, 23407, 310, 5146, 292,
#               23990, 577, 306, 29915, 345, 8459, 304, 10674, 271, 297, 263,
#               20810, 5260, 28402, 3787, 1363, 372, 756, 20947, 310, 9687, 363,
#               592, 304, 17545, 746, 278, 3787, 29915, 29879, 5764, 29889, 1128,
#               508, 306, 10674, 271, 297, 263, 5214, 1728, 5019, 451, 18499,
#               29973, 319, 1799, 9047, 13566, 29901
#           ],
#           [
#               32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#               32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#               32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#               32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#               32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 1,
#               22815, 29940, 4214, 8079, 8707, 5348, 29903, 8098, 29901, 3148,
#               1001, 29901, 2106, 263, 1051, 310, 2600, 1662, 3838, 319, 1799,
#               9047, 13566, 29901
#           ],
#           [
#               32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#               32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#               32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
#               32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 1,
#               22815, 29940, 4214, 8079, 8707, 5348, 29903, 8098, 29901, 3148,
#               1001, 29901, 306, 29915, 29885, 13858, 2534, 590, 8750, 9942,
#               29883, 3368, 29889, 1724, 437, 366, 1348, 29973, 319, 1799, 9047,
#               13566, 29901
#           ]]
# input_ids = np.array(input_ids, dtype=np.int64)
# attn_mask = input_ids != 32000
# inputs = {"input_ids": input_ids, "attention_mask": attn_mask}


def to_pd_tensor(data):
    import paddle

    return map_structure(lambda x: paddle.to_tensor(x.numpy() if not isinstance(x, np.ndarray) else x).cuda(), data)


def to_th_tensor(data):
    import torch

    # data = inputs
    return map_structure(lambda x: torch.from_numpy(x.numpy() if not isinstance(x, np.ndarray) else x).cuda(), data)


def get_pd_nograd():
    import paddle

    return paddle.no_grad


def get_th_nograd():
    import torch

    return torch.no_grad


local_vars = locals()
get_model = local_vars[f"get_{runtime}_model"]
get_dataloader = local_vars[f"get_{runtime}_dataloader"]  # get_th_dataloader
to_tensor = local_vars[f"to_{runtime}_tensor"]  # to_th_tensor
get_no_grad = local_vars[f"get_{runtime}_nograd"]  # get_th_nograd
get_reward_model = local_vars[f"get_{runtime}_reward_model"]  # get_th_reward_model
# get_model = get_th_model
# get_dataloader = get_th_dataloader
# to_tensor = to_th_tensor
# get_no_grad = get_th_nograd
# get_reward_model = get_th_reward_model


def get_prefer_score(model, batch):
    higher_end_rewards = model(
        batch["better_input_ids"], attention_mask=batch["better_attention_mask"], return_dict=True
    ).end_scores.squeeze(dim=-1)
    lower_end_rewards = model(
        batch["worse_input_ids"], attention_mask=batch["worse_attention_mask"], return_dict=True
    ).end_scores.squeeze(dim=-1)
    return higher_end_rewards, lower_end_rewards


def seed_everything(seed: int) -> None:
    """Set global random seed for reproducibility."""
    import random

    import numpy as np
    import paddle
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    paddle.seed(seed)
    # core = paddle.framework.core
    # core.default_cuda_generator(0).manual_seed(seed * 2)
    # paddle.framework.core.default_cpu_generator().manual_seed(seed * 2)


def main():
    # Arguments
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     model_args, data_args, training_args = parser.parse_json_file(
    #         json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses(
    #     )
    arg_dict = {
        # "eval_datasets": "PKU-SafeRLHF/test",
        "eval_datasets": "alpaca",
        "actor_model_name_or_path":
        # "/root/paddlejob/workspace/guosheng/out/ppo-stage3-ep2/",
        # "/root/paddlejob/workspace/guosheng/out/ppo-stage3/",
        # "PKU-Alignment/beaver-7b-v1.0",
        # "PKU-Alignment/alpaca-7b-reproduced",
        # "/root/paddlejob/workspace/guosheng/alpaca-7b-reproduced/",
        model_path_dict["sft"][runtime],
        "reward_model_name_or_path": None,  # model_path_dict["reward"][runtime],
        "max_length": 512,
        "temperature": 1.0,
        "num_return_sequences": 1,
        "repetition_penalty": 1.0,
        "per_device_eval_batch_size": 4,
        "output_dir": "eval_output",
        "seed": 42,
    }
    model_args, data_args, training_args = parser.parse_dict(arg_dict)
    model_args.max_length = data_args.max_length
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    seed_everything(training_args.seed)

    # Load model
    model, tokenizer, generation_config = get_model(model_args, training_args)
    print("=" * 20, "lm_head.weight after create", model.lm_head.weight, model.lm_head.weight.dtype)
    print("=" * 20, model.get_decoder().layers[0].mlp.down_proj.weight)
    print("=" * 20, model.get_decoder().layers[-1].mlp.down_proj.weight)
    if model_args.reward_model_name_or_path:
        reward_model, reward_tokenizer, _ = get_reward_model(model_args, training_args)
    # exit(0)
    # print("=" * 20, "weight dtype", model.get_decoder().layers[0].mlp.gate_proj.weight.dtype)
    # dataloader = get_dataloader(data_args, training_args, tokenizer)
    dataloader = get_dataloader(data_args, training_args, tokenizer, data_type="PromptOnlyDataset")
    # dataloader = get_dataloader(data_args,
    #                             training_args,
    #                             tokenizer,
    #                             data_type="PreferenceDataset")
    # dataloader = get_dataloader(data_args,
    #                             training_args,
    #                             tokenizer,
    #                             data_type="SupervisedDataset")

    out_file = os.path.join(training_args.output_dir, f"generated-{runtime}-tmp.txt")
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    f = open(out_file, "w")

    for i, batch in enumerate(
        tqdm(
            dataloader,
            desc="Evaluating",
            disable=False,
        ),
        start=1,
    ):
        batch = to_tensor(batch)
        with get_no_grad()():
            # out = model(**batch, return_dict=True).loss
            # print(out)
            # exit(0)
            output_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                synced_gpus=False,
                generation_config=generation_config,
            )
            output_ids = output_ids[0] if isinstance(output_ids, tuple) else output_ids
            print(output_ids.shape, output_ids)

            if model_args.reward_model_name_or_path:
                attention_mask = (output_ids != tokenizer.pad_token_id) & (output_ids != tokenizer.unk_token_id)
                reward_output_ids = output_ids
                reward_attention_mask = attention_mask
                reward_score = reward_model(
                    reward_output_ids, attention_mask=reward_attention_mask, return_dict=True
                ).end_scores.squeeze(-1)
            else:
                reward_score = [None] * len(output_ids)

            gathered_output_ids = output_ids
            sentences = tokenizer.batch_decode(gathered_output_ids, skip_special_tokens=True)
            for seq, reward in zip(sentences, reward_score):
                print(seq)
                print(reward)
                f.write(seq + "\n")
                # f.write(f"REWARD: {reward if reward is None else reward.item()}" + "\n")

            # assert model_args.reward_model_name_or_path is not None
            # higher_end_rewards, lower_end_rewards = get_prefer_score(
            #     reward_model, batch)
            # better_res = tokenizer.batch_decode(batch['better_input_ids'],
            #                        skip_special_tokens=True)
            # worse_res = tokenizer.batch_decode(batch['worse_input_ids'],
            #                                    skip_special_tokens=True)
            # for b_s, w_s, h_r, l_r in zip(better_res, worse_res, higher_end_rewards,
            #                     lower_end_rewards):
            #     h = h_r.item()
            #     l = l_r.item()
            #     s = f"higher_end_rewards: {h}, lower_end_rewards: {l}, diff: {h - l}"
            #     print(s)
            #     f.write("BETTER_INPUTS: " + b_s + "\n")
            #     f.write("WORSE_INPUTS: " + w_s + "\n")
            #     f.write(s + "\n")

            if i == 1:
                break

    f.close()


if __name__ == "__main__":
    main()
