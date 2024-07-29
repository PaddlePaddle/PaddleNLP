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

import copy
from dataclasses import dataclass, field

import numpy
import paddle
from paddle.distributed import fleet
from ppo_trainer import Trainer, data_group_merge, data_group_split, group_rank_guard

from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
)


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    test_mode: str = field(default="export", metadata={"help": "export data_split or rank_guard."})


def test_group_rank_guard(group):
    @group_rank_guard(group=group, rank=0)
    def func():
        tensor = paddle.randn([4, 64])
        return tensor

    t = func()
    ret = []
    paddle.distributed.stream.all_gather(ret, t, group=group)

    for x in ret:
        assert x._md5sum() == t._md5sum(), f"{x} {t}"


def main():
    # Arguments
    parser = PdArgumentParser((ModelArgument, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    hcg = fleet.get_hybrid_communicate_group()
    pp_group = hcg.get_pipe_parallel_group()
    tp_group = hcg.get_model_parallel_group()

    if model_args.test_mode == "rank_guard":
        test_group_rank_guard(tp_group)
        return 0

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        tensor_parallel_output=False,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        dtype="float32",
    )

    model_class = AutoModelForCausalLM
    if training_args.pipeline_parallel_degree > 1:
        model_class = AutoModelForCausalLMPipe

    actor_model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
    )

    if True:  # test export_evaluate_model
        # 随机初始化
        config = copy.deepcopy(model_config)
        if training_args.pipeline_parallel_degree <= 1:
            config.tensor_parallel_degree = -1
            config.tensor_parallel_rank = 0

        actor_eval_model = AutoModelForCausalLM.from_config(config)
        # ground truth模型
        actor_gt_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config)

        trainer = Trainer(
            model=actor_model,
            args=training_args,
        )
        trainer.export_evaluate_model(actor_model, actor_eval_model)

        gp_state = actor_gt_model.state_dict()
        export_state = actor_eval_model.state_dict()

        for k, v in gp_state.items():
            assert (
                v._md5sum() == export_state[k]._md5sum()
            ), f"{k} groud_truth: {v.shape}, export: {export_state[k].shape}"

        split_group = tp_group
        if training_args.pipeline_parallel_degree > 1:
            split_group = pp_group

        input_ids = paddle.randint(low=1, high=50, shape=[8, 64])
        paddle.distributed.broadcast(input_ids, src=0)

        split_input_ids = data_group_split(input_ids, group=split_group)
        ret = actor_eval_model(input_ids=split_input_ids, return_dict=True)
        eval_loggits = data_group_merge(ret.logits, group=split_group)

        gt_ret = actor_gt_model(input_ids=input_ids, return_dict=True)
        gt_loggits = gt_ret.logits
        numpy.testing.assert_almost_equal(eval_loggits.numpy(), gt_loggits.numpy(), decimal=5)


if __name__ == "__main__":
    main()
