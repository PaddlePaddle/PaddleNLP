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
import time

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddlenlp.transformers import AutoTokenizer



def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", default="./chatglm_2048/", help="The directory of model."
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="chatglm",
        help="The model and params file prefix.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Type of inference device, support 'cpu' or 'gpu'.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The batch size of data."
    )
    parser.add_argument(
        "--src_length", type=int, default=1024, help="The batch size of data."
    )
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


def init_dist_env(world_size, seed=20):
    # start to init distributed env
    strategy = fleet.DistributedStrategy()

    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": world_size,
        "pp_degree": 1,
        "sharding_degree": 1,
    }

    # Set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": seed}

    fleet.init(is_collective=True, strategy=strategy)


class Predictor(object):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.batch_size = args.batch_size
        self.src_length = args.src_length

        if dist.get_world_size() > 1:
            init_dist_env(dist.get_world_size())
            self.nranks = fleet.worker_num()
            self.rank = fleet.worker_index()
        else:
            self.nranks = 1
            self.rank = 0

        self.predictor = self.create_predictor(args)

    def create_predictor(self, args):
        infer_model_path = os.path.join(args.model_dir, args.model_prefix)

        config = paddle.inference.Config(
            infer_model_path + ".pdmodel", infer_model_path + ".pdiparams"
        )
        config.enable_memory_optim()
        config.switch_ir_optim(True)
        device_id = int(os.environ.get("FLAGS_selected_gpus", 0))
        config.enable_use_gpu(1000, device_id)
        # config.disable_glog_info()

        if self.nranks > 1:
            trainer_endpoints = fleet.worker_endpoints()
            current_endpoint = trainer_endpoints[self.rank]

            dist_config = config.dist_config()
            dist_config.set_ranks(self.nranks, self.rank)
            dist_config.set_endpoints(trainer_endpoints, current_endpoint)
            dist_config.enable_dist_model(True)

            dist_config.set_comm_init_config(
                os.path.join(args.model_dir, "rank_mapping.csv")
            )
            config.set_dist_config(dist_config)

        predictor = paddle.inference.create_predictor(config)
        return predictor

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            padding=True,
            max_length=self.src_length,
            truncation=True,
            truncation_side="left",
        )
        use_pre_caches = True

        import numpy as np
        if use_pre_caches:
            pre_caches_numpy = np.load("./prefix_tuning/pre_caches.npy")
            pre_caches = np.split(pre_caches_numpy, 28)
            for i in range(28):
                inputs["pre_cache_{}".format(i)] = pre_caches[i].transpose(1, 0, 2, 3, 4).astype("float16")
        else:
            for i in range(28):
                inputs["pre_cache_{}".format(i)] = np.ones([1]).astype("float16")
        

        input_ids_shape = inputs["input_ids"].shape
        prefix_attention_mask = np.zeros(
            [input_ids_shape[0], 1, input_ids_shape[-1], 64], dtype="int64"
        )
        inputs["use_pre_caches"] = np.array([use_pre_caches])
        if use_pre_caches:
            inputs["attention_mask"] = np.concatenate((prefix_attention_mask, inputs["attention_mask"]), axis=3)
        
        return inputs

    def infer(self, inputs):
        input_handles = {}
        for i in range(3):
            start = time.perf_counter()
            for name in self.predictor.get_input_names():
                input_handles[name] = self.predictor.get_input_handle(name)
                input_handles[name].copy_from_cpu(inputs[name])

            self.predictor.run()
            output_names = self.predictor.get_output_names()
            output_handle = self.predictor.get_output_handle(output_names[0])
            results = output_handle.copy_to_cpu()
            hf_cost = (time.perf_counter() - start) * 1000
            print("Speed Paddle:", hf_cost)


        return results

    def postprocess(self, infer_data):
        result = []
        for x in infer_data.tolist():
            text = self.tokenizer.decode(x, skip_special_tokens=True)
            result.append(text)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        print(infer_result.shape)
        output = self.postprocess(infer_result)
        return output


if __name__ == "__main__":
    args = parse_arguments()
    paddle.seed(100)
    predictor = Predictor(args)
    all_texts = [
        # "你好",
        # "小明有15本漫画书，他每天阅读3本。请问他可以连续阅读几天？",
        "[Round 1]\n问:你需要根据以下任务中的描述进行角色扮演，你只能以任务角色的身份应答，而不是语言模型。\n\n任务：大模型应用助手\n\n请基于以下已知信息回答我的问题，不允许进行编造与作假，如仍无法回答，请说你不知道如何作答。\n\n\n我的问题：你是谁？\n答:我是大模型应用助手,一名由清华大学 KEG 实验室和智谱AI训练的大型语言模型。我被设计用于回答用户提出的问题,并提供有用的信息和建议。\n[Round 2]\n问:请介绍一下你自己\n答:我是一个大型语言模型,被训练用于回答用户提出的问题。我可以通过分析大量的文本数据来学习语言模式和知识,并为用户提供相关的信息和建议。我可以回答各种各样的问题,例如学术、技术、娱乐、健康等方面的问题。\n[Round 3]\n问:chatglm好还是baichuan好\n答:"
    ]

    batch_texts = batchfy_text(all_texts, args.batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for text, result in zip(texts, outputs["result"]):
            print("========================")
            print("{} \n\n {}".format(text, result))
