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

import custom_setup_ops
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from utils import dybatch_preprocess, get_infer_model_path, load_real_time_tokens
import time
from paddlenlp.transformers import AutoTokenizer, LlamaConfig


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="The directory of model.")
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="llama",
        help="The model and params file prefix.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Type of inference device, support 'cpu' or 'gpu'.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=1024, help="The batch size of data.")
    parser.add_argument("--tgt_length", type=int, default=100, help="The batch size of data.")
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
        self.args = args

        if dist.get_world_size() > 1:
            init_dist_env(dist.get_world_size())
            self.nranks = fleet.worker_num()
            self.rank = fleet.worker_index()
        else:
            self.nranks = 1
            self.rank = 0

        self.config = LlamaConfig.from_pretrained(args.model_dir)
        self.config.tensor_parallel_degree = self.nranks

        self.cache_kvs = []
        for _ in range(self.config.num_hidden_layers):
            self.cache_kvs.append(
                paddle.cast(
                    paddle.to_tensor(
                        np.zeros(
                            (
                                2,
                                args.batch_size,
                                self.config.num_attention_heads // self.nranks,
                                args.src_length + args.tgt_length + 1,
                                self.config.hidden_size // self.config.num_attention_heads,
                            ),
                            dtype="float32",
                        )
                    ),
                    "float16",
                )
            )
        self.pre_ids = paddle.to_tensor(np.full((args.batch_size, args.tgt_length + 1), -1, dtype="int64"))
        self.attention_mask = paddle.zeros(
            shape=(args.batch_size, 1, args.src_length, args.src_length),
            dtype="float16",
        )
        self.tgt_generation_mask = paddle.zeros(
            shape=[args.batch_size, 1, 1, args.src_length + args.tgt_length + 1],
            dtype="float16",
        )

        self.predictor = self.create_predictor(args)

    def create_predictor(self, args):
        infer_model_path = get_infer_model_path(args.model_dir, args.model_prefix)

        config = paddle.inference.Config(infer_model_path + ".pdmodel", infer_model_path + ".pdiparams")
        # config.enable_memory_optim()
        config.switch_ir_optim(False)
        device_id = int(os.environ.get("FLAGS_selected_gpus", 0))
        config.enable_use_gpu(100, device_id)
        # config.disable_glog_info()

        if self.nranks > 1:
            trainer_endpoints = fleet.worker_endpoints()
            current_endpoint = trainer_endpoints[self.rank]

            dist_config = config.dist_config()
            dist_config.set_ranks(self.nranks, self.rank)
            dist_config.set_endpoints(trainer_endpoints, current_endpoint)
            dist_config.enable_dist_model(True)

            dist_config.set_comm_init_config(os.path.join(args.model_dir, "rank_mapping.csv"))
            config.set_dist_config(dist_config)

        predictor = paddle.inference.create_predictor(config)
        return predictor

    def preprocess(self, input_text):
        inputs = dybatch_preprocess(self.tokenizer, input_text, self.config, self.args)
        # print(inputs["input_ids"].shape)
        for i in range(inputs["input_ids"].shape[0]):
            length = inputs["seq_len_encoder"][i][0]
            self.attention_mask[i, 0, :length, :length] = paddle.tril(
                paddle.ones(shape=(length, length), dtype="float16")
            )
            self.tgt_generation_mask[i, 0, 0, :length] = paddle.ones(shape=[1, length], dtype="float16")
        inputs["attention_mask"] = self.attention_mask
        inputs["tgt_generation_mask"] = self.tgt_generation_mask
        return inputs

    def infer(self, inputs):
        for k, v in inputs.items():
            input_tensor = self.predictor.get_input_handle(k)
            if "mask" in k or "position" in k:
                input_tensor.share_external_data(v)
            else:
                input_tensor.copy_from_cpu(v)
        for i in range(self.config.num_hidden_layers):
            input_tensor = self.predictor.get_input_handle("cache_kvs_" + str(i))
            input_tensor.share_external_data(self.cache_kvs[i])
        input_tensor = self.predictor.get_input_handle("pre_ids")
        input_tensor.share_external_data(self.pre_ids)

        self.predictor.run()

    def postprocess(self, infer_data):
        if paddle.distributed.get_rank() == 0:
            tokens = load_real_time_tokens()
            result = []
            for x in tokens.tolist():
                # print("Out shape is: ", len(x)) 
                res = self.tokenizer.decode(x, skip_special_tokens=True)
                result.append(res)
            out_dict = {"result": result}
        else:
            out_dict = {"result": "not first rank"}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


if __name__ == "__main__":
    args = parse_arguments()
    paddle.seed(100)
    predictor = Predictor(args)
    all_texts = [
        "My name is",
        "Today is",
    ]

    src_length = args.src_length
    warmup_times = 2 
    test_times = 10
    print("Src length is: ", src_length)
    # infer_dials = ["My name is " + "<PAD>" * (src_length-3)]
    # infer_dials = ["My " * src_length]
    # infer_dials = ["My name is My name is My name is My name is My name is My name is My name is My name is My name is My name is My name is My name is My name is My name is My name is "]
    
    test_text = "<pad>"* (src_length // 2 - 3) + "My name is "
    # test_text = "My name is"
    infer_dials = [test_text for _ in range(args.batch_size)]

    print(infer_dials)
    for _ in range(warmup_times): 
        for idx in range(0, len(infer_dials), args.batch_size):
            batch_dials = infer_dials[idx : idx + args.batch_size]
            # print("Batch dials is: ", batch_dials)
            batch_outputs = predictor.predict(batch_dials)
            # print("Batch outputs is: ", batch_outputs)
    

    start_time = time.time()
    for _ in range(test_times): 
        each_step_start_time = time.time()
        for idx in range(0, len(infer_dials), args.batch_size): 
            batch_dials = infer_dials[idx : idx + args.batch_size]
            batch_outputs = predictor.predict(batch_dials)
        print(f"Average step {_}, elapse {(time.time() - each_step_start_time)}")
        
    print(f"Average step elapse {(time.time() - start_time) / test_times}")

