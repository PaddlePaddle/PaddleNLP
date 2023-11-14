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

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddlenlp.transformers import AutoTokenizer, GPTTokenizer, LlamaTokenizer
import numpy as np

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="The directory of model.")
    parser.add_argument("--model_prefix", type=str, default="infer", help="The model and params file prefix.")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Type of inference device, support 'cpu' or 'gpu'.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=50, help="The batch size of data.")
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
        self.tokenizer = LlamaTokenizer.from_pretrained(args.model_dir)
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
        infer_model_path = "./checkpoints/infer"

        config = paddle.inference.Config(
            infer_model_path + ".pdmodel", infer_model_path + ".pdiparams"
        )
        config.enable_memory_optim()
        config.switch_ir_optim(True)
        device_id = int(os.environ.get("FLAGS_selected_gpus", 0))
        config.enable_use_gpu(100, device_id)
        config.disable_glog_info()

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

    def preprocess(self, input_text, vit_image):
        inputs = self.tokenizer(
            input_text,
            padding=True,
            return_tensors="np",
            max_length=self.src_length,
            return_attention_mask=True,
            return_position_ids=True,
        )
        inputs={}
        inputs["pixel_values"] = vit_image.astype("float32")
        inputs["first_input_ids"] = np.array([[1]]).astype("int64")
        inputs["first_attention_mask"] = np.array([[1]]).astype("int64")
        # inputs["max_length"] = np.array([20]).astype("int64")
        return inputs

    def infer(self, inputs):
        input_handles = {}
        for name in self.predictor.get_input_names():
            input_handles[name] = self.predictor.get_input_handle(name)
            input_handles[name].copy_from_cpu(inputs[name])

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = output_handle.copy_to_cpu()
        # import pdb;pdb.set_trace()
        # print("outputs", results)
        # print("results", results.shape)

        output = []
        for i in range(100):
            if results[0][i] != 2:
                output.append(results[0][i])
            if results[0][i] == 2:
                break
        # print(output)
        msg = self.tokenizer.convert_tokens_to_string(np.array(output).tolist())
        print("Inference result: ", msg)

        return results



    def predict(self, texts):
        path_data = "./vit_numpy.npy"
        vit_data = np.load(path_data, allow_pickle=True)
        
        for i in range(222):
            input_map = self.preprocess(texts, vit_data[i])
            output = self.infer(input_map)
        return output


if __name__ == "__main__":
    args = parse_arguments()
    paddle.seed(100)
    predictor = Predictor(args)
    all_texts = [
        "My name is",
        "Today is"
    ]

    batch_texts = batchfy_text(all_texts, args.batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for text, result in zip(texts, outputs["result"]):
            print("========================")
            print("{} \n\n {}".format(text, result))