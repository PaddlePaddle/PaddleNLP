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

import argparse
import os
import sys
import time

import numpy as np
import paddle.distributed.fleet as fleet

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../", "../")))

from ppfleetx.core.engine.inference_engine import InferenceEngine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", default=128, type=int, required=False, help="seq length of inputs")
    parser.add_argument("--iter", default=100, type=int, help="run iterations for timing")
    parser.add_argument("--mp_degree", default=1, type=int, help="")
    parser.add_argument("--model_dir", default="output", type=str, help="model directory")

    args = parser.parse_args()
    return args


def predict(engine, data, args):

    with engine._static_guard:
        for d, name in zip(data, engine.input_names()):
            handle = engine.predictor.get_input_handle(name)
            handle.copy_from_cpu(d)

        for _ in range(10):
            engine.predictor.run()
        engine.predictor.get_output_handle(engine.output_names()[0]).copy_to_cpu()

        start = time.perf_counter()
        for _ in range(args.iter):
            engine.predictor.run()
        end = time.perf_counter()
        print(f"batch {data.shape} run time: {1000 * (end - start) / args.iter}ms")

        return {name: engine.predictor.get_output_handle(name).copy_to_cpu() for name in engine.output_names()}


def main():

    args = parse_args()

    fleet.init(is_collective=True)
    infer_engine = InferenceEngine(args.model_dir, args.mp_degree)
    ids = [100] * args.seq_len

    # run test
    for batch in [1, 2, 4, 8, 16]:

        whole_data = [ids] * batch
        whole_data = np.array(whole_data, dtype="int64").reshape(1, batch, -1)

        _ = predict(infer_engine, whole_data, args)


if __name__ == "__main__":
    main()
