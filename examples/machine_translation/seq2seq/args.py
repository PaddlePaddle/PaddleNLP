# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001,
                        help="learning rate for optimizer")

    parser.add_argument("--num_layers",
                        type=int,
                        default=1,
                        help="layers number of encoder and decoder")

    parser.add_argument("--hidden_size",
                        type=int,
                        default=100,
                        help="hidden size of encoder and decoder")

    parser.add_argument("--batch_size",
                        type=int,
                        help="batch size of each step")

    parser.add_argument("--max_epoch",
                        type=int,
                        default=12,
                        help="max epoch for the training")

    parser.add_argument("--max_len",
                        type=int,
                        default=50,
                        help="max length for source and target sentence")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.2,
                        help="drop probability")

    parser.add_argument("--init_scale",
                        type=float,
                        default=0.0,
                        help="init scale for parameter")

    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=5.0,
                        help="max grad norm for global norm clip")

    parser.add_argument("--log_freq",
                        type=int,
                        default=100,
                        help="The frequency to print training logs")

    parser.add_argument("--model_path",
                        type=str,
                        default='model',
                        help="model path for model to save")

    parser.add_argument("--infer_output_file",
                        type=str,
                        default='infer_output',
                        help="file name for inference output")

    parser.add_argument("--beam_size",
                        type=int,
                        default=10,
                        help="file name for inference")

    parser.add_argument("--device",
                        default="gpu",
                        choices=["gpu", "cpu", "xpu"],
                        help="Device selected for inference.")

    parser.add_argument("--init_from_ckpt",
                        type=str,
                        default=None,
                        help="The path of checkpoint to be loaded.")

    parser.add_argument(
        "--export_path",
        type=str,
        default=None,
        help="The output file prefix used to save the exported inference model."
    )

    args = parser.parse_args()
    return args
