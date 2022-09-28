#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

    parser.add_argument("--dataset",
                        type=str,
                        help="Dataset name. Now ptb|yahoo is supported.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001,
                        help="Learning rate of optimizer.")

    parser.add_argument("--num_layers",
                        type=int,
                        default=1,
                        help="The number of layers of encoder and decoder.")

    parser.add_argument("--embed_dim",
                        type=int,
                        default=256,
                        help="Embedding dim of encoder and decoder.")

    parser.add_argument("--hidden_size",
                        type=int,
                        default=256,
                        help="Hidden size of encoder and decoder.")

    parser.add_argument("--latent_size",
                        type=int,
                        default=32,
                        help="Latent size of Variational Auto Encoder.")

    parser.add_argument("--batch_size", type=int, help="Batch size.")

    parser.add_argument("--max_epoch",
                        type=int,
                        default=20,
                        help="Max epoch of training.")

    parser.add_argument("--max_len",
                        type=int,
                        default=1280,
                        help="Max length of source and target sentence.")

    parser.add_argument("--log_freq",
                        type=int,
                        default=200,
                        help="Log frequency")

    parser.add_argument("--dec_dropout",
                        type=float,
                        default=0.5,
                        help="Drop probability of decoder")

    parser.add_argument("--enc_dropout",
                        type=float,
                        default=0.,
                        help="Drop probability of encoder.")

    parser.add_argument("--init_scale",
                        type=float,
                        default=0.0,
                        help="Init scale for parameter.")

    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=5.0,
                        help="Max grad norm of global norm clip.")

    parser.add_argument("--model_path",
                        type=str,
                        default='model',
                        help="Model path for model to save.")

    parser.add_argument("--infer_output_file",
                        type=str,
                        default='infer_output.txt',
                        help="File name to save inference output.")

    parser.add_argument("--beam_size",
                        type=int,
                        default=1,
                        help="Beam size for Beam search.")

    parser.add_argument("--device",
                        default="gpu",
                        choices=["gpu", "cpu", "xpu"],
                        help="Device selected for inference.")

    parser.add_argument("--warm_up",
                        type=int,
                        default=10,
                        help='The number of warm up epoch for KL.')

    parser.add_argument("--kl_start",
                        type=float,
                        default=0.1,
                        help='KL start value, up to 1.0.')

    parser.add_argument("--init_from_ckpt",
                        type=str,
                        default=None,
                        help="The path of checkpoint to be loaded.")

    args = parser.parse_args()
    return args
